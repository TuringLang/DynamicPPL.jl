struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
DynamicPPL.maybe_invlink_before_eval!!(vi::OnlyAccsVarInfo, ::Model) = vi
DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) = OnlyAccsVarInfo(accs)

struct RangeAndLinked
    # indices that the variable corresponds to in the vectorised parameter
    range::UnitRange{Int}
    # whether it's linked
    is_linked::Bool
end

struct FastLDFContext{N<:NamedTuple,T<:AbstractVector{<:Real}} <: AbstractContext
    # The ranges of identity VarNames are stored in a NamedTuple for improved performance
    # (it's around 1.5x faster).
    iden_varname_ranges::N
    # This Dict stores the ranges for all other VarNames
    varname_ranges::Dict{VarName,RangeAndLinked}
    # The full parameter vector which we index into to get variable values
    params::T
end
DynamicPPL.NodeTrait(::FastLDFContext) = IsLeaf()
function get_range_and_linked(
    ctx::FastLDFContext, ::VarName{sym,typeof(identity)}
) where {sym}
    return ctx.iden_varname_ranges[sym]
end
function get_range_and_linked(ctx::FastLDFContext, vn::VarName)
    return ctx.varname_ranges[vn]
end

function tilde_assume!!(
    ctx::FastLDFContext, right::Distribution, vn::VarName, vi::OnlyAccsVarInfo
)
    # Don't need to read the data from the varinfo at all since it's
    # all inside the context.
    range_and_linked = get_range_and_linked(ctx, vn)
    y = @view ctx.params[range_and_linked.range]
    f = if range_and_linked.is_linked
        from_linked_vec_transform(right)
    else
        from_vec_transform(right)
    end
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end

function tilde_observe!!(
    ::FastLDFContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::OnlyAccsVarInfo,
)
    # This is the same as for DefaultContext
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end

struct FastLDF{
    M<:Model,
    F<:Function,
    N<:NamedTuple,
    AD<:Union{ADTypes.AbstractADType,Nothing},
    ADP<:Union{Nothing,DI.GradientPrep},
}
    _model::M
    _getlogdensity::F
    # See FastLDFContext for explanation of these two fields
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
    _adtype::AD
    _adprep::ADP

    function FastLDF(
        model::Model,
        getlogdensity::Function=getlogjoint_internal,
        # This only works with typed Metadata-varinfo.
        # Obviously, this can be generalised later.
        varinfo::VarInfo{<:NamedTuple{syms}}=VarInfo(model);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    ) where {syms}
        # Figure out which variable corresponds to which index, and
        # which variables are linked.
        all_iden_ranges = NamedTuple()
        all_ranges = Dict{VarName,RangeAndLinked}()
        offset = 1
        for sym in syms
            md = varinfo.metadata[sym]
            for (vn, idx) in md.idcs
                len = length(md.ranges[idx])
                is_linked = md.is_transformed[idx]
                range = offset:(offset + len - 1)
                if AbstractPPL.getoptic(vn) === identity
                    all_iden_ranges = merge(
                        all_iden_ranges,
                        NamedTuple((
                            AbstractPPL.getsym(vn) => RangeAndLinked(range, is_linked),
                        )),
                    )
                else
                    all_ranges[vn] = RangeAndLinked(range, is_linked)
                end
                offset += len
            end
        end
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = tweak_adtype(adtype, model, varinfo)
            x = [val for val in varinfo[:]]
            DI.prepare_gradient(
                FastLogDensityAt(model, getlogdensity, all_iden_ranges, all_ranges),
                adtype,
                x,
            )
        end

        return new{
            typeof(model),
            typeof(getlogdensity),
            typeof(all_iden_ranges),
            typeof(adtype),
            typeof(prep),
        }(
            model, getlogdensity, all_iden_ranges, all_ranges, adtype, prep
        )
    end
end

function _evaluate!!(
    model::Model{F,A,D,M,TA,TD,<:FastLDFContext}, varinfo::OnlyAccsVarInfo
) where {F,A,D,M,TA,TD}
    args = map(maybe_deepcopy, model.args)
    return model.f(model, varinfo, args...; model.defaults...)
end
maybe_deepcopy(@nospecialize(x)) = x
function maybe_deepcopy(x::AbstractArray{T}) where {T}
    if T >: Missing
        # avoid overwriting missing elements of model arguments when
        # evaluating the model.
        deepcopy(x)
    else
        x
    end
end

struct FastLogDensityAt{M<:Model,F<:Function,N<:NamedTuple}
    _model::M
    _getlogdensity::F
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
end
function (f::FastLogDensityAt)(params::AbstractVector{<:Real})
    ctx = FastLDFContext(f._iden_varname_ranges, f._varname_ranges, params)
    model = DynamicPPL.setleafcontext(f._model, ctx)
    # This can obviously also be optimised for the case where not
    # all accumulators are needed.
    accs = AccumulatorTuple((
        LogPriorAccumulator(), LogLikelihoodAccumulator(), LogJacobianAccumulator()
    ))
    _, vi = _evaluate!!(model, OnlyAccsVarInfo(accs))
    return f._getlogdensity(vi)
end

function LogDensityProblems.logdensity(fldf::FastLDF, params::AbstractVector{<:Real})
    return FastLogDensityAt(
        fldf._model, fldf._getlogdensity, fldf._iden_varname_ranges, fldf._varname_ranges
    )(
        params
    )
end

function LogDensityProblems.logdensity_and_gradient(
    fldf::FastLDF, params::AbstractVector{<:Real}
)
    return DI.value_and_gradient(
        FastLogDensityAt(
            fldf._model,
            fldf._getlogdensity,
            fldf._iden_varname_ranges,
            fldf._varname_ranges,
        ),
        fldf._adprep,
        fldf._adtype,
        params,
    )
end
