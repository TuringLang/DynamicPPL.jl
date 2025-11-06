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

struct FastLDFContext{T<:AbstractVector{<:Real}} <: AbstractContext
    varname_ranges::Dict{VarName,RangeAndLinked}
    params::T
end
DynamicPPL.NodeTrait(::FastLDFContext) = IsLeaf()

function tilde_assume!!(
    ctx::FastLDFContext, right::Distribution, vn::VarName, vi::OnlyAccsVarInfo
)
    # Don't need to read the data from the varinfo at all since it's
    # all inside the context.
    range_and_linked = ctx.varname_ranges[vn]
    y = @view ctx.params[range_and_linked.range]
    is_linked = range_and_linked.is_linked
    f = if is_linked
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
    AD<:Union{ADTypes.AbstractADType,Nothing},
    ADP<:Union{Nothing,DI.GradientPrep},
}
    _model::M
    _getlogdensity::F
    _varname_ranges::Dict{VarName,RangeAndLinked}
    _adtype::AD
    _adprep::ADP

    function FastLDF(
        model::Model,
        getlogdensity::Function,
        # This only works with typed Metadata-varinfo.
        # Obviously, this can be generalised later.
        varinfo::VarInfo{<:NamedTuple{syms}};
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    ) where {syms}
        # Figure out which variable corresponds to which index, and
        # which variables are linked.
        all_ranges = Dict{VarName,RangeAndLinked}()
        offset = 1
        for sym in syms
            md = varinfo.metadata[sym]
            for (vn, idx) in md.idcs
                len = length(md.ranges[idx])
                is_linked = md.is_transformed[idx]
                range = offset:(offset + len - 1)
                all_ranges[vn] = RangeAndLinked(range, is_linked)
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
                FastLogDensityAt(model, getlogdensity, all_ranges), adtype, x
            )
        end

        return new{typeof(model),typeof(getlogdensity),typeof(adtype),typeof(prep)}(
            model, getlogdensity, all_ranges, adtype, prep
        )
    end
end

struct FastLogDensityAt{M<:Model,F<:Function}
    _model::M
    _getlogdensity::F
    _varname_ranges::Dict{VarName,RangeAndLinked}
end
function (f::FastLogDensityAt)(params::AbstractVector{<:Real})
    ctx = FastLDFContext(f._varname_ranges, params)
    model = DynamicPPL.setleafcontext(f._model, ctx)
    # This can obviously also be optimised for the case where not
    # all accumulators are needed.
    accs = AccumulatorTuple((
        LogPriorAccumulator(), LogLikelihoodAccumulator(), LogJacobianAccumulator()
    ))
    _, vi = DynamicPPL._evaluate!!(model, OnlyAccsVarInfo(accs))
    return f._getlogdensity(vi)
end

function LogDensityProblems.logdensity(fldf::FastLDF, params::AbstractVector{<:Real})
    return FastLogDensityAt(fldf._model, fldf._getlogdensity, fldf._varname_ranges)(params)
end

function LogDensityProblems.logdensity_and_gradient(
    fldf::FastLDF, params::AbstractVector{<:Real}
)
    return DI.value_and_gradient(
        FastLogDensityAt(fldf._model, fldf._getlogdensity, fldf._varname_ranges),
        fldf._adprep,
        fldf._adtype,
        params,
    )
end
