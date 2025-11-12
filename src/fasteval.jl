using DynamicPPL:
    AbstractVarInfo,
    AccumulatorTuple,
    InitContext,
    InitFromParams,
    LogJacobianAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    Model,
    ThreadSafeVarInfo,
    VarInfo,
    OnlyAccsVarInfo,
    RangeAndLinked,
    VectorWithRanges,
    Metadata,
    VarNamedVector,
    default_accumulators,
    float_type_with_fallback,
    getlogjoint,
    getlogjoint_internal,
    getloglikelihood,
    getlogprior,
    getlogprior_internal
using ADTypes: ADTypes
using BangBang: BangBang
using AbstractPPL: AbstractPPL, VarName
using LogDensityProblems: LogDensityProblems
import DifferentiationInterface as DI
using Random: Random

"""
    FastLDF(
        model::Model,
        getlogdensity::Function=getlogjoint_internal,
        varinfo::AbstractVarInfo=VarInfo(model);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )

A struct which contains a model, along with all the information necessary to:

 - calculate its log density at a given point;
 - and if `adtype` is provided, calculate the gradient of the log density at that point.

This information can be extracted using the LogDensityProblems.jl interface, specifically,
using `LogDensityProblems.logdensity` and `LogDensityProblems.logdensity_and_gradient`. If
`adtype` is nothing, then only `logdensity` is implemented. If `adtype` is a concrete AD
backend type, then `logdensity_and_gradient` is also implemented.

There are several options for `getlogdensity` that are 'supported' out of the box:

- [`getlogjoint_internal`](@ref): calculate the log joint, including the log-Jacobian term
  for any variables that have been linked in the provided VarInfo.
- [`getlogprior_internal`](@ref): calculate the log prior, including the log-Jacobian term
  for any variables that have been linked in the provided VarInfo.
- [`getlogjoint`](@ref): calculate the log joint in the model space, ignoring any effects of
  linking
- [`getlogprior`](@ref): calculate the log prior in the model space, ignoring any effects of
  linking
- [`getloglikelihood`](@ref): calculate the log likelihood (this is unaffected by linking,
  since transforms are only applied to random variables)

!!! note
    By default, `FastLDF` uses `getlogjoint_internal`, i.e., the result of
    `LogDensityProblems.logdensity(f, x)` will depend on whether the `FastLDF` was created
    with a linked or unlinked VarInfo. This is done primarily to ease interoperability with
    MCMC samplers.

If you provide one of these functions, a `VarInfo` will be automatically created for you. If
you provide a different function, you have to manually create a VarInfo and pass it as the
third argument.

If the `adtype` keyword argument is provided, then this struct will also store the adtype
along with other information for efficient calculation of the gradient of the log density.
Note that preparing a `FastLDF` with an AD type `AutoBackend()` requires the AD backend
itself to have been loaded (e.g. with `import Backend`).

## Fields

Note that it is undefined behaviour to access any of a `FastLDF`'s fields, apart from:

- `fastldf.model`: The original model from which this `FastLDF` was constructed.
- `fastldf.adtype`: The AD type used for gradient calculations, or `nothing` if no AD
  type was provided.

# Extended help

Up until DynamicPPL v0.38, there have been two ways of evaluating a DynamicPPL model at a
given set of parameters:

1. With `unflatten` + `evaluate!!` with `DefaultContext`: this stores a vector of parameters
   inside a VarInfo's metadata, then reads parameter values from the VarInfo during evaluation.

2. With `InitFromParams`: this reads parameter values from a NamedTuple or a Dict, and stores
   them inside a VarInfo's metadata.

In general, both of these approaches work fine, but the fact that they modify the VarInfo's
metadata can often be quite wasteful. In particular, it is very common that the only outputs
we care about from model evaluation are those which are stored in accumulators, such as log
probability densities, or `ValuesAsInModel`.

To avoid this issue, we use `OnlyAccsVarInfo`, which is a VarInfo that only contains
accumulators. It implements enough of the `AbstractVarInfo` interface to not error during
model evaluation.

Because `OnlyAccsVarInfo` does not store any parameter values, when evaluating a model with
it, it is mandatory that parameters are provided from outside the VarInfo, namely via
`InitContext`.

The main problem that we face is that it is not possible to directly implement
`DynamicPPL.init(rng, vn, dist, strategy)` for `strategy::InitFromParams{<:AbstractVector}`.
In particular, it is not clear:

 - which parts of the vector correspond to which random variables, and
 - whether the variables are linked or unlinked.

Traditionally, this problem has been solved by `unflatten`, because that function would
place values into the VarInfo's metadata alongside the information about ranges and linking.
That way, when we evaluate with `DefaultContext`, we can read this information out again.
However, we want to avoid using a metadata. Thus, here, we _extract this information from
the VarInfo_ a single time when constructing a `FastLDF` object. Inside the FastLDF, we
store a mapping from VarNames to ranges in that vector, along with link status.

For VarNames with identity optics, this is stored in a NamedTuple for efficiency. For all
other VarNames, this is stored in a Dict. The internal data structure used to represent this
could almost certainly be optimised further. See e.g. the discussion in
https://github.com/TuringLang/DynamicPPL.jl/issues/1116.

When evaluating the model, this allows us to combine the parameter vector together with those
ranges to create an `InitFromParams{VectorWithRanges}`, which lets us very quickly read
parameter values from the vector.

Note that this assumes that the ranges and link status are static throughout the lifetime of
the `FastLDF` object. Therefore, a `FastLDF` object cannot handle models which have variable
numbers of parameters, or models which may visit random variables in different orders depending
on stochastic control flow. **Indeed, silent errors may occur with such models.** This is a
general limitation of vectorised parameters: the original `unflatten` + `evaluate!!`
approach also fails with such models.
"""
struct FastLDF{
    M<:Model,
    AD<:Union{ADTypes.AbstractADType,Nothing},
    F<:Function,
    N<:NamedTuple,
    ADP<:Union{Nothing,DI.GradientPrep},
}
    model::M
    adtype::AD
    _getlogdensity::F
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
    _adprep::ADP

    function FastLDF(
        model::Model,
        getlogdensity::Function=getlogjoint_internal,
        varinfo::AbstractVarInfo=VarInfo(model);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        # Figure out which variable corresponds to which index, and
        # which variables are linked.
        all_iden_ranges, all_ranges = get_ranges_and_linked(varinfo)
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = DynamicPPL.tweak_adtype(adtype, model, varinfo)
            x = [val for val in varinfo[:]]
            DI.prepare_gradient(
                FastLogDensityAt(model, getlogdensity, all_iden_ranges, all_ranges),
                adtype,
                x,
            )
        end
        return new{
            typeof(model),
            typeof(adtype),
            typeof(getlogdensity),
            typeof(all_iden_ranges),
            typeof(prep),
        }(
            model, adtype, getlogdensity, all_iden_ranges, all_ranges, prep
        )
    end
end

###################################
# LogDensityProblems.jl interface #
###################################
"""
    fast_ldf_accs(getlogdensity::Function)

Determine which accumulators are needed for fast evaluation with the given
`getlogdensity` function.
"""
fast_ldf_accs(::Function) = default_accumulators()
fast_ldf_accs(::typeof(getlogjoint_internal)) = default_accumulators()
function fast_ldf_accs(::typeof(getlogjoint))
    return AccumulatorTuple((LogPriorAccumulator(), LogLikelihoodAccumulator()))
end
function fast_ldf_accs(::typeof(getlogprior_internal))
    return AccumulatorTuple((LogPriorAccumulator(), LogJacobianAccumulator()))
end
fast_ldf_accs(::typeof(getlogprior)) = AccumulatorTuple((LogPriorAccumulator(),))
fast_ldf_accs(::typeof(getloglikelihood)) = AccumulatorTuple((LogLikelihoodAccumulator(),))

struct FastLogDensityAt{M<:Model,F<:Function,N<:NamedTuple}
    _model::M
    _getlogdensity::F
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
end
function (f::FastLogDensityAt)(params::AbstractVector{<:Real})
    ctx = InitContext(
        Random.default_rng(),
        InitFromParams(
            VectorWithRanges(f._iden_varname_ranges, f._varname_ranges, params), nothing
        ),
    )
    model = DynamicPPL.setleafcontext(f._model, ctx)
    accs = fast_ldf_accs(f._getlogdensity)
    # Calling `evaluate!!` would be fine, but would lead to an extra call to resetaccs!!,
    # which is unnecessary. So we shortcircuit this by simply calling `_evaluate!!`
    # directly. To preserve thread-safety we need to reproduce the ThreadSafeVarInfo logic
    # here.
    # TODO(penelopeysm): This should _not_ check Threads.nthreads(). I still don't know what
    # it _should_ do, but this is wrong regardless.
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1086
    vi = if Threads.nthreads() > 1
        Tlogp = float_type_with_fallback(eltype(params))
        OnlyAccsVarInfo(DynamicPPL.default_atomic_accumulators(Tlogp))
    else
        OnlyAccsVarInfo(accs)
    end
    _, vi = DynamicPPL._evaluate!!(model, vi)
    return f._getlogdensity(vi)
end

function LogDensityProblems.logdensity(fldf::FastLDF, params::AbstractVector{<:Real})
    return FastLogDensityAt(
        fldf.model, fldf._getlogdensity, fldf._iden_varname_ranges, fldf._varname_ranges
    )(
        params
    )
end

function LogDensityProblems.logdensity_and_gradient(
    fldf::FastLDF, params::AbstractVector{<:Real}
)
    return DI.value_and_gradient(
        FastLogDensityAt(
            fldf.model, fldf._getlogdensity, fldf._iden_varname_ranges, fldf._varname_ranges
        ),
        fldf._adprep,
        fldf.adtype,
        params,
    )
end

######################################################
# Helper functions to extract ranges and link status #
######################################################

# This fails for SimpleVarInfo, but honestly there is no reason to support that here. The
# fact is that evaluation doesn't use a VarInfo, it only uses it once to generate the ranges
# and link status. So there is no motivation to use SimpleVarInfo inside a
# LogDensityFunction any more, we can just always use typed VarInfo. In fact one could argue
# that there is no purpose in supporting untyped VarInfo either.
"""
    get_ranges_and_linked(varinfo::VarInfo)

Given a `VarInfo`, extract the ranges of each variable in the vectorised parameter
representation, along with whether each variable is linked or unlinked.

This function should return a tuple containing:

- A NamedTuple mapping VarNames with identity optics to their corresponding `RangeAndLinked`
- A Dict mapping all other VarNames to their corresponding `RangeAndLinked`.
"""
function get_ranges_and_linked(varinfo::VarInfo{<:NamedTuple{syms}}) where {syms}
    all_iden_ranges = NamedTuple()
    all_ranges = Dict{VarName,RangeAndLinked}()
    offset = 1
    for sym in syms
        md = varinfo.metadata[sym]
        this_md_iden, this_md_others, offset = get_ranges_and_linked_metadata(md, offset)
        all_iden_ranges = merge(all_iden_ranges, this_md_iden)
        all_ranges = merge(all_ranges, this_md_others)
    end
    return all_iden_ranges, all_ranges
end
function get_ranges_and_linked(varinfo::VarInfo{<:Union{Metadata,VarNamedVector}})
    all_iden, all_others, _ = get_ranges_and_linked_metadata(varinfo.metadata, 1)
    return all_iden, all_others
end
function get_ranges_and_linked_metadata(md::Metadata, start_offset::Int)
    all_iden_ranges = NamedTuple()
    all_ranges = Dict{VarName,RangeAndLinked}()
    offset = start_offset
    for (vn, idx) in md.idcs
        is_linked = md.is_transformed[idx]
        range = md.ranges[idx] .+ (start_offset - 1)
        if AbstractPPL.getoptic(vn) === identity
            all_iden_ranges = merge(
                all_iden_ranges,
                NamedTuple((AbstractPPL.getsym(vn) => RangeAndLinked(range, is_linked),)),
            )
        else
            all_ranges[vn] = RangeAndLinked(range, is_linked)
        end
        offset += length(range)
    end
    return all_iden_ranges, all_ranges, offset
end
function get_ranges_and_linked_metadata(vnv::VarNamedVector, start_offset::Int)
    all_iden_ranges = NamedTuple()
    all_ranges = Dict{VarName,RangeAndLinked}()
    offset = start_offset
    for (vn, idx) in vnv.varname_to_index
        is_linked = vnv.is_unconstrained[idx]
        range = vnv.ranges[idx] .+ (start_offset - 1)
        if AbstractPPL.getoptic(vn) === identity
            all_iden_ranges = merge(
                all_iden_ranges,
                NamedTuple((AbstractPPL.getsym(vn) => RangeAndLinked(range, is_linked),)),
            )
        else
            all_ranges[vn] = RangeAndLinked(range, is_linked)
        end
        offset += length(range)
    end
    return all_iden_ranges, all_ranges, offset
end
