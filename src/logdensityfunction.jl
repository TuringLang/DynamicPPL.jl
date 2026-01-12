using DynamicPPL:
    AbstractVarInfo,
    AccumulatorTuple,
    InitContext,
    InitFromParams,
    AbstractInitStrategy,
    LogJacobianAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    Model,
    ThreadSafeVarInfo,
    VarInfo,
    OnlyAccsVarInfo,
    RangeAndLinked,
    VectorWithRanges,
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
    DynamicPPL.LogDensityFunction(
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
    By default, `LogDensityFunction` uses `getlogjoint_internal`, i.e., the result of
    `LogDensityProblems.logdensity(f, x)` will depend on whether the `LogDensityFunction`
    was created with a linked or unlinked VarInfo. This is done primarily to ease
    interoperability with MCMC samplers.

If you provide one of these functions, a `VarInfo` will be automatically created for you. If
you provide a different function, you have to manually create a VarInfo and pass it as the
third argument.

If the `adtype` keyword argument is provided, then this struct will also store the adtype
along with other information for efficient calculation of the gradient of the log density.
Note that preparing a `LogDensityFunction` with an AD type `AutoBackend()` requires the AD
backend itself to have been loaded (e.g. with `import Backend`).

## Fields

Note that it is undefined behaviour to access any of a `LogDensityFunction`'s fields, apart
from:

- `ldf.model`: The original model from which this `LogDensityFunction` was constructed.
- `ldf.adtype`: The AD type used for gradient calculations, or `nothing` if no AD
  type was provided.

# Extended help

Up until DynamicPPL v0.38, there have been two ways of evaluating a DynamicPPL model at a
given set of parameters:

1. With `unflatten` + `evaluate!!` with `DefaultContext`: this stores a vector of parameters
   inside a VarInfo's metadata, then reads parameter values from the VarInfo during
   evaluation.

2. With `InitFromParams`: this reads parameter values from a NamedTuple or a Dict, and
   stores them inside a VarInfo's metadata.

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
the VarInfo_ a single time when constructing a `LogDensityFunction` object. Inside the
LogDensityFunction, we store a mapping from VarNames to ranges in that vector, along with
link status.

When evaluating the model, this allows us to combine the parameter vector together with
those ranges to create an `InitFromParams{VectorWithRanges}`, which lets us very quickly
read parameter values from the vector.

Note that this assumes that the ranges and link status are static throughout the lifetime of
the `LogDensityFunction` object. Therefore, a `LogDensityFunction` object cannot handle
models which have variable numbers of parameters, or models which may visit random variables
in different orders depending on stochastic control flow. **Indeed, silent errors may occur
with such models.** This is a general limitation of vectorised parameters: the original
`unflatten` + `evaluate!!` approach also fails with such models.
"""
struct LogDensityFunction{
    # true if all variables are linked; false if all variables are unlinked; nothing if
    # mixed
    Tlink,
    M<:Model,
    AD<:Union{ADTypes.AbstractADType,Nothing},
    F<:Function,
    VNT<:VarNamedTuple,
    ADP<:Union{Nothing,DI.GradientPrep},
    # type of the vector passed to logdensity functions
    X<:AbstractVector,
}
    model::M
    adtype::AD
    _getlogdensity::F
    _varname_ranges::VNT
    _adprep::ADP
    _dim::Int

    function LogDensityFunction(
        model::Model,
        getlogdensity::Function=getlogjoint_internal,
        varinfo::AbstractVarInfo=VarInfo(model);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        # Figure out which variable corresponds to which index, and
        # which variables are linked.
        all_ranges = get_ranges_and_linked(varinfo)
        # Figure out if all variables are linked, unlinked, or mixed
        link_statuses = Bool[]
        for vn in keys(all_ranges)
            push!(link_statuses, all_ranges[vn].is_linked)
        end
        Tlink = if all(link_statuses)
            true
        elseif all(!s for s in link_statuses)
            false
        else
            nothing
        end
        x = [val for val in varinfo[:]]
        dim = length(x)
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = DynamicPPL.tweak_adtype(adtype, model, varinfo)
            DI.prepare_gradient(
                LogDensityAt{Tlink}(model, getlogdensity, all_ranges), adtype, x
            )
        end
        return new{
            Tlink,
            typeof(model),
            typeof(adtype),
            typeof(getlogdensity),
            typeof(all_ranges),
            typeof(prep),
            typeof(x),
        }(
            model, adtype, getlogdensity, all_ranges, prep, dim
        )
    end
end

function _get_input_vector_type(::LogDensityFunction{T,M,A,G,I,P,X}) where {T,M,A,G,I,P,X}
    return X
end

###################################
# LogDensityProblems.jl interface #
###################################
"""
    ldf_accs(getlogdensity::Function)

Determine which accumulators are needed for fast evaluation with the given
`getlogdensity` function.
"""
ldf_accs(::Function) = default_accumulators()
ldf_accs(::typeof(getlogjoint_internal)) = default_accumulators()
function ldf_accs(::typeof(getlogjoint))
    return AccumulatorTuple((LogPriorAccumulator(), LogLikelihoodAccumulator()))
end
function ldf_accs(::typeof(getlogprior_internal))
    return AccumulatorTuple((LogPriorAccumulator(), LogJacobianAccumulator()))
end
ldf_accs(::typeof(getlogprior)) = AccumulatorTuple((LogPriorAccumulator(),))
ldf_accs(::typeof(getloglikelihood)) = AccumulatorTuple((LogLikelihoodAccumulator(),))

struct LogDensityAt{Tlink,M<:Model,F<:Function,VNT<:VarNamedTuple}
    model::M
    getlogdensity::F
    varname_ranges::VNT

    function LogDensityAt{Tlink}(
        model::M, getlogdensity::F, varname_ranges::N
    ) where {Tlink,M,F,N}
        return new{Tlink,M,F,N}(model, getlogdensity, varname_ranges)
    end
end
function (f::LogDensityAt{Tlink})(params::AbstractVector{<:Real}) where {Tlink}
    strategy = InitFromParams(VectorWithRanges{Tlink}(f.varname_ranges, params), nothing)
    accs = ldf_accs(f.getlogdensity)
    _, vi = DynamicPPL.init!!(f.model, OnlyAccsVarInfo(accs), strategy)
    return f.getlogdensity(vi)
end

function LogDensityProblems.logdensity(
    ldf::LogDensityFunction{Tlink}, params::AbstractVector{<:Real}
) where {Tlink}
    return LogDensityAt{Tlink}(ldf.model, ldf._getlogdensity, ldf._varname_ranges)(params)
end

function LogDensityProblems.logdensity_and_gradient(
    ldf::LogDensityFunction{Tlink}, params::AbstractVector{<:Real}
) where {Tlink}
    params = convert(_get_input_vector_type(ldf), params)
    return DI.value_and_gradient(
        LogDensityAt{Tlink}(ldf.model, ldf._getlogdensity, ldf._varname_ranges),
        ldf._adprep,
        ldf.adtype,
        params,
    )
end

function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{T,M,Nothing}}
) where {T,M}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{T,M,<:ADTypes.AbstractADType}}
) where {T,M}
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.dimension(ldf::LogDensityFunction)
    return ldf._dim
end

"""
    tweak_adtype(
        adtype::ADTypes.AbstractADType,
        model::Model,
        varinfo::AbstractVarInfo,
    )

Return an 'optimised' form of the adtype. This is useful for doing
backend-specific optimisation of the adtype (e.g., for ForwardDiff, calculating
the chunk size: see the method override in `ext/DynamicPPLForwardDiffExt.jl`).
The model is passed as a parameter in case the optimisation depends on the
model.

By default, this just returns the input unchanged.
"""
tweak_adtype(adtype::ADTypes.AbstractADType, ::Model, ::AbstractVarInfo) = adtype

######################################################
# Helper functions to extract ranges and link status #
######################################################

"""
    get_ranges_and_linked(varinfo::VarInfo)

Given a `VarInfo`, extract the ranges of each variable in the vectorised parameter
representation, along with whether each variable is linked or unlinked.

This function returns a VarNamedTuple mapping all VarNames to their corresponding
`RangeAndLinked`.
"""
function get_ranges_and_linked(vi::VNTVarInfo)
    # TODO(mhauru) Check that the closure doesn't cause type instability here.
    vnt = VarNamedTuple()
    vnt, _ = mapreduce(
        identity,
        function ((vnt, offset), pair)
            vn, tv = pair
            val = tv.val
            range = offset:(offset + length(val) - 1)
            offset += length(val)
            ral = RangeAndLinked(range, tv.linked, tv.size)
            vnt = setindex!!(vnt, ral, vn)
            return vnt, offset
        end,
        vi.values;
        init=(VarNamedTuple(), 1),
    )
    return vnt
end
