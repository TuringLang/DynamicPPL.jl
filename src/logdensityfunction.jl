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
        getlogdensity::Any=getlogjoint_internal,
        varinfo::AbstractVarInfo=VarInfo(model)
        accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=DynamicPPL.ldf_accs(getlogdensity);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )

A struct which contains a model, along with all the information necessary to:

 - calculate its log density at a given point;
 - and if `adtype` is provided, calculate the gradient of the log density at that point.

This information can be extracted using the LogDensityProblems.jl interface, specifically,
using `LogDensityProblems.logdensity` and `LogDensityProblems.logdensity_and_gradient`. If
`adtype` is nothing, then only `logdensity` is implemented. If `adtype` is a concrete AD
backend type, then `logdensity_and_gradient` is also implemented.

`getlogdensity` should be a callable which takes a single argument: a `VarInfo`, and returns
a `Real` corresponding to the log density of interest. There are several functions in
DynamicPPL that are 'supported' out of the box:

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

`accs` allows you to specify an `AccumulatorTuple` or a tuple of `AbstractAccumulator`s
which will be used _when evaluating the log density_`. (Note that the accumulators from the
`VarInfo` argument are discarded.) By default, this uses an internal function,
`DynamicPPL.ldf_accs`, which attempts to choose an appropriate set of accumulators based on
which kind of log-density is being calculated.

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
- `ldf.transform_strategy`: The transform strategy that specifies which variables in the
  LogDensityFunction are linked or unlinked.

# Extended help

Up until DynamicPPL v0.38, there have been two ways of evaluating a DynamicPPL model at a
given set of parameters:

1. With `unflatten!!` + `evaluate!!` with `DefaultContext`: this stores a vector of
   parameters inside a VarInfo's metadata, then reads parameter values from the VarInfo
   during evaluation.

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

Traditionally, this problem has been solved by `unflatten!!`, because that function would
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
`unflatten!!` + `evaluate!!` approach also fails with such models.
"""
struct LogDensityFunction{
    M<:Model,
    AD<:Union{ADTypes.AbstractADType,Nothing},
    L<:AbstractTransformStrategy,
    F,
    VNT<:VarNamedTuple,
    ADP<:Union{Nothing,DI.GradientPrep},
    # type of the vector passed to logdensity functions
    X<:AbstractVector,
    AC<:AccumulatorTuple,
}
    model::M
    adtype::AD
    transform_strategy::L
    _getlogdensity::F
    _varname_ranges::VNT
    _adprep::ADP
    _dim::Int
    _accs::AC

    function LogDensityFunction(
        model::Model,
        getlogdensity::Any=getlogjoint_internal,
        # TODO(penelopeysm): It is a bit redundant to pass a VarInfo, as well as the
        # accumulators, into here. The truth is that the VarInfo is used ONLY for generating
        # the ranges and link status, so arguably we should only pass in a metadata; or when
        # VNT is done, we should pass in only a VNT.
        varinfo::AbstractVarInfo=VarInfo(model),
        accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(
            getlogdensity
        );
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        all_ranges = get_ranges_and_linked(varinfo)
        # Figure out if all variables are linked, unlinked, or mixed
        linked_vns = Set{VarName}()
        unlinked_vns = Set{VarName}()
        for vn in keys(all_ranges)
            if all_ranges[vn].is_linked
                push!(linked_vns, vn)
            else
                push!(unlinked_vns, vn)
            end
        end
        transform_strategy = if isempty(unlinked_vns)
            LinkAll()
        elseif isempty(linked_vns)
            UnlinkAll()
        else
            # We could have a marginal performance optimisation here by checking whether
            # linked_vns or unlinked_vns is smaller, and then using LinkSome or UnlinkSome
            # accordingly, so that there are fewer `subsumes` checks. However, in practice,
            # the mixed linking case performance is going to be a lot worse than in the
            # fully linked or fully unlinked cases anyway, so this would be a bit of a
            # premature optimisation.
            LinkSome(linked_vns, UnlinkAll())
        end
        x = [val for val in varinfo[:]]
        dim = length(x)
        # convert to AccumulatorTuple if needed
        accs = AccumulatorTuple(accs)
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = DynamicPPL.tweak_adtype(adtype, model, varinfo)
            args = (model, getlogdensity, all_ranges, transform_strategy, accs)
            if _use_closure(adtype)
                DI.prepare_gradient(LogDensityAt(args...), adtype, x)
            else
                DI.prepare_gradient(logdensity_at, adtype, x, map(DI.Constant, args)...)
            end
        end
        return new{
            typeof(model),
            typeof(adtype),
            typeof(transform_strategy),
            typeof(getlogdensity),
            typeof(all_ranges),
            typeof(prep),
            typeof(x),
            typeof(accs),
        }(
            model, adtype, transform_strategy, getlogdensity, all_ranges, prep, dim, accs
        )
    end
end

function _get_input_vector_type(::LogDensityFunction{M,A,L,G,R,P,X}) where {M,A,L,G,R,P,X}
    return X
end

###################################
# LogDensityProblems.jl interface #
###################################
"""
    ldf_accs(getlogdensity::Any)

Determine which accumulators are needed for fast evaluation with the given
`getlogdensity` callable.
"""
ldf_accs(::Any) = default_accumulators()
ldf_accs(::typeof(getlogjoint_internal)) = default_accumulators()
function ldf_accs(::typeof(getlogjoint))
    return AccumulatorTuple((LogPriorAccumulator(), LogLikelihoodAccumulator()))
end
function ldf_accs(::typeof(getlogprior_internal))
    return AccumulatorTuple((LogPriorAccumulator(), LogJacobianAccumulator()))
end
ldf_accs(::typeof(getlogprior)) = AccumulatorTuple((LogPriorAccumulator(),))
ldf_accs(::typeof(getloglikelihood)) = AccumulatorTuple((LogLikelihoodAccumulator(),))

"""
    logdensity_at(
        params::AbstractVector{<:Real},
        model::Model,
        getlogdensity::Any,
        varname_ranges::VarNamedTuple,
        transform_strategy::AbstractTransformStrategy,
        accs::AccumulatorTuple,
    )

Calculate the log density at the given `params`, using the provided information extracted
from a `LogDensityFunction`.
"""
function logdensity_at(
    params::AbstractVector{<:Real},
    model::Model,
    getlogdensity::Any,
    varname_ranges::VarNamedTuple,
    transform_strategy::AbstractTransformStrategy,
    accs::AccumulatorTuple,
)
    init_strategy = InitFromParams(
        VectorWithRanges(varname_ranges, params, transform_strategy), nothing
    )
    _, vi = DynamicPPL.init!!(
        model, OnlyAccsVarInfo(accs), init_strategy, transform_strategy
    )
    return getlogdensity(vi)
end

"""
    LogDensityAt(
        model::Model,
        getlogdensity::Any,
        varname_ranges::VarNamedTuple,
        transform_strategy::AbstractTransformStrategy,
        accs::AccumulatorTuple,
    )

A callable struct that behaves in the same way as `logdensity_at`, but stores the model and
other information internally. Having two separate functions/structs allows for better
performance with AD backends.
"""
struct LogDensityAt{
    M<:Model,F,V<:VarNamedTuple,L<:AbstractTransformStrategy,A<:AccumulatorTuple
}
    model::M
    getlogdensity::F
    varname_ranges::V
    transform_strategy::L
    accs::A

    function LogDensityAt(
        model::M, getlogdensity::F, varname_ranges::V, transform_strategy::L, accs::A
    ) where {M,F,V,L,A}
        return new{M,F,V,L,A}(
            model, getlogdensity, varname_ranges, transform_strategy, accs
        )
    end
end
function (f::LogDensityAt)(params::AbstractVector{<:Real})
    return logdensity_at(
        params, f.model, f.getlogdensity, f.varname_ranges, f.transform_strategy, f.accs
    )
end

function LogDensityProblems.logdensity(
    ldf::LogDensityFunction, params::AbstractVector{<:Real}
)
    return logdensity_at(
        params,
        ldf.model,
        ldf._getlogdensity,
        ldf._varname_ranges,
        ldf.transform_strategy,
        ldf._accs,
    )
end

function LogDensityProblems.logdensity_and_gradient(
    ldf::LogDensityFunction, params::AbstractVector{<:Real}
)
    # `params` has to be converted to the same vector type that was used for AD preparation,
    # otherwise the preparation will not be valid.
    params = convert(_get_input_vector_type(ldf), params)
    return if _use_closure(ldf.adtype)
        DI.value_and_gradient(
            LogDensityAt(
                ldf.model,
                ldf._getlogdensity,
                ldf._varname_ranges,
                ldf.transform_strategy,
                ldf._accs,
            ),
            ldf._adprep,
            ldf.adtype,
            params,
        )
    else
        DI.value_and_gradient(
            logdensity_at,
            ldf._adprep,
            ldf.adtype,
            params,
            DI.Constant(ldf.model),
            DI.Constant(ldf._getlogdensity),
            DI.Constant(ldf._varname_ranges),
            DI.Constant(ldf.transform_strategy),
            DI.Constant(ldf._accs),
        )
    end
end

function LogDensityProblems.capabilities(::Type{<:LogDensityFunction{M,Nothing}}) where {M}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{M,<:ADTypes.AbstractADType}}
) where {M}
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

"""
    _use_closure(adtype::ADTypes.AbstractADType)

In LogDensityProblems, we want to calculate the derivative of `logdensity(f, x)` with
respect to x, where f is the model (in our case LogDensityFunction or its arguments ) and is
a constant. However, DifferentiationInterface generally expects a single-argument function
g(x) to differentiate.

There are two ways of dealing with this:

1. Construct a closure over the model, i.e. let g = Base.Fix1(logdensity, f)

2. Use a constant DI.Context. This lets us pass a two-argument function to DI, as long as we
   also give it the 'inactive argument' (i.e. the model) wrapped in `DI.Constant`.

The relative performance of the two approaches, however, depends on the AD backend used.
Some benchmarks are provided here: https://github.com/TuringLang/DynamicPPL.jl/pull/1172

This function is used to determine whether a given AD backend should use a closure or a
constant. If `use_closure(adtype)` returns `true`, then the closure approach will be used.
By default, this function returns `false`, i.e. the constant approach will be used.
"""
# For these AD backends both closure and no closure work, but it is just faster to not use a
# closure (see link in the docstring).
_use_closure(::ADTypes.AutoForwardDiff) = false
_use_closure(::ADTypes.AutoMooncake) = false
_use_closure(::ADTypes.AutoMooncakeForward) = false
# For ReverseDiff, with the compiled tape, you _must_ use a closure because otherwise with
# DI.Constant arguments the tape will always be recompiled upon each call to
# value_and_gradient. For non-compiled ReverseDiff, it is faster to not use a closure.
_use_closure(::ADTypes.AutoReverseDiff{compile}) where {compile} = !compile
# For AutoEnzyme it allows us to avoid setting function_annotation
_use_closure(::ADTypes.AutoEnzyme) = false
# Since for most backends it's faster to not use a closure, we set that as the default
# for unknown AD backends
_use_closure(::ADTypes.AbstractADType) = false

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
function get_ranges_and_linked(vi::VarInfo)
    vnt, _ = mapreduce(
        identity,
        function ((vnt, offset), pair)
            vn, tv = pair
            val = tv.val
            range = offset:(offset + length(val) - 1)
            offset += length(val)
            ral = RangeAndLinked(range, tv isa LinkedVectorValue, tv.size)
            template = vi.values.data[AbstractPPL.getsym(vn)]
            vnt = templated_setindex!!(vnt, ral, vn, template)
            return vnt, offset
        end,
        vi.values;
        init=(VarNamedTuple(), 1),
    )
    return vnt
end
