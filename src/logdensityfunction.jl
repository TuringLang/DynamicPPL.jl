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
the VarInfo_ a single time when constructing a `LogDensityFunction` object. Inside the
LogDensityFunction, we store a mapping from VarNames to ranges in that vector, along with
link status.

For VarNames with identity optics, this is stored in a NamedTuple for efficiency. For all
other VarNames, this is stored in a Dict. The internal data structure used to represent this
could almost certainly be optimised further. See e.g. the discussion in
https://github.com/TuringLang/DynamicPPL.jl/issues/1116.

When evaluating the model, this allows us to combine the parameter vector together with those
ranges to create an `InitFromParams{VectorWithRanges}`, which lets us very quickly read
parameter values from the vector.

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
    N<:NamedTuple,
    ADP<:Union{Nothing,DI.GradientPrep},
    # type of the vector passed to logdensity functions
    X<:AbstractVector,
}
    model::M
    adtype::AD
    _getlogdensity::F
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
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
        all_iden_ranges, all_ranges = get_ranges_and_linked(varinfo)
        # Figure out if all variables are linked, unlinked, or mixed
        link_statuses = Bool[]
        for ral in all_iden_ranges
            push!(link_statuses, ral.is_linked)
        end
        for (_, ral) in all_ranges
            push!(link_statuses, ral.is_linked)
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
                LogDensityAt{Tlink}(model, getlogdensity, all_iden_ranges, all_ranges),
                adtype,
                x,
            )
        end
        return new{
            Tlink,
            typeof(model),
            typeof(adtype),
            typeof(getlogdensity),
            typeof(all_iden_ranges),
            typeof(prep),
            typeof(x),
        }(
            model, adtype, getlogdensity, all_iden_ranges, all_ranges, prep, dim
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

struct LogDensityAt{Tlink,M<:Model,F<:Function,N<:NamedTuple}
    model::M
    getlogdensity::F
    iden_varname_ranges::N
    varname_ranges::Dict{VarName,RangeAndLinked}

    function LogDensityAt{Tlink}(
        model::M,
        getlogdensity::F,
        iden_varname_ranges::N,
        varname_ranges::Dict{VarName,RangeAndLinked},
    ) where {Tlink,M,F,N}
        return new{Tlink,M,F,N}(model, getlogdensity, iden_varname_ranges, varname_ranges)
    end
end
function (f::LogDensityAt{Tlink})(params::AbstractVector{<:Real}) where {Tlink}
    strategy = InitFromParams(
        VectorWithRanges{Tlink}(f.iden_varname_ranges, f.varname_ranges, params), nothing
    )
    accs = ldf_accs(f.getlogdensity)
    _, vi = DynamicPPL.init!!(f.model, OnlyAccsVarInfo(accs), strategy)
    return f.getlogdensity(vi)
end

function LogDensityProblems.logdensity(
    ldf::LogDensityFunction{Tlink}, params::AbstractVector{<:Real}
) where {Tlink}
    return LogDensityAt{Tlink}(
        ldf.model, ldf._getlogdensity, ldf._iden_varname_ranges, ldf._varname_ranges
    )(
        params
    )
end

function LogDensityProblems.logdensity_and_gradient(
    ldf::LogDensityFunction{Tlink}, params::AbstractVector{<:Real}
) where {Tlink}
    params = convert(_get_input_vector_type(ldf), params)
    return DI.value_and_gradient(
        LogDensityAt{Tlink}(
            ldf.model, ldf._getlogdensity, ldf._iden_varname_ranges, ldf._varname_ranges
        ),
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

####################################################
# Generate new parameters for a LogDensityFunction #
####################################################
# Previously, when LogDensityFunction contained a full VarInfo, it was easy to generate
# new 'trial' parameters for a LogDensityFunction by doing
#
#     new_vi = last(DynamicPPL.init!!(rng, ldf.model, ldf.varinfo, strategy))
#
# This is useful e.g. when initialising MCMC sampling.
#
# However, now that LogDensityFunction only contains ranges and link status, we need to
# provide some functionality to generate new parameter vectors (and also return their
# logp).

struct LDFValuesAccumulator{T<:Real,N<:NamedTuple} <: AbstractAccumulator
    # These are copied over from the LogDensityFunction
    iden_varname_ranges::N
    varname_ranges::Dict{VarName,RangeAndLinked}
    # These are the actual outputs
    values::Dict{UnitRange{Int},Vector{T}}
    # This is the forward log-Jacobian term
    fwd_logjac::T
end
function LDFValuesAccumulator(ldf::LogDensityFunction)
    nt = ldf._iden_varname_ranges
    T = eltype(_get_input_vector_type(ldf))
    return LDFValuesAccumulator{T,typeof(nt)}(
        nt, ldf._varname_ranges, Dict{UnitRange{Int},Vector{T}}(), zero(T)
    )
end

const _LDFValuesAccName = :LDFValues
accumulator_name(::Type{<:LDFValuesAccumulator}) = _LDFValuesAccName
accumulate_observe!!(acc::LDFValuesAccumulator, dist, val, vn) = acc
function accumulate_assume!!(acc::LDFValuesAccumulator, val, logjac, vn::VarName, dist)
    ral = if DynamicPPL.getoptic(vn) === identity
        acc.iden_varname_ranges[DynamicPPL.getsym(vn)]
    else
        acc.varname_ranges[vn]
    end
    range = ral.range
    # Since `val` is always unlinked, we need to regenerate the vectorised value. This is a
    # bit wasteful if `tilde_assume!!` also did the invlinking, but in general, this is not
    # guaranteed: indeed, `tilde_assume!!` may never have seen a linked vector at all (e.g.
    # if it was called with `InitContext{rng,<:Union{InitFromPrior,InitFromUniform}}`; which
    # is the most likely situation where this accumulator will be used).
    y, fwd_logjac = if ral.is_linked
        with_logabsdet_jacobian(DynamicPPL.to_linked_vec_transform(dist), val)
    else
        with_logabsdet_jacobian(DynamicPPL.to_vec_transform(dist), val)
    end
    acc.values[range] = y
    return LDFValuesAccumulator(
        acc.iden_varname_ranges, acc.varname_ranges, acc.values, acc.fwd_logjac + fwd_logjac
    )
end
function reset(acc::LDFValuesAccumulator{T}) where {T}
    return LDFValuesAccumulator(
        acc.iden_varname_ranges,
        acc.varname_ranges,
        Dict{UnitRange{Int},Vector{T}}(),
        zero(T),
    )
end
function Base.copy(acc::LDFValuesAccumulator)
    return LDFValuesAccumulator(
        acc.iden_varname_ranges, copy(acc.varname_ranges), copy(acc.values), acc.fwd_logjac
    )
end
function split(acc::LDFValuesAccumulator{T}) where {T}
    return LDFValuesAccumulator(
        acc.iden_varname_ranges,
        acc.varname_ranges,
        Dict{UnitRange{Int},Vector{T}}(),
        zero(T),
    )
end
function combine(acc::LDFValuesAccumulator{T}, acc2::LDFValuesAccumulator{T}) where {T}
    if acc.iden_varname_ranges != acc2.iden_varname_ranges ||
        acc.varname_ranges != acc2.varname_ranges
        throw(
            ArgumentError(
                "cannot combine LDFValuesAccumulators with different varname ranges"
            ),
        )
    end
    combined_values = merge(acc.values, acc2.values)
    combined_logjac = acc.fwd_logjac + acc2.fwd_logjac
    return LDFValuesAccumulator(
        acc.iden_varname_ranges, acc.varname_ranges, combined_values, combined_logjac
    )
end

"""
    DynamicPPL.rand_with_logdensity(
        [rng::Random.AbstractRNG,]
        ldf::LogDensityFunction,
        strategy::AbstractInitStrategy=InitFromPrior(),
    )

Given a LogDensityFunction, generate a new parameter vector by sampling from the model using
the given strategy. Returns a tuple of (new parameters, logdensity).

This function therefore provides an interface to sample from the model, even though the
LogDensityFunction no longer carries a full VarInfo with it which would ordinarily be
required for such sampling.

If `ldf` was generated using the call `LogDensityFunction(model, getlogdensity, vi)`, then
as long as `model` does not involve any indeterministic operations that use the `rng`
argument (e.g. parallel sampling with multiple threads), then the outputs of

```julia
new_params, new_logp = rand_with_logdensity(rng, ldf, strategy)
```

and

```julia
_, new_vi = DynamicPPL.init!!(rng, model, vi, strategy)
```

are guaranteed to be related in that

```julia
new_params ≈ new_vi[:]               # (1)
new_logp = getlogdensity(new_vi)     # (2)
```

Furthermore, it is also guaranteed that

```julia
LogDensityProblems.logdensity(ldf, new_params) ≈ new_logp  # (3)
```

If there are indeterministic operations, then (1) and (2) may not _exactly_ hold (for
example, since variables may be sampled in a different order), but (3) will always remain
true. In other words, `new_params` will always be an element of the set of valid parameters
that could have been generated given the indeterminacy, and `new_logp` is the corresponding
log-density.
"""
function rand_with_logdensity(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    strategy::AbstractInitStrategy=InitFromPrior(),
)
    # Create a VarInfo with the necessary accumulators to generate both parameters and logp
    accs = (ldf_accs(ldf._getlogdensity)..., LDFValuesAccumulator(ldf))
    vi = OnlyAccsVarInfo(accs)
    # Initialise the model with the given strategy
    _, new_vi = DynamicPPL.init!!(rng, ldf.model, vi, strategy)
    # Extract the new parameters into a vector
    x = Vector{eltype(_get_input_vector_type(ldf))}(
        undef, LogDensityProblems.dimension(ldf)
    )
    values_acc = DynamicPPL.getacc(new_vi, Val(_LDFValuesAccName))
    for (range, val) in values_acc.values
        x[range] = val
    end
    # This ignores the logjac if there is no LogJacobianAccumulator, which is the correct
    # behaviour
    new_vi = if haskey(getaccs(new_vi), Val(:LogJacobian))
        acclogjac!!(new_vi, values_acc.fwd_logjac)
    else
        new_vi
    end
    lp = ldf._getlogdensity(new_vi)
    return x, lp
end
function rand_with_logdensity(
    ldf::LogDensityFunction, strategy::AbstractInitStrategy=InitFromPrior()
)
    return rand_with_logdensity(Random.default_rng(), ldf, strategy)
end
