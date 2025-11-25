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
}
    model::M
    adtype::AD
    _getlogdensity::F
    _iden_varname_ranges::N
    _varname_ranges::Dict{VarName,RangeAndLinked}
    _adprep::ADP
    _dim::Int

    """
        function LogDensityFunction(
            model::Model,
            getlogdensity::Function=getlogjoint_internal,
            link::Union{Bool,Set{VarName}}=false;
            adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
        )

    Generate a `LogDensityFunction` for the given model.

    The `link` argument specifies which VarNames in the model should be linked. This can
    either be a Bool (if `link=true` all variables are linked; if `link=false` all variables
    are unlinked); or a `Set{VarName}` specifying exactly which variables should be linked.
    Any sub-variables of the set's elements will be linked.
    """
    function LogDensityFunction(
        model::Model,
        getlogdensity::Function=getlogjoint_internal,
        link::Union{Bool,Set{VarName}}=false;
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        # Run the model once to determine variable ranges and linking. Because the
        # parameters stored in the LogDensityFunction are never used, we can just use
        # InitFromPrior to create new values. The actual values don't matter, only the
        # length, since that's used for gradient prep.
        vi = OnlyAccsVarInfo(AccumulatorTuple((RangeLinkedValueAcc(link),)))
        _, vi = DynamicPPL.init!!(model, vi, InitFromPrior())
        rlvacc = first(vi.accs)
        Tlink, all_iden_ranges, all_ranges, x = get_data(rlvacc)
        @info Tlink, all_iden_ranges, all_ranges, x
        # That gives us all the information we need to create the LogDensityFunction.
        dim = length(x)
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = DynamicPPL.tweak_adtype(adtype, model, x)
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
        }(
            model, adtype, getlogdensity, all_iden_ranges, all_ranges, prep, dim
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
    accs = fast_ldf_accs(f.getlogdensity)
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
        params::AbstractVector
    )

Return an 'optimised' form of the adtype. This is useful for doing
backend-specific optimisation of the adtype (e.g., for ForwardDiff, calculating
the chunk size: see the method override in `ext/DynamicPPLForwardDiffExt.jl`).
The model is passed as a parameter in case the optimisation depends on the
model.

By default, this just returns the input unchanged.
"""
tweak_adtype(adtype::ADTypes.AbstractADType, ::Model, ::AbstractVector) = adtype

##############################
# RangeLinkedVal accumulator #
##############################

struct RangeLinkedValueAcc{L<:Union{Bool,Set{VarName}},N<:NamedTuple} <: AbstractAccumulator
    should_link::L
    current_index::Int
    iden_varname_ranges::N
    varname_ranges::Dict{VarName,RangeAndLinked}
    values::Vector{Any}
end
function RangeLinkedValueAcc(should_link::Union{Bool,Set{VarName}})
    return RangeLinkedValueAcc(should_link, 1, (;), Dict{VarName,RangeAndLinked}(), Any[])
end

function get_data(rlvacc::RangeLinkedValueAcc)
    link_statuses = Bool[]
    for ral in rlvacc.iden_varname_ranges
        push!(link_statuses, ral.is_linked)
    end
    for (_, ral) in rlvacc.varname_ranges
        push!(link_statuses, ral.is_linked)
    end
    Tlink = if all(link_statuses)
        true
    elseif all(!s for s in link_statuses)
        false
    else
        nothing
    end
    return (
        Tlink, rlvacc.iden_varname_ranges, rlvacc.varname_ranges, [v for v in rlvacc.values]
    )
end

accumulator_name(::Type{<:RangeLinkedValueAcc}) = :RangeLinkedValueAcc
accumulate_observe!!(acc::RangeLinkedValueAcc, dist, val, vn) = acc
function accumulate_assume!!(
    acc::RangeLinkedValueAcc, val, logjac, vn::VarName{sym}, dist::Distribution
) where {sym}
    link_this_vn = if acc.should_link isa Bool
        acc.should_link
    else
        # Set{VarName}
        any(should_link_vn -> subsumes(should_link_vn, vn), acc.should_link)
    end
    val = if link_this_vn
        to_linked_vec_transform(dist)(val)
    else
        to_vec_transform(dist)(val)
    end
    new_values = vcat(acc.values, val)
    len = length(val)
    range = (acc.current_index):(acc.current_index + len - 1)
    ral = RangeAndLinked(range, link_this_vn)
    iden_varnames, other_varnames = if getoptic(vn) === identity
        merge(acc.iden_varname_ranges, (sym => ral,)), acc.varname_ranges
    else
        acc.varname_ranges[vn] = ral
        acc.iden_varname_ranges, acc.varname_ranges
    end
    return RangeLinkedValueAcc(
        acc.should_link, acc.current_index + len, iden_varnames, other_varnames, new_values
    )
end
function Base.copy(acc::RangeLinkedValueAcc)
    return RangeLinkedValueAcc(
        acc.should_link,
        acc.current_index,
        acc.iden_varname_ranges,
        copy(acc.varname_ranges),
        copy(acc.values),
    )
end
_zero(acc::RangeLinkedValueAcc) = RangeLinkedValueAcc(acc.should_link)
reset(acc::RangeLinkedValueAcc) = _zero(acc)
split(acc::RangeLinkedValueAcc) = _zero(acc)
function combine(acc1::RangeLinkedValueAcc, acc2::RangeLinkedValueAcc)
    new_values = vcat(acc1.values, acc2.values)
    new_current_index = acc1.current_index + acc2.current_index - 1
    acc2_iden_varnames_shifted = NamedTuple(
        k => RangeAndLinked((ral.range .+ (acc1.current_index - 1)), ral.is_linked) for
        (k, ral) in pairs(acc2.iden_varname_ranges)
    )
    new_iden_varname_ranges = merge(acc1.iden_varname_ranges, acc2_iden_varnames_shifted)
    acc2_varname_ranges_shifted = Dict{VarName,RangeAndLinked}()
    for (k, ral) in acc2.varname_ranges
        acc2_varname_ranges_shifted[k] = RangeAndLinked(
            (ral.range .+ (acc1.current_index - 1)), ral.is_linked
        )
    end
    new_varname_ranges = merge(acc1.varname_ranges, acc2_varname_ranges_shifted)
    return RangeLinkedValueAcc(
        # TODO: using acc1.should_link is not really 'correct', but `should_link` only
        # affects model evaluation and `combine` only runs at the end of model evaluation,
        # so it shouldn't matter
        acc1.should_link,
        new_current_index,
        new_iden_varname_ranges,
        new_varname_ranges,
        new_values,
    )
end
