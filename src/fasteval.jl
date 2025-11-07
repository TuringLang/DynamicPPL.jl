"""
fasteval.jl
-----------

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

To avoid this issue, we implement here `OnlyAccsVarInfo`, which is a VarInfo that only
contains accumulators. When evaluating a model with `OnlyAccsVarInfo`, it is mandatory that
the model's leaf context is a `FastEvalContext`, which provides extremely fast access to
parameter values. No writing of values into VarInfo metadata is performed at all.

Vector parameters
-----------------

We first consider the case of parameter vectors, i.e., the case which would normally be
handled by `unflatten` and `evaluate!!`. Unfortunately, it is not enough to just store
the vector of parameters in the `FastEvalContext`, because it is not clear:

 - which parts of the vector correspond to which random variables, and
 - whether the variables are linked or unlinked.

Traditionally, this problem has been solved by `unflatten`, because that function would
place values into the VarInfo's metadata alongside the information about ranges and linking.
However, we want to avoid doing this. Thus, here, we _extract this information from the
VarInfo_ a single time when constructing a `FastLDF` object.

Note that this assumes that the ranges and link status are static throughout the lifetime of
the `FastLDF` object. Therefore, a `FastLDF` object cannot handle models which have variable
numbers of parameters, or models which may visit random variables in different orders depending
on stochastic control flow. **Indeed, silent errors may occur with such models.** This is a
general limitation of vectorised parameters: the original `unflatten` + `evaluate!!`
approach also fails with such models.

NamedTuple and Dict parameters
------------------------------

Fast evaluation has not yet been extended to NamedTuple and Dict parameters. Such
representations are capable of handling models with variable sizes and stochastic control
flow.

However, the path towards implementing these is straightforward: just make `InitContext` work
correctly with `OnlyAccsVarInfo`. There will probably be a few functions that need to be
overloaded to make this work: for example `push!!` on `OnlyAccsVarInfo` can just be defined
as a no-op.
"""

using DynamicPPL:
    AbstractContext,
    AbstractVarInfo,
    AccumulatorTuple,
    LogJacobianAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    Metadata,
    Model,
    ThreadSafeVarInfo,
    VarInfo,
    VarNamedVector,
    accumulate_assume!!,
    accumulate_observe!!,
    default_accumulators,
    float_type_with_fallback,
    from_linked_vec_transform,
    from_vec_transform,
    getlogjoint,
    getlogjoint_internal,
    getloglikelihood,
    getlogprior,
    getlogprior_internal,
    leafcontext
using ADTypes: ADTypes
using BangBang: BangBang
using Bijectors: with_logabsdet_jacobian
using AbstractPPL: AbstractPPL, VarName
using Distributions: Distribution
using DocStringExtensions: TYPEDFIELDS
using LogDensityProblems: LogDensityProblems
import DifferentiationInterface as DI

"""
    OnlyAccsVarInfo

This is a wrapper around an `AccumulatorTuple` that implements the minimal `AbstractVarInfo`
interface to work with the `accumulate_assume!!` and `accumulate_observe!!` functions.

Note that this does not implement almost every other AbstractVarInfo interface function, and
so using this outside of FastLDF will lead to errors.

Conceptually, one can also think of this as a VarInfo that doesn't contain a metadata field.
That is because values for random variables are obtained by reading from a separate entity
(such as a `FastLDFContext`), rather than from the VarInfo itself.
"""
struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
OnlyAccsVarInfo() = OnlyAccsVarInfo(default_accumulators())
DynamicPPL.maybe_invlink_before_eval!!(vi::OnlyAccsVarInfo, ::Model) = vi
DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) = OnlyAccsVarInfo(accs)
@inline Base.haskey(::OnlyAccsVarInfo, ::VarName) = false
@inline DynamicPPL.is_transformed(::OnlyAccsVarInfo) = false
@inline BangBang.push!!(vi::OnlyAccsVarInfo, vn, y, dist) = vi
function DynamicPPL.get_param_eltype(
    ::Union{OnlyAccsVarInfo,ThreadSafeVarInfo{<:OnlyAccsVarInfo}}, model::Model
)
    # Because the VarInfo has no parameters stored in it, we need to get the eltype from the
    # model's leaf context. This is only possible if said leaf context is indeed a FastEval
    # context.
    leaf_ctx = DynamicPPL.leafcontext(model.context)
    if leaf_ctx isa FastEvalVectorContext
        return eltype(leaf_ctx.params)
    elseif leaf_ctx isa InitContext{<:Any,<:InitFromParams}
        eltype = DynamicPPL.infer_nested_eltype(leaf_ctx.strategy.params)
        @info "Inferring parameter eltype as $eltype from InitContext"
        return eltype
    else
        error(
            "OnlyAccsVarInfo can only be used with FastEval contexts, found $(typeof(leaf_ctx))",
        )
    end
end

"""
    RangeAndLinked

Suppose we have vectorised parameters `params::AbstractVector{<:Real}`. Each random variable
in the model will in general correspond to a sub-vector of `params`. This struct stores
information about that range, as well as whether the sub-vector represents a linked value or
an unlinked value.

$(TYPEDFIELDS)
"""
struct RangeAndLinked
    # indices that the variable corresponds to in the vectorised parameter
    range::UnitRange{Int}
    # whether it's linked
    is_linked::Bool
end

"""
    FastEvalVectorContext(
        iden_varname_ranges::NamedTuple,
        varname_ranges::Dict{VarName,RangeAndLinked},
        params::AbstractVector{<:Real},
    ) <: AbstractContext

A context that wraps a vector of parameter values, plus information about how random
variables map to ranges in that vector.

In the simplest case, this could be accomplished only with a single dictionary mapping
VarNames to ranges and link status. However, for performance reasons, we separate out
VarNames with identity optics into a NamedTuple (`iden_varname_ranges`). All
non-identity-optic VarNames are stored in the `varname_ranges` Dict.

It would be nice to unify the NamedTuple and Dict approach. See, e.g.
https://github.com/TuringLang/DynamicPPL.jl/issues/1116.
"""
struct FastEvalVectorContext{N<:NamedTuple,T<:AbstractVector{<:Real}} <: AbstractContext
    # This NamedTuple stores the ranges for identity VarNames
    iden_varname_ranges::N
    # This Dict stores the ranges for all other VarNames
    varname_ranges::Dict{VarName,RangeAndLinked}
    # The full parameter vector which we index into to get variable values
    params::T
end
DynamicPPL.NodeTrait(::FastEvalVectorContext) = DynamicPPL.IsLeaf()

function get_range_and_linked(
    ctx::FastEvalVectorContext, ::VarName{sym,typeof(identity)}
) where {sym}
    return ctx.iden_varname_ranges[sym]
end
function get_range_and_linked(ctx::FastEvalVectorContext, vn::VarName)
    return ctx.varname_ranges[vn]
end

function DynamicPPL.tilde_assume!!(
    ctx::FastEvalVectorContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    # Note that this function does not use the metadata field of `vi` at all.
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

function DynamicPPL.tilde_observe!!(
    ::FastEvalVectorContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    # This is the same as for DefaultContext
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end

########################################
# Log-density functions using FastEval #
########################################

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

## Extended help

`FastLDF` uses `FastEvalVectorContext` internally to provide extremely rapid evaluation of
the model given a vector of parameters.

Because it is common to call `LogDensityProblems.logdensity` and
`LogDensityProblems.logdensity_and_gradient` within tight loops, it is beneficial for us to
pre-compute as much of the information as possible when constructing the `FastLDF` object.
In particular, we use the provided VarInfo's metadata to extract the mapping from VarNames
to ranges and link status, and store this mapping inside the `FastLDF` object. We can later
use this to construct a FastEvalVectorContext, without having to look into a metadata again.
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
    # See FastLDFContext for explanation of these two fields.
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
    ctx = FastEvalVectorContext(f._iden_varname_ranges, f._varname_ranges, params)
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
        accs = map(
            acc -> DynamicPPL.convert_eltype(float_type_with_fallback(eltype(params)), acc),
            accs,
        )
        ThreadSafeVarInfo(OnlyAccsVarInfo(accs))
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

# TODO: Fails for SimpleVarInfo. Do I really care enough? Ehhh, honestly, debatable.

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
