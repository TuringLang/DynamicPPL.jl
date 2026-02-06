module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedCollections, OrderedDict
using Printf: Printf

using AbstractMCMC: AbstractMCMC
using ADTypes: ADTypes
using BangBang: BangBang, push!!, empty!!, setindex!!
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Accessors: Accessors
using LogDensityProblems: LogDensityProblems

using LinearAlgebra: LinearAlgebra, Cholesky

using DocStringExtensions

using Random: Random

# For extending
import AbstractPPL: predict, hasvalue, getvalue

# TODO: Remove these when it's possible.
import Bijectors: link, invlink

import Base:
    Symbol,
    ==,
    hash,
    getindex,
    setindex!,
    push!,
    show,
    isempty,
    empty!,
    getproperty,
    setproperty!,
    keys,
    haskey

# VarInfo
export AbstractVarInfo,
    VarInfo,
    VarNamedTuple,
    @vnt,
    map_pairs!!,
    map_values!!,
    apply!!,
    push!!,
    empty!!,
    subset,
    getlogp,
    getlogjoint,
    getlogprior,
    getloglikelihood,
    getlogjac,
    getlogjoint_internal,
    getlogprior_internal,
    setlogp!!,
    setlogprior!!,
    setlogjac!!,
    setloglikelihood!!,
    acclogp,
    acclogp!!,
    acclogjac!!,
    acclogprior!!,
    accloglikelihood!!,
    is_transformed,
    set_transformed!!,
    # Accumulators
    AbstractAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    LogJacobianAccumulator,
    RawValueAccumulator,
    get_raw_values,
    PriorDistributionAccumulator,
    accumulate_assume!!,
    accumulate_observe!!,
    accumulator_name,
    reset,
    split,
    combine,
    VNTAccumulator,
    DoNotAccumulate,
    getacc,
    setacc!!,
    setaccs!!,
    deleteacc!!,
    # Working with internal values as vectors
    unflatten!!,
    internal_values_as_vector,
    # VarName (reexport from AbstractPPL)
    VarName,
    subsumes,
    @varname,
    # Compiler
    @model,
    # Utilities
    OrderedDict,
    typed_identity,
    # Model
    Model,
    getmissings,
    getargnames,
    setthreadsafe,
    requires_threadsafe,
    extract_priors,
    PriorDistributionAccumulator,
    # evaluation
    evaluate!!,
    init!!,
    # LogDensityFunction
    LogDensityFunction,
    OnlyAccsVarInfo,
    # Leaf contexts
    AbstractContext,
    contextualize,
    DefaultContext,
    InitContext,
    # Parent contexts
    AbstractParentContext,
    childcontext,
    setchildcontext,
    leafcontext,
    setleafcontext,
    # Tilde pipeline
    tilde_assume!!,
    tilde_observe!!,
    # Initialisation
    AbstractInitStrategy,
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    get_param_eltype,
    init,
    # Transformed values
    VectorValue,
    LinkedVectorValue,
    UntransformedValue,
    get_transform,
    get_internal_value,
    set_internal_value,
    # Linking
    link,
    link!!,
    invlink,
    invlink!!,
    update_link_status!!,
    AbstractTransformStrategy,
    LinkAll,
    UnlinkAll,
    LinkSome,
    UnlinkSome,
    target_transform,
    apply_transform_strategy,
    AbstractTransform,
    DynamicLink,
    Unlink,
    # Pseudo distributions
    NamedDist,
    NoDist,
    # Convenience functions
    logprior,
    logjoint,
    pointwise_prior_logdensities,
    pointwise_logdensities,
    pointwise_loglikelihoods,
    condition,
    conditioned,
    decondition,
    fix,
    fixed,
    unfix,
    predict,
    marginalize,
    prefix,
    returned,
    to_submodel,
    # Struct to hold model outputs
    ParamsWithStats,
    # Convenience macros
    @addlogprob!,
    check_model,
    check_model_and_trace,
    # Deprecated.
    @logprob_str,
    @prob_str,
    generated_quantities

# Reexport
using Distributions: loglikelihood
export loglikelihood

# TODO: Remove once we feel comfortable people aren't using it anymore.
macro logprob_str(str)
    return :(error(
        "The `@logprob_str` macro is no longer supported. See https://turinglang.org/dev/docs/using-turing/guide/#querying-probabilities-from-model-or-chain for information on how to query probabilities, and https://github.com/TuringLang/DynamicPPL.jl/issues/356 for information regarding its removal.",
    ))
end

macro prob_str(str)
    return :(error(
        "The `@prob_str` macro is no longer supported. See https://turinglang.org/dev/docs/using-turing/guide/#querying-probabilities-from-model-or-chain for information on how to query probabilities, and https://github.com/TuringLang/DynamicPPL.jl/issues/356 for information regarding its removal.",
    ))
end

"""
    AbstractHasAccs

An abstract type for anything that contains accumulators. Many functions for extracting
model outputs, such as [`getlogjoint`](@ref), depend only on the presence of certain
accumulators. Thus, these functions are defined for the abstract type `AbstractHasAccs`
rather than `AbstractVarInfo`, which is a more specific type that contains accumulators but
also contains other information such as the values of random variables.

This is a supertype of both [`AccumulatorTuple`](@ref) as well as [`AbstractVarInfo`](@ref).

To subtype this, you only need to implement two functions:

- `DynamicPPL.getaccs(aha::AbstractHasAccs)`, which should return an `AccumulatorTuple`
- `DynamicPPL.setaccs!!(aha::AbstractHasAccs, accs::AccumulatorTuple)`, which should return
   a new `AbstractHasAccs` with the accumulators set to `accs`.

All extra functionality of `AbstractHasAccs` is solely defined in terms of these two
functions.
"""
abstract type AbstractHasAccs <: AbstractModelTrace end

"""
    AbstractVarInfo <: AbstractHasAccs

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

Since `VarInfo`s also contain accumulators, they are also subtypes of `AbstractHasAccs`.
"""
abstract type AbstractVarInfo <: AbstractHasAccs end

# Necessary forward declarations
include("utils.jl")
include("varnamedtuple.jl")
using .VarNamedTuples:
    VarNamedTuples,
    VarNamedTuple,
    map_pairs!!,
    map_values!!,
    apply!!,
    templated_setindex!!,
    NoTemplate,
    SkipTemplate,
    @vnt

include("transformed_values.jl")
include("contexts.jl")
include("contexts/default.jl")
include("contexts/init.jl")
include("contexts/prefix.jl")
include("contexts/conditionfix.jl")  # Must come after contexts/prefix.jl
include("model.jl")
include("varname.jl")
include("distribution_wrappers.jl")
include("submodel.jl")
include("accumulators.jl")
include("accumulators/default.jl")
include("accumulators/vnt.jl")
include("accumulators/vector_values.jl")
include("accumulators/priors.jl")
include("accumulators/raw_values.jl")
include("accumulators/pointwise_logdensities.jl")

include("abstract_hasaccs.jl")
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("onlyaccs.jl")
include("compiler.jl")
include("logdensityfunction.jl")
include("model_utils.jl")
include("chains.jl")
include("bijector.jl")

include("debug_utils.jl")
using .DebugUtils
include("test_utils.jl")

include("deprecated.jl")

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Same for MarginalLogDensities.jl
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            requires_mld =
                exc.f === DynamicPPL.marginalize &&
                length(argtypes) == 2 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractVector{<:Union{Symbol,<:VarName}}
            if requires_mld
                printstyled(
                    io,
                    "\n\n    `$(exc.f)` requires MarginalLogDensities.jl to be loaded.\n    Please run `using MarginalLogDensities` before calling `$(exc.f)`.\n";
                    color=:cyan,
                    bold=true,
                )
            end
        end

        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            is_evaluate_three_arg =
                exc.f === AbstractPPL.evaluate!! &&
                length(argtypes) == 3 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractVarInfo &&
                argtypes[3] <: AbstractContext
            if is_evaluate_three_arg
                print(
                    io,
                    "\n\nThe method `evaluate!!(model, varinfo, new_ctx)` has been removed. Instead, you should store the `new_ctx` in the `model.context` field using `new_model = contextualize(model, new_ctx)`, and then call `evaluate!!(new_model, varinfo)` on the new model. (Note that, if the model already contained a non-default context, you will need to wrap the existing context.)",
                )
            end
        end
    end
end

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

# Extended in MarginalLogDensitiesExt
function marginalize end

end # module
