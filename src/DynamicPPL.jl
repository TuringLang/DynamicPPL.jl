module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedCollections, OrderedDict

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

import AbstractPPL: predict

# VarInfo
export AbstractVarInfo,
    VarInfo,
    UntypedVarInfo,
    TypedVarInfo,
    SimpleVarInfo,
    push!!,
    empty!!,
    subset,
    getlogp,
    setlogp!!,
    acclogp!!,
    resetlogp!!,
    get_num_produce,
    set_num_produce!,
    reset_num_produce!,
    increment_num_produce!,
    set_retained_vns_del!,
    is_flagged,
    set_flag!,
    unset_flag!,
    setorder!,
    istrans,
    link,
    link!!,
    invlink,
    invlink!!,
    values_as,
    # VarName (reexport from AbstractPPL)
    VarName,
    subsumes,
    @varname,
    # Compiler
    @model,
    # Utilities
    init,
    OrderedDict,
    # Model
    Model,
    getmissings,
    getargnames,
    extract_priors,
    values_as_in_model,
    # Samplers
    Sampler,
    SampleFromPrior,
    SampleFromUniform,
    # LogDensityFunction
    LogDensityFunction,
    # Contexts
    SamplingContext,
    DefaultContext,
    LikelihoodContext,
    PriorContext,
    MiniBatchContext,
    PrefixContext,
    ConditionContext,
    assume,
    observe,
    tilde_assume,
    tilde_observe,
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
    decondition,
    fix,
    unfix,
    predict,
    prefix,
    returned,
    to_submodel,
    # Convenience macros
    @addlogprob!,
    @submodel,
    value_iterator_from_chain,
    check_model,
    check_model_and_trace,
    # Benchmarking
    make_benchmark_suite,
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
    AbstractVarInfo

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

See also: [`VarInfo`](@ref), [`SimpleVarInfo`](@ref).
"""
abstract type AbstractVarInfo <: AbstractModelTrace end

# Necessary forward declarations
include("utils.jl")
include("chains.jl")
include("model.jl")
include("sampler.jl")
include("varname.jl")
include("distribution_wrappers.jl")
include("contexts.jl")
include("varnamedvector.jl")
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("simple_varinfo.jl")
include("context_implementations.jl")
include("compiler.jl")
include("pointwise_logdensities.jl")
include("submodel_macro.jl")
include("test_utils.jl")
include("transforming.jl")
include("logdensityfunction.jl")
include("model_utils.jl")
include("extract_priors.jl")
include("values_as_in_model.jl")

include("debug_utils.jl")
using .DebugUtils

include("experimental.jl")
include("deprecated.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

# Better error message if users forget to load JET
if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            requires_jet =
                exc.f === DynamicPPL.Experimental._determine_varinfo_jet &&
                length(argtypes) >= 2 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractContext
            requires_jet |=
                exc.f === DynamicPPL.Experimental.is_suitable_varinfo &&
                length(argtypes) >= 3 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractContext &&
                argtypes[3] <: AbstractVarInfo
            if requires_jet
                print(
                    io,
                    "\n$(exc.f) requires JET.jl to be loaded. Please run `using JET` before calling $(exc.f).",
                )
            end
        end
    end
end

# DynamicPPLForwardDiffExt
# Improves stacktraces, see https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

# DynamicPPLBenchmarkToolsExt
function make_benchmark_suite end

end # module
