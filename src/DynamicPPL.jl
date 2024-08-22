module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedDict

using AbstractMCMC: AbstractMCMC
using ADTypes: ADTypes
using BangBang: BangBang, push!!, empty!!, setindex!!
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Accessors: Accessors
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD

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

# VarInfo
export AbstractVarInfo,
    VarInfo,
    UntypedVarInfo,
    TypedVarInfo,
    VectorVarInfo,
    SimpleVarInfo,
    VarNamedVector,
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
    set_retained_vns_del_by_spl!,
    is_flagged,
    set_flag!,
    unset_flag!,
    setgid!,
    updategid!,
    setorder!,
    istrans,
    link,
    link!,
    link!!,
    invlink,
    invlink!,
    invlink!!,
    values_as,
    # VarName (reexport from AbstractPPL)
    VarName,
    inspace,
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
    generated_quantities,
    extract_priors,
    values_as_in_model,
    # Samplers
    Sampler,
    SampleFromPrior,
    SampleFromUniform,
    # Contexts
    SamplingContext,
    DefaultContext,
    LikelihoodContext,
    PriorContext,
    MiniBatchContext,
    PrefixContext,
    ConditionContext,
    assume,
    dot_assume,
    observe,
    dot_observe,
    tilde_assume,
    tilde_observe,
    dot_tilde_assume,
    dot_tilde_observe,
    # Pseudo distributions
    NamedDist,
    NoDist,
    # Convenience functions
    logprior,
    logjoint,
    pointwise_loglikelihoods,
    condition,
    decondition,
    fix,
    unfix,
    # Convenience macros
    @addlogprob!,
    @submodel,
    value_iterator_from_chain,
    check_model,
    check_model_and_trace,
    # Deprecated.
    @logprob_str,
    @prob_str

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

# Used here and overloaded in Turing
function getspace end

"""
    AbstractVarInfo

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

See also: [`VarInfo`](@ref), [`SimpleVarInfo`](@ref).
"""
abstract type AbstractVarInfo <: AbstractModelTrace end

const LEGACY_WARNING = """
!!! warning
    This method is considered legacy, and is likely to be deprecated in the future.
"""

# Necessary forward declarations
include("utils.jl")
include("selector.jl")
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
include("loglikelihoods.jl")
include("submodel_macro.jl")
include("test_utils.jl")
include("transforming.jl")
include("logdensityfunction.jl")
include("model_utils.jl")
include("extract_priors.jl")
include("values_as_in_model.jl")

include("debug_utils.jl")
using .DebugUtils

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include(
            "../ext/DynamicPPLChainRulesCoreExt.jl"
        )
        @require EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869" include(
            "../ext/DynamicPPLEnzymeCoreExt.jl"
        )
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include(
            "../ext/DynamicPPLForwardDiffExt.jl"
        )
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" include(
            "../ext/DynamicPPLMCMCChainsExt.jl"
        )
        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include(
            "../ext/DynamicPPLReverseDiffExt.jl"
        )
        @require ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444" include(
            "../ext/DynamicPPLZygoteRulesExt.jl"
        )
    end
end

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

end # module
