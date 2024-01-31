module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedDict

using AbstractMCMC: AbstractMCMC
using BangBang: BangBang, push!!, empty!!, setindex!!
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Setfield: Setfield
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
    vectorize,
    OrderedDict,
    # Model
    Model,
    getmissings,
    getargnames,
    generated_quantities,
    extract_priors,
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
    # Prob macros
    @prob_str,
    @logprob_str,
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
    value_iterator_from_chain

# Reexport
using Distributions: loglikelihood
export loglikelihood

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
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("simple_varinfo.jl")
include("context_implementations.jl")
include("compiler.jl")
include("prob_macro.jl")
include("loglikelihoods.jl")
include("submodel_macro.jl")
include("test_utils.jl")
include("transforming.jl")
include("logdensityfunction.jl")
include("model_utils.jl")
include("extract_priors.jl")

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
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" include(
            "../ext/DynamicPPLMCMCChainsExt.jl"
        )
        @require ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444" include(
            "../ext/DynamicPPLZygoteRulesExt.jl"
        )
    end
end

end # module
