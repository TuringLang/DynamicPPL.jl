module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Distributions
using OrderedCollections: OrderedDict

using AbstractMCMC: AbstractMCMC
using BangBang: BangBang, push!!, empty!!, setindex!!
using ChainRulesCore: ChainRulesCore
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Setfield: Setfield
using ZygoteRules: ZygoteRules
using LogDensityProblems: LogDensityProblems

using DocStringExtensions

using Random: Random

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
    getlogp,
    resetlogp!,
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
    link!,
    link!!,
    invlink!,
    invlink!!,
    tonamedtuple,
    values_as,
    # VarName (reexport from AbstractPPL)
    VarName,
    inspace,
    subsumes,
    @varname,
    # Compiler
    @model,
    # Utilities
    vectorize,
    reconstruct,
    reconstruct!,
    Sample,
    init,
    vectorize,
    OrderedDict,
    # Model
    Model,
    getmissings,
    getargnames,
    generated_quantities,
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
    WrappedDistribution,
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
    # Convenience macros
    @addlogprob!,
    @submodel

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
include("compat/ad.jl")
include("loglikelihoods.jl")
include("submodel_macro.jl")
include("test_utils.jl")
include("transforming.jl")
include("logdensityfunction.jl")

end # module
