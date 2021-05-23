module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Distributions
using Bijectors

using AbstractMCMC: AbstractMCMC
using ChainRulesCore: ChainRulesCore
using NaturalSort: NaturalSort
using MacroTools: MacroTools
using ZygoteRules: ZygoteRules

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
    getlogp,
    setlogp!,
    acclogp!,
    resetlogp!,
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
    invlink!,
    tonamedtuple,
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
    EvaluationContext,
    SamplingContext,
    LikelihoodContext,
    PriorContext,
    MiniBatchContext,
    PrefixContext,
    assume,
    dot_assume,
    observer,
    dot_observe,
    tilde,
    dot_tilde,
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
    # Convenience macros
    @addlogprob!,
    @submodel

# Reexport
using Distributions: loglikelihood
export loglikelihood

# Used here and overloaded in Turing
function getspace end

# Necessary forward declarations
abstract type AbstractVarInfo <: AbstractModelTrace end
abstract type AbstractContext end

include("utils.jl")
include("selector.jl")
include("model.jl")
include("sampler.jl")
include("varname.jl")
include("distribution_wrappers.jl")
include("contexts.jl")
include("varinfo.jl")
include("threadsafe.jl")
include("context_implementations.jl")
include("compiler.jl")
include("prob_macro.jl")
include("compat/ad.jl")
include("loglikelihoods.jl")
include("submodel_macro.jl")

end # module
