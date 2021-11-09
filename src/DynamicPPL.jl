module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Distributions
using Bijectors

using AbstractMCMC: AbstractMCMC
using ChainRulesCore: ChainRulesCore
using MacroTools: MacroTools
using ZygoteRules: ZygoteRules
using BangBang: BangBang
using Setfield: Setfield

using Setfield: Setfield
using BangBang: BangBang

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

import BangBang: push!!, empty!!, setindex!!

# VarInfo
export AbstractVarInfo,
    VarInfo,
    UntypedVarInfo,
    TypedVarInfo,
    SimpleVarInfo,
    push!!,
    empty!!,
    getlogp,
    setlogp!,
    acclogp!,
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
    set_flag!!,
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
    # Convenience macros
    @addlogprob!,
    @submodel

# Reexport
using Distributions: loglikelihood
export loglikelihood

# Used here and overloaded in Turing
function getspace end

# Necessary forward declarations
"""
    AbstractVarInfo

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

See also: [`VarInfo`](@ref)
"""
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
include("simple_varinfo.jl")
include("context_implementations.jl")
include("compiler.jl")
include("prob_macro.jl")
include("compat/ad.jl")
include("loglikelihoods.jl")
include("submodel_macro.jl")
include("test_utils.jl")

# Deprecations
@deprecate empty!(vi::VarInfo) empty!!(vi::VarInfo)
@deprecate push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution) push!!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution
)
@deprecate push!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, sampler::AbstractSampler
) push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, sampler::AbstractSampler)
@deprecate push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Selector) push!!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Selector
)
@deprecate push!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Set{Selector}
) push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Set{Selector})

@deprecate setlogp!(vi, logp) setlogp!!(vi, logp)
@deprecate acclogp!(vi, logp) acclogp!!(vi, logp)
@deprecate resetlogp!(vi) resetlogp!!(vi)

@deprecate initialize_parameters!(vi, init_params, spl) initialize_parameters!!(
    vi, init_params, spl
)

end # module
