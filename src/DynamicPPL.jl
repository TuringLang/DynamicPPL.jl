module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains, AbstractModel
using Distributions
using Bijectors

import AbstractMCMC
import ChainRulesCore
import NaturalSort
import MacroTools

import Random

import Base: Symbol,
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
export  AbstractVarInfo,
        VarInfo,
        UntypedVarInfo,
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
        unset_flag!,
        setgid!,
        updategid!,
        setorder!,
        istrans,
        link!,
        invlink!,
        tonamedtuple,
#VarName
        VarName,
        inspace,
        subsumes,
# Compiler
        @model,
        @varname,
# Utilities
        vectorize,
        reconstruct,
        reconstruct!,
        Sample,
        init,
        vectorize,
        set_resume!,
# Model
        Model,
        getmissings,
        getargnames,
# Samplers
        Sampler,
        SampleFromPrior,
        SampleFromUniform,
# Contexts
        DefaultContext,
        LikelihoodContext,
        PriorContext,
        MiniBatchContext,
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
# Convenience macros
        @addlogprob!

# Reexport
using Distributions: loglikelihood
export loglikelihood

# Used here and overloaded in Turing
function getspace end

# Necessary forward declarations
abstract type AbstractVarInfo end
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

end # module
