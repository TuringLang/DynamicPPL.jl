module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains, AbstractModel
using Distributions: UnivariateDistribution,
                     MultivariateDistribution,
                     MatrixDistribution,
                     Distribution
using Bijectors: link, invlink
using MacroTools

import Base: string,
             Symbol,
             ==,
             hash,
             in,
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
export  VarName,
        AbstractVarInfo,
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
# Compiler
        ModelGen,
        @model,
        @varname,
        @varinfo,
        @logpdf,
        @sampler,
# Utilities
        vectorize,
        reconstruct,
        reconstruct!,
        Sample,
        Chain,
        init,
        vectorize,
        set_resume!,
# Model
        Model,
        getmissing,
        runmodel!,
# Samplers
        Sampler,
        SampleFromPrior,
        SampleFromUniform,
# Contexts
        DefaultContext,
        LikelihoodContext,
        PriorContext,
        MiniBatchContext,
# Prob macros
        @prob_str,
        @logprob_str

# Used here and overloaded in Turing
function getspace end
function tilde end
function dot_tilde end

include("utils.jl")
include("selector.jl")
include("model.jl")
include("sampler.jl")
include("varname.jl")
include("contexts.jl")
include("varinfo.jl")
include("compiler.jl")
include("prob_macro.jl")

end # module
