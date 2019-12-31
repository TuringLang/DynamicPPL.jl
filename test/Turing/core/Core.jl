module Core

using Libtask, ForwardDiff, Random
using Distributions, LinearAlgebra
using ..Utilities, Reexport
using Tracker: Tracker
using ..Turing: Turing, Model, runmodel!,
    AbstractSampler, Sampler, SampleFromPrior
using LinearAlgebra: copytri!
using DistributionsAD
using StatsFuns: logsumexp, softmax
using Bijectors: PDMatDistribution

using DynamicPPL

include("container.jl")
include("ad.jl")

export  @model,
        @varname,
        ParticleContainer,
        Particle,
        Trace,
        fork,
        forkr,
        current_trace,
        getweights,
        effectiveSampleSize,
        increase_logweight,
        inrease_logevidence,
        resample!,
        ResampleWithESSThreshold,
        ADBackend,
        setadbackend,
        setadsafe,
        ForwardDiffAD,
        TrackerAD,
        value,
        gradient_logp,
        CHUNKSIZE,
        ADBACKEND,
        setchunksize,
        verifygrad,
        gradient_logp_forward,
        gradient_logp_reverse,
        @varinfo,
        @logpdf,
        @sampler

end # module
