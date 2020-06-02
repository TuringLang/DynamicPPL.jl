"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct SampleFromUniform{Tvi <: AbstractVarInfo} <: AbstractSampler
    vi::Tvi
end
function SampleFromUniform(model::AbstractModel; specialize_after=1)
    return SampleFromUniform(VarInfo(model; specialize_after=specialize_after))
end
SampleFromUniform() = SampleFromUniform(VarInfo())

struct SampleFromPrior{Tvi <: AbstractVarInfo} <: AbstractSampler
    vi::Tvi
end
function SampleFromPrior(model::AbstractModel; specialize_after=1)
    return SampleFromPrior(VarInfo(model; specialize_after=specialize_after))
end
SampleFromPrior() = SampleFromPrior(VarInfo())

getspace(::Union{SampleFromPrior, SampleFromUniform}) = ()

# Initializations.
init(dist, ::SampleFromPrior) = rand(dist)
init(dist, ::SampleFromUniform) = istransformable(dist) ? inittrans(dist) : rand(dist)

init(dist, ::SampleFromPrior, n::Int) = rand(dist, n)
function init(dist, ::SampleFromUniform, n::Int)
    return istransformable(dist) ? inittrans(dist, n) : rand(dist, n)
end

"""
    has_eval_num(spl::AbstractSampler)

Check whether `spl` has a field called `eval_num` in its state variables or not.
"""
has_eval_num(spl::SampleFromUniform) = false
has_eval_num(spl::SampleFromPrior) = false
has_eval_num(spl::AbstractSampler) = :eval_num in fieldnames(typeof(spl.state))

"""
An abstract type that mutable sampler state structs inherit from.
"""
abstract type AbstractSamplerState end

"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

DynamicPPL translates models to chunks that call the modelling functions at specified points.
The dispatch is based on the value of a `sampler` variable.
To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
mutable struct Sampler{T, S<:AbstractSamplerState} <: AbstractSampler
    alg      ::  T
    info     ::  Dict{Symbol, Any} # sampler infomation
    selector ::  Selector
    state    ::  S
end
Sampler(alg) = Sampler(alg, Selector())
Sampler(alg, model::Model; specialize_after=1) = Sampler(alg, model, Selector(); specialize_after=specialize_after)

# AbstractMCMC interface for SampleFromUniform and SampleFromPrior

function AbstractMCMC.step!(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Union{SampleFromUniform,SampleFromPrior},
    ::Integer,
    transition;
    kwargs...
)
    empty!(sampler.vi)
    model(sampler.vi, sampler)
    return sampler.vi
end
getinferred(spl::Sampler) = getinferred(spl.state.vi)
Base.empty!(spl::Sampler) = empty!(spl.state.vi)