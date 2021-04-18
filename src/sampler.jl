# TODO: Make `UniformSampling` and `Prior` algs + just use `Sampler`
# That would let us use all defaults for Sampler, combine it with other samplers etc.
"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct SampleFromUniform <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

getspace(::Union{SampleFromPrior, SampleFromUniform}) = ()

# Initializations.
init(rng, dist, ::SampleFromPrior) = rand(rng, dist)
function init(rng, dist, ::SampleFromUniform)
    return istransformable(dist) ? inittrans(rng, dist) : rand(rng, dist)
end

init(rng, dist, ::SampleFromPrior, n::Int) = rand(rng, dist, n)
function init(rng, dist, ::SampleFromUniform, n::Int)
    return istransformable(dist) ? inittrans(rng, dist, n) : rand(rng, dist, n)
end

"""
    Sampler{T}

Generic sampler type for inference algorithms of type `T` in DynamicPPL.

`Sampler` should implement the AbstractMCMC interface, and in particular
[`AbstractMCMC.step`](@ref). A default implementation of the initial sampling step is
provided that supports resuming sampling from a previous state and setting initial
parameter values. It requires to overload [`loadstate`](@ref) and [`initialstep`](@ref)
for loading previous states and actually performing the initial sampling step,
respectively. Additionally, sometimes one might want to implement [`initialsampler`](@ref)
that specifies how the initial parameter values are sampled if they are not provided.
By default, values are sampled from the prior.
"""
struct Sampler{T} <: AbstractSampler
    alg::T
    selector::Selector # Can we remove it?
    # TODO: add space such that we can integrate existing external samplers in DynamicPPL
end
Sampler(alg) = Sampler(alg, Selector())
Sampler(alg, model::Model) = Sampler(alg, model, Selector())
Sampler(alg, model::Model, s::Selector) = Sampler(alg, s)

# AbstractMCMC interface for SampleFromUniform and SampleFromPrior
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Union{SampleFromUniform,SampleFromPrior},
    state = nothing;
    kwargs...
)
    vi = VarInfo()
    model(rng, vi, sampler)
    return vi, nothing
end

# initial step: general interface for resuming and
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler;
    resume_from = nothing,
    kwargs...
)
    if resume_from !== nothing
        state = loadstate(resume_from)
        return AbstractMCMC.step(rng, model, spl, state; kwargs...)
    end

    # Sample initial values.
    _spl = initialsampler(spl)
    vi = VarInfo(rng, model, _spl)

    # Update the parameters if provided.
    if haskey(kwargs, :init_params)
        initialize_parameters!(vi, kwargs[:init_params], spl)

        # Update joint log probability.
        # TODO: fix properly by using sampler and evaluation contexts
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        if _spl isa SampleFromUniform
            model(rng, vi, SampleFromPrior())
        else
            model(rng, vi, _spl)
        end
    end

    return initialstep(rng, model, spl, vi; kwargs...)
end

"""
    loadstate(data)

Load sampler state from `data`.
"""
function loadstate end

"""
    initialsampler(sampler::Sampler)

Return the sampler that is used for generating the initial parameters when sampling with
`sampler`.

By default, it returns an instance of [`SampleFromPrior`](@ref).
"""
initialsampler(spl::Sampler) = SampleFromPrior()

function initialize_parameters!(vi::AbstractVarInfo, init_params, spl::Sampler)
    @debug "Using passed-in initial variable values" init_params

    # Flatten parameters.
    init_theta = mapreduce(vcat, init_params) do x
        vec([x;])
    end

    # Get all values.
    linked = islinked(vi, spl)
    linked && invlink!(vi, spl)
    theta = vi[spl]
    length(theta) == length(init_theta) ||
        error("Provided initial value doesn't match the dimension of the model")

    # Update values that are provided.
    for i in 1:length(init_theta)
        x = init_theta[i]
        if x !== missing
            theta[i] = x
        end
    end

    # Update in `vi`.
    vi[spl] = theta
    linked && link!(vi, spl)

    return
end

"""
    initialstep(rng, model, sampler, varinfo; kwargs...)

Perform the initial sampling step of the `sampler` for the `model`.

The `varinfo` contains the initial samples, which can be provided by the user or
sampled randomly.
"""
function initialstep end
