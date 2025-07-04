# TODO(mhauru) Could we get rid of Sampler now that it's just a wrapper around `alg`?
# (Selector has been removed).
"""
    Sampler{T}

Generic sampler type for inference algorithms of type `T` in DynamicPPL.

`Sampler` should implement the AbstractMCMC interface, and in particular
`AbstractMCMC.step`. A default implementation of the initial sampling step is
provided that supports resuming sampling from a previous state and setting initial
parameter values. It requires to overload [`loadstate`](@ref) and [`initialstep`](@ref)
for loading previous states and actually performing the initial sampling step,
respectively. Additionally, sometimes one might want to implement [`initialsampler`](@ref)
that specifies how the initial parameter values are sampled if they are not provided.
By default, values are sampled from the prior.
"""
struct Sampler{T} <: AbstractSampler
    alg::T
end

"""
    default_varinfo(rng, model, sampler)

Return a default varinfo object for the given `model` and `sampler`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function default_varinfo(rng::Random.AbstractRNG, model::Model, sampler::AbstractSampler)
    init_sampler = initialsampler(sampler)
    return typed_varinfo(rng, model, init_sampler)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Sampler,
    N::Integer;
    chain_type=default_chain_type(sampler),
    resume_from=nothing,
    initial_state=loadstate(resume_from),
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng, model, sampler, N; chain_type, initial_state, kwargs...
    )
end

# initial step: general interface for resuming and
function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Sampler; initial_params=nothing, kwargs...
)
    # Sample initial values.
    vi = default_varinfo(rng, model, spl)

    # Update the parameters if provided.
    if initial_params !== nothing
        vi = initialize_parameters!!(vi, initial_params, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(evaluate!!(model, vi))
    end

    return initialstep(rng, model, spl, vi; initial_params, kwargs...)
end

"""
    loadstate(data)

Load sampler state from `data`.

By default, `data` is returned.
"""
loadstate(data) = data

"""
    default_chain_type(sampler)

Default type of the chain of posterior samples from `sampler`.
"""
default_chain_type(sampler::Sampler) = Any

"""
    init_strategy(sampler)

Define the initialisation strategy used for generating initial values when
sampling with `sampler`. Defaults to `PriorInit()`, but can be overridden.
"""
init_strategy(::Sampler) = PriorInit()

"""
    initialstep(rng, model, sampler, varinfo; kwargs...)

Perform the initial sampling step of the `sampler` for the `model`.

The `varinfo` contains the initial samples, which can be provided by the user or
sampled randomly.
"""
function initialstep end
