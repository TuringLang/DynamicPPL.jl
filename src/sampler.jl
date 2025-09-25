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
respectively. Additionally, sometimes one might want to implement an [`init_strategy`](@ref)
that specifies how the initial parameter values are sampled if they are not provided.
By default, values are sampled from the prior.
"""
struct Sampler{T} <: AbstractSampler
    alg::T
end

"""
    default_varinfo(rng, model, sampler)

Return a default varinfo object for the given `model` and `sampler`.

The default method for this returns a NTVarInfo (i.e. 'typed varinfo').

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function default_varinfo(rng::Random.AbstractRNG, model::Model, ::AbstractSampler)
    # Note that in `AbstractMCMC.step`, the values in the varinfo returned here are
    # immediately overwritten by a subsequent call to `init!!`. The reason why we
    # _do_ create a varinfo with parameters here (as opposed to simply returning
    # an empty `typed_varinfo(VarInfo())`) is to avoid issues where pushing to an empty
    # typed VarInfo would fail. This can happen if two VarNames have different types
    # but share the same symbol (e.g. `x.a` and `x.b`).
    # TODO(mhauru) Fix push!! to work with arbitrary lens types, and then remove the arguments
    # and return an empty VarInfo instead.
    return typed_varinfo(VarInfo(rng, model))
end

"""
    init_strategy(sampler::AbstractSampler)

Define the initialisation strategy used for generating initial values when
sampling with `sampler`. Defaults to `InitFromPrior()`, but can be overridden.
"""
init_strategy(::AbstractSampler) = InitFromPrior()

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Sampler,
    N::Integer;
    chain_type=default_chain_type(sampler),
    resume_from=nothing,
    initial_params=init_strategy(sampler),
    initial_state=loadstate(resume_from),
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng, model, sampler, N; chain_type, initial_params, initial_state, kwargs...
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Sampler,
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    chain_type=default_chain_type(sampler),
    initial_params=fill(init_strategy(sampler), nchains),
    resume_from=nothing,
    initial_state=loadstate(resume_from),
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        parallel,
        N,
        nchains;
        chain_type,
        initial_params,
        initial_state,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler;
    initial_params::AbstractInitStrategy=init_strategy(spl),
    kwargs...,
)
    # Generate the default varinfo. Note that any parameters inside this varinfo
    # will be immediately overwritten by the next call to `init!!`.
    vi = default_varinfo(rng, model, spl)

    # Fill it with initial parameters. Note that, if `InitFromParams` is used, the
    # parameters provided must be in unlinked space (when inserted into the
    # varinfo, they will be adjusted to match the linking status of the
    # varinfo).
    _, vi = init!!(rng, model, vi, initial_params)

    # Call the actual function that does the first step.
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
default_chain_type(::Sampler) = Any

"""
    initialstep(rng, model, sampler, varinfo; kwargs...)

Perform the initial sampling step of the `sampler` for the `model`.

The `varinfo` contains the initial samples, which can be provided by the user or
sampled randomly.
"""
function initialstep end
