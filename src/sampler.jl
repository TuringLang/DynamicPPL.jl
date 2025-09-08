"""
    default_varinfo(rng, model, sampler)

Return a default varinfo object for the given `model` and `sampler`.

The default method for this returns an empty NTVarInfo (i.e. 'typed varinfo').

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function default_varinfo(::Random.AbstractRNG, ::Model, ::AbstractSampler)
    # Note that variable values are unconditionally initialized later, so no
    # point putting them in now.
    return typed_varinfo(VarInfo())
end

"""
    init_strategy(sampler)

Define the initialisation strategy used for generating initial values when
sampling with `sampler`. Defaults to `PriorInit()`, but can be overridden.
"""
init_strategy(::AbstractMCMC.AbstractSampler) = PriorInit()

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
default_chain_type(::AbstractMCMC.AbstractSampler) = Any
