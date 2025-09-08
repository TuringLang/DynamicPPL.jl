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

"""
    _convert_initial_params(initial_params)

Convert `initial_params` to an `AbstractInitStrategy` if it is not already one.
"""
_convert_initial_params(initial_params::AbstractInitStrategy) = initial_params
function _convert_initial_params(nt::NamedTuple)
    @info "Using a NamedTuple for `initial_params` will be deprecated in a future release. Please use `InitFromParams(namedtuple)` instead."
    return InitFromParams(nt)
end
function _convert_initial_params(d::AbstractDict{<:VarName})
    @info "Using a Dict for `initial_params` will be deprecated in a future release. Please use `InitFromParams(dict)` instead."
    return InitFromParams(d)
end
function _convert_initial_params(::AbstractVector)
    errmsg = "`initial_params` must be a `NamedTuple`, an `AbstractDict{<:VarName}`, or ideally an `AbstractInitStrategy`. Using a vector of parameters for `initial_params` is no longer supported. Please see https://turinglang.org/docs/usage/sampling-options/#specifying-initial-parameters for details on how to update your code."
    throw(ArgumentError(errmsg))
end

"""
    loadstate(chain::AbstractChains)

Load sampler state from an `AbstractChains` object. This function should be overloaded by a
concrete Chains implementation.
"""
function loadstate end
loadstate(data) = data

"""
    default_chain_type(sampler)

Default type of the chain of posterior samples from `sampler`.
"""
default_chain_type(::AbstractSampler) = Any
