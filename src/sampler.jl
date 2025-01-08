# TODO: Make `UniformSampling` and `Prior` algs + just use `Sampler`
# That would let us use all defaults for Sampler, combine it with other samplers etc.
"""
    SampleFromUniform

Sampling algorithm that samples unobserved random variables from a uniform distribution.

# References

[Stan reference manual](https://mc-stan.org/docs/2_28/reference-manual/initialization.html#random-initial-values)
"""
struct SampleFromUniform <: AbstractSampler end

"""
    SampleFromPrior

Sampling algorithm that samples unobserved random variables from their prior distribution.
"""
struct SampleFromPrior <: AbstractSampler end

getspace(::Union{SampleFromPrior,SampleFromUniform}) = ()

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
    state=nothing;
    kwargs...,
)
    vi = VarInfo()
    model(rng, vi, sampler)
    return vi, nothing
end

"""
    default_varinfo(rng, model, sampler[, context])

Return a default varinfo object for the given `model` and `sampler`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.
- `context::AbstractContext`: Context in which the model is evaluated.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function default_varinfo(rng::Random.AbstractRNG, model::Model, sampler::AbstractSampler)
    return default_varinfo(rng, model, sampler, DefaultContext())
end
function default_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler,
    context::AbstractContext,
)
    init_sampler = initialsampler(sampler)
    return VarInfo(rng, model, init_sampler, context)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Sampler,
    N::Integer;
    chain_type=default_chain_type(sampler),
    initial_state=nothing,
    kwargs...,
)
    if haskey(kwargs, :resume_from)
        throw(
            ArgumentError(
                "The `resume_from` keyword argument is no longer supported. Please use `initial_state=loadstate(chain)` instead of `resume_from=chain`.",
            ),
        )
    end
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
        vi = initialize_parameters!!(vi, initial_params, spl, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(evaluate!!(model, vi, DefaultContext()))
    end

    return initialstep(rng, model, spl, vi; initial_params, kwargs...)
end

"""
    loadstate(data)

Load sampler state from `data`.

If `data` isa MCMCChains.Chains object, this attempts to fetch the last state
of the sampler from the metadata stored inside the Chains object. This requires
you to have passed the `save_state=true` keyword argument to the `sample()`
when generating the chain.

This function can be overloaded for specific types of `data` if desired. If
there is no specific implementation for a given type, it falls back to just
returning `data`, i.e. acts as an identity function.
"""
loadstate(data) = data

"""
    default_chain_type(sampler)

Default type of the chain of posterior samples from `sampler`.
"""
default_chain_type(sampler::Sampler) = Any

"""
    initialsampler(sampler::Sampler)

Return the sampler that is used for generating the initial parameters when sampling with
`sampler`.

By default, it returns an instance of [`SampleFromPrior`](@ref).
"""
initialsampler(spl::Sampler) = SampleFromPrior()

function set_values!!(
    varinfo::AbstractVarInfo, initial_params::AbstractVector, spl::AbstractSampler
)
    throw(
        ArgumentError(
            "`initial_params` must be a vector of type `Union{Real,Missing}`. " *
            "If `initial_params` is a vector of vectors, please flatten it (e.g. using `vcat`) first.",
        ),
    )
end

function set_values!!(
    varinfo::AbstractVarInfo,
    initial_params::AbstractVector{<:Union{Real,Missing}},
    spl::AbstractSampler,
)
    flattened_param_vals = varinfo[spl]
    length(flattened_param_vals) == length(initial_params) || throw(
        DimensionMismatch(
            "Provided initial value size ($(length(initial_params))) doesn't match " *
            "the model size ($(length(flattened_param_vals))).",
        ),
    )

    # Update values that are provided.
    for i in eachindex(initial_params)
        x = initial_params[i]
        if x !== missing
            flattened_param_vals[i] = x
        end
    end

    # Update in `varinfo`.
    return setindex!!(varinfo, flattened_param_vals, spl)
end

function set_values!!(
    varinfo::AbstractVarInfo, initial_params::NamedTuple, spl::AbstractSampler
)
    vars_in_varinfo = keys(varinfo)
    for v in keys(initial_params)
        vn = VarName{v}()
        if !(vn in vars_in_varinfo)
            for vv in vars_in_varinfo
                if subsumes(vn, vv)
                    throw(
                        ArgumentError(
                            "The current model contains sub-variables of $v, such as ($vv). " *
                            "Using NamedTuple for initial_params is not supported in such a case. " *
                            "Please use AbstractVector for initial_params instead of NamedTuple.",
                        ),
                    )
                end
            end
            throw(ArgumentError("Variable $v not found in the model."))
        end
    end
    initial_params = NamedTuple(k => v for (k, v) in pairs(initial_params) if v !== missing)
    return update_values!!(
        varinfo, initial_params, map(k -> VarName{k}(), keys(initial_params))
    )
end

function initialize_parameters!!(
    vi::AbstractVarInfo, initial_params, spl::AbstractSampler, model::Model
)
    @debug "Using passed-in initial variable values" initial_params

    # `link` the varinfo if needed.
    linked = islinked(vi, spl)
    if linked
        vi = invlink!!(vi, spl, model)
    end

    # Set the values in `vi`.
    vi = set_values!!(vi, initial_params, spl)

    # `invlink` if needed.
    if linked
        vi = link!!(vi, spl, model)
    end

    return vi
end

"""
    initialstep(rng, model, sampler, varinfo; kwargs...)

Perform the initial sampling step of the `sampler` for the `model`.

The `varinfo` contains the initial samples, which can be provided by the user or
sampled randomly.
"""
function initialstep end
