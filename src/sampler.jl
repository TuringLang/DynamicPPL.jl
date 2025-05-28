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

# Initializations.
init(rng, dist, ::SampleFromPrior) = rand(rng, dist)
function init(rng, dist, ::SampleFromUniform)
    return istransformable(dist) ? inittrans(rng, dist) : rand(rng, dist)
end

init(rng, dist, ::SampleFromPrior, n::Int) = rand(rng, dist, n)
function init(rng, dist, ::SampleFromUniform, n::Int)
    return istransformable(dist) ? inittrans(rng, dist, n) : rand(rng, dist, n)
end

# TODO(mhauru) Could we get rid of Sampler now that it's just a wrapper around `alg`?
# (Selector has been removed).
"""
    Sampler{T}

Generic sampler type for inference algorithms of type `T` in DynamicPPL.

`Sampler` should implement the AbstractMCMC interface, and in particular
`AbstractMCMC.step`. A default implementation of the initial sampling step is
provided that supports resuming sampling from a previous state and setting
initial parameter values. It requires you to to overload [`loadstate`](@ref)
for loading previous states. Additionally, sometimes one might want to
implement [`initialsampler`](@ref) that specifies how the initial parameter
values are sampled if they are not provided. By default, values are sampled
from the prior.
"""
struct Sampler{T} <: AbstractSampler
    alg::T
end

# AbstractMCMC interface for SampleFromUniform and SampleFromPrior
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    sampler::Union{SampleFromUniform,SampleFromPrior},
    state=nothing;
    kwargs...,
)
    ctx = SamplingContext(rng, sampler)
    _, vi = DynamicPPL.evaluate!!(ldf.model, ldf.varinfo, ctx)
    return vi, nothing
end

"""
    default_varinfo(rng, model, sampler[, context])

Return a default varinfo object for the given `model` and `sampler`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.
- `initial_params::Union{AbstractVector,Nothing}`: Initial parameter values to
be set in the varinfo object.
- `link::Bool`: Whether to link the varinfo.
- `context::AbstractContext`: Context in which the model is evaluated. Defaults
to `DefaultContext()`.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function default_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler,
    initial_params::Union{AbstractVector,Nothing}=nothing,
    link::Bool=false,
    context::AbstractContext=DefaultContext(),
)
    init_sampler = initialsampler(sampler)
    vi = typed_varinfo(rng, model, init_sampler, context)

    # Update the parameters if provided.
    if initial_params !== nothing
        vi = initialize_parameters!!(vi, initial_params, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(evaluate!!(model, vi, DefaultContext()))
    end

    return vi
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
    initialsampler(sampler::Sampler)

Return the sampler that is used for generating the initial parameters when sampling with
`sampler`.

By default, it returns an instance of [`SampleFromPrior`](@ref).
"""
initialsampler(spl::Sampler) = SampleFromPrior()

"""
    set_initial_values(varinfo::AbstractVarInfo, initial_params::AbstractVector)
    set_initial_values(varinfo::AbstractVarInfo, initial_params::NamedTuple)

Take the values inside `initial_params`, replace the corresponding values in
the given VarInfo object, and return a new VarInfo object with the updated values.

This differs from `DynamicPPL.unflatten` in two ways:

1. It works with `NamedTuple` arguments.
2. For the `AbstractVector` method, if any of the elements are missing, it will not
overwrite the original value in the VarInfo (it will just use the original
value instead).
"""
function set_initial_values(varinfo::AbstractVarInfo, initial_params::AbstractVector)
    throw(
        ArgumentError(
            "`initial_params` must be a vector of type `Union{Real,Missing}`. " *
            "If `initial_params` is a vector of vectors, please flatten it (e.g. using `vcat`) first.",
        ),
    )
end

function set_initial_values(
    varinfo::AbstractVarInfo, initial_params::AbstractVector{<:Union{Real,Missing}}
)
    flattened_param_vals = varinfo[:]
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
    new_varinfo = unflatten(varinfo, flattened_param_vals)
    return new_varinfo
end

function set_initial_values(varinfo::AbstractVarInfo, initial_params::NamedTuple)
    varinfo = deepcopy(varinfo)
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

function initialize_parameters!!(vi::AbstractVarInfo, initial_params, model::Model)
    @debug "Using passed-in initial variable values" initial_params

    # `link` the varinfo if needed.
    linked = islinked(vi)
    if linked
        vi = invlink!!(vi, model)
    end

    # Set the values in `vi`.
    vi = set_initial_values(vi, initial_params)

    # `invlink` if needed.
    if linked
        vi = link!!(vi, model)
    end

    return vi
end

# TODO: Get rid of this
function initialstep(args...; kwargs...)
    return error("no initialstep")
end
