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

Generic sampler type for inference algorithms in DynamicPPL.
"""
struct Sampler{T} <: AbstractSampler
    alg::T
    # TODO: remove selector & add space
    selector::Selector
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
        model(rng, vi, _spl)
    end

    return initialstep(rng, model, spl, vi; kwargs...)
end

function loadstate end

initialsampler(spl::Sampler) = SampleFromPrior()

function initialstep end

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
    length(theta) == length(init_theta_flat) ||
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
