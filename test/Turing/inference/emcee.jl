###
### Sampler states
###

struct Emcee{space, E<:AMH.Ensemble} <: InferenceAlgorithm
    ensemble::E
end

function Emcee(n_walkers::Int, stretch_length=2.0)
    # Note that the proposal distribution here is just a Normal(0,1)
    # because we do not need AdvancedMH to know the proposal for
    # ensemble sampling.
    prop = AMH.StretchProposal(nothing, stretch_length)
    ensemble = AMH.Ensemble(n_walkers, prop)
    return Emcee{(), typeof(ensemble)}(ensemble)
end

alg_str(::Sampler{<:Emcee}) = "Emcee"

struct EmceeState{V<:AbstractVarInfo,S}
    vi::V
    states::S
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee};
    resume_from = nothing,
    kwargs...
)
    if resume_from !== nothing
        state = loadstate(resume_from)
        return AbstractMCMC.step(rng, model, spl, state; kwargs...)
    end

    # Sample from the prior
    n = spl.alg.ensemble.n_walkers
    vis = [VarInfo(rng, model) for _ in 1:n]

    # Update the parameters if provided.
    if haskey(kwargs, :init_params)
        for vi in vis
            initialize_parameters!(vi, kwargs[:init_params], spl)
        end
    end

    # Update log probability and transform to unconstrained space.
    for vi in vis
        model(rng, vi, spl)
        DynamicPPL.link!(vi, spl)
    end

    # Compute initial transition and states.
    transition = map(vis) do vi
        Transition(vi)
    end
    state = EmceeState(
        vis[1], map(vis) do vi
        AMH.Transition(vi[spl], getlogp(vi))
        end
    )

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    state::EmceeState;
    kwargs...
)
    # Generate a log joint function.
    vi = state.vi
    densitymodel = AMH.DensityModel(gen_logπ(vi, DynamicPPL.SampleFromPrior(), model))

    # Compute the next states.
    states = last(AbstractMCMC.step(rng, densitymodel, spl.alg.ensemble, state.states))

    # Compute the next transition and state.
    transition = map(states) do _state
        DynamicPPL.link!(vi, spl)
        vi[spl] = _state.params
        DynamicPPL.invlink!(vi, spl)
        return Transition(tonamedtuple(vi), _state.lp)
    end
    newstate = EmceeState(vi, states)

    return transition, newstate
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Vector},
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Emcee},
    state::EmceeState,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    kwargs...
)
    # Convert transitions to array format.
    # Also retrieve the variable names.
    params_vec = map(_params_to_array, ts_transform)

    # Extract names and values separately.
    nms = params_vec[1][1]
    vals_vec = [p[2] for p in params_vec]

    # Get the values of the extra parameters in each transition.
    extra_vec = map(get_transition_extras, ts_transform)

    # Get the extra parameter names & values.
    extra_params = extra_vec[1][1]
    extra_values_vec = [e[2] for e in extra_vec]

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = map(x -> hcat(x[1], x[2]), zip(vals_vec, extra_values_vec))
    parray = cat(parray..., dims=3)

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(ts, state, spl)

    # Set up the info tuple.
    if save_state
        info = (range = rng, model = model, spl = spl, samplerstate = state)
    else
        info = NamedTuple()
    end

    # Concretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    return MCMCChains.Chains(
        parray,
        nms,
        extra_params;
        evidence=le,
        info=info,
    ) |> sort
end
