function StatsBase.predict(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    data::InferenceObjects.Dataset;
    coords=(;),
    kwargs...,
)
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    iters = Iterators.product(axes(data, :draw), axes(data, :chain))
    values = map(iters) do (draw_id, chain_id)
        # Set variables present in `data` and mark those NOT present in data to be resampled.
        DynamicPPL.setval_and_resample!(vi, data, draw_id, chain_id)
        model(rng, vi, spl)
        return map(concretize, DynamicPPL.values_as(vi, NamedTuple))
    end
    coords = merge(coords, dims2coords(Dimensions.dims(data, (:draw, :chain))))
    predictions = InferenceObjects.convert_to_dataset(
        collect(eachcol(values)); group=:posterior_predictive, coords=coords, kwargs...
    )
    pred_keys = filter(∉(keys(data)), keys(predictions))
    isempty(pred_keys) && return nothing
    return predictions[pred_keys]
end
function StatsBase.predict(
    model::DynamicPPL.Model, data::InferenceObjects.Dataset; kwargs...
)
    return StatsBase.predict(Random.GLOBAL_RNG, model, data; kwargs...)
end

function StatsBase.predict(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    data::InferenceObjects.InferenceData;
    coords=(;),
    kwargs...,
)
    if haskey(data, :observed_data)
        coords = merge(coords, dims2coords(Dimensions.dims(data.observed_data)))
    end
    new_groups = Dict{Symbol,InferenceObjects.Dataset}()
    if haskey(data, :posterior)
        posterior_predictive = StatsBase.predict(
            rng, model, data.posterior; coords=coords, kwargs...
        )
        if posterior_predictive === nothing
            @warn "No predictions were made based on posterior. Has the model been deconditioned?"
        else
            new_groups[:posterior_predictive] = posterior_predictive
        end
    end
    if haskey(data, :prior)
        prior_predictive = StatsBase.predict(
            rng, model, data.prior; coords=coords, kwargs...
        )
        if prior_predictive === nothing
            @warn "No predictions were made based on prior. Has the model been deconditioned?"
        else
            new_groups[:prior_predictive] = prior_predictive
        end
    end
    if !(haskey(data, :posterior) || haskey(data, :prior))
        @warn "No posterior or prior found in InferenceData. Returning unmodified input."
        return data
    end
    return merge(data, InferenceObjects.InferenceData(; new_groups...))
end
function StatsBase.predict(
    model::DynamicPPL.Model, data::InferenceObjects.InferenceData; kwargs...
)
    return StatsBase.predict(Random.GLOBAL_RNG, model, data; kwargs...)
end
