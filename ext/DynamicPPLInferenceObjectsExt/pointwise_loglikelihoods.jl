function DynamicPPL.pointwise_loglikelihoods(
    model::DynamicPPL.Model, data::InferenceObjects.Dataset; coords=(;), kwargs...
)
    # Get the data by executing the model once
    vi = DynamicPPL.VarInfo(model)
    context = DynamicPPL.PointwiseLikelihoodContext(Dict{String,Vector{Float64}}())

    iters = Iterators.product(axes(data, :draw), axes(data, :chain))
    for (draw, chain) in iters
        # Update the values
        DynamicPPL.setval!(vi, data, draw, chain)

        # Execute model
        model(vi, context)
    end

    ndraws = size(data, :draw)
    nchains = size(data, :chain)
    # TODO: optionally post-process idata to convert index variables like Symbol("y[1]") to Symbol("y")
    loglikelihoods = Dict(
        varname => reshape(logliks, ndraws, nchains) for
        (varname, logliks) in context.loglikelihoods
    )
    isempty(loglikelihoods) && return nothing
    coords = merge(coords, dims2coords(Dimensions.dims(data, (:draw, :chain))))
    return InferenceObjects.convert_to_dataset(
        loglikelihoods; group=:log_likelihood, coords=coords, kwargs...
    )
end
function DynamicPPL.pointwise_loglikelihoods(
    model::DynamicPPL.Model, data::InferenceObjects.InferenceData; kwargs...
)
    log_likelihood = DynamicPPL.pointwise_loglikelihoods(model, data.posterior; kwargs...)
    return merge(data, InferenceObjects.InferenceData(; log_likelihood=log_likelihood))
end
