function DynamicPPL.generated_quantities(
    mod::DynamicPPL.Model, data::InferenceObjects.Dataset; coords=(;), kwargs...
)
    sample_dims = Dimensions.dims(data, (:draw, :chain))
    diminds = DimensionalData.DimIndices(sample_dims)
    values = map(diminds) do dims
        DynamicPPL.generated_quantities(mod, data[dims...])
    end
    coords = merge(coords, dims2coords(sample_dims))
    return InferenceObjects.convert_to_dataset(
        collect(eachcol(values)); coords=coords, kwargs...
    )
end

function DynamicPPL.generated_quantities(
    mod::DynamicPPL.Model, idata::InferenceObjects.InferenceData; kwargs...
)
    new_groups = Dict{Symbol,InferenceObjects.Dataset}()
    for k in (:posterior, :prior)
        if haskey(idata, k)
            data = idata[k]
            new_groups[k] = merge(
                DynamicPPL.generated_quantities(mod, data; kwargs...), data
            )
        end
    end
    return merge(idata, InferenceObjects.InferenceData(; new_groups...))
end
