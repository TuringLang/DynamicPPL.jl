function AbstractPPL.condition(
    context::AbstractPPL.AbstractContext, data::InferenceObjects.Dataset
)
    return AbstractPPL.condition(context, NamedTuple(data))
end
function AbstractPPL.condition(
    context::AbstractPPL.AbstractContext, data::InferenceObjects.InferenceData
)
    return AbstractPPL.condition(context, data.posterior)
end
