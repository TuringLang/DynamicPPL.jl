struct TransformedModel{M,F} <: AbstractModel
    model::M
    transform::F
end

function Bijectors.transformed(model::AbstractModel, b::Bijectors.AbstractBijector)
    return TransformedModel(model, b)
end

(model::TransformedModel)(args...) = (first ∘ evaluate!!)(model, args...)

function evaluate!!(
    model::TransformedModel, varinfo::AbstractVarInfo, context::AbstractContext
)
    transformed_values, logp = forward(model.transform, varinfo.values)
    varinfo_transformed = Setfield.@set(varinfo.values = transformed_values)
    retval, varinfo_new = evaluate!!(model.model, varinfo_transformed, context)

    return retval, acclogp!!(varinfo_new, logp)
end

# Methods we just defer to the underlying `model` property.
MacroTools.@forward TransformedModel.model getargnames, nameof, getmissings
MacroTools.@forward(
    TransformedModel.model, (AbstractPPL.condition, conditioned, AbstractPPL.decondition)
)

function contextualize(model::TransformedModel{<:Model}, context::AbstractContext)
    return Setfield.@set model.model = contextualize(model.model, context)
end
