"""
    struct DynamicTransformationContext{isinverse} <: AbstractContext

When a model is evaluated with this context, transform the accompanying `AbstractVarInfo` to
constrained space if `isinverse` or unconstrained if `!isinverse`.

Note that some `AbstractVarInfo` types, must notably `VarInfo`, override the
`DynamicTransformationContext` methods with more efficient implementations.
`DynamicTransformationContext` is a fallback for when we need to evaluate the model to know
how to do the transformation, used by e.g. `SimpleVarInfo`.
"""
struct DynamicTransformationContext{isinverse} <: AbstractContext end
NodeTrait(::DynamicTransformationContext) = IsLeaf()

function tilde_assume(
    ::DynamicTransformationContext{isinverse}, right, vn, vi
) where {isinverse}
    # vi[vn, right] always provides the value in unlinked space.
    x = vi[vn, right]

    if istrans(vi, vn)
        isinverse || @warn "Trying to link an already transformed variable ($vn)"
    else
        isinverse && @warn "Trying to invlink a non-transformed variable ($vn)"
    end

    transform = isinverse ? identity : link_transform(right)
    y, logjac = with_logabsdet_jacobian(transform, x)
    vi = accumulate_assume!!(vi, x, logjac, vn, right)
    vi = setindex!!(vi, y, vn)
    return x, vi
end

function tilde_observe!!(::DynamicTransformationContext, right, left, vn, vi)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end

function link!!(t::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return _transform!!(t, DynamicTransformationContext{false}(), vi, model)
end

function invlink!!(::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return _transform!!(NoTransformation(), DynamicTransformationContext{true}(), vi, model)
end

function _transform!!(
    t::AbstractTransformation,
    ctx::DynamicTransformationContext,
    vi::AbstractVarInfo,
    model::Model,
)
    # To transform using DynamicTransformationContext, we evaluate the model using that as the leaf context:
    model = contextualize(model, setleafcontext(model.context, ctx))
    vi = settrans!!(last(evaluate!!(model, vi)), t)
    return vi
end

function link(t::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return link!!(t, deepcopy(vi), model)
end

function invlink(t::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return invlink!!(t, deepcopy(vi), model)
end
