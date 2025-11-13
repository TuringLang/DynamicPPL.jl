"""
    struct DynamicTransformationContext{isinverse} <: AbstractContext

When a model is evaluated with this context, transform the accompanying `AbstractVarInfo` to
constrained space if `isinverse` or unconstrained if `!isinverse`.

Note that some `AbstractVarInfo` types, must notably `VarInfo`, override the
`DynamicTransformationContext` methods with more efficient implementations.
`DynamicTransformationContext` is a fallback for when we need to evaluate the model to know
how to do the transformation.
"""
struct DynamicTransformationContext{isinverse} <: AbstractContext end

function tilde_assume!!(
    ::DynamicTransformationContext{isinverse},
    right::Distribution,
    vn::VarName,
    vi::AbstractVarInfo,
) where {isinverse}
    # vi[vn, right] always provides the value in unlinked space.
    x = vi[vn, right]

    if is_transformed(vi, vn)
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

function tilde_observe!!(
    ::DynamicTransformationContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
