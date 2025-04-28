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
    r = vi[vn, right]
    lp = Bijectors.logpdf_with_trans(right, r, !isinverse)

    if istrans(vi, vn)
        isinverse || @warn "Trying to link an already transformed variable ($vn)"
    else
        isinverse && @warn "Trying to invlink a non-transformed variable ($vn)"
    end

    # Only transform if `!isinverse` since `vi[vn, right]`
    # already performs the inverse transformation if it's transformed.
    r_transformed = isinverse ? r : link_transform(right)(r)
    if hasacc(vi, Val(:LogPrior))
        vi = acclogprior!!(vi, lp)
    end
    return r, setindex!!(vi, r_transformed, vn)
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
    # To transform using DynamicTransformationContext, we evaluate the model, but we do not
    # need to use any accumulators other than LogPriorAccumulator (which is affected by the Jacobian of
    # the transformation).
    accs = getaccs(vi)
    has_logprior = haskey(accs, Val(:LogPrior))
    if has_logprior
        old_logprior = getacc(accs, Val(:LogPrior))
        vi = setaccs!!(vi, (old_logprior,))
    end
    vi = settrans!!(last(evaluate!!(model, vi, ctx)), t)
    # Restore the accumulators.
    if has_logprior
        new_logprior = getacc(vi, Val(:LogPrior))
        accs = setacc!!(accs, new_logprior)
    end
    vi = setaccs!!(vi, accs)
    return vi
end

function link(t::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return link!!(t, deepcopy(vi), model)
end

function invlink(t::DynamicTransformation, vi::AbstractVarInfo, model::Model)
    return invlink!!(t, deepcopy(vi), model)
end
