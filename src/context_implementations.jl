# assume
function tilde_assume!!(context::AbstractContext, prefix, right::Distribution, vn, vi)
    return tilde_assume!!(childcontext(context), prefix, right, vn, vi)
end
function tilde_assume!!(::DefaultContext, prefix, right::Distribution, vn, vi)
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end

"""
    tilde_assume!!(context, prefix, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value and updated `vi`.
"""
function tilde_assume!!(context, prefix, right::DynamicPPL.Submodel, vn, vi)
    return _evaluate!!(right, vi, context, prefix, vn)
end

# observe
function tilde_observe!!(context::AbstractContext, right, left, vn, vi)
    return tilde_observe!!(childcontext(context), right, left, vn, vi)
end

"""
    tilde_observe!!(context, right, left, vn, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value and updated `vi`.

Falls back to `tilde_observe!!(context, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!!(::DefaultContext, right::Distribution, left, vn, vi)
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end

function tilde_observe!!(::DefaultContext, ::DynamicPPL.Submodel, left, vn, vi)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is observed"))
end
