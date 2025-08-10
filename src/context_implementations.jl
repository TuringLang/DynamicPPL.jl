# assume
function tilde_assume!!(context::AbstractContext, right::Distribution, vn, vi)
    return tilde_assume!!(childcontext(context), right, vn, vi)
end
function tilde_assume!!(::DefaultContext, right::Distribution, vn, vi)
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end
function tilde_assume!!(context::PrefixContext, right::Distribution, vn, vi)
    # Note that we can't use something like this here:
    #     new_vn = prefix(context, vn)
    #     return tilde_assume!!(childcontext(context), right, new_vn, vi)
    # This is because `prefix` applies _all_ prefixes in a given context to a
    # variable name. Thus, if we had two levels of nested prefixes e.g.
    # `PrefixContext{:a}(PrefixContext{:b}(DefaultContext()))`, then the
    # first call would apply the prefix `a.b._`, and the recursive call
    # would apply the prefix `b._`, resulting in `b.a.b._`.
    # This is why we need a special function, `prefix_and_strip_contexts`.
    new_vn, new_context = prefix_and_strip_contexts(context, vn)
    return tilde_assume!!(new_context, right, new_vn, vi)
end

"""
    tilde_assume!!(context, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value and updated `vi`.
"""
function tilde_assume!!(context, right::DynamicPPL.Submodel, vn, vi)
    return _evaluate!!(right, vi, context, vn)
end

# observe
function tilde_observe!!(context::AbstractContext, right, left, vn, vi)
    return tilde_observe!!(childcontext(context), right, left, vn, vi)
end

# `PrefixContext`
function tilde_observe!!(context::PrefixContext, right, left, vn, vi)
    # In the observe case, unlike assume, `vn` may be `nothing` if the LHS is a literal
    # value. For the need for prefix_and_strip_contexts rather than just prefix, see the
    # comment in `tilde_assume!!`.
    new_vn, new_context = if vn !== nothing
        prefix_and_strip_contexts(context, vn)
    else
        vn, childcontext(context)
    end
    return tilde_observe!!(new_context, right, left, new_vn, vi)
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
