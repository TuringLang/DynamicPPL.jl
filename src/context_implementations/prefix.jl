function tilde_assume(ctx::PrefixContext, right, left, vn, inds, vi)
    return tilde_assume(childcontext(ctx), right, left, prefix(ctx, vn), inds, vi)
end

function tilde_observe(ctx::PrefixContext, right, left, vi)
    return tilde_observe(childcontext(ctx), right, left, vi)
end

function dot_tilde_assume(ctx::PrefixContext, right, left, vn, inds, vi)
    return dot_tilde_assume(
        childcontext(ctx), right, left, map(Base.Fix1(prefix, ctx), vn), inds, vi
    )
end

function dot_tilde_observe(ctx::PrefixContext, right, left, vi)
    return dot_tilde_observe(childcontext(ctx), right, left, vi)
end
