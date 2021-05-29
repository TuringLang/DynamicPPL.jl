function tilde_assume(ctx::MiniBatchContext, right, left, vn, inds, vi)
    return tilde_assume(childcontext(ctx), right, left, vn, inds, vi)
end

function tilde_observe(ctx::MiniBatchContext, right, left, vi)
    return ctx.loglike_scalar * tilde_observe(childcontext(ctx), right, left, vi)
end

function dot_tilde_assume(ctx::MiniBatchContext, right, left, vn, inds, vi)
    return dot_tilde_assume(childcontext(ctx), right, left, vn, inds, vi)
end

function dot_tilde_observe(ctx::MiniBatchContext, right, left, vi)
    return ctx.loglike_scalar * dot_tilde_observe(childcontext(ctx), right, left, vi)
end
