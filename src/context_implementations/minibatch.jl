function tilde_assume(
    rng, ctx::MiniBatchContext, sampler, right, left, vn, inds, vi
)
    return tilde_assume(rng, childcontext(ctx), sampler, right, left, vn, inds, vi)
end

function tilde_observe(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * tilde_observe(childcontext(ctx), sampler, right, left, vi)
end

function dot_tilde_assume(
    rng, ctx::MiniBatchContext, sampler, right, left, vn, inds, vi
)
    return dot_tilde_assume(rng, childcontext(ctx), sampler, right, left, vn, inds, vi)
end

function dot_tilde_observe(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar *
           dot_tilde_observe(childcontext(ctx), sampler, right, left, vi)
end
