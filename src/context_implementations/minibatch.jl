function tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * tilde(ctx.ctx, sampler, right, left, vi)
end

function tilde(rng, ctx::MiniBatchContext, sampler, right, left, vn::VarName, inds, vi)
    return tilde(rng, ctx.ctx, sampler, right, left, vn, inds, vi)
end

function dot_tilde(rng, ctx::MiniBatchContext, sampler, right, left, vn::VarName, inds, vi)
    return dot_tilde(rng, ctx.ctx, sampler, right, left, vn, inds, vi)
end

function dot_tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * dot_tilde(ctx.ctx, sampler, right, left, vi)
end

