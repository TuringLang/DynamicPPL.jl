function tilde(rng, ctx::PrefixContext, sampler, right, left, vn::VarName, inds, vi)
    return tilde(rng, ctx.ctx, sampler, right, left, prefix(ctx, vn), inds, vi)
end

function tilde(ctx::PrefixContext, sampler, right, left, vi)
    return tilde(ctx.ctx, sampler, right, left, vi)
end
