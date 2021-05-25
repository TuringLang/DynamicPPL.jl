function tilde_assume(rng, ctx::PrefixContext, sampler, right, left, vn, inds, vi)
    return tilde_assume(
        rng, childcontext(ctx), sampler, right, left, prefix(ctx, vn), inds, vi
    )
end

function tilde_observe(ctx::PrefixContext, sampler, right, left, vi)
    return tilde_observe(childcontext(ctx), sampler, right, left, vi)
end

function dot_tilde_assume(
    rng::Random.AbstractRNG, ctx::PrefixContext, sampler, right, left, vn, inds, vi
)
    return dot_tilde_assume(
        rng,
        childcontext(ctx),
        sampler,
        right,
        left,
        map(Base.Fix1(prefix, ctx), vn),
        inds,
        vi,
    )
end

function dot_tilde_observe(ctx::PrefixContext, sampler, right, left, vi)
    return dot_tilde_observe(childcontext(ctx), sampler, right, left, vi)
end
