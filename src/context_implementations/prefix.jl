function tilde(rng, ctx::PrefixContext, sampler, right, left, vn::VarName, inds, vi)
    return tilde(rng, childcontext(ctx), sampler, right, left, prefix(ctx, vn), inds, vi)
end

function tilde(ctx::PrefixContext, sampler, right, left, vi)
    return tilde(childcontext(ctx), sampler, right, left, vi)
end

function dot_tilde(ctx::PrefixContext, sampler, right, left, vi)
    return dot_tilde(childcontext(ctx), sampler, right, left, vi)
end
function dot_tilde(
    rng::Random.AbstractRNG, ctx::PrefixContext, sampler, right, left, vn, inds, vi
)
    return dot_tilde(
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
