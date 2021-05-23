function tilde(rng, ctx::LikelihoodContext, sampler, right, left, vn::VarName, inds, vi)
    if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_primitive(
        rng,
        rewrap(childcontext(ctx), EvaluateContext()),
        sampler,
        NoDist(right),
        left,
        vn,
        vi,
    )
end

function tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return tilde_primitive(sampler, right, left, vi)
end

function dot_tilde(rng, ctx::LikelihoodContext, sampler, right, left, vn::VarName, inds, vi)
    if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return dot_tilde_primitive(
        rng,
        rewrap(childcontext(ctx), EvaluateContext()),
        sampler,
        NoDist.(dist),
        left,
        vns,
        vi,
    )
end

function dot_tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return dot_tilde_primitive(sampler, right, left, vi)
end
