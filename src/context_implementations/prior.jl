function tilde(rng, ctx::PriorContext, sampler, right, left, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_primitive(rng, childcontext(ctx), sampler, right, left, vn, vi)
end

function tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end

function dot_tilde(rng, ctx::PriorContext, sampler, right, left, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return dot_tilde_primitive(rng, childcontext(ctx), sampler, dist, left, vns, vi)
end

function dot_tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
