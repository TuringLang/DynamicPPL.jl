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

function dot_tilde(
    rng,
    ctx::PriorContext,
    sampler,
    right,
    left,
    vns::AbstractArray{<:VarName{sym}},
    inds,
    vi,
) where {sym}
    var = if ctx.vars isa NamedTuple && haskey(ctx.vars, sym)
        _getindex(getfield(ctx.vars, sym), inds)
    else
        vi[vns]
    end
    return dot_tilde_primitive(rng, childcontext(ctx), sampler, right, var, vns, vi)
end

function dot_tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
