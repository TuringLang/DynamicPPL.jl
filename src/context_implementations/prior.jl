function tilde(rng, ctx::PriorContext, sampler, right, left, vn::VarName, inds, vi)
    var = if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        _getvalue(ctx.vars, getsym(vn), inds)
    else
        left
    end
    return tilde_primitive(rng, childcontext(ctx), sampler, right, var, vn, vi)
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
        _getvalue(ctx.vars, getsym(vn), inds)
    else
        left
    end
    return dot_tilde_primitive(rng, childcontext(ctx), sampler, right, var, vns, vi)
end

function dot_tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
