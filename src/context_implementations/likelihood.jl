function tilde(rng, ctx::LikelihoodContext, sampler, right, left, vn::VarName, inds, vi)
    if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_primitive(
        rng,
        rewrap(childcontext(ctx), EvaluationContext()),
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

function dot_tilde(
    rng,
    ctx::LikelihoodContext,
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
    return dot_tilde_primitive(
        rng,
        rewrap(childcontext(ctx), EvaluationContext()),
        sampler,
        NoDist.(right),
        var,
        vns,
        vi,
    )
end

function dot_tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return dot_tilde_primitive(sampler, right, left, vi)
end
