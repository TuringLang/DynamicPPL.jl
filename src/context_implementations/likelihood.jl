function tilde_assume(
    rng, ctx::LikelihoodContext, sampler, right, left, vn::VarName, inds, vi
)
    var = if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        _getvalue(ctx.vars, getsym(vn), inds)
    else
        left
    end
    return tilde_assume(
        rng,
        rewrap(childcontext(ctx), EvaluationContext()),
        sampler,
        NoDist(right),
        var,
        vn,
        inds,
        vi,
    )
end

function tilde_observe(ctx::LikelihoodContext, sampler, right, left, vi)
    return tilde_observe(childcontext(ctx), sampler, right, left, vi)
end

function dot_tilde_assume(
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
        _getvalue(ctx.vars, sym, inds)
    else
        left
    end
    return dot_tilde_assume(
        rng,
        rewrap(childcontext(ctx), EvaluationContext()),
        sampler,
        NoDist.(right),
        var,
        vns,
        inds,
        vi,
    )
end

function dot_tilde_observe(ctx::LikelihoodContext, sampler, right, left, vi)
    return dot_tilde_observe(childcontext(ctx), sampler, right, left, vi)
end
