function tilde_assume(ctx::PriorContext, right, left, vn, inds, vi)
    var = if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        _getvalue(ctx.vars, getsym(vn), inds)
    else
        left
    end
    return tilde_assume(childcontext(ctx), right, var, vn, inds, vi)
end

function tilde_observe(ctx::PriorContext, right, left, vi)
    return 0
end

function dot_tilde_assume(
    ctx::PriorContext, right, left, vns::AbstractArray{<:VarName{sym}}, inds, vi
) where {sym}
    var = if ctx.vars isa NamedTuple && haskey(ctx.vars, sym)
        _getvalue(ctx.vars, getsym(vn), inds)
    else
        left
    end
    return dot_tilde_assume(childcontext(ctx), right, var, vns, inds, vi)
end

function dot_tilde_observe(ctx::PriorContext, right, left, vi)
    return 0
end
