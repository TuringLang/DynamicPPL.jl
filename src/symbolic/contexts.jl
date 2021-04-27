struct SymbolicContext{Ctx} <: DynamicPPL.AbstractContext
    ctx::Ctx
    vn2var::Dict
    vn2rights::Dict
end
SymbolicContext() = SymbolicContext(DefaultContext())
SymbolicContext(ctx) = SymbolicContext(ctx, Dict(), Dict())

# assume
function DynamicPPL.tilde(rng, ctx::SymbolicContext, sampler, right, vn::VarName, inds, vi)
    if Symbolic.issym(right) || (haskey(vi, vn) && Symbolic.issym(vi[vn]))
        # Distribution is symbolic OR variable is.
        ctx.vn2var[vn] = vi[vn]
        ctx.vn2rights[vn] = right
    end

    return DynamicPPL.tilde(rng, ctx.ctx, sampler, right, vn, inds, vi)
end


# TODO: Make it more useful when working with symbolic observations.
# observe
function DynamicPPL.tilde(ctx::SymbolicContext, sampler, right, left, vi)
    if Symbolic.issym(right) || Symbolic.issym(left)
        # TODO: implement
    end

    return DynamicPPL.tilde(ctx.ctx, sampler, right, left, vi)
end
