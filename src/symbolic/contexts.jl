struct SymbolicContext{Ctx} <: DynamicPPL.AbstractContext
    ctx::Ctx
    vn2var::Dict
    vn2rights::Dict
end
SymbolicContext() = SymbolicContext(DefaultContext())
SymbolicContext(ctx) = SymbolicContext(ctx, Dict(), Dict())

# assume
function DynamicPPL.tilde_assume(rng, ctx::SymbolicContext, sampler, right, vn, inds, vi)
    if Symbolic.issym(right) || (haskey(vi, vn) && Symbolic.issym(vi[vn]))
        # Distribution is symbolic OR variable is.
        ctx.vn2var[vn] = vi[vn]
        ctx.vn2rights[vn] = right
    end

    return DynamicPPL.tilde_assume(rng, ctx.ctx, sampler, right, vn, inds, vi)
end

# TODO: Make it more useful when working with symbolic observations.
# observe
function DynamicPPL.tilde_observe(ctx::SymbolicContext, sampler, right, left, vi)
    if Symbolic.issym(right) || Symbolic.issym(left)
        # TODO: implement
    end

    return DynamicPPL.tilde_observe(ctx.ctx, sampler, right, left, vi)
end

function DynamicPPL.assume(dist::Symbolics.Num, vn::VarName, vi)
    if !haskey(vi, vn)
        error("variable $vn does not exist")
    end
    r = vi[vn]
    return r, Bijectors.logpdf_with_trans(dist, vi[vn], DynamicPPL.istrans(vi, vn))
end

function DynamicPPL.observe(right::Symbolics.Num, left, vi)
    return Distributions.loglikelihood(right, left)
end

Symbolics.@register Distributions.loglikelihood(dist, x)
