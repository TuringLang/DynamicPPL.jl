module Symbolic

import ..DynamicPPL
import ..DynamicPPL: Model, VarInfo, AbstractSampler, SampleFromPrior, VarName, DefaultContext

import Random
import Bijectors
using Distributions
import Symbolics
import Symbolics: SymbolicUtils

issym(x::Union{Symbolics.Num, SymbolicUtils.Symbolic}) = true
issym(x) = false

include("rules.jl")
include("contexts.jl")

symbolize(args...; kwargs...) = symbolize(Random.GLOBAL_RNG, args...; kwargs...)
function symbolize(
    rng::Random.AbstractRNG,
    m::Model,
    vi::VarInfo = VarInfo(m);
    spl = SampleFromPrior(),
    ctx = DefaultContext(),
    include_data = false
)
    m(rng, vi, spl, ctx);
    θ_orig = vi[spl]

    # Symbolic `logpdf` for fixed observations.
    Symbolics.@variables θ[1:length(θ_orig)]
    vi = VarInfo{Real}(vi, spl, θ, 0.0);
    m(vi, ctx);

    return vi, θ
end

function dependencies(ctx::SymbolicContext, vn::VarName)
    right = ctx.vn2rights[vn]
    r = Symbolics.value(right)

    if !issym(r)
        # No dependencies.
        return []
    end

    args = SymbolicUtils.arguments(r)
    return mapreduce(vcat, args) do a
        Symbolics.get_variables(a)
    end
end
function dependencies(ctx::SymbolicContext, symbolic = false)
    vn2var = ctx.vn2var
    var2vn = Dict(values(vn2var) .=> keys(vn2var))
    return Dict(
        (symbolic ? vn2var[vn] : vn) => map(x -> symbolic ? x : var2vn[x], dependencies(ctx, vn))
        for vn in keys(ctx.vn2var)
    )
end

function dependencies(m::Model, symbolic = false)
    ctx = SymbolicContext(DefaultContext())
    vi = symbolize(m, VarInfo(m), ctx = ctx)

    return dependencies(ctx, symbolic)
end


function symbolic_logp(m::Model)
    vi, θ = symbolize(m)
    lp = DynamicPPL.getlogp(vi)
    lp_analytic = analytic_rw(Symbolics.value(lp))
    lp_analytic_num = addnum_rw(lp_analytic)

    return lp_analytic_num, θ
end
end
