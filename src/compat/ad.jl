# Prevent Zygote from differentiating push!
# See https://github.com/TuringLang/Turing.jl/issues/1199
ZygoteRules.@adjoint function push!(
    vi::VarInfo,
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector}
)
    return push!(vi, vn, r, dist, gidset), _ -> nothing
end

# Multithreaded evaluation is not compatible with Zygote.
ZygoteRules.@adjoint function (model::Model)(
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    ctx::AbstractContext
)
    function evaluate(vi, spl, ctx)
        return evaluate_singlethreaded(model, vi, spl, ctx)
    end
    return ZygoteRules.pullback(evaluate, vi, spl, ctx)
end

