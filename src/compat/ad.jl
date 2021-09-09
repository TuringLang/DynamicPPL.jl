# See https://github.com/TuringLang/Turing.jl/issues/1199
ChainRulesCore.@non_differentiable push!!(
    vi::VarInfo, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)

ChainRulesCore.@non_differentiable updategid!(
    vi::AbstractVarInfo, vn::VarName, spl::Sampler
)

# https://github.com/TuringLang/Turing.jl/issues/1595
ZygoteRules.@adjoint function dot_observe(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::AbstractArray{<:Distribution},
    value::AbstractArray,
    vi,
)
    function dot_observe_fallback(spl, dists, value, vi)
        increment_num_produce!(vi)
        return sum(map(Distributions.loglikelihood, dists, value))
    end
    return ZygoteRules.pullback(__context__, dot_observe_fallback, spl, dists, value, vi)
end
