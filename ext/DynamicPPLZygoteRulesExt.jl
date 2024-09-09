module DynamicPPLZygoteRulesExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL, Distributions
    using ZygoteRules: ZygoteRules
else
    using ..DynamicPPL: DynamicPPL, Distributions
    using ..ZygoteRules: ZygoteRules
end

# https://github.com/TuringLang/Turing.jl/issues/1595
ZygoteRules.@adjoint function DynamicPPL.dot_observe(
    spl::Union{DynamicPPL.SampleFromPrior,DynamicPPL.SampleFromUniform},
    dists::AbstractArray{<:Distributions.Distribution},
    value::AbstractArray,
    vi,
)
    function dot_observe_fallback(spl, dists, value, vi)
        DynamicPPL.increment_num_produce!(vi)
        return sum(map(Distributions.loglikelihood, dists, value)), vi
    end
    return ZygoteRules.pullback(dot_observe_fallback, __context__, spl, dists, value, vi)
end

end # module
