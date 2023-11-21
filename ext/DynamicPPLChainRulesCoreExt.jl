module DynamicPPLChainRulesCoreExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL, BangBang, Distributions
    using ChainRulesCore: ChainRulesCore
else
    using ..DynamicPPL: DynamicPPL, BangBang, Distributions
    using ..ChainRulesCore: ChainRulesCore
end

# See https://github.com/TuringLang/Turing.jl/issues/1199
ChainRulesCore.@non_differentiable BangBang.push!!(
    vi::DynamicPPL.VarInfo,
    vn::DynamicPPL.VarName,
    r,
    dist::Distributions.Distribution,
    gidset::Set{DynamicPPL.Selector},
)

ChainRulesCore.@non_differentiable DynamicPPL.updategid!(
    vi::DynamicPPL.AbstractVarInfo, vn::DynamicPPL.VarName, spl::DynamicPPL.Sampler
)

end # module
