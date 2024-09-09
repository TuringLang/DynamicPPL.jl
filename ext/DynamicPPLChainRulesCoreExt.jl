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

# No need + causes issues for some AD backends, e.g. Zygote.
ChainRulesCore.@non_differentiable DynamicPPL.infer_nested_eltype(x)

ChainRulesCore.@non_differentiable DynamicPPL.recontiguify_ranges!(ranges)
# TODO(mhauru) A workaround for https://github.com/FluxML/Zygote.jl/issues/1523, remove when
# fixed. Note that this is some serious type piracy. Needed because VarNamedVector uses
# BitVectors.
ChainRulesCore.@non_differentiable BitVector(a, b)

end # module
