# See https://github.com/TuringLang/Turing.jl/issues/1199
ChainRulesCore.@non_differentiable push!(
    vi::VarInfo,
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector}
)

ChainRulesCore.@non_differentiable updategid!(
    vi::AbstractVarInfo,
    vn::VarName,
    spl::Sampler,
)
