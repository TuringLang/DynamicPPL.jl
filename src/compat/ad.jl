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
ZygoteRules.@adjoint function zygote_setval!(vi, val, vn)
    return zygote_setval!(vi, val, vn), _ -> nothing
end