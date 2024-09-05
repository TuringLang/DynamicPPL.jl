module TestPredict

using Test
using DynamicPPL
using AbstractMCMC
using MCMCChains
using Distributions
using Random
using LogDensityProblemsAD
using AdvancedHMC
using Tapir
using ForwardDiff

@model function linear_reg(x, y, σ=0.1)
    β ~ Normal(0, 1)
    for i in eachindex(y)
        y[i] ~ Normal(β * x[i], σ)
    end
end

@model function linear_reg_vec(x, y, σ=0.1)
    β ~ Normal(0, 1)
    return y ~ MvNormal(β .* x, σ^2 * I)
end

f(x) = 2 * x + 0.1 * randn()

Δ = 0.1
xs_train = 0:Δ:10
ys_train = f.(xs_train)
xs_test = [10 + Δ, 10 + 2 * Δ]
ys_test = f.(xs_test)

model = linear_reg(xs_train, ys_train)

m_lin_reg = linear_reg(xs_train, ys_train)
ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.VarInfo(model))
ad_ldf = LogDensityProblemsAD.ADgradient(Val(:Tapir), ldf; safety_on=false)
chain = AbstractMCMC.sample(
    ad_ldf, AdvancedHMC.NUTS(0.6), 1000; chain_type=MCMCChains.Chains, param_names=[:β]
)

DynamicPPL.predict(test_model, chain)

# LKJ example
@model demo_lkj() = x ~ LKJCholesky(2, 1.0)

model = demo_lkj()

ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.SimpleVarInfo(model))
ad_ldf = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ldf)

chain = AbstractMCMC.sample(
    ad_ldf, AdvancedHMC.NUTS(0.6), 1000; chain_type=MCMCChains.Chains, param_names=[:Σ, :x]
)

end # module
