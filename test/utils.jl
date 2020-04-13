using DynamicPPL
using DynamicPPL: getargs_dottilde, getargs_tilde

using Test

@testset "getargs_dottilde" begin
    # Some things that are not expressions.
    @test getargs_dottilde(:x) === nothing
    @test getargs_dottilde(1.0) === nothing
    @test getargs_dottilde([1.0, 2.0, 4.0]) === nothing

    # Some expressions.
    @test getargs_dottilde(:(x ~ Normal(μ, σ))) === nothing
    @test getargs_dottilde(:((.~)(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
    @test getargs_dottilde(:((~).(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
    @test getargs_dottilde(:(@. x ~ Normal(μ, σ))) === nothing
    @test getargs_dottilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) === nothing
    @test getargs_dottilde(:(@~ Normal.(μ, σ))) === nothing
end

@testset "getargs_tilde" begin
    # Some things that are not expressions.
    @test getargs_tilde(:x) === nothing
    @test getargs_tilde(1.0) === nothing
    @test getargs_tilde([1.0, 2.0, 4.0]) === nothing

    # Some expressions.
    @test getargs_tilde(:(x ~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
    @test getargs_tilde(:((.~)(x, Normal(μ, σ)))) === nothing
    @test getargs_tilde(:((~).(x, Normal(μ, σ)))) === nothing
    @test getargs_tilde(:(@. x ~ Normal(μ, σ))) === nothing
    @test getargs_tilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) === nothing
    @test getargs_tilde(:(@~ Normal.(μ, σ))) === nothing
end
