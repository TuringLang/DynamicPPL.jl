using DynamicPPL
using DynamicPPL: apply_dotted, getargs_dottilde, getargs_tilde

using Test

@testset "apply_dotted" begin
    # Some things that are not expressions.
    @test apply_dotted(:x) === :x
    @test apply_dotted(1.0) === 1.0
    @test apply_dotted([1.0, 2.0, 4.0]) == [1.0, 2.0, 4.0]

    # Some expressions.
    @test apply_dotted(:(x ~ Normal(μ, σ))) == :(x ~ Normal(μ, σ))
    @test apply_dotted(:((.~)(x, Normal(μ, σ)))) == :((.~)(x, Normal(μ, σ)))
    @test apply_dotted(:((~).(x, Normal(μ, σ)))) == :((~).(x, Normal(μ, σ)))
    @test apply_dotted(:(@. x ~ Normal(μ, σ))) == :((~).(x, Normal.(μ, σ)))
    @test apply_dotted(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) ==
        :((~).(x, Normal.(μ, sqrt(v))))
    @test apply_dotted(:(@~ Normal.(μ, σ))) == :(@~ Normal.(μ, σ))
end

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
