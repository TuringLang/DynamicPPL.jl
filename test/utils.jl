@testset "utils.jl" begin
    @testset "addlogprob!" begin
        @model function testmodel()
            global lp_before = getlogp(__varinfo__)
            @addlogprob!(42)
            return global lp_after = getlogp(__varinfo__)
        end

        model = testmodel()
        varinfo = VarInfo(model)
        @test iszero(lp_before)
        @test getlogp(varinfo) == lp_after == 42
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
        @test getargs_dottilde(:(x .~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
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

    @testset "vectorize" begin
        dist = LKJCholesky(2, 1)
        x = rand(dist)
        @test vectorize(dist, x) == vec(x.UL)
    end
end
