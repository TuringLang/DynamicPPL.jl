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

    @testset "BangBang.possible" begin
        a = zeros(3, 3, 3, 3) # also allow varname concretization
        svi = SimpleVarInfo(Dict(@varname(a) => a))
        DynamicPPL.setindex!!(svi, ones(3, 2), @varname(a[1, 1:3, 1, 1:2]))
        @test eltype(svi[@varname(a)]) != Any

        DynamicPPL.setindex!!(svi, ones(3), @varname(a[1, 1, :, 1]))
        @test eltype(svi[@varname(a)]) != Any

        DynamicPPL.setindex!!(svi, [1, 2], @varname(a[[5, 8]]))
        @test eltype(svi[@varname(a)]) != Any

        DynamicPPL.setindex!!(
            svi,
            [1, 2],
            @varname(a[[CartesianIndex(1, 1, 3, 1), CartesianIndex(1, 1, 3, 2)]])
        )
        @test eltype(svi[@varname(a)]) != Any

        svi = SimpleVarInfo(Dict(@varname(b) => [zeros(2), zeros(3)]))
        DynamicPPL.setindex!!(svi, ones(2), @varname(b[1]))
        @test eltype(svi[@varname(b)][1]) != Any

        DynamicPPL.setindex!!(svi, ones(2), @varname(b[2][1:2]))
        @test eltype(svi[@varname(b)][2]) != Any
    end
end
