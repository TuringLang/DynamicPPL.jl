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

    @testset "tovec" begin
        dist = LKJCholesky(2, 1)
        x = rand(dist)
        @test DynamicPPL.tovec(x) == vec(x.UL)

        nt = (a=[1, 2], b=3.0)
        @test DynamicPPL.tovec(nt) == [1, 2, 3.0]

        t = (2.0, [3.0, 4.0])
        @test DynamicPPL.tovec(t) == [2.0, 3.0, 4.0]
    end

    @testset "unique_syms" begin
        vns = (@varname(x), @varname(y[1]), @varname(x.a), @varname(z[15]), @varname(y[2]))
        @inferred DynamicPPL.unique_syms(vns)
        @inferred DynamicPPL.unique_syms(())
        @test DynamicPPL.unique_syms(vns) == (:x, :y, :z)
        @test DynamicPPL.unique_syms(()) == ()
    end

    @testset "group_varnames_by_symbol" begin
        vns_tuple = (
            @varname(x), @varname(y[1]), @varname(x.a), @varname(z[15]), @varname(y[2])
        )
        vns_vec = collect(vns_tuple)
        vns_nt = (;
            x=[@varname(x), @varname(x.a)],
            y=[@varname(y[1]), @varname(y[2])],
            z=[@varname(z[15])],
        )
        vns_vec_single_symbol = [@varname(x.a), @varname(x.b), @varname(x[1])]
        @inferred DynamicPPL.group_varnames_by_symbol(vns_tuple)
        @test DynamicPPL.group_varnames_by_symbol(vns_tuple) == vns_nt
    end
end
