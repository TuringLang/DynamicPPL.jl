module DynamicPPLChainsTests

using DynamicPPL
using Distributions
using Test

@testset "ParamsWithStats" begin
    @model function f(z)
        x ~ Normal()
        y := x + 1
        return z ~ Normal(y)
    end
    z = 1.0
    model = f(z)

    @testset "with reevaluation" begin
        ps = ParamsWithStats(VarInfo(model), model)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        @test haskey(ps.stats, :logprior)
        @test haskey(ps.stats, :loglikelihood)
        @test haskey(ps.stats, :lp)
        @test length(ps.stats) == 3
        @test ps.stats.lp ≈ ps.stats.logprior + ps.stats.loglikelihood
        @test ps.params[@varname(y)] ≈ ps.params[@varname(x)] + 1
        @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
        @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(y)]), z)
    end

    @testset "without colon_eq" begin
        ps = ParamsWithStats(VarInfo(model), model; include_colon_eq=false)
        @test haskey(ps.params, @varname(x))
        @test length(ps.params) == 1
        @test haskey(ps.stats, :logprior)
        @test haskey(ps.stats, :loglikelihood)
        @test haskey(ps.stats, :lp)
        @test length(ps.stats) == 3
        @test ps.stats.lp ≈ ps.stats.logprior + ps.stats.loglikelihood
        @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
        @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(x)] + 1), z)
    end

    @testset "without log probs" begin
        ps = ParamsWithStats(VarInfo(model), model; include_log_probs=false)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        @test isempty(ps.stats)
    end

    @testset "no reevaluation" begin
        # Without VAIM, it should error
        @test_throws ErrorException ParamsWithStats(VarInfo(model))
        # With VAIM, it should work
        vi = DynamicPPL.setaccs!!(
            VarInfo(model), (DynamicPPL.ValuesAsInModelAccumulator(true),)
        )
        vi = last(DynamicPPL.evaluate!!(model, vi))
        ps = ParamsWithStats(vi)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        # Because we didn't evaluate with log prob accumulators, there should be no stats
        @test isempty(ps.stats)
    end
end

end # module
