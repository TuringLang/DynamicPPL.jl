module DynamicPPLChainsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra
using Test

@testset "ParamsWithStats from VarInfo" begin
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
        @test haskey(ps.stats, :logjoint)
        @test length(ps.stats) == 3
        @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
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
        @test haskey(ps.stats, :logjoint)
        @test length(ps.stats) == 3
        @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
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
        vi = DynamicPPL.setaccs!!(VarInfo(model), (DynamicPPL.RawValueAccumulator(true),))
        vi = last(DynamicPPL.init!!(model, vi, InitFromPrior(), UnlinkAll()))
        ps = ParamsWithStats(vi)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        # Because we didn't evaluate with log prob accumulators, there should be no stats
        @test isempty(ps.stats)
    end
end

@testset "ParamsWithStats from LogDensityFunction" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        @testset "$transform_strategy" for transform_strategy in (UnlinkAll(), LinkAll())
            # Get the ParamsWithStats using LogDensityFunction
            ldf = LogDensityFunction(m, getlogjoint, transform_strategy)
            param_vector = rand(ldf)
            # This will give us a VNT of values.params`.
            actual_vnt = ParamsWithStats(param_vector, ldf).params
            # We should make sure that those values line up with the values inside the vector.
            accs = OnlyAccsVarInfo(RawValueAccumulator(true))
            _, accs = DynamicPPL.init!!(
                m, accs, InitFromVector(param_vector, ldf), transform_strategy
            )
            expected_vnt = DynamicPPL.densify!!(get_raw_values(accs))

            # Iterate over all variables to check that their values match
            @test Set(keys(actual_vnt)) == Set(keys(expected_vnt))
            for vn in keys(actual_vnt)
                @test actual_vnt[vn] == expected_vnt[vn]
            end
        end
    end
end

@testset "ParamsWithStats from LogDensityFunction with fixed transforms" begin
    # Note: can't use ALL_MODELS here because that contains a model with dynamic transforms,
    # which would yield incorrect results with fix_transforms.
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        @testset "$transform_strategy" for transform_strategy in (UnlinkAll(), LinkAll())
            ldf_fixed = LogDensityFunction(
                m, getlogjoint_internal, transform_strategy; fix_transforms=true
            )
            ldf_dynamic = LogDensityFunction(m, getlogjoint_internal, transform_strategy)
            param_vector = rand(ldf_fixed)

            # Fast path (no log probs, no colon eq): should match the model-evaluation path
            fast = ParamsWithStats(
                param_vector, ldf_fixed; include_log_probs=false, include_colon_eq=false
            )
            slow = ParamsWithStats(
                param_vector, ldf_dynamic; include_log_probs=false, include_colon_eq=false
            )
            @test fast == slow
        end
    end

    @testset "check that model is actually not evaluated" begin
        should_error = false
        @model function prickly()
            x ~ Normal()
            return should_error && error("nope")
        end
        # need to construct LDF without erroring
        ldf = LogDensityFunction(
            prickly(), getlogjoint_internal, LinkAll(); fix_transforms=true
        )
        # now make the model error
        should_error = true
        @test_throws ErrorException prickly()()
        # check that ParamsWithStats doesn't error
        @test ParamsWithStats(
            [0.5], ldf; include_log_probs=false, include_colon_eq=false
        ) isa Any
        # but it does if you set either of them to true
        for (ilp, ice) in ((true, false), (false, true), (true, true))
            @test_throws ErrorException ParamsWithStats(
                [0.5], ldf; include_log_probs=ilp, include_colon_eq=ice
            )
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
