using Test, DynamicPPL, ADTypes, LogDensityProblems, ForwardDiff

@testset "`getmodel` and `setmodel`" begin
    @testset "$(nameof(model))" for model in DynamicPPL.TestUtils.DEMO_MODELS
        model = DynamicPPL.TestUtils.DEMO_MODELS[1]
        ℓ = DynamicPPL.LogDensityFunction(model)
        @test DynamicPPL.getmodel(ℓ) == model
        @test DynamicPPL.setmodel(ℓ, model).model == model
    end
end

@testset "LogDensityFunction" begin
    @testset "$(nameof(model))" for model in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)
        vns = DynamicPPL.TestUtils.varnames(model)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(model, example_values, vns)

        vi = first(varinfos)
        theta = vi[:]
        ldf_joint = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.logdensity(ldf_joint, theta) ≈ logjoint(model, vi)
        ldf_prior = DynamicPPL.LogDensityFunction(model, getlogprior)
        @test LogDensityProblems.logdensity(ldf_prior, theta) ≈ logprior(model, vi)
        ldf_likelihood = DynamicPPL.LogDensityFunction(model, getloglikelihood)
        @test LogDensityProblems.logdensity(ldf_likelihood, theta) ≈
            loglikelihood(model, vi)

        @testset "$(varinfo)" for varinfo in varinfos
            logdensity = DynamicPPL.LogDensityFunction(model, getlogjoint, varinfo)
            θ = varinfo[:]
            @test LogDensityProblems.logdensity(logdensity, θ) ≈ logjoint(model, varinfo)
            @test LogDensityProblems.dimension(logdensity) == length(θ)
        end
    end

    @testset "capabilities" begin
        model = DynamicPPL.TestUtils.DEMO_MODELS[1]
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{0}()

        ldf_with_ad = DynamicPPL.LogDensityFunction(model; adtype=AutoForwardDiff())
        @test LogDensityProblems.capabilities(typeof(ldf_with_ad)) ==
            LogDensityProblems.LogDensityOrder{1}()
    end
end
