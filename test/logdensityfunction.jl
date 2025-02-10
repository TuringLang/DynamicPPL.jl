using Test, DynamicPPL, ADTypes, LogDensityProblems, ReverseDiff

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

        @testset "$(varinfo)" for varinfo in varinfos
            logdensity = DynamicPPL.LogDensityFunction(model, varinfo)
            θ = varinfo[:]
            @test LogDensityProblems.logdensity(logdensity, θ) ≈ logjoint(model, varinfo)
            @test LogDensityProblems.dimension(logdensity) == length(θ)
        end
    end
end
