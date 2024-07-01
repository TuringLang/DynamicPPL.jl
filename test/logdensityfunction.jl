using Test, DynamicPPL, LogDensityProblems

@testset "`getmodel` and `setmodel`" begin
    # TODO: does it worth to test all demo models?
    model = DynamicPPL.TestUtils.DEMO_MODELS[1]
    ℓ = DynamicPPL.LogDensityFunction(model)
    @test DynamicPPL.getmodel(ℓ) == model
    @test DynamicPPL.setmodel(ℓ, model).model == model

    # ReverseDiff related
    ∇ℓ = LogDensityProblems.ADgradient(:ReverseDiff, ℓ; compile=Val(false))
    @test DynamicPPL.getmodel(∇ℓ) == model
    @test getmodel(DynamicPPL.setmodel(∇ℓ, model)) == model
    
    ∇ℓ = LogDensityProblems.ADgradient(:ReverseDiff, ℓ; compile=Val(true))
    new_∇ℓ = DynamicPPL.setmodel(∇ℓ, model)
    @test DynamicPPL.getmodel(new_∇ℓ) == model
    @test new_∇ℓ.ℓ.compiledtape != ∇ℓ.ℓ.compiledtape
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
