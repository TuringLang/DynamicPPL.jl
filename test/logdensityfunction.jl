using Test, DynamicPPL, ADTypes, LogDensityProblems, LogDensityProblemsAD, ReverseDiff

@testset "`getmodel` and `setmodel`" begin
    @testset "$(nameof(model))" for model in TU.DEMO_MODELS
        model = TU.DEMO_MODELS[1]
        ℓ = DynamicPPL.LogDensityFunction(model)
        @test DynamicPPL.getmodel(ℓ) == model
        @test DynamicPPL.setmodel(ℓ, model).model == model

        # ReverseDiff related
        ∇ℓ = LogDensityProblemsAD.ADgradient(:ReverseDiff, ℓ; compile=Val(false))
        @test DynamicPPL.getmodel(∇ℓ) == model
        @test DynamicPPL.getmodel(DynamicPPL.setmodel(∇ℓ, model, AutoReverseDiff())) ==
            model
        ∇ℓ = LogDensityProblemsAD.ADgradient(:ReverseDiff, ℓ; compile=Val(true))
        new_∇ℓ = DynamicPPL.setmodel(∇ℓ, model, AutoReverseDiff())
        @test DynamicPPL.getmodel(new_∇ℓ) == model
        # HACK(sunxd): rely on internal implementation detail, i.e., naming of `compiledtape`
        @test new_∇ℓ.compiledtape != ∇ℓ.compiledtape
    end
end

@testset "LogDensityFunction" begin
    @testset "$(nameof(model))" for model in TU.DEMO_MODELS
        example_values = TU.rand_prior_true(model)
        vns = TU.varnames(model)
        varinfos = TU.setup_varinfos(model, example_values, vns)

        @testset "$(varinfo)" for varinfo in varinfos
            logdensity = DynamicPPL.LogDensityFunction(model, varinfo)
            θ = varinfo[:]
            @test LogDensityProblems.logdensity(logdensity, θ) ≈ logjoint(model, varinfo)
            @test LogDensityProblems.dimension(logdensity) == length(θ)
        end
    end
end
