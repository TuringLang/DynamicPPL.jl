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
    @testset "construction from $(nameof(model))" for model in
                                                      DynamicPPL.TestUtils.DEMO_MODELS
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

    @testset "LogDensityProblems interface" begin
        @model demo_simple() = x ~ Normal()
        model = demo_simple()

        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{0}()
        @test LogDensityProblems.logdensity(ldf, [1.0]) isa Any

        # Set AD type on model, then reconstruct LDF
        model_with_ad = Model(model, AutoForwardDiff())
        ldf_with_ad = DynamicPPL.LogDensityFunction(model_with_ad)
        @test LogDensityProblems.capabilities(typeof(ldf_with_ad)) ==
            LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.logdensity(ldf_with_ad, [1.0]) isa Any
        @test LogDensityProblems.logdensity_and_gradient(ldf_with_ad, [1.0]) isa Any

        # Set AD type on LDF directly
        ldf_with_ad2 = DynamicPPL.LogDensityFunction(ldf, AutoForwardDiff())
        @test LogDensityProblems.capabilities(typeof(ldf_with_ad2)) ==
            LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.logdensity(ldf_with_ad2, [1.0]) isa Any
        @test LogDensityProblems.logdensity_and_gradient(ldf_with_ad2, [1.0]) isa Any
    end
end
