using Test, DynamicPPL, LogDensityProblems

@testset "LogDensityFunction" begin
    @testset "$(nameof(model))" for model in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = rand(NamedTuple, model)
        vns = DynamicPPL.TestUtils.varnames(model)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(model, example_values, vns)

        @testset "$(varinfo)" for varinfo in varinfos
            logdensity = LogDensityFunction(model, varinfo)
            θ = varinfo[:]
            @test LogDensityProblems.logdensity(logdensity, θ) ≈ logjoint(model, varinfo)
        end
    end
end
