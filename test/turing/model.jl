@testset "model.jl" begin
    @testset "setval! & generated_quantities" begin
        @testset "$model" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = sample(model, Prior(), 10)
            # A simple way of checking that the computation is determinstic: run twice and compare.
            res1 = generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
            res2 = generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
            @test all(res1 .== res2)
            test_setval!(model, MCMCChains.get_sections(chain, :parameters))
        end
    end
end
