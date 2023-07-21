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

    @testset "value_iterator_from_chain" begin
        @testset "$model" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = sample(model, Prior(), 10; progress=false)
            for (i, d) in enumerate(value_iterator_from_chain(model, chain))
                for vn in keys(d)
                    val = DynamicPPL.getvalue(d, vn)
                    for vn_leaf in DynamicPPL.varname_leaves(vn, val)
                        val_leaf = DynamicPPL.getvalue(d, vn_leaf)
                        @test val_leaf == chain[i, Symbol(vn_leaf), 1]
                    end
                end
            end
        end
    end
end
