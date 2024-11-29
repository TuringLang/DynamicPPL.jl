@testset "model_utils.jl" begin
    @testset "value_iterator_from_chain" begin
        @testset "$model" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = make_chain_from_prior(model, 10)
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
