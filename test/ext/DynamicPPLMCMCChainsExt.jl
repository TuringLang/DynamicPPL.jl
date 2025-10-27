@testset "DynamicPPLMCMCChainsExt" begin
    @model demo() = x ~ Normal()
    model = demo()

    chain = MCMCChains.Chains(
        randn(1000, 2, 1),
        [:x, :y],
        Dict(:internals => [:y]);
        info=(; varname_to_symbol=Dict(@varname(x) => :x)),
    )
    chain_generated = @test_nowarn returned(model, chain)
    @test size(chain_generated) == (1000, 1)
    @test mean(chain_generated) ≈ 0 atol = 0.1
end

# test for `predict` is in `test/model.jl`
