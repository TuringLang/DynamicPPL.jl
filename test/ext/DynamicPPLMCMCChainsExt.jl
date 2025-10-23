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

    @testset "varinfos_to_chains" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end

        z = 1.0
        model = f(z)

        @testset "vector" begin
            ps = [ParamsWithStats(VarInfo(model), model) for _ in 1:50]
            c = DynamicPPL.to_chains(MCMCChains.Chains, ps)
            @test c isa MCMCChains.Chains
            @test size(c, 1) == 50
            @test size(c, 3) == 1
            @test Set(c.name_map.parameters) == Set([:x, :y])
            @test Set(c.name_map.internals) == Set([:logprior, :loglikelihood, :lp])
            @test logpdf.(Normal(), c[:x]) ≈ c[:logprior]
            @test c.info.varname_to_symbol[@varname(x)] == :x
            @test c.info.varname_to_symbol[@varname(y)] == :y
        end

        @testset "matrix" begin
            ps = [ParamsWithStats(VarInfo(model), model) for _ in 1:50, _ in 1:3]
            c = DynamicPPL.to_chains(MCMCChains.Chains, ps)
            @test c isa MCMCChains.Chains
            @test size(c, 1) == 50
            @test size(c, 3) == 3
            @test Set(c.name_map.parameters) == Set([:x, :y])
            @test Set(c.name_map.internals) == Set([:logprior, :loglikelihood, :lp])
            @test logpdf.(Normal(), c[:x]) ≈ c[:logprior]
            @test c.info.varname_to_symbol[@varname(x)] == :x
            @test c.info.varname_to_symbol[@varname(y)] == :y
        end
    end
end

# test for `predict` is in `test/model.jl`
