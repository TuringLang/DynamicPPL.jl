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

    @testset "to_chains" begin
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

    @testset "from_chains" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end

        z = 1.0
        model = f(z)
        ps = [ParamsWithStats(VarInfo(model), model) for _ in 1:50]
        c = DynamicPPL.to_chains(MCMCChains.Chains, ps)

        @testset "OrderedDict" begin
            arr_dicts = DynamicPPL.from_chains(OrderedDict{VarName,Any}, c)
            @test size(arr_dicts) == (50, 1)
            for i in 1:50
                dict = arr_dicts[i, 1]
                @test dict isa OrderedDict{VarName,Any}
                p = ps[i].params
                @test dict[@varname(x)] == p[@varname(x)]
                @test dict[@varname(y)] == p[@varname(y)]
                @test length(dict) == 2
            end
        end

        @testset "NamedTuple" begin
            arr_nts = DynamicPPL.from_chains(NamedTuple, c)
            @test size(arr_nts) == (50, 1)
            for i in 1:50
                nt = arr_nts[i, 1]
                @test length(nt) == 5
                p = ps[i]
                @test nt.x == p.params[@varname(x)]
                @test nt.y == p.params[@varname(y)]
                @test nt.lp == p.stats.lp
                @test nt.logprior == p.stats.logprior
                @test nt.loglikelihood == p.stats.loglikelihood
            end
        end

        @testset "ParamsWithStats" begin
            arr_pss = DynamicPPL.from_chains(ParamsWithStats, c)
            @test size(arr_pss) == (50, 1)
            for i in 1:50
                new_p = arr_pss[i, 1]
                p = ps[i]
                @test new_p.params == p.params
                @test new_p.stats == p.stats
            end
        end
    end
end

# test for `predict` is in `test/model.jl`
