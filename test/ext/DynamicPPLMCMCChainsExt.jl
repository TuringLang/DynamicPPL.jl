module DynamicPPLMCMCChainsExtTests

using DynamicPPL, Distributions, MCMCChains, Test
using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL

@testset "DynamicPPLMCMCChainsExt" begin
    @testset "from_samples" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end

        z = 1.0
        model = f(z)

        @testset "matrix" begin
            ps = [ParamsWithStats(VarInfo(model), model) for _ in 1:50, _ in 1:3]
            c = AbstractMCMC.from_samples(MCMCChains.Chains, ps)
            @test c isa MCMCChains.Chains
            @test size(c, 1) == 50
            @test size(c, 3) == 3
            @test Set(c.name_map.parameters) == Set([:x, :y])
            @test Set(c.name_map.internals) == Set([:logprior, :loglikelihood, :logjoint])
            @test logpdf.(Normal(), c[:x]) ≈ c[:logprior]
            @test c.info.varname_to_symbol[@varname(x)] == :x
            @test c.info.varname_to_symbol[@varname(y)] == :y
        end
    end

    @testset "to_samples" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end
        # Make the chain first
        z = 1.0
        model = f(z)
        ps = hcat([ParamsWithStats(VarInfo(model), model) for _ in 1:50])
        c = AbstractMCMC.from_samples(MCMCChains.Chains, ps)
        # Then convert back to ParamsWithStats
        arr_pss = AbstractMCMC.to_samples(ParamsWithStats, c, model)
        @test size(arr_pss) == (50, 1)
        for i in 1:50
            new_p = arr_pss[i, 1]
            p = ps[i]
            @test new_p.params == p.params
            @test new_p.stats == p.stats
        end
    end

    @testset "returned (basic)" begin
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

    @testset "returned() on `LKJCholesky`" begin
        n = 10
        d = 2
        model = DynamicPPL.TestUtils.demo_lkjchol(d)
        xs = [model().x for _ in 1:n]

        # Extract varnames and values.
        vns_and_vals_xs = map(
            collect ∘ Base.Fix1(AbstractPPL.varname_and_value_leaves, @varname(x)), xs
        )
        vns = map(first, first(vns_and_vals_xs))
        vals = map(vns_and_vals_xs) do vns_and_vals
            map(last, vns_and_vals)
        end

        # Construct the chain.
        syms = map(Symbol, vns)
        vns_to_syms = OrderedDict{VarName,Any}(zip(vns, syms))

        chain = MCMCChains.Chains(
            permutedims(stack(vals)), syms; info=(varname_to_symbol=vns_to_syms,)
        )

        # Test!
        results = returned(model, chain)
        for (x_true, result) in zip(xs, results)
            @test x_true.UL == result.x.UL
        end

        # With variables that aren't in the `model`.
        vns_to_syms_with_extra = let d = deepcopy(vns_to_syms)
            d[@varname(y)] = :y
            d
        end
        vals_with_extra = map(enumerate(vals)) do (i, v)
            vcat(v, i)
        end
        chain_with_extra = MCMCChains.Chains(
            permutedims(stack(vals_with_extra)),
            vcat(syms, [:y]);
            info=(varname_to_symbol=vns_to_syms_with_extra,),
        )
        # Test!
        results = returned(model, chain_with_extra)
        for (x_true, result) in zip(xs, results)
            @test x_true.UL == result.x.UL
        end
    end

    @testset "returned: errors on missing variable" begin
        # Create a chain that only has `m`.
        @model function m_only()
            return m ~ Normal()
        end
        model_m_only = m_only()
        chain_m_only = AbstractMCMC.from_samples(
            MCMCChains.Chains,
            hcat([ParamsWithStats(VarInfo(model_m_only), model_m_only) for _ in 1:50]),
        )

        # Define a model that needs both `m` and `s`.
        @model function f()
            m ~ Normal()
            s ~ Exponential()
            return y ~ Normal(m, s)
        end
        model = f() | (; y=1.0)
        @test_throws "No value was provided" returned(model, chain_m_only)
    end
end

# test for `predict` is in `test/model.jl`

end # module
