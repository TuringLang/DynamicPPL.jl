@testset "prob_macro.jl" begin
    @testset "scalar" begin
        @model function demo(x)
            m ~ Normal()
            return x ~ Normal(m, 1)
        end

        mval = 3
        xval = 2
        iters = 1000

        model = demo(xval)
        varinfo = VarInfo(model)
        chain = MCMCChains.get_sections(
            sample(model, IS(), iters; save_state=true), :parameters
        )
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())
        lps = logpdf.(Normal.(chain["m"], 1), xval)
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = model" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = model, varinfo = varinfo" == lps

        # multiple chains
        pchain = chainscat(chain, chain)
        pchain2 = chainscat(chain2, chain2)
        plps = repeat(lps, 1, 2)
        @test logprob"x = xval | chain = pchain" == plps
        @test logprob"x = xval | chain = pchain2, model = model" == plps
        @test logprob"x = xval | chain = pchain, varinfo = varinfo" == plps
        @test logprob"x = xval | chain = pchain2, model = model, varinfo = varinfo" == plps
    end
    @testset "vector" begin
        n = 5
        @model function demo(x, n=n)
            m ~ MvNormal(n, 1.0)
            return x ~ MvNormal(m, 1.0)
        end
        mval = rand(n)
        xval = rand(n)
        iters = 1000

        model = demo(xval)
        varinfo = VarInfo(model)
        chain = MCMCChains.get_sections(
            sample(model, HMC(0.5, 1), iters; save_state=true), :parameters
        )
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())

        names = namesingroup(chain, "m")
        lps = [
            logpdf(MvNormal(chain.value[i, names, j], 1.0), xval) for i in 1:size(chain, 1),
            j in 1:size(chain, 3)
        ]
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = model" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = model, varinfo = varinfo" == lps

        # multiple chains
        pchain = chainscat(chain, chain)
        pchain2 = chainscat(chain2, chain2)
        plps = repeat(lps, 1, 2)
        @test logprob"x = xval | chain = pchain" == plps
        @test logprob"x = xval | chain = pchain2, model = model" == plps
        @test logprob"x = xval | chain = pchain, varinfo = varinfo" == plps
        @test logprob"x = xval | chain = pchain2, model = model, varinfo = varinfo" == plps
    end
    @testset "issue#137" begin
        @model function model1(y, group, n_groups)
            σ ~ truncated(Cauchy(0, 1), 0, Inf)
            α ~ filldist(Normal(0, 10), n_groups)
            μ = α[group]
            return y ~ MvNormal(μ, σ)
        end

        y = randn(100)
        group = rand(1:4, 100)
        n_groups = 4

        chain1 = MCMCChains.get_sections(
            sample(model1(y, group, n_groups), NUTS(0.65), 2_000; save_state=true),
            :parameters,
        )
        logprob"y = y[[1]] | group = group[[1]], n_groups = n_groups, chain = chain1"

        @model function model2(y, group, n_groups)
            σ ~ truncated(Cauchy(0, 1), 0, Inf)
            α ~ filldist(Normal(0, 10), n_groups)
            for i in 1:length(y)
                y[i] ~ Normal(α[group[i]], σ)
            end
        end

        chain2 = MCMCChains.get_sections(
            sample(model2(y, group, n_groups), NUTS(0.65), 2_000; save_state=true),
            :parameters,
        )
        logprob"y = y[[1]] | group = group[[1]], n_groups = n_groups, chain = chain2"
    end
end
