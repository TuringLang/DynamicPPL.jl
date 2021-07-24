@testset "loglikelihoods.jl" begin
    @model function demo(xs, y)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(xs)
            xs[i] ~ Normal(m, √s)
        end

        return y ~ Normal(m, √s)
    end

    xs = randn(3)
    y = randn()
    model = demo(xs, y)
    chain = sample(model, MH(), MCMCThreads(), 100, 2)
    var_to_likelihoods = pointwise_loglikelihoods(
        model, MCMCChains.get_sections(chain, :parameters)
    )
    @test haskey(var_to_likelihoods, "xs[1]")
    @test haskey(var_to_likelihoods, "xs[2]")
    @test haskey(var_to_likelihoods, "xs[3]")
    @test haskey(var_to_likelihoods, "y")

    for chain_idx in MCMCChains.chains(chain)
        for (i, (s, m)) in enumerate(zip(chain[:, :s, chain_idx], chain[:, :m, chain_idx]))
            @test logpdf(Normal(m, √s), xs[1]) == var_to_likelihoods["xs[1]"][i, chain_idx]
            @test logpdf(Normal(m, √s), xs[2]) == var_to_likelihoods["xs[2]"][i, chain_idx]
            @test logpdf(Normal(m, √s), xs[3]) == var_to_likelihoods["xs[3]"][i, chain_idx]
            @test logpdf(Normal(m, √s), y) == var_to_likelihoods["y"][i, chain_idx]
        end
    end

    var_info = VarInfo(model)
    results = pointwise_loglikelihoods(model, var_info)
    var_to_likelihoods = Dict(string(vn) => ℓ for (vn, ℓ) in results)
    s, m = var_info[SampleFromPrior()]
    @test [logpdf(Normal(m, √s), xs[1])] == var_to_likelihoods["xs[1]"]
    @test [logpdf(Normal(m, √s), xs[2])] == var_to_likelihoods["xs[2]"]
    @test [logpdf(Normal(m, √s), xs[3])] == var_to_likelihoods["xs[3]"]
    @test [logpdf(Normal(m, √s), y)] == var_to_likelihoods["y"]
end
