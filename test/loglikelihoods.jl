using .Turing

@testset "loglikelihoods" begin
    @model function demo(xs, y)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(xs)
            xs[i] ~ Normal(m, √s)
        end

        y ~ Normal(m, √s)
    end

    xs = randn(3);
    y = randn();
    model = demo(xs, y);
    chain = sample(model, MH(), MCMCThreads(), 100, 2);
    results = elementwise_loglikelihoods(model, chain)
    var_to_likelihoods = Dict(
        string(varname) => logliks for (varname, logliks) in results
    )
    @test haskey(var_to_likelihoods, "xs[1]")
    @test haskey(var_to_likelihoods, "xs[2]")
    @test haskey(var_to_likelihoods, "xs[3]")
    @test haskey(var_to_likelihoods, "y")

    for chain_idx in MCMCChains.chains(chain)

        for (i, (s, m)) in enumerate(zip(chain[:s], chain[:m]))
            @test logpdf(Normal(m, √s), xs[1]) == var_to_likelihoods["xs[1]"][chain_idx, i]
            @test logpdf(Normal(m, √s), xs[2]) == var_to_likelihoods["xs[2]"][chain_idx, i]
            @test logpdf(Normal(m, √s), xs[3]) == var_to_likelihoods["xs[3]"][chain_idx, i]
            @test logpdf(Normal(m, √s), y) == var_to_likelihoods["y"][chain_idx, i]
        end
    end

    var_info = VarInfo(model)
    results = DynamicPPL.elementwise_loglikelihoods(model, var_info)
    var_to_likelihoods = Dict(string(vn) => ℓ for (vn, ℓ) in results)
    s, m = var_info[SampleFromPrior()]
    @test logpdf(Normal(m, √s), xs[1]) == var_to_likelihoods["xs[1]"]
    @test logpdf(Normal(m, √s), xs[2]) == var_to_likelihoods["xs[2]"]
    @test logpdf(Normal(m, √s), xs[3]) == var_to_likelihoods["xs[3]"]
    @test logpdf(Normal(m, √s), y) == var_to_likelihoods["y"]
end
