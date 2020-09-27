using DynamicPPL
using Turing
using Test

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
    chain = sample(model, MH(), 100);
    results = DynamicPPL.elementwise_loglikelihoods(model, chain)
    var_to_likelihoods = Dict(string(varname) => logliks for (varname, logliks) in results)
    @test haskey(var_to_likelihoods, "xs[1]")
    @test haskey(var_to_likelihoods, "xs[2]")
    @test haskey(var_to_likelihoods, "xs[3]")
    @test haskey(var_to_likelihoods, "y")

    for (i, (s, m)) in enumerate(zip(chain[:s], chain[:m]))
        @test logpdf(Normal(m, √s), xs[1]) == var_to_likelihoods["xs[1]"][i]
        @test logpdf(Normal(m, √s), xs[2]) == var_to_likelihoods["xs[2]"][i]
        @test logpdf(Normal(m, √s), xs[3]) == var_to_likelihoods["xs[3]"][i]
        @test logpdf(Normal(m, √s), y) == var_to_likelihoods["y"][i]
    end
end
