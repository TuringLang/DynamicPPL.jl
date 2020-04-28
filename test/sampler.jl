using DynamicPPL
using Distributions
using AbstractMCMC: sample

using Random
using Statistics
using Test

Random.seed!(100)

@testset "AbstractMCMC interface" begin
    @model gdemo(x, y) = begin
        s ~ InverseGamma(2, 3)
        m ~ Normal(2.0, sqrt(s))
        x ~ Normal(m, sqrt(s))
        y ~ Normal(m, sqrt(s))
    end

    model = gdemo(1.0, 2.0)
    N = 1_000

    chains = sample(model, SampleFromPrior(), N; progress = false)
    @test chains isa Vector{<:VarInfo}
    @test length(chains) == N

    # Expected value of ``X`` where ``X ~ N(2, ...)`` is 2.
    @test mean(vi[@varname(m)] for vi in chains) ≈ 2 atol = 0.1

    # Expected value of ``X`` where ``X ~ IG(2, 3)`` is 3.
    @test mean(vi[@varname(s)] for vi in chains) ≈ 3 atol = 0.1

    chains = sample(model, SampleFromUniform(), N; progress = false)
    @test chains isa Vector{<:VarInfo}
    @test length(chains) == N

    # Expected value of ``X`` where ``X ~ U[-2, 2]`` is ≈ 0.
    @test mean(vi[@varname(m)] for vi in chains) ≈ 0 atol = 0.1

    # Expected value of ``exp(X)`` where ``X ~ U[-2, 2]`` is ≈ 1.8.
    @test mean(vi[@varname(s)] for vi in chains) ≈ 1.8 atol = 0.1
end

