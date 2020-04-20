using DynamicPPL
using Distributions
using AbstractMCMC: sample

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "AbstractMCMC interface" begin
    @model gdemo(x, y) = begin
        s ~ InverseGamma(2, 3)
        m ~ Normal(0.0, sqrt(s))
        x ~ Normal(m, sqrt(s))
        y ~ Normal(m, sqrt(s))
    end

    model = gdemo(1.0, 2.0)
    N = 10_000

    chains = sample(model, SampleFromPrior(), N; progress = false)
    @test chains isa Vector{<:VarInfo}
    @test length(chains) == N
    @test mean(vi[@varname(m)] for vi in chains) ≈ 0 atol = 0.1
    @test mean(vi[@varname(s)] for vi in chains) ≈ 3 atol = 0.1

    chains = sample(model, SampleFromUniform(), N; progress = false)
    @test chains isa Vector{<:VarInfo}
    @test length(chains) == N
    @test mean(vi[@varname(m)] for vi in chains) ≈ 1 atol = 0.1
    @test mean(vi[@varname(s)] for vi in chains) ≈ 3.3 atol = 0.1
end

