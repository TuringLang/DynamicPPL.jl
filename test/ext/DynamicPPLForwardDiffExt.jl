module DynamicPPLForwardDiffExtTests

using DynamicPPL
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using Distributions: MvNormal, Normal
using LinearAlgebra: I
using Test: @test, @test_logs, @testset

@testset "ForwardDiff tweak_adtype" begin
    MODEL_SIZE = 10
    @model f() = x ~ MvNormal(zeros(MODEL_SIZE), I)
    model = f()
    x = randn(MODEL_SIZE)

    @testset "Chunk size setting" for chunksize in (nothing, 0)
        base_adtype = AutoForwardDiff(; chunksize=chunksize)
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype isa AutoForwardDiff{MODEL_SIZE}
    end

    @testset "Tag setting" begin
        base_adtype = AutoForwardDiff()
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype.tag isa ForwardDiff.Tag{DynamicPPL.DynamicPPLTag}
    end
end

@testset "input dependency warning" begin
    @model function derived_observation(y, sigma)
        m ~ Normal(0, sigma)
        v = exp(y) + sigma
        return v ~ Normal(m, sigma)
    end
    @test_logs (:warn, r"v.*derived from a model input.*classified as latent") check_model(
        derived_observation(1.0, 1.0)
    )

    @model function ordinary_latent(y)
        x ~ Normal(y)
        return x
    end
    @test_logs check_model(ordinary_latent(1.0))

    @model function preallocated_latent()
        x = zeros(2)
        return x ~ MvNormal(zeros(2), I)
    end
    @test_logs check_model(preallocated_latent())
end

end
