# NOTE: These tests use `@everywhere`, which makes it impossible to put the tests inside a
# module. They will just have to live in the `Main` scope.

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Serialization: serialize, deserialize
using Distributions
using Distributed: addprocs, nworkers, rmprocs, @everywhere, pmap
using Test

@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
gdemo_def = gdemo_d()

@testset "serialization.jl" begin
    @testset "saving and loading" begin
        # Save model.
        file = joinpath(mktempdir(), "gdemo_default.jls")
        serialize(file, gdemo_def)

        # Sample from deserialized model.
        gdemo_default_copy = deserialize(file)
        samples = [gdemo_default_copy() for _ in 1:1_000]
        samples_s = first.(samples)
        samples_m = last.(samples)

        @test mean(samples_s) ≈ 3 atol = 0.2
        @test mean(samples_m) ≈ 0 atol = 0.15
    end

    @testset "pmap" begin
        # Add worker processes.
        pids = addprocs()
        @info "serialization test: using $(nworkers()) processes"

        # Load packages on all processes.
        @everywhere begin
            using DynamicPPL
            using Distributions
        end

        # Define model on all proceses.
        @everywhere @model function model()
            return m ~ Normal(0, 1)
        end

        # Generate `Model` objects on all processes.
        models = pmap(_ -> model(), 1:100)
        @test models isa Vector{<:Model}
        @test length(models) == 100

        # Sample from model on all processes.
        n = 1_000
        samples1 = pmap(_ -> model()(), 1:n)
        m = model()
        samples2 = pmap(_ -> m(), 1:n)

        for samples in (samples1, samples2)
            @test samples isa Vector{Float64}
            @test length(samples) == n
            @test mean(samples) ≈ 0 atol = 0.15
            @test std(samples) ≈ 1 atol = 0.1
        end

        # Remove processes
        rmprocs(pids...)
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."
