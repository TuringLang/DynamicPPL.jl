@testset "serialization.jl" begin
    @testset "saving and loading" begin
        # Save model.
        file = joinpath(mktempdir(), "gdemo_default.jls")
        serialize(file, gdemo_default)

        # Sample from deserialized model.
        gdemo_default_copy = deserialize(file)
        samples = [gdemo_default_copy() for _ in 1:1_000]
        samples_s = first.(samples)
        samples_m = last.(samples)

        @test mean(samples_s) ≈ 3 atol = 0.2
        @test mean(samples_m) ≈ 0 atol = 0.1
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
