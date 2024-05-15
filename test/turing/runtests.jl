using DynamicPPL
using Turing
using LinearAlgebra
using ReverseDiff

using Random
using Test

setprogress!(false)

Random.seed!(100)

# load test utilities
include(joinpath(pathof(DynamicPPL), "..", "..", "test", "test_util.jl"))
include(joinpath(pathof(Turing), "..", "..", "test", "test_utils", "numerical_tests.jl"))

@testset "Turing" begin
    include("compiler.jl")
    include("loglikelihoods.jl")
    include("model.jl")
    include("varinfo.jl")
end
