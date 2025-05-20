using Accessors
using ADTypes
using DynamicPPL
using AbstractMCMC
using AbstractPPL
using Bijectors
using DifferentiationInterface
using Distributions
using DistributionsAD
using ForwardDiff
using LogDensityProblems
using MacroTools
using MCMCChains
using StableRNGs
using ReverseDiff
using Zygote
using Compat

using Distributed
using LinearAlgebra
using Pkg
using Random
using Serialization
using Test
using Distributions
using LinearAlgebra # Diagonal

using JET: JET

using Combinatorics: combinations
using OrderedCollections: OrderedSet

using DynamicPPL: getargs_dottilde, getargs_tilde

# These flags are set in CI
const GROUP = get(ENV, "GROUP", "All")
const AQUA = get(ENV, "AQUA", "true") == "true"

# Detect if prerelease version, if so, we skip some tests
const IS_PRERELEASE = !isempty(VERSION.prerelease)
if !IS_PRERELEASE
    Pkg.add("Mooncake")
    using Mooncake: Mooncake
end

Random.seed!(100)
include("test_util.jl")

@testset verbose = true "DynamicPPL.jl" begin
    # The tests are split into two groups so that CI can run in parallel. The
    # groups are chosen to make both groups take roughly the same amount of
    # time, but beyond that there is no particular reason for the split.
    if GROUP == "All" || GROUP == "Group1"
        if AQUA
            include("Aqua.jl")
        end
        include("utils.jl")
        include("compiler.jl")
        include("varnamedvector.jl")
        include("varinfo.jl")
        include("simple_varinfo.jl")
        include("model.jl")
        include("sampler.jl")
        include("independence.jl")
        include("distribution_wrappers.jl")
        include("logdensityfunction.jl")
        include("linking.jl")
        include("serialization.jl")
        include("pointwise_logdensities.jl")
        include("lkj.jl")
        include("contexts.jl")
        include("context_implementations.jl")
        include("threadsafe.jl")
        include("debug_utils.jl")
        include("deprecated.jl")
        include("submodels.jl")
        include("bijector.jl")
    end

    if GROUP == "All" || GROUP == "Group2"
        @testset "compat" begin
            include(joinpath("compat", "ad.jl"))
        end
        @testset "extensions" begin
            include("ext/DynamicPPLMCMCChainsExt.jl")
            include("ext/DynamicPPLJETExt.jl")
        end
        @testset "ad" begin
            include("ext/DynamicPPLForwardDiffExt.jl")
            if !IS_PRERELEASE
                include("ext/DynamicPPLMooncakeExt.jl")
            end
            include("ad.jl")
        end
        @testset "prob and logprob macro" begin
            @test_throws ErrorException prob"..."
            @test_throws ErrorException logprob"..."
        end
    end
end
