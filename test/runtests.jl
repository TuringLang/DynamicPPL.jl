using Accessors
using ADTypes
using DynamicPPL
using AbstractMCMC
using AbstractPPL
using BangBang: delete!!, setindex!!
using Bijectors
using DifferentiationInterface
using Distributions
using DistributionsAD
using Documenter
using ForwardDiff
using LogDensityProblems
using MacroTools
using MCMCChains
using Mooncake
using StableRNGs
using ReverseDiff
using Mooncake
using Zygote

using Distributed
using LinearAlgebra
using Pkg
using Random
using Serialization
using Test
using Distributions
using LinearAlgebra # Diagonal

using Combinatorics: combinations
using OrderedCollections: OrderedSet

using DynamicPPL: getargs_dottilde, getargs_tilde

# These flags are set in CI
const GROUP = get(ENV, "GROUP", "All")
const AQUA = get(ENV, "AQUA", "true") == "true"

Random.seed!(100)
include("test_util.jl")

@testset verbose = true "DynamicPPL.jl" begin
    # The tests are split into two groups so that CI can run in parallel. The
    # groups are chosen to make both groups take roughly the same amount of
    # time, but beyond that there is no particular reason for the split.
    if GROUP == "All" || GROUP == "Group1"
        # if AQUA
        #     include("Aqua.jl")
        # end
        # include("utils.jl")
        # include("accumulators.jl")
        # include("compiler.jl")
        # include("varnamedtuple.jl")
        include("varinfo.jl")
        # include("model.jl")
        # include("distribution_wrappers.jl")
        # include("linking.jl")
        # include("serialization.jl")
        # include("pointwise_logdensities.jl")
        # include("lkj.jl")
        # include("contexts.jl")
        # include("context_implementations.jl")
        # include("threadsafe.jl")
        # include("debug_utils.jl")
        # include("submodels.jl")
        # include("chains.jl")
    end

    # if GROUP == "All" || GROUP == "Group2"
    #     include("bijector.jl")
    #     include("logdensityfunction.jl")
    #     @testset "extensions" begin
    #         include("ext/DynamicPPLMCMCChainsExt.jl")
    #         include("ext/DynamicPPLMarginalLogDensitiesExt.jl")
    #     end
    #     @testset "ad" begin
    #         include("ext/DynamicPPLForwardDiffExt.jl")
    #         include("ext/DynamicPPLMooncakeExt.jl")
    #     end
    #     @testset "prob and logprob macro" begin
    #         @test_throws ErrorException prob"..."
    #         @test_throws ErrorException logprob"..."
    #     end
    # end
    #
    # if GROUP == "All" || GROUP == "Doctests"
    #     DocMeta.setdocmeta!(
    #         DynamicPPL, :DocTestSetup, :(using DynamicPPL, Distributions); recursive=true
    #     )
    #     doctestfilters = [
    #         # Ignore the source of a warning in the doctest output, since this is dependent on host.
    #         # This is a line that starts with "└ @ " and ends with the line number.
    #         r"└ @ .+:[0-9]+",
    #     ]
    #
    #     doctest(DynamicPPL; manual=false, doctestfilters=doctestfilters)
    # end
end
