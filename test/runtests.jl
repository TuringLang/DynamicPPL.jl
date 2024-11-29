using Accessors
using ADTypes
using DynamicPPL
using AbstractMCMC
using AbstractPPL
using Bijectors
using DifferentiationInterface
using Distributions
using DistributionsAD
using Documenter
using ForwardDiff
using LogDensityProblems, LogDensityProblemsAD
using MacroTools
using MCMCChains
using Mooncake: Mooncake
using StableRNGs
using Tracker
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

using Combinatorics: combinations

using DynamicPPL: getargs_dottilde, getargs_tilde, Selector

Random.seed!(100)

include("test_util.jl")

@testset "DynamicPPL.jl" begin
    @testset "interface" begin
        include("utils.jl")
        include("compiler.jl")
        include("varnamedvector.jl")
        include("varinfo.jl")
        include("simple_varinfo.jl")
        include("model.jl")
        include("sampler.jl")
        include("independence.jl")
        include("distribution_wrappers.jl")
        include("contexts.jl")
        include("context_implementations.jl")
        include("logdensityfunction.jl")
        include("linking.jl")
        include("threadsafe.jl")
        include("serialization.jl")
        include("pointwise_logdensities.jl")
        include("lkj.jl")
        include("debug_utils.jl")
    end

    @testset "compat" begin
        include(joinpath("compat", "ad.jl"))
    end

    @testset "extensions" begin
        include("ext/DynamicPPLMCMCChainsExt.jl")
    end

    @testset "ad" begin
        include("ext/DynamicPPLForwardDiffExt.jl")
        include("ext/DynamicPPLMooncakeExt.jl")
        include("ad.jl")
    end

    @testset "prob and logprob macro" begin
        @test_throws ErrorException prob"..."
        @test_throws ErrorException logprob"..."
    end

    @testset "doctests" begin
        DocMeta.setdocmeta!(
            DynamicPPL, :DocTestSetup, :(using DynamicPPL, Distributions); recursive=true
        )
        doctestfilters = [
            # Older versions will show "0 element Array" instead of "Type[]".
            r"(Any\[\]|0-element Array{.+,[0-9]+})",
            # Older versions will show "Array{...,1}" instead of "Vector{...}".
            r"(Array{.+,\s?1}|Vector{.+})",
            # Older versions will show "Array{...,2}" instead of "Matrix{...}".
            r"(Array{.+,\s?2}|Matrix{.+})",
            # Errors from macros sometimes result in `LoadError: LoadError:`
            # rather than `LoadError:`, depending on Julia version.
            r"ERROR: (LoadError:\s)+",
            # Older versions do not have `;;]` but instead just `]` at end of the line
            # => need to treat `;;]` and `]` as the same, i.e. ignore them if at the end of a line
            r"(;;){0,1}\]$"m,
        ]
        doctest(DynamicPPL; manual=false, doctestfilters=doctestfilters)
    end
end
