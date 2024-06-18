using Accessors
using ADTypes
using DynamicPPL
using AbstractMCMC
using AbstractPPL
using Bijectors
using Distributions
using DistributionsAD
using Documenter
using ForwardDiff
using LogDensityProblems, LogDensityProblemsAD
using MacroTools
using MCMCChains
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

using DynamicPPL: getargs_dottilde, getargs_tilde, Selector

const DIRECTORY_DynamicPPL = dirname(dirname(pathof(DynamicPPL)))
const DIRECTORY_Turing_tests = joinpath(DIRECTORY_DynamicPPL, "test", "turing")
const GROUP = get(ENV, "GROUP", "All")

Random.seed!(100)

include("test_util.jl")

@testset "DynamicPPL.jl" begin
    if GROUP == "All" || GROUP == "DynamicPPL"
        @testset "interface" begin
            include("utils.jl")
            include("compiler.jl")
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

            include("loglikelihoods.jl")

            include("lkj.jl")
        end

        @testset "compat" begin
            include(joinpath("compat", "ad.jl"))
        end

        @testset "extensions" begin
            include("ext/DynamicPPLMCMCChainsExt.jl")
        end

        @testset "ad" begin
            include("ext/DynamicPPLForwardDiffExt.jl")
            include("ad.jl")
        end

        @testset "prob and logprob macro" begin
            @test_throws ErrorException prob"..."
            @test_throws ErrorException logprob"..."
        end

        @testset "doctests" begin
            DocMeta.setdocmeta!(
                DynamicPPL,
                :DocTestSetup,
                :(using DynamicPPL, Distributions);
                recursive=true,
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
            ]
            doctest(DynamicPPL; manual=false, doctestfilters=doctestfilters)
        end
    end

    if GROUP == "All" || GROUP == "Downstream"
        @testset "turing" begin
            try
                # activate separate test environment
                Pkg.activate(DIRECTORY_Turing_tests)
                Pkg.develop(PackageSpec(; path=DIRECTORY_DynamicPPL))
                Pkg.instantiate()

                # make sure that the new environment is considered `using` and `import` statements
                # (not added automatically on Julia 1.3, see e.g. PR #209)
                if !(joinpath(DIRECTORY_Turing_tests, "Project.toml") in Base.load_path())
                    pushfirst!(LOAD_PATH, DIRECTORY_Turing_tests)
                end

                include(joinpath("turing", "runtests.jl"))
            catch err
                err isa Pkg.Resolve.ResolverError || rethrow()
                # If we can't resolve that means this is incompatible by SemVer and this is fine
                # It means we marked this as a breaking change, so we don't need to worry about
                # Mistakenly introducing a breaking change, as we have intentionally made one
                @info "Not compatible with this release. No problem." exception = err
            end
        end
    end
end
