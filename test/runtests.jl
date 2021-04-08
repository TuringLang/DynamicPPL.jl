using DynamicPPL
using AbstractMCMC
using AbstractPPL
using Bijectors
using Distributions
using DistributionsAD
using Documenter
using ForwardDiff
using MacroTools
using MCMCChains
using Tracker
using Zygote

using Distributed
using Pkg
using Random
using Serialization
using Test

using DynamicPPL: getargs_dottilde, getargs_tilde, Selector

const DIRECTORY_DynamicPPL = dirname(dirname(pathof(DynamicPPL)))
const DIRECTORY_Turing_tests = joinpath(DIRECTORY_DynamicPPL, "test", "turing")

Random.seed!(100)

include("test_util.jl")

@testset "DynamicPPL.jl" begin
    include("utils.jl")
    include("compiler.jl")
    include("varinfo.jl")
    include("model.jl")
    include("sampler.jl")
    include("prob_macro.jl")
    include("independence.jl")
    include("distribution_wrappers.jl")
    include("context_implementations.jl")

    include("threadsafe.jl")

    include("serialization.jl")

    @testset "compat" begin
        include(joinpath("compat", "ad.jl"))
    end

    @testset "turing" begin
        # activate separate test environment
        Pkg.activate(DIRECTORY_Turing_tests)
        Pkg.develop(PackageSpec(path=DIRECTORY_DynamicPPL))
        Pkg.instantiate()

        # make sure that the new environment is considered `using` and `import` statements
        # (not added automatically on Julia 1.3, see e.g. PR #209)
        if !(joinpath(DIRECTORY_Turing_tests, "Project.toml") in Base.load_path())
            pushfirst!(LOAD_PATH, DIRECTORY_Turing_tests)
        end

        include(joinpath("turing", "runtests.jl"))
    end

    @testset "doctests" begin
        DocMeta.setdocmeta!(
            DynamicPPL,
            :DocTestSetup,
            :(using DynamicPPL);
            recursive=true,
        )
        doctest(DynamicPPL; manual=false)
    end
end
