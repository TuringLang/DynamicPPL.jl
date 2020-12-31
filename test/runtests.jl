using DynamicPPL
using Bijectors
using Distributions
using DistributionsAD
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

using DynamicPPL: vsym, vinds, getargs_dottilde, getargs_tilde, Selector

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

    #include("serialization.jl")

    @testset "compat" begin
        include(joinpath("compat", "ad.jl"))
    end

    @testset "turing" begin
        Pkg.activate("turing")
        Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
        Pkg.instantiate()
        include(joinpath("turing", "runtests.jl"))
    end
end
