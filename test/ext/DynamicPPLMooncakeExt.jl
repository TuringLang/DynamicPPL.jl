module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Mooncake: Mooncake
using ADTypes: AutoMooncake
using Distributions: Normal
using ForwardDiff: ForwardDiff
using StableRNGs: StableRNG
using DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad
using Test: @test, @testset

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123456),
        is_transformed,
        VarInfo(VectorValueAccumulator());
        unsafe_perturb=true,
        interface_only=true,
    )

    @testset "evaluation-local partial binding" begin
        @model function child(y)
            for i in eachindex(y)
                y[i] ~ Normal()
            end
            return sum(y)
        end
        @model function parent(make_array)
            m ~ Normal()
            a ~ to_submodel(condition(child(make_array(m)), @varname(y[1]) => 2m))
            return z ~ Normal(a)
        end
        for make_array in (m -> [m], m -> Real[m])
            @test run_ad(
                parent(make_array), AutoMooncake(); params=[0.3, 0.5], rng=StableRNG(123456)
            ) isa DynamicPPL.TestUtils.AD.ADResult
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
