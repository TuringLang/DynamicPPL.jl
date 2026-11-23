module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using ADTypes: AutoMooncake
using Distributions: MvNormal, Normal
using LinearAlgebra: I
using LogDensityProblems: dimension, logdensity_and_gradient
using Mooncake: Mooncake
using StableRNGs: StableRNG
using DynamicPPL:
    @model,
    @varname,
    condition,
    fix,
    is_transformed,
    LogDensityFunction,
    to_submodel,
    VarInfo,
    VectorValueAccumulator
using Test: @testset, @test, @test_broken, @inferred

# At top level: a local child model would be captured by the parent's body and lose its type.
@model observed_child(m, x) = (x ~ Normal(m); x)
@model function unprefixed_parent()
    m ~ Normal()
    return _ignore ~ to_submodel(observed_child(m, missing), false)
end
@model pair_child(m, x, y) = (x ~ Normal(m); y ~ MvNormal(fill(m, 2), I); (x, y))
@model function body_binding(binding)
    m ~ Normal()
    return a ~ to_submodel(binding(pair_child(m, missing, missing)))
end
@model function tuple_loop(y)
    for i in eachindex(y)
        y[i] ~ Normal()
    end
    return y
end

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123456),
        is_transformed,
        VarInfo(VectorValueAccumulator());
        unsafe_perturb=true,
        interface_only=true,
    )

    @testset "unprefixed child with parent bindings infers" begin
        model = condition(unprefixed_parent(), @varname(x) => 1.5)
        ldf = LogDensityFunction(model; adtype=AutoMooncake())
        @test @inferred(logdensity_and_gradient(ldf, [0.3])) isa
            Tuple{Float64,Vector{Float64}}
    end

    @testset "whole-root bindings in a model body infer" begin
        for binding in (
            c -> condition(c; x=1.5),
            c -> fix(c, @varname(x) => 1.5),
            c -> condition(c, (y=[0.5, 1.0],); x=1.5),
            c -> condition(condition(c; x=0.5); x=1.5),
        )
            ldf = LogDensityFunction(body_binding(binding); adtype=AutoMooncake())
            @test @inferred(logdensity_and_gradient(ldf, zeros(dimension(ldf)))) isa
                Tuple{Float64,Vector{Float64}}
        end
    end

    @testset "loop over a tuple argument with missing entries" begin
        # Gradient preparation already fails, with a `MethodError` for `increment!!`.
        model = tuple_loop((missing, missing))
        @test_broken LogDensityFunction(model; adtype=AutoMooncake()) isa LogDensityFunction
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
