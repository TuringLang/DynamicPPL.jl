module DynamicPPLForwardDiffExtTests

using DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using Distributions: MvNormal, Normal, logpdf
using LinearAlgebra: I
using Test: @test, @testset

@testset "ForwardDiff tweak_adtype" begin
    MODEL_SIZE = 10
    @model f() = x ~ MvNormal(zeros(MODEL_SIZE), I)
    model = f()
    x = randn(MODEL_SIZE)

    @testset "Chunk size setting" for chunksize in (nothing, 0)
        base_adtype = AutoForwardDiff(; chunksize=chunksize)
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype isa AutoForwardDiff{MODEL_SIZE}
    end

    @testset "Tag setting" begin
        base_adtype = AutoForwardDiff()
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype.tag isa ForwardDiff.Tag{DynamicPPL.DynamicPPLTag}
    end
end

@testset "argument bindings" begin
    @model function argument_binding(x)
        μ ~ Normal()
        for i in eachindex(x)
            x[i] ~ Normal(μ, 1)
        end
        return x
    end
    original = argument_binding([1.0, 2.0])
    replaced = condition(original; x=[2.0, 3.0])
    @test run_ad(
        replaced,
        AutoForwardDiff();
        params=[0.5],
        test=WithExpectedResult(
            logpdf(Normal(), 0.5) +
            logpdf(Normal(0.5, 1), 2.0) +
            logpdf(Normal(0.5, 1), 3.0),
            [3.5],
        ),
    ) isa Any
    partial = decondition(original, @varname(x[1]))
    @test run_ad(
        partial,
        AutoForwardDiff();
        params=[0.5, 0.25],
        test=WithExpectedResult(
            logpdf(Normal(), 0.5) +
            logpdf(Normal(0.5, 1), 0.25) +
            logpdf(Normal(0.5, 1), 2.0),
            [0.75, 0.25],
        ),
    ) isa Any
end

@testset "zero-dimensional LHS gradients" begin
    @model scalar_array() = (x = fill(0.0); x[] ~ Normal(); x)
    @model argument_array(x) = x[] ~ Normal()
    for model in (scalar_array(), decondition(argument_array(fill(0.0)), @varname(x[])))
        @test run_ad(
            model,
            AutoForwardDiff();
            params=[0.5],
            test=WithExpectedResult(logpdf(Normal(), 0.5), [-0.5]),
        ) isa Any
    end
end

@model function observed_input(y)
    x ~ Normal()
    y ~ Normal(x)
    0.0 ~ Normal(x)
    return x
end
@model nested_input(y) = a ~ to_submodel(observed_input(y))

@testset "Context parameter types come from inputs" begin
    for T in (Float32, Float64, BigFloat)
        y = T(2)
        x = T(0.5)
        for (model, vn) in
            ((observed_input(y), @varname(x)), (nested_input(y), @varname(a.x)))
            for threaded in (false, true)
                m = setthreadsafe(model, threaded)
                function density(value)
                    values = VarNamedTuple((vn => TransformedValue([value], Unlink()),))
                    _, output = evaluate!!(
                        m,
                        Context(
                            InitFromParams(values, nothing),
                            DynamicPPL.infer_transform_strategy_from_values(values),
                        ),
                        VarInfo(),
                    )
                    return getlogjoint(output)
                end
                @test ForwardDiff.derivative(density, x) ≈ y - 3x
            end
        end
    end
end

end
