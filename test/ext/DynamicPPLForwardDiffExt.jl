module DynamicPPLForwardDiffExtTests

using DynamicPPL
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using Distributions: MvNormal, Normal
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

@model function observed_input(y)
    x ~ Normal()
    y ~ Normal(x)
    0.0 ~ Normal(x)
    return x
end
@model nested_input(y) = a ~ to_submodel(observed_input(y))

@testset "DefaultContext parameter types come from inputs" begin
    for T in (Float32, Float64, BigFloat)
        y = T(2)
        x = T(0.5)
        for (model, vn) in
            ((observed_input(y), @varname(x)), (nested_input(y), @varname(a.x)))
            for threaded in (false, true)
                m = setthreadsafe(model, threaded)
                function density(value)
                    values = VarNamedTuple((vn => TransformedValue([value], Unlink()),))
                    _, output = evaluate!!(m, DefaultContext(values), OnlyAccsVarInfo())
                    return getlogjoint(output)
                end
                @test ForwardDiff.derivative(density, x) ≈ y - 3x
            end
        end
    end
end

end
