module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Mooncake: Mooncake
using StableRNGs: StableRNG
using DynamicPPL: is_transformed, VarInfo, VectorValueAccumulator
using Test: @testset, @test, @inferred
using DynamicPPL: DynamicPPL, @varname
using ADTypes: AutoMooncake
using DifferentiationInterface: DifferentiationInterface

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123456),
        is_transformed,
        VarInfo(VectorValueAccumulator());
        unsafe_perturb=true,
        interface_only=true,
    )
end

@testset "dense scalar argument overlay rule" begin
    ext = Base.get_extension(DynamicPPL, :DynamicPPLMooncakeExt)
    mode = isdefined(Mooncake, :ReverseMode) ? (; mode=Mooncake.ReverseMode) : (;)
    function is_overlay_primitive(bindings, template, vn)
        sig = Tuple{
            typeof(DynamicPPL._model_argument_value),
            typeof(bindings),
            typeof(template),
            typeof(vn),
        }
        return if isdefined(Mooncake, :ReverseMode)
            Mooncake.is_primitive(
                Mooncake.DefaultCtx, Mooncake.ReverseMode, sig, Base.get_world_counter()
            )
        else
            Mooncake.is_primitive(Mooncake.DefaultCtx, sig, Base.get_world_counter())
        end
    end
    function check_overlay(values, mask, template, vn)
        bindings = DynamicPPL.VarNamedTuples.PartialArray(values, mask)
        @test is_overlay_primitive(bindings, template, vn)
        _, pullback = @inferred Mooncake.rrule!!(
            Mooncake.zero_fcodual(DynamicPPL._model_argument_value),
            Mooncake.zero_fcodual(bindings),
            Mooncake.zero_fcodual(template),
            Mooncake.zero_fcodual(vn),
        )
        @inferred pullback(Mooncake.NoRData())
        return Mooncake.TestUtils.test_rule(
            StableRNG(123456),
            DynamicPPL._model_argument_value,
            bindings,
            template,
            vn;
            unsafe_perturb=true,
            mode...,
        )
    end
    @testset "primitive boundaries" begin
        for T in (Float16, Float32, Float64, BigFloat, Int)
            data = [DynamicPPL.ModelValue{DynamicPPL.Condition}(T(1))]
            bindings = DynamicPPL.VarNamedTuples.PartialArray(data, [true])
            @test is_overlay_primitive(bindings, T[0], @varname(x)) == (T <: Base.IEEEFloat)
            @test !is_overlay_primitive(bindings, T[0], @varname(x[1]))
            @test !is_overlay_primitive(bindings, view(T[0], :), @varname(x))
            @test !is_overlay_primitive(
                bindings, (T === Float64 ? Float32 : Float64)[0], @varname(x)
            )
        end
        data = [DynamicPPL.ModelValue{DynamicPPL.Condition}([1.0])]
        bindings = DynamicPPL.VarNamedTuples.PartialArray(data, [true])
        @test !is_overlay_primitive(bindings, [0.0], @varname(x))
    end
    @testset "$T" for T in (Float32, Float64)
        for role in (DynamicPPL.ArgumentCondition, DynamicPPL.Condition, DynamicPPL.Fix)
            values = [DynamicPPL.ModelValue{role}(T(i)) for i in 1:3]
            for mask in ([true, true, true], [false, true, false], [false, false, false]),
                n in (3, 5)

                check_overlay(values, mask, T.(1:n), @varname(x))
            end
        end
        values = ext.ScalarArgumentBinding{T}[
            DynamicPPL.ModelValue{DynamicPPL.Condition}(T(1)),
            DynamicPPL.ModelValue{DynamicPPL.Fix}(T(2)),
            DynamicPPL.ModelValue{DynamicPPL.ArgumentCondition}(T(3)),
        ]
        for (mask, template) in (
            ([true, true, true], T[]),
            ([false, true, true], T[0]),
            ([true, false, true], zeros(T, 3)),
            ([true, false, true], zeros(T, 5)),
        )
            check_overlay(values, mask, template, nothing)
        end
        check_overlay(ext.ScalarArgumentBinding{T}[], Bool[], T[1], nothing)
        # Unset binding slots must not be read, even when their data is uninitialised.
        partial = Vector{ext.ScalarArgumentBinding{T}}(undef, 3)
        partial[2] = values[2]
        check_overlay(partial, [false, true, false], zeros(T, 3), @varname(x))

        # Both the bound payloads and the surviving template entries are differentiable.
        function objective(z)
            data = [DynamicPPL.ModelValue{DynamicPPL.Condition}(z[i]) for i in 1:3]
            bindings = DynamicPPL.VarNamedTuples.PartialArray(data, [true, false, true])
            result = DynamicPPL._model_argument_value(bindings, z[4:8], @varname(x))
            return sum(i * result[i] for i in eachindex(result))
        end
        x = T.(1:8)
        prep = DifferentiationInterface.prepare_gradient(objective, AutoMooncake(), x)
        for z in (x, -x, x)
            gradient = DifferentiationInterface.gradient(objective, prep, AutoMooncake(), z)
            @test eltype(gradient) === T
            @test gradient ≈ [1, 0, 3, 0, 2, 0, 0, 0]
            h = cbrt(eps(T))
            numerical = map(eachindex(z)) do i
                plus, minus = copy(z), copy(z)
                plus[i] += h
                minus[i] -= h
                (objective(plus) - objective(minus)) / (2h)
            end
            @test isapprox(gradient, numerical; atol=10h^2, rtol=10h^2)
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
