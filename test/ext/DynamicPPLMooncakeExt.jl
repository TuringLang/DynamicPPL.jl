module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Mooncake: Mooncake
using ADTypes: AutoMooncake, AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface
using Distributions: Normal, MvNormal, logpdf
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems, logdensity_and_gradient, dimension
using OffsetArrays: OffsetArray
using StableRNGs: StableRNG
using DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult
using Test: @test, @testset, @inferred, @test_throws

struct RuntimeBindingRecord{A,B}
    a::A
    b::B
end
@model function runtime_record_child(p)
    p.a ~ Normal()
    p.b ~ Normal(p.a)
    return p.a + p.b
end
@model function runtime_record_parent(op, wrap)
    m ~ Normal()
    child = if op === decondition
        decondition(runtime_record_child(wrap(m, 2m)), @varname(p.a))
    elseif op === unfix
        unfix(fix(decondition(runtime_record_child(wrap(m, 2m))); p=wrap(m, 2m)), @varname(p.a))
    else
        op(runtime_record_child(wrap(m, 2m)), @varname(p.a) => 3m)
    end
    a ~ to_submodel(child)
    return z ~ Normal(a)
end
@testset "runtime record bindings" begin
    for op in (condition, fix, decondition, unfix)
        x = op in (condition, fix) ? [0.3, 0.5] : [0.3, 0.4, 0.5]
        expected = if op in (condition, fix)
            (
                logpdf(Normal(), x[1]) +
                (op === condition ? logpdf(Normal(), 3x[1]) : 0) +
                logpdf(Normal(3x[1]), 2x[1]) +
                logpdf(Normal(5x[1]), x[2]),
                [
                    (op === condition ? -10x[1] : -x[1]) - x[1] + 5(x[2] - 5x[1]),
                    5x[1] - x[2],
                ],
            )
        elseif op === decondition
            (-3.945754132818691, [-1.7, -0.7, 0.5])
        else
            (
                logpdf(Normal(), x[1]) +
                logpdf(Normal(), x[2]) +
                logpdf(Normal(x[2] + 2x[1]), x[3]),
                [-1.3, -0.9, 0.5],
            )
        end
        for ad in (AutoForwardDiff(), AutoMooncake())
            @testset "$op $ad" begin
                @test_throws ArgumentError runtime_record_parent(op, RuntimeBindingRecord)(
                    StableRNG(1)
                )
                model = runtime_record_parent(op, (a, b) -> (; a, b))
                _, vi = DynamicPPL.init!!(
                    StableRNG(1), model, VarInfo(VectorValueAccumulator()), InitFromPrior()
                )
                ldf = LogDensityFunction(model, getlogjoint_internal, vi; adtype=ad)
                value, gradient = logdensity_and_gradient(ldf, x)
                @test value ≈ expected[1]
                @test gradient ≈ expected[2]
            end
        end
    end
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

@model function partial_observations(y)
    m ~ Normal()
    for i in eachindex(y)
        y[i] ~ Normal(m)
    end
end

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123456),
        is_transformed,
        VarInfo(VectorValueAccumulator());
        unsafe_perturb=true,
        interface_only=true,
    )

    @testset "runtime partial binding" begin
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

    @testset "runtime abstract argument arrays" begin
        @model function abstract_child(y)
            for i in eachindex(y)
                y[i] ~ Normal()
            end
            return sum(y)
        end
        @model function abstract_parent(::Val{T}; slice=false) where {T}
            m ~ Normal()
            a ~ to_submodel(
                condition(
                    abstract_child(
                        T === Tuple ? (zero(m), m, 0, 0.0f0) : T[zero(m), m, 0, 0.0f0]
                    ),
                    (slice ? @varname(y[1:2]) => [2m, 3m] : @varname(y[1]) => 2m),
                ),
            )
            return z ~ Normal(a + m)
        end
        @test_throws ArgumentError abstract_parent(Val(Tuple))(StableRNG(1))
        for (T, slice) in ((Real, false), (Any, false), (Real, true), (Any, true))
            model = abstract_parent(Val(T); slice=slice)
            _, vi = DynamicPPL.init!!(
                StableRNG(123456),
                model,
                VarInfo(VectorValueAccumulator()),
                InitFromPrior(),
                UnlinkAll(),
            )
            mc = LogDensityFunction(model, getlogjoint_internal, vi; adtype=AutoMooncake())
            fd = LogDensityFunction(
                model, getlogjoint_internal, vi; adtype=AutoForwardDiff()
            )
            for x in ([0.3, 0.5], [-0.4, -0.2], [0.3, 0.5])
                value, gradient = logdensity_and_gradient(mc, x)
                fd_value, fd_gradient = logdensity_and_gradient(fd, x)
                @test value ≈ fd_value
                @test gradient ≈ fd_gradient
                h = 1e-5
                numerical = map(eachindex(x)) do i
                    plus, minus = copy(x), copy(x)
                    plus[i] += h
                    minus[i] -= h
                    (
                        LogDensityProblems.logdensity(fd, plus) -
                        LogDensityProblems.logdensity(fd, minus)
                    ) / (2h)
                end
                @test gradient ≈ numerical
            end
        end
    end

    @testset "partial binding gradients" begin
        original = partial_observations([1.0, 2.0, 3.0])
        for model in (
            condition(original, @varname(y[1]) => 0.0),
            fix(original, @varname(y[1]) => 0.0),
            decondition(original, @varname(y[1])),
        )
            mc = LogDensityFunction(model; adtype=AutoMooncake())
            fd = LogDensityFunction(model; adtype=AutoForwardDiff())
            x = fill(0.5, dimension(mc))
            mc_value, mc_gradient = logdensity_and_gradient(mc, x)
            fd_value, fd_gradient = logdensity_and_gradient(fd, x)
            @test mc_value ≈ fd_value
            @test mc_gradient ≈ fd_gradient
        end
    end
end

@testset "runtime bindings preserve AD values" begin
    @model runtime_child(y) = (y[1] ~ Normal(); y[2] ~ Normal())
    @model function runtime_parent(bind, partial)
        m ~ Normal()
        child = runtime_child(fill(zero(m), 2))
        bound = partial ? bind(child, @varname(y[1]) => 2m) : bind(child; y=[2m, zero(m)])
        a ~ to_submodel(bound)
        return m
    end
    for bind in (condition, fix), partial in (false, true)
        ldf = LogDensityFunction(runtime_parent(bind, partial); adtype=AutoMooncake())
        _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3])
        @test gradient ≈ [bind === condition ? -1.5 : -0.3]
    end
end

@testset "aliased latent argument storage" begin
    @model function aliased_argument(x)
        x.a[1] ~ Normal()
        return 0.0 ~ Normal(x.b[1], 1)
    end
    # Abstract storage can accept AD numbers without replacing either shared array.
    value = Real[0.0]
    model = decondition(aliased_argument((a=value, b=value)))
    _, vi = init!!(
        StableRNG(123456),
        model,
        VarInfo(VectorValueAccumulator()),
        InitFromPrior(),
        UnlinkAll(),
    )
    for adtype in (AutoForwardDiff(), AutoMooncake())
        ldf = LogDensityFunction(model, getlogjoint_internal, vi; adtype)
        density, gradient = logdensity_and_gradient(ldf, [2.0])
        @test density ≈ logjoint(model, (x=(a=[2.0],),))
        @test gradient ≈ [-4.0]
    end
end

mutable struct InnerConstructorADState{T}
    x::T
    InnerConstructorADState(x::T) where {T} = new{T}(x)
end

@testset "latent inner-constructor argument gradients" begin
    @model inner_state(s) = (y ~ Normal(s.x); s.x ~ Normal(); nothing)
    @model function outer_state()
        m ~ Normal()
        return a ~ to_submodel(decondition(inner_state(InnerConstructorADState(m))))
    end
    model = outer_state()
    _, vi = init!!(
        StableRNG(123456),
        model,
        VarInfo(VectorValueAccumulator()),
        InitFromPrior(),
        UnlinkAll(),
    )
    for adtype in (AutoForwardDiff(), AutoMooncake())
        ldf = LogDensityFunction(model, getlogjoint_internal, vi; adtype)
        _, gradient = logdensity_and_gradient(ldf, [0.3, 0.7, 0.9])
        @test gradient ≈ [0.1, -0.4, -0.9]
    end
end

mutable struct ConstantCopyState{T}
    const offset::T
    x::Real
    ConstantCopyState(offset::T) where {T} = new{T}(offset, zero(offset))
end
@testset "copy preserves AD identity" begin
    @model function alias_copy(x)
        x.a[1] ~ Normal()
        x.c ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    data = Real[0.0]
    alias_model = condition(
        decondition(alias_copy((a=data, b=data, c=0.0))), @varname(x.c) => 0.0
    )
    @model function const_copy(s)
        s.x ~ Normal()
        return 0.0 ~ Normal(s.x + s.offset)
    end
    @model function runtime_const_copy()
        m ~ Normal()
        return a ~ to_submodel(decondition(const_copy(ConstantCopyState(m))))
    end
    @model function copy_child(y)
        y[1] ~ Normal()
        y[2] ~ Normal()
        return sum(y)
    end
    @model function runtime_copy(make_array)
        m ~ Normal()
        a ~ to_submodel(condition(copy_child(make_array(m)), @varname(y[1]) => 2m))
        return z ~ Normal(a)
    end
    @model function nested_copy_child(x)
        y ~ Normal(x.b[1])
        x.a[1] ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    @model function nested_copy_parent()
        m ~ MvNormal(zeros(1), ones(1))
        return a ~ to_submodel(decondition(nested_copy_child((a=m, b=m))))
    end
    @model named_copy_child(p) = p.m ~ Normal(1)
    @model function named_copy_parent()
        m ~ Normal()
        return a ~ to_submodel(condition(named_copy_child((m=m,)), @varname(p.m) => 2m))
    end
    for adtype in (AutoForwardDiff(), AutoMooncake())
        nested = LogDensityFunction(nested_copy_parent(); adtype)
        _, nested_gradient = logdensity_and_gradient(nested, [0.3, 0.7, 0.9])
        @test nested_gradient ≈ [0.1, -0.4, -1.8]
        named = LogDensityFunction(named_copy_parent(); adtype)
        for m in (0.0, 0.3)
            _, named_gradient = logdensity_and_gradient(named, [m])
            @test named_gradient ≈ [2 - 5m]
        end
        ldf = LogDensityFunction(alias_model; adtype)
        for x in ([2.0], [-0.4], [2.0])
            density, gradient = logdensity_and_gradient(ldf, x)
            @test density ≈ 3logpdf(Normal(), 0.0) - x[1]^2
            @test gradient ≈ -2x
        end
        ldf = LogDensityFunction(runtime_const_copy(); adtype)
        for x in ([0.3, 0.7], [-0.4, 0.2], [0.3, 0.7])
            density, gradient = logdensity_and_gradient(ldf, x)
            @test density ≈
                logpdf(Normal(), x[1]) +
                  logpdf(Normal(), x[2]) +
                  logpdf(Normal(x[1] + x[2]), 0)
            @test gradient ≈ [-2x[1] - x[2], -x[1] - 2x[2]]
        end
        for make_array in (m -> [m, m], m -> Real[m, m])
            ldf = LogDensityFunction(runtime_copy(make_array); adtype)
            density, gradient = logdensity_and_gradient(ldf, [0.3, 0.5])
            @test density ≈
                logpdf(Normal(), 0.3) +
                  logpdf(Normal(), 0.6) +
                  logpdf(Normal(), 0.3) +
                  logpdf(Normal(0.9), 0.5)
            @test gradient ≈ [-3.0, 0.4]
        end
    end
end

@model function replaced_argument(x=missing)
    x === missing && (x = zeros(2))
    x[1] ~ Normal()
    return x[2] ~ Normal(x[1])
end
@model function replaced_keyword(; x=missing)
    x === missing && (x = zeros(2))
    x[1] ~ Normal()
    return x[2] ~ Normal(x[1])
end
@model function partly_missing(x)
    x[1] ~ Normal()
    x[2] ~ Normal(x[1])
    x[3] ~ Normal(x[2])
    return x
end
@model function partly_missing_parent(make_data)
    m ~ Normal()
    a ~ to_submodel(partly_missing(make_data(m)))
    return 0.5 ~ Normal(a[3])
end
@testset "placeholder argument gradients" begin
    a, b, c = 0.3, -0.4, 0.8
    chain = (logpdf(Normal(), a) + logpdf(Normal(a), b), [b - 2a, a - b])
    # The parent observes `a.x[2] = d` with `dd = ∂d/∂m`; parameters are `m`, `a.x[1]`, `a.x[3]`.
    function parent_result(d, dd)
        value = sum(logpdf.(Normal(), [a, b])) + logpdf(Normal(b), d) + logpdf(Normal(d), c)
        value += logpdf(Normal(c), 0.5)
        return value, [-a + dd * (b - d + c - d), d - 2b, d - 2c + 0.5]
    end
    data = Union{Missing,Float64}[missing, 1.5, missing]
    cases = (
        (replaced_argument(), [a, b], chain...),
        (replaced_keyword(), [a, b], chain...),
        (
            partly_missing(data),
            [a, b],
            logpdf(Normal(), a) + logpdf(Normal(a), 1.5) + logpdf(Normal(1.5), b),
            [1.5 - 2a, 1.5 - b],
        ),
        (partly_missing_parent(_ -> data), [a, b, c], parent_result(1.5, 0)...),
        (
            partly_missing_parent(m -> Union{Missing,typeof(m)}[missing, 2m, missing]),
            [a, b, c],
            parent_result(2a, 2)...,
        ),
    )
    for (model, params, value, gradient) in cases,
        adtype in (AutoForwardDiff(), AutoMooncake())

        test = WithExpectedResult(value, gradient)
        @test run_ad(model, adtype; params, test, verbose=false) isa Any
    end
end

@testset "latent branches beside immutable observations" begin
    @model function partial_fields(x)
        x.a[1][1] ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    model = decondition(
        partial_fields((a=Any[[0.0], [1.0, 2.0]], b=1.0:2.0)), @varname(x.a)
    )
    for adtype in (AutoForwardDiff(), AutoMooncake())
        ldf = LogDensityFunction(model; adtype)
        value, gradient = logdensity_and_gradient(ldf, [0.3])
        @test value ≈ logpdf(Normal(), 0.3) + logpdf(Normal(1.0), 0.0)
        @test gradient ≈ [-0.3]
    end
end

@testset "latent arguments holding arrays of arrays" begin
    @model function nested_argument(x)
        for i in eachindex(x), j in eachindex(x[i])
            x[i][j] ~ Normal(i)
        end
    end
    for x in (
        Any[[0.0], [0.0, 0.0]],
        view([[0.0], [0.0, 0.0]], 1:2),
        OffsetArray([[0.0], [0.0, 0.0]], 1:2),
        [[0.0], [0.0, 0.0]],
    )
        model = decondition(nested_argument(x))
        ldf = LogDensityFunction(model; adtype=AutoMooncake())
        @test logdensity_and_gradient(ldf, fill(0.1, 3))[2] ≈ [0.9, 1.9, 1.9]
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

@testset "zero-dimensional LHS gradients" begin
    @model scalar_array() = (x = fill(0.0); x[] ~ Normal(); x)
    @model argument_array(x) = x[] ~ Normal()
    for model in (scalar_array(), decondition(argument_array(fill(0.0)), @varname(x[])))
        @test run_ad(
            model,
            AutoMooncake();
            params=[0.5],
            test=WithExpectedResult(logpdf(Normal(), 0.5), [-0.5]),
        ) isa Any
    end
end

end # module
