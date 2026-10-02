module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Mooncake: Mooncake
using ADTypes: AutoMooncake, AutoForwardDiff
using Distributions: Normal
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems, logdensity_and_gradient, dimension
using StableRNGs: StableRNG
using DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad
using Test: @test, @testset

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
        for (T, slice) in
            ((Real, false), (Any, false), (Tuple, false), (Real, true), (Any, true))
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
        0.0 ~ Normal(x.b[1], 1)
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

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
