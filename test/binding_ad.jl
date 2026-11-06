module BindingADTests

using ADTypes: AutoForwardDiff, AutoMooncake, AutoReverseDiff
using DynamicPPL, Distributions, Test
using DifferentiationInterface: DifferentiationInterface
using AbstractPPL: @of, of
using ForwardDiff: ForwardDiff
using StaticArrays: SVector
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using LogDensityProblems: LogDensityProblems

@model function observed_array(x)
    mu ~ Normal()
    for i in eachindex(x)
        x[i] ~ Normal(mu)
    end
    return x
end
@model function copied_child(x)
    y ~ Normal(x.a[1])
    x.b[1] ~ Normal()
    return 0.0 ~ Normal(x.a[1])
end
@model function copied_parent()
    m ~ Normal()
    return a ~ to_submodel(decondition(copied_child((a=[m], b=[m]))))
end
@model function dynamic_support()
    x ~ Normal()
    y ~ Uniform(0, exp(x))
    return y
end
@model scalar_storage(x) = (x[] ~ Normal(); x)

@testset "binding gradients and repeated buffers" begin
    original = [1.0, 2.0]
    @testset "$adtype" for adtype in (
        AutoForwardDiff(),
        AutoMooncake(),
        AutoReverseDiff(),
        AutoReverseDiff(; compile=true),
    )
        for (model, params, density, gradient) in (
            (
                condition(observed_array(original); x=[2.0, 3.0]),
                [0.5],
                logpdf(Normal(), 0.5) + sum(logpdf.(Normal(0.5), [2.0, 3.0])),
                [3.5],
            ),
            (
                decondition(observed_array(original), @varname(x[1])),
                [0.5, 0.25],
                logpdf(Normal(), 0.5) + sum(logpdf.(Normal(0.5), [0.25, 2.0])),
                [0.75, 0.25],
            ),
            (decondition(scalar_storage(fill(0.0))), [0.5], logpdf(Normal(), 0.5), [-0.5]),
        )
            ldf = LogDensityFunction(model, getlogjoint_internal, UnlinkAll(); adtype)
            for _ in 1:3
                value, grad = LogDensityProblems.logdensity_and_gradient(ldf, params)
                @test value ≈ density
                @test grad ≈ gradient
            end
        end
        ldf = LogDensityFunction(copied_parent(), getlogjoint_internal, UnlinkAll(); adtype)
        for x in ([0.3, 0.7, 0.9], [-0.4, 0.2, -0.3], [0.3, 0.7, 0.9])
            value, grad = LogDensityProblems.logdensity_and_gradient(ldf, x)
            @test value ≈ sum(logpdf.(Normal(), [x[1], x[2] - x[1], x[3], x[1]]))
            @test grad ≈ [x[2] - 3x[1], x[1] - x[2], -x[3]]
        end
        @test original == [1.0, 2.0]
    end
    @testset "$adtype" for adtype in (AutoForwardDiff(), AutoMooncake(), AutoReverseDiff())
        ldf = LogDensityFunction(
            dynamic_support(), getlogjoint_internal, UnlinkAll(); adtype
        )
        for x in ([0.3, 0.2], [-0.4, 0.1])
            value, grad = LogDensityProblems.logdensity_and_gradient(ldf, x)
            @test value ≈ logpdf(Normal(), x[1]) - x[1]
            @test grad ≈ [-x[1] - 1, 0.0]
        end
    end
end

@model runtime_storage(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
@testset "runtime binding types and threaded parity" begin
    for T in (Float32, Float64, BigFloat),
        bind in (condition, fix),
        threaded in (false, true)

        function density(theta)
            model = bind(
                runtime_storage(Vector{Union{Missing,typeof(theta)}}(missing, 2)),
                @varname(x[1]) => theta,
                @of(x = of(Array, typeof(theta), 2))
            )
            model = setthreadsafe(model, threaded)
            return logjoint(model, (; x=fill(theta, 2)))
        end
        theta = T(0.5)
        @test density(theta) ≈ (bind === condition ? 2 : 1) * logpdf(Normal(), theta)
        @test ForwardDiff.derivative(density, theta) ≈
            (bind === condition ? -2 : -1) * theta
    end
end

@model bound_child(x) = (x ~ Normal(); x)
@model function binding_parent(bind)
    x ~ Normal()
    return y ~ to_submodel(bind(bound_child(missing); x=x))
end
@testset "gradients through bindings constructed in a model" begin
    for (bind, weight) in ((condition, 2), (fix, 1))
        ldf = LogDensityFunction(
            binding_parent(bind), getlogjoint_internal, UnlinkAll(); adtype=AutoMooncake()
        )
        value, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.5])
        @test value ≈ weight * logpdf(Normal(), 0.5)
        @test gradient ≈ [-weight * 0.5]
    end
end

@model tuple_latent(x) = (x[1] ~ Normal(); x[2] ~ Normal(x[1]); x)
@testset "positional argument gradients" begin
    for data in ((1.0, 2.0), SVector(1.0, 2.0)),
        adtype in (AutoForwardDiff(), AutoMooncake())

        for fixed_position in (false, true)
            model = decondition(tuple_latent(data))
            model = fixed_position ? fix(model, @varname(x[2]) => 0.7) : model
            params = fixed_position ? [0.3] : [0.3, 0.7]
            ldf = LogDensityFunction(model, getlogjoint_internal, UnlinkAll(); adtype)
            value, gradient = LogDensityProblems.logdensity_and_gradient(ldf, params)
            @test value ≈
                logpdf(Normal(), 0.3) + (fixed_position ? 0 : logpdf(Normal(0.3), 0.7))
            @test gradient ≈ (fixed_position ? [-0.3] : [0.1, -0.4])
        end
    end
end
end
