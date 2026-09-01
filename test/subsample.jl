module DynamicPPLSubsampleTests

using Distributions:
    Beta,
    BernoulliLogit,
    ContinuousUnivariateDistribution,
    Distributions,
    MatrixNormal,
    MvNormal,
    Normal,
    insupport,
    logpdf,
    loglikelihood,
    product_distribution,
    truncated
using DynamicPPL
using FillArrays: Fill
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using LogDensityProblems: LogDensityProblems
using Random: Random
using StableRNGs: StableRNG
using Test

@model function normal_location()
    μ ~ Normal(0, 10)
    return x ~ independent_distribution(Normal(μ, 1))
end

struct CountingDistribution <: ContinuousUnivariateDistribution
    evaluations::Base.RefValue{Int}
end
function Distributions.logpdf(dist::CountingDistribution, value::Real)
    dist.evaluations[] += 1
    return -abs2(value) / 2
end

mutable struct CountingRNG{R<:Random.AbstractRNG} <: Random.AbstractRNG
    rng::R
    draws::Int
end
function Random.rand(rng::CountingRNG, values::UnitRange{Int})
    rng.draws += 1
    return rand(rng.rng, values)
end

function independent_problem(data; transform_strategy=UnlinkAll())
    model = normal_location() | (x=data,)
    return independent_problem(model, size(data, ndims(data)); transform_strategy)
end

function independent_problem(model, dataset_size; transform_strategy=UnlinkAll())
    return subsample(
        StableRNG(1), model, (rng, N) -> collect(1:N), dataset_size; transform_strategy
    )
end

@testset "Subsampling" begin
    @testset "full and batched densities" begin
        data = [-2.0, -1.0, 0.5, 3.0]
        model = normal_location() | (x=data,)
        problem = independent_problem(data)
        params = [0.25]

        expected =
            logpdf(Normal(0, 10), only(params)) +
            loglikelihood(Normal(only(params), 1), data)
        @test @inferred(LogDensityProblems.logdensity(problem, params)) ≈ expected
        @test LogDensityProblems.dimension(problem) == 1
        @test get_input_vector_type(problem) === Vector{Float64}
        @test length(rand(StableRNG(5), problem)) == 1
        @test LogDensityProblems.capabilities(typeof(problem)) ==
            LogDensityProblems.LogDensityOrder{0}()

        batch_indices = [1, 3]
        batch = subsample(StableRNG(2), model, (rng, N) -> batch_indices, length(data))
        selected_batch = subsample(StableRNG(2), model, batch_indices, length(data))
        range_batch = subsample(StableRNG(2), model, 1:2, length(data))
        direct_batch = subsample(StableRNG(2), model, length(data), length(data))
        expected_batch =
            logpdf(Normal(0, 10), only(params)) +
            (length(data)//length(batch_indices)) *
            loglikelihood(Normal(only(params), 1), data[batch_indices])
        @test LogDensityProblems.logdensity(batch, params) ≈ expected_batch
        @test LogDensityProblems.logdensity(selected_batch, params) ≈ expected_batch
        @test typeof(range_batch) === typeof(selected_batch)
        @test LogDensityProblems.logdensity(direct_batch, params) ≈ expected
        @test direct_batch isa LogDensityFunction
        one_observation = subsample(StableRNG(2), model, 1, length(data))
        one_logdensity = LogDensityProblems.logdensity(one_observation, params)
        possible_logdensities = map(data) do x
            logpdf(Normal(0, 10), only(params)) +
            length(data) * logpdf(Normal(only(params), 1), x)
        end
        @test any(Base.Fix1(isapprox, one_logdensity), possible_logdensities)
        with_replacement = subsample(StableRNG(4), model, (rng, N) -> [2, 2], length(data))
        expected_replacement =
            logpdf(Normal(0, 10), only(params)) +
            2 * loglikelihood(Normal(only(params), 1), data[[2, 2]])
        @test LogDensityProblems.logdensity(with_replacement, params) ≈ expected_replacement

        gradient = ForwardDiff.gradient(
            p -> LogDensityProblems.logdensity(batch, p), params
        )
        expected_gradient = [-only(params) / 100 + 2 * sum(data[batch_indices] .- params)]
        @test gradient ≈ expected_gradient
    end

    @testset "public interface" begin
        data = [-2.0, -1.0, 0.5, 3.0]
        model = normal_location() | (x=data,)
        params = [0.25]
        expected =
            logpdf(Normal(0, 10), only(params)) +
            loglikelihood(Normal(only(params), 1), data)

        full_batch = subsample(model, length(data), length(data))
        @test LogDensityProblems.logdensity(full_batch, params) ≈ expected

        calls = Ref(0)
        received_size = Ref(0)
        resampler = function (rng, dataset_size)
            calls[] += 1
            received_size[] = dataset_size
            return [1, 3]
        end
        resampled = subsample(model, resampler, length(data))
        @test calls[] == 1
        @test received_size[] == length(data)
        expected_batch =
            logpdf(Normal(0, 10), only(params)) +
            2 * loglikelihood(Normal(only(params), 1), data[[1, 3]])
        @test LogDensityProblems.logdensity(resampled, params) ≈ expected_batch
        @test calls[] == 1

        @model function constrained_location()
            μ ~ Beta(2, 2)
            return x ~ independent_distribution(Normal(μ))
        end
        constrained_model = constrained_location() | (x=data,)
        linked = subsample(
            constrained_model,
            (rng, N) -> [1, 3],
            length(data);
            transform_strategy=LinkAll(),
        )
        @test linked.transform_strategy isa LinkAll
        @test isfinite(LogDensityProblems.logdensity(linked, [0.0]))
    end

    @testset "distribution metadata" begin
        scalar = independent_distribution(Normal())
        vector = independent_distribution(MvNormal(zeros(2), I))
        covariance = Matrix{Float64}(I, 2, 2)
        matrix = independent_distribution(MatrixNormal(zeros(2, 2), covariance, covariance))

        @test Distributions.variate_form(typeof(scalar)) === Distributions.Multivariate
        @test Distributions.variate_form(typeof(vector)) === Distributions.Matrixvariate
        @test Distributions.variate_form(typeof(matrix)) ===
            Distributions.ArrayLikeVariate{3}
        @test_throws ArgumentError size(scalar)

        data = [-1.0, 0.5, 2.0]
        product = product_distribution(Fill(Normal(), length(data)))
        @test logpdf(scalar, data) == logpdf(product, data)
        @test loglikelihood(scalar, data) == loglikelihood(product, data)
        @test insupport(scalar, data) == insupport(product, data)

        empty_data = Float64[]
        empty_product = product_distribution(Fill(Normal(), 0))
        @test logpdf(scalar, empty_data) == logpdf(empty_product, empty_data)
        @test loglikelihood(scalar, empty_data) == loglikelihood(empty_product, empty_data)
        @test insupport(scalar, empty_data) == insupport(empty_product, empty_data)
        @test logjoint(normal_location() | (x=empty_data,), (; μ=0.25)) ≈
            logpdf(Normal(0, 10), 0.25)

        indexed = independent_distribution(i -> Normal(i), 3)
        indexed_product = product_distribution([Normal(1), Normal(2), Normal(3)])
        @test size(indexed) == size(indexed_product)
        @test logpdf(indexed, data) == logpdf(indexed_product, data)
        @test loglikelihood(indexed, data) == loglikelihood(indexed_product, data)
        @test insupport(indexed, data) == insupport(indexed_product, data)
        @test rand(StableRNG(1), indexed) == rand(StableRNG(1), indexed_product)

        calls = Ref(0)
        large_indexed = independent_distribution(i -> (calls[] += 1; Normal(i)), 10_000)
        calls[] = 0
        @test size(large_indexed) == (10_000,)
        @test calls[] == 0
        indexed_vector = independent_distribution(i -> MvNormal(fill(i, 2), I), 3)
        @test size(indexed_vector) == (2, 3)

        @model function indexed_latent()
            return x ~ independent_distribution(i -> Normal(i), 3)
        end
        @model function product_latent()
            return x ~ product_distribution([Normal(i) for i in 1:3])
        end
        indexed_ldf = LogDensityFunction(indexed_latent(), getlogjoint_internal, LinkAll())
        product_ldf = LogDensityFunction(product_latent(), getlogjoint_internal, LinkAll())
        @test LogDensityProblems.dimension(indexed_ldf) ==
            LogDensityProblems.dimension(product_ldf)
        @test LogDensityProblems.logdensity(indexed_ldf, zeros(3)) ==
            LogDensityProblems.logdensity(product_ldf, zeros(3))
    end

    @testset "sampling cost depends on batch size" begin
        rng = CountingRNG(StableRNG(1), 0)
        data = zeros(1_000_000)
        problem = subsample(rng, normal_location() | (x=data,), 3, length(data))
        @test problem isa LogDensityFunction
        @test rng.draws == 3
    end

    @testset "full-data validation is constant-time" begin
        visits = Ref(0)
        @model function scans_observation(visits)
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            for _ in eachindex(x)
                visits[] += 1
            end
        end

        model = scans_observation(visits) | (x=zeros(1_000_000),)
        @test_throws ArgumentError subsample(StableRNG(1), model, 3, 1_000_000)
        @test visits[] == 0
    end

    @testset "observation-specific distributions" begin
        @model function logistic_regression(X)
            β ~ MvNormal(zeros(size(X, 2)), I)
            return y ~ independent_distribution(
                i -> BernoulliLogit(sum(@view(X[i, :]) .* β)), size(X, 1)
            )
        end

        X = [-1.0 0.5; 0.0 1.0; 1.0 -0.5; 2.0 1.0]
        observations = [false, true, true, false]
        model = logistic_regression(X) | (y=observations,)
        indices = [1, 3]
        problem = subsample(StableRNG(1), model, indices, length(observations))
        params = [0.2, -0.1]
        expected =
            logpdf(MvNormal(zeros(2), I), params) +
            2 * sum(
                i -> logpdf(BernoulliLogit(sum(@view(X[i, :]) .* params)), observations[i]),
                indices,
            )
        @test LogDensityProblems.logdensity(problem, params) ≈ expected
        @test all(
            isfinite,
            ForwardDiff.gradient(p -> LogDensityProblems.logdensity(problem, p), params),
        )
    end

    @testset "validation does not evaluate densities" begin
        evaluations = Ref(0)
        @model function counted_model(evaluations)
            μ ~ Normal()
            return x ~ independent_distribution(CountingDistribution(evaluations))
        end

        data = collect(1.0:4.0)
        model = counted_model(evaluations) | (x=data,)
        batch = subsample(StableRNG(1), model, (rng, N) -> [1, 3], length(data))
        @test evaluations[] == 0
        LogDensityProblems.logdensity(batch, [0.0])
        @test evaluations[] == 2
    end

    @testset "data ownership" begin
        data = [-2.0, -1.0, 0.5, 3.0]
        original = copy(data)
        problem = independent_problem(data)
        data .= 100
        params = [0.25]
        @test LogDensityProblems.logdensity(problem, params) ≈
            logpdf(Normal(0, 10), only(params)) +
              loglikelihood(Normal(only(params), 1), original)
    end

    @testset "multivariate observations" begin
        @model function multivariate_location()
            μ ~ MvNormal(zeros(2), I)
            return x ~ independent_distribution(MvNormal(μ, I))
        end
        data = [-1.0 0.0 2.0; 0.5 1.0 -2.0]
        model = condition(multivariate_location(), @varname(x) => data)
        problem = independent_problem(model, 3)
        params = [0.25, -0.5]
        batch = subsample(StableRNG(2), model, (rng, N) -> [1, 3], 3)
        @test LogDensityProblems.logdensity(problem, params) ≈
            logpdf(MvNormal(zeros(2), I), params) +
              loglikelihood(MvNormal(params, I), data)
        @test LogDensityProblems.logdensity(batch, params) ≈
            logpdf(MvNormal(zeros(2), I), params) +
              (3//2) * loglikelihood(MvNormal(params, I), data[:, [1, 3]])

        @model function matrix_observation()
            μ ~ Normal()
            covariance = Matrix{typeof(μ)}(I, 2, 2)
            return x ~ independent_distribution(
                MatrixNormal(fill(μ, 2, 2), covariance, covariance)
            )
        end
        matrix_data = reshape(collect(1.0:12.0), 2, 2, 3)
        matrix_model = condition(matrix_observation(), @varname(x) => matrix_data)
        matrix_batch = subsample(StableRNG(2), matrix_model, (rng, N) -> [1, 3], 3)
        matrix_params = [0.25]
        covariance = Matrix{eltype(matrix_params)}(I, 2, 2)
        matrix_dist = MatrixNormal(fill(only(matrix_params), 2, 2), covariance, covariance)
        @test LogDensityProblems.logdensity(matrix_batch, matrix_params) ≈
            logpdf(Normal(), only(matrix_params)) +
              (3//2) * loglikelihood(matrix_dist, matrix_data[:, :, [1, 3]])
    end

    @testset "conditioning constraints" begin
        data = [0.0, 1.0]
        @test_throws ArgumentError independent_problem(normal_location(), 2)

        @model function argument_observation(x)
            μ ~ Normal()
            return x ~ independent_distribution(Normal(μ))
        end
        argument_model = condition(argument_observation(data), @varname(x) => data)
        @test_throws ArgumentError independent_problem(argument_model, 2)

        model = condition(normal_location(), @varname(x) => data, @varname(z) => 1.0)
        @test_throws ArgumentError independent_problem(model, 2)

        partial = @vnt begin
            @template x = zeros(2)
            x[1] := 0.0
        end
        model = condition(normal_location(), partial)
        @test_throws ArgumentError independent_problem(model, 2)
    end

    @testset "evaluation constraints" begin
        data = [0.0, 1.0]

        @model function ordinary_observation()
            μ ~ Normal()
            return x ~ Normal(μ)
        end
        model = condition(ordinary_observation(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function observed_before(y)
            y ~ Normal()
            μ ~ Normal()
            return x ~ independent_distribution(Normal(μ))
        end
        model = condition(observed_before(0.0), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function observed_after(y)
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return y ~ Normal(μ)
        end
        model = condition(observed_after(0.0), @varname(x) => data)
        μ = 0.25
        @test logjoint(model, (; μ)) ≈
            logpdf(Normal(), μ) + loglikelihood(Normal(μ), data) + logpdf(Normal(μ), 0.0)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function latent_after()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return z ~ Normal(μ)
        end
        model = condition(latent_after(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function explicit_likelihood()
            μ ~ Normal()
            @addlogprob! (; loglikelihood=-1.0)
            return x ~ independent_distribution(Normal(μ))
        end
        model = condition(explicit_likelihood(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function explicit_prior()
            μ ~ Normal()
            @addlogprob! (; logprior=-1.0)
            return x ~ independent_distribution(Normal(μ))
        end
        model = condition(explicit_prior(), @varname(x) => data)
        prior_problem = independent_problem(model, 2)
        @test LogDensityProblems.logdensity(prior_problem, [0.0]) ≈
            logpdf(Normal(), 0.0) - 1 + loglikelihood(Normal(), data)

        @model function explicit_prior_after()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            @addlogprob! (; logprior=-1.0)
        end
        model = condition(explicit_prior_after(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function explicit_logjac_after()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return __varinfo__ = acclogjac!!(
                __varinfo__, -1.0; ignore_missing_accumulator=true
            )
        end
        model = condition(explicit_logjac_after(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function replace_prior()
            μ ~ truncated(Normal(); lower=0)
            x ~ independent_distribution(Normal(μ))
            if μ < 0
                return __varinfo__ = setlogprior!!(__varinfo__, -1.0)
            end
        end
        model = condition(replace_prior(), @varname(x) => data)
        problem = independent_problem(model, 2)
        @test_throws ArgumentError LogDensityProblems.logdensity(problem, [-1.0])

        @model function replace_logjac()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return __varinfo__ = setlogjac!!(__varinfo__, -1.0)
        end
        model = condition(replace_logjac(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function replace_likelihood()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return __varinfo__ = setloglikelihood!!(__varinfo__, -1.0)
        end
        model = condition(replace_likelihood(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function no_observation()
            return μ ~ Normal()
        end
        model = condition(no_observation(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)

        @model function repeated_observation()
            μ ~ Normal()
            x ~ independent_distribution(Normal(μ))
            return x ~ independent_distribution(Normal(μ))
        end
        model = condition(repeated_observation(), @varname(x) => data)
        @test_throws ArgumentError independent_problem(model, 2)
    end

    @testset "data and batch validation" begin
        @test_throws ArgumentError VarInfo(normal_location())
        @test_throws DimensionMismatch independent_problem(zeros(2, 2))
        @test_throws ArgumentError independent_problem(Float64[])

        model = condition(normal_location(), @varname(x) => [0.0, 1.0])
        @test_throws ArgumentError subsample(StableRNG(1), model, 0, 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, true, 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, 1, false)
        @test_throws ArgumentError subsample(StableRNG(1), model, 3, 2)
        @test_throws DimensionMismatch subsample(StableRNG(1), model, 1, 3)
        @test_throws ArgumentError subsample(StableRNG(1), model, (rng, N) -> [1.0], 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, (rng, N) -> Int[], 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, (rng, N) -> Bool[true], 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, (rng, N) -> [0], 2)
        @test_throws ArgumentError subsample(StableRNG(1), model, (rng, N) -> [3], 2)

        @model function wrong_event_shape()
            μ ~ MvNormal(zeros(2), I)
            return x ~ independent_distribution(MvNormal(μ, I))
        end
        model = condition(wrong_event_shape(), @varname(x) => zeros(3, 2))
        @test_throws DimensionMismatch independent_problem(model, 2)

        storage = [0.0, 1.0]
        view_model = normal_location() | (x=@view(storage[:]),)
        view_batch = subsample(StableRNG(1), view_model, 1, 2)
        @test isfinite(LogDensityProblems.logdensity(view_batch, [0.0]))
    end

    @testset "latent layout constraints" begin
        @model scalar_latent() = a ~ Normal()
        @model renamed_latent() = b ~ Normal()
        scalar = LogDensityFunction(scalar_latent(), getlogjoint_internal, UnlinkAll())
        renamed = LogDensityFunction(renamed_latent(), getlogjoint_internal, UnlinkAll())
        @test_throws DimensionMismatch DynamicPPL._check_independent_layout(scalar, renamed)

        @model vector_latent() = a ~ MvNormal(zeros(2), I)
        vector = LogDensityFunction(vector_latent(), getlogjoint_internal, UnlinkAll())
        @test_throws DimensionMismatch DynamicPPL._check_independent_layout(scalar, vector)

        @model constrained_latent() = a ~ Beta(2, 2)
        linked = LogDensityFunction(constrained_latent(), getlogjoint_internal, LinkAll())
        unlinked = LogDensityFunction(
            constrained_latent(), getlogjoint_internal, UnlinkAll()
        )
        @test_throws ArgumentError DynamicPPL._check_independent_layout(linked, unlinked)

        ranges = get_all_ranges_and_transforms(scalar)
        wrong_dimension = LogDensityFunction(
            scalar_latent(), getlogjoint_internal, ranges, zeros(2)
        )
        @test_throws DimensionMismatch DynamicPPL._check_independent_layout(
            scalar, wrong_dimension
        )
        wrong_vector_type = LogDensityFunction(
            scalar_latent(), getlogjoint_internal, ranges, zeros(Float32, 1)
        )
        @test_throws ArgumentError DynamicPPL._check_independent_layout(
            scalar, wrong_vector_type
        )

        @model function parameter_dependent_layout()
            z ~ truncated(Normal(); lower=0)
            if z > 0
                y ~ Normal()
            end
            return x ~ independent_distribution(Normal(z))
        end
        dynamic = subsample(parameter_dependent_layout() | (x=[0.0, 1.0],), [1], 2)
        @test_throws DimensionMismatch LogDensityProblems.logdensity(dynamic, [-1.0, 0.0])
    end

    @testset "nested conditioning contexts" begin
        data = zeros(100_000)
        model = prefix(condition(normal_location(); x=data), :a)
        problem = subsample(model, [1, 3], length(data))
        @test Base.summarysize(problem) < Base.summarysize(data) ÷ 2
        @test LogDensityProblems.logdensity(problem, [0.0]) ≈
            logpdf(Normal(0, 10), 0.0) +
              (length(data)//2) * loglikelihood(Normal(), data[[1, 3]])
    end

    @testset "thread-safe evaluation" begin
        model = setthreadsafe(condition(normal_location(), @varname(x) => [0.0, 1.0]), true)
        @test_throws ArgumentError independent_problem(model, 2)
    end
end

end
