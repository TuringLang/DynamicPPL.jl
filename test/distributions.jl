module DynamicPPLDistributionsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions:
    Distributions, Binomial, Normal, Poisson, logpdf, loglikelihood, pdf, sampler, support
using Bijectors: Bijectors
using ForwardDiff: ForwardDiff
using LogExpFunctions: logistic
using LogDensityProblems: LogDensityProblems
using Random: Random
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_throws, @testset

@testset "distributions.jl" begin
    @testset "Flat" begin
        d = Flat()
        @test minimum(d) == -Inf
        @test maximum(d) == Inf
        @test @inferred(logpdf(d, 0.0)) === 0.0
        @test logpdf(d, 1.0f6) === 0.0f0
        @test logpdf(d, [1.0, 2.0]) == [0.0, 0.0]
        @test loglikelihood(d, [1.0, -1.0]) === 0.0
        x = rand(StableRNG(1), d)
        @test x isa Float64
        @test rand(StableRNG(1), d) == x
    end

    @testset "FlatPos" begin
        d = FlatPos(1.0)
        @test minimum(d) == 1.0
        @test maximum(d) == Inf
        @test @inferred(logpdf(d, 2.0)) === 0.0
        @test logpdf(d, 1.0) === -Inf
        @test logpdf(d, 0.5) === -Inf
        @test logpdf(FlatPos(1.0f0), 2.0f0) === 0.0f0
        @test loglikelihood(d, [2.0, 3.0]) === 0.0
        @test loglikelihood(d, [2.0, 0.5]) === -Inf
        @test rand(StableRNG(1), d) > 1.0
        # The generic Bijectors fallback picks the transform up from the support.
        b = Bijectors.bijector(d)
        @test b(2.0) isa Real
        @test Bijectors.inverse(b)(b(2.0)) ≈ 2.0
    end

    @testset "BinomialLogit" begin
        d = BinomialLogit(10, -0.3)
        @test minimum(d) == 0
        @test maximum(d) == 10
        @test_throws ErrorException BinomialLogit(-1, 0.5)

        # Matches Binomial with the same success probability.
        for logitp in (-5.0, -0.3, 0.0, 2.7), k in -1:11
            reference = logpdf(Binomial(10, logistic(logitp)), k)
            @test logpdf(BinomialLogit(10, logitp), k) ≈ reference
        end
        # Stays finite for extreme logits, which is the reason it exists.
        @test isfinite(logpdf(BinomialLogit(10, 40.0), 10))
        @test isfinite(logpdf(BinomialLogit(10, -40.0), 0))
        @test logpdf(d, -1) === -Inf
        @test logpdf(d, 11) === -Inf
        @test @inferred(logpdf(d, 3)) isa Float64

        @test rand(StableRNG(2), d) in 0:10
        @test rand(StableRNG(2), sampler(d)) in 0:10
        n = 100_000
        xs = rand(StableRNG(3), d, n)
        @test sum(xs) / n ≈ 10 * logistic(-0.3) rtol = 0.05

        g = ForwardDiff.derivative(lp -> logpdf(BinomialLogit(10, lp), 3), -0.3)
        h = 1e-6
        fd =
            (
                logpdf(BinomialLogit(10, -0.3 + h), 3) -
                logpdf(BinomialLogit(10, -0.3 - h), 3)
            ) / 2h
        @test g ≈ fd rtol = 1e-6
    end

    @testset "OrderedLogistic" begin
        d = OrderedLogistic(-2.0, [-1.0, 1.0])
        @test minimum(d) == 1
        @test maximum(d) == 3
        @test support(d) == 1:3
        @test_throws ErrorException OrderedLogistic(0.0, [1.0, -1.0])

        # The mass function sums to one and matches the logistic differences.
        for η in (-2.0, 0.0, 3.5), cutpoints in ([-1.0, 1.0], [0.0, 0.5, 1.5, 4.0])
            dist = OrderedLogistic(η, cutpoints)
            K = length(cutpoints) + 1
            ps = [pdf(dist, k) for k in 1:K]
            @test sum(ps) ≈ 1.0
            manual = [
                1 - logistic(η - cutpoints[1])
                [
                    logistic(η - cutpoints[k - 1]) - logistic(η - cutpoints[k]) for
                    k in 2:(K - 1)
                ]
                logistic(η - cutpoints[K - 1])
            ]
            @test ps ≈ manual
        end
        @test logpdf(d, 0) === -Inf
        @test logpdf(d, 4) === -Inf
        @test @inferred(logpdf(d, 2)) isa Float64

        n = 100_000
        xs = rand(StableRNG(4), d, n)
        for k in 1:3
            @test sum(==(k), xs) / n ≈ pdf(d, k) atol = 0.01
        end
        @test rand(StableRNG(4), sampler(d)) in 1:3

        g = ForwardDiff.derivative(η -> logpdf(OrderedLogistic(η, [-1.0, 1.0]), 2), -2.0)
        h = 1e-6
        fd =
            (
                logpdf(OrderedLogistic(-2.0 + h, [-1.0, 1.0]), 2) -
                logpdf(OrderedLogistic(-2.0 - h, [-1.0, 1.0]), 2)
            ) / 2h
        @test g ≈ fd rtol = 1e-6
        gc = ForwardDiff.gradient(c -> logpdf(OrderedLogistic(-2.0, c), 2), [-1.0, 1.0])
        @test all(isfinite, gc)
    end

    @testset "LogPoisson" begin
        d = LogPoisson(log(2.5))
        @test minimum(d) == 0
        @test maximum(d) == Inf

        for logλ in (-3.0, 0.0, 1.7), k in 0:10
            @test logpdf(LogPoisson(logλ), k) ≈ logpdf(Poisson(exp(logλ)), k)
        end
        @test logpdf(d, -1) === -Inf
        @test @inferred(logpdf(d, 3)) isa Float64

        n = 100_000
        xs = rand(StableRNG(5), d, n)
        @test sum(xs) / n ≈ 2.5 rtol = 0.05
        @test rand(StableRNG(5), sampler(d)) isa Int

        g = ForwardDiff.derivative(l -> logpdf(LogPoisson(l), 3), log(2.5))
        h = 1e-6
        fd =
            (logpdf(LogPoisson(log(2.5) + h), 3) - logpdf(LogPoisson(log(2.5) - h), 3)) / 2h
        @test g ≈ fd rtol = 1e-6
    end

    @testset "use in models" begin
        # Flat priors link through the generic bijector fallback.
        @model function flat_model(y)
            m ~ Flat()
            s ~ FlatPos(0.0)
            return y ~ Normal(m, s)
        end
        model = flat_model(0.5)
        vi = last(
            DynamicPPL.init!!(StableRNG(6), model, VarInfo(), InitFromPrior(), LinkAll())
        )
        @test isfinite(DynamicPPL.getlogjoint(vi))
        @test isfinite(DynamicPPL.getlogjac(vi))

        # The discrete distributions work as likelihoods, with gradients.
        @model function discrete_obs(k_binom, k_ord, k_pois)
            η ~ Normal()
            k_binom ~ BinomialLogit(10, η)
            k_ord ~ OrderedLogistic(η, [-1.0, 1.0])
            return k_pois ~ LogPoisson(η)
        end
        model = discrete_obs(3, 2, 4)
        vi = last(
            DynamicPPL.init!!(
                StableRNG(7),
                model,
                VarInfo(VectorValueAccumulator()),
                InitFromPrior(),
                LinkAll(),
            ),
        )
        ldf = DynamicPPL.LogDensityFunction(
            model,
            DynamicPPL.getlogjoint_internal,
            vi;
            adtype=DynamicPPL.ADTypes.AutoForwardDiff(),
        )
        lp, grad = LogDensityProblems.logdensity_and_gradient(ldf, [0.2])
        @test isfinite(lp)
        @test length(grad) == 1 && isfinite(only(grad))
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
