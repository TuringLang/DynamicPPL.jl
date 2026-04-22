module DynamicPPLPointwiseLogDensitiesTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL
using Distributions
using DynamicPPL
using LinearAlgebra
using MCMCChains: MCMCChains
using Random: Random
using Test

function make_chain_from_prior(rng::Random.AbstractRNG, model::Model, n_iters::Int)
    vi = DynamicPPL.OnlyAccsVarInfo((
        DynamicPPL.default_accumulators()..., DynamicPPL.RawValueAccumulator(false)
    ))
    ps = hcat([
        DynamicPPL.ParamsWithStats(
            last(DynamicPPL.init!!(rng, model, vi, InitFromPrior(), UnlinkAll()))
        ) for _ in 1:n_iters
    ])
    return AbstractMCMC.from_samples(MCMCChains.Chains, ps)
end
function make_chain_from_prior(model::Model, n_iters::Int)
    return make_chain_from_prior(Random.default_rng(), model, n_iters)
end

@testset "pointwise_logdensities with values" begin
    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)

        # Instantiate a `VarInfo` with the example values.
        vi = last(init!!(model, VarInfo(), InitFromParams(example_values), UnlinkAll()))

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(
            model, example_values...
        )
        logprior_true = logprior(model, vi)

        # Compute the pointwise loglikelihoods.
        lls = pointwise_loglikelihoods(model, InitFromParams(vi.values))
        if isempty(lls)
            # One of the models with literal observations, so we'll set this to 0 for subsequent comparisons.
            loglikelihood_true = 0.0
        else
            @test [:x] == unique(DynamicPPL.getsym.(keys(lls)))
            loglikelihood_sum = sum(sum, values(lls))
            @test loglikelihood_sum ≈ loglikelihood_true
        end

        # Compute the pointwise logdensities of the priors.
        lps_prior = pointwise_prior_logdensities(model, InitFromParams(vi.values))
        @test :x ∉ DynamicPPL.getsym.(keys(lps_prior))
        logp = sum(sum, values(lps_prior))
        @test logp ≈ logprior_true

        # Compute both likelihood and logdensity of prior
        # using the default DefaultContext
        lps = pointwise_logdensities(model, InitFromParams(vi.values))
        logp = sum(sum, values(lps))
        @test logp ≈ (logprior_true + loglikelihood_true)
    end

    @testset "with factorize=true" begin
        @model function f(z)
            x ~ MvNormal(zeros(2), I)
            y ~ product_distribution((; a=Normal(), b=Normal()))
            return z ~ MvNormal(x, I)
        end
        model = f(randn(2))

        logdensities = pointwise_logdensities(model, InitFromPrior(); factorize=true)
        @test logdensities isa DynamicPPL.VarNamedTuple
        @test logdensities[@varname(x)] isa Vector{Float64}
        @test length(logdensities[@varname(x)]) == 2
        @test logdensities[@varname(z)] isa Vector{Float64}
        @test length(logdensities[@varname(z)]) == 2

        log_prior_densities = pointwise_prior_logdensities(
            model, InitFromPrior(); factorize=true
        )
        @test logdensities isa DynamicPPL.VarNamedTuple
        @test log_prior_densities[@varname(x)] isa Vector{Float64}
        @test length(log_prior_densities[@varname(x)]) == 2
        @test !haskey(log_prior_densities, @varname(z))

        log_likelihoods = pointwise_loglikelihoods(model, InitFromPrior(); factorize=true)
        @test log_likelihoods isa DynamicPPL.VarNamedTuple
        @test log_likelihoods[@varname(z)] isa Vector{Float64}
        @test length(log_likelihoods[@varname(z)]) == 2
        @test !haskey(log_likelihoods, @varname(x))
    end
end

@testset "pointwise_logdensities with chain" begin
    @testset "correctness" begin
        model = DynamicPPL.TestUtils.demo_assume_index_observe()
        vns = DynamicPPL.TestUtils.varnames(model)
        num_iters = 10
        chain = make_chain_from_prior(model, num_iters)

        # Compute the different pointwise logdensities.
        logjoints_pointwise = pointwise_logdensities(model, chain)
        logpriors_pointwise = pointwise_prior_logdensities(model, chain)
        loglikelihoods_pointwise = pointwise_loglikelihoods(model, chain)

        # Check output type
        @test logjoints_pointwise isa MCMCChains.Chains
        @test logpriors_pointwise isa MCMCChains.Chains
        @test loglikelihoods_pointwise isa MCMCChains.Chains

        # Check that they contain the correct variables.
        @test all(Symbol(vn) in keys(logjoints_pointwise) for vn in vns)
        @test all(Symbol(vn) in keys(logpriors_pointwise) for vn in vns)
        @test !any(Base.Fix1(startswith, "x"), String.(keys(logpriors_pointwise)))
        @test !any(Symbol(vn) in keys(loglikelihoods_pointwise) for vn in vns)
        @test all(Base.Fix1(startswith, "x"), String.(keys(loglikelihoods_pointwise)))

        # Get the sum of the logjoints for each of the iterations.
        logjoints = [
            sum(logjoints_pointwise[vn][idx] for vn in keys(logjoints_pointwise)) for
            idx in 1:num_iters
        ]
        logpriors = [
            sum(logpriors_pointwise[vn][idx] for vn in keys(logpriors_pointwise)) for
            idx in 1:num_iters
        ]
        loglikelihoods = [
            sum(loglikelihoods_pointwise[vn][idx] for vn in keys(loglikelihoods_pointwise))
            for idx in 1:num_iters
        ]

        expected_logjoints = logjoint(model, chain)
        expected_logpriors = logprior(model, chain)
        expected_loglikelihoods = loglikelihood(model, chain)

        @test logjoints ≈ expected_logjoints
        @test logpriors ≈ expected_logpriors
        @test loglikelihoods ≈ expected_loglikelihoods
    end

    @testset "errors when variables are missing" begin
        # Create a chain that only has `m`.
        @model function m_only()
            return m ~ Normal()
        end
        model_m_only = m_only()
        chain_m_only = AbstractMCMC.from_samples(
            MCMCChains.Chains,
            hcat([DynamicPPL.ParamsWithStats(InitFromPrior(), model_m_only) for _ in 1:50]),
        )

        # Define a model that needs both `m` and `s`.
        @model function f()
            m ~ Normal()
            s ~ Exponential()
            return y ~ Normal(m, s)
        end
        model = f() | (; y=1.0)
        @test_throws "No value was provided" pointwise_logdensities(model, chain_m_only)
        @test_throws "No value was provided" pointwise_loglikelihoods(model, chain_m_only)
        @test_throws "No value was provided" pointwise_prior_logdensities(
            model, chain_m_only
        )
    end

    @testset "with factorize=true" begin
        @model function f(z)
            x ~ MvNormal(zeros(2), I)
            # NOTE: This product_distribution doesn't work with MCMCChains because in the
            # chain the elements of `y` are stored as `y.a` and `y.b`. When we call
            # pointwise_logdensities(model, chn), DynamicPPL has to reconstruct the values
            # and it doesn't realise that a and b should be part of the same distribution.
            # Thus it recreates a VNT that looks like
            #
            # VarNamedTuple
            # ├─ x => [-0.37054531061901747, -0.3947539032928444]
            # └─ y => VarNamedTuple
            #         ├─ a => 0.015729054630598562
            #         └─ b => -2.152129714223151
            #
            # rather than
            #
            # VarNamedTuple
            # ├─ x => [-0.14269319184188936, -1.6481452899699949]
            # └─ y => (a = -0.5455556873253069, b = -1.0051579479025197)
            #
            # This is an inherent failure of MCMCChains and not something that we can fix
            # here.
            # y ~ product_distribution((; a=Normal(), b=Normal()))
            return z ~ MvNormal(x, I)
        end
        model = f(randn(2))
        chn = make_chain_from_prior(model, 10)
        logdensities = pointwise_logdensities(model, chn; factorize=true)
        for k in [Symbol("x[1]"), Symbol("x[2]"), Symbol("z[1]"), Symbol("z[2]")]
            @test k in keys(logdensities)
        end
        log_prior_densities = pointwise_prior_logdensities(model, chn; factorize=true)
        for k in [Symbol("x[1]"), Symbol("x[2]")]
            @test k in keys(logdensities)
        end
        log_likelihoods = pointwise_loglikelihoods(model, chn; factorize=true)
        for k in [Symbol("z[1]"), Symbol("z[2]")]
            @test k in keys(logdensities)
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
