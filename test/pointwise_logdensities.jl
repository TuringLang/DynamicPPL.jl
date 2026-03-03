module DynamicPPLPointwiseLogDensitiesTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL
using Distributions: Normal, Exponential
using DynamicPPL
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

@testset "pointwise_logdensities.jl" begin
    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)

        # Instantiate a `VarInfo` with the example values.
        vi = last(init!!(model, VarInfo(), InitFromParams(example_values), UnlinkAll()))

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(
            model, example_values...
        )
        logprior_true = logprior(model, vi)

        # Compute the pointwise loglikelihoods.
        lls = pointwise_loglikelihoods(model, vi)
        if isempty(lls)
            # One of the models with literal observations, so we'll set this to 0 for subsequent comparisons.
            loglikelihood_true = 0.0
        else
            @test [:x] == unique(DynamicPPL.getsym.(keys(lls)))
            loglikelihood_sum = sum(sum, values(lls))
            @test loglikelihood_sum ≈ loglikelihood_true
        end

        # Compute the pointwise logdensities of the priors.
        lps_prior = pointwise_prior_logdensities(model, vi)
        @test :x ∉ DynamicPPL.getsym.(keys(lps_prior))
        logp = sum(sum, values(lps_prior))
        @test logp ≈ logprior_true

        # Compute both likelihood and logdensity of prior
        # using the default DefaultContext
        lps = pointwise_logdensities(model, vi)
        logp = sum(sum, values(lps))
        @test logp ≈ (logprior_true + loglikelihood_true)
    end
end

@testset "pointwise_logdensities chain" begin
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
            hcat([
                DynamicPPL.ParamsWithStats(VarInfo(model_m_only), model_m_only) for
                _ in 1:50
            ]),
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
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
