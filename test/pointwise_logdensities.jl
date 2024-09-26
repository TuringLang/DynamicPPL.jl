@testset "logdensities_likelihoods.jl" begin
    mod_ctx = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.2)
    mod_ctx2 = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.4, mod_ctx)
    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(model)
        for vn in DynamicPPL.TestUtils.varnames(model)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(
            model, example_values...
        )
        logp_true = logprior(model, vi)

        # Compute the pointwise loglikelihoods.
        lls = pointwise_loglikelihoods(model, vi)
        if isempty(lls)
            # One of the models with literal observations, so we'll set this to 0 for subsequent comparisons.
            loglikelihood_true = 0.0
        else
            @test [:x] ==  unique(DynamicPPL.getsym.(keys(lls)))
            loglikelihood_sum = sum(sum, values(lls))
            @test loglikelihood_sum ≈ loglikelihood_true
        end

        # Compute the pointwise logdensities of the priors.
        lps_prior = pointwise_prior_logdensities(model, vi)
        @test :x ∉ DynamicPPL.getsym.(keys(lps_prior))
        logp = sum(sum, values(lps_prior))
        @test logp ≈ logp_true

        # Compute both likelihood and logdensity of prior
        # using the default DefaultContex        
        lps = pointwise_logdensities(model, vi)
        logp = sum(sum, values(lps))
        @test logp ≈ (logp_true + loglikelihood_true)

        # Test that modifications of Setup are picked up
        lps = pointwise_logdensities(model, vi, mod_ctx2)
        logp = sum(sum, values(lps))
        @test logp ≈ (logp_true + loglikelihood_true) * 1.2 * 1.4
    end
end

@testset "pointwise_logdensities chain" begin
    # We'll just test one, since `pointwise_logdensities(::Model, ::AbstractVarInfo)` is tested extensively,
    # and this is what is used to implement `pointwise_logdensities(::Model, ::Chains)`. This test suite is just
    # to ensure that we don't accidentally break the the version on `Chains`.
    model = DynamicPPL.TestUtils.demo_dot_assume_dot_observe()
    # FIXME(torfjelde): Make use of `varname_and_value_leaves` once we've introduced
    # an impl of this for containers.
    vns = DynamicPPL.TestUtils.varnames(model)
    # Get some random `NamedTuple` samples from the prior.
    vals = [DynamicPPL.TestUtils.rand_prior_true(model) for _ = 1:5]
    # Concatenate the vector representations and create a `Chains` from it.
    vals_arr = reduce(hcat, (mapreduce(DynamicPPL.tovec, vcat, values(nt) for nt in vals))
    chain = Chains(permutedims(vals_arr), map(Symbol, vns))
    logjoints_pointwise = pointwise_logdensities(model, chain)
    # Get the sum of the logjoints for each of the iterations.
    logjoints = [
        sum(logjoints_pointwise[vn][idx] for vn in vns)
        for idx = 1:5
    ]
    for (val, logp) in zip(vals, logjoints)
        # Compare true logjoint with the one obtained from `pointwise_logdensities`.
        logjoint_true = DynamicPPL.TestUtils.logjoint_true(model, val...)
        @test logp ≈ logjoint_true
    end
end;
