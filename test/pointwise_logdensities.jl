@testset "logdensities_likelihoods.jl" begin
    mod_ctx = TU.TestLogModifyingChildContext(1.2)
    mod_ctx2 = TU.TestLogModifyingChildContext(1.4, mod_ctx)
    @testset "$(model.f)" for model in TU.DEMO_MODELS
        example_values = TU.rand_prior_true(model)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(model)
        for vn in TU.varnames(model)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        loglikelihood_true = TU.loglikelihood_true(
            model, example_values...
        )
        logp_true = logprior(model, vi)

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
    model = TU.demo_dot_assume_dot_observe()
    # FIXME(torfjelde): Make use of `varname_and_value_leaves` once we've introduced
    # an impl of this for containers.
    # NOTE(torfjelde): This only returns the varnames of the _random_ variables, i.e. excl. observed.
    vns = TU.varnames(model)
    # Get some random `NamedTuple` samples from the prior.
    num_iters = 3
    vals = [TU.rand_prior_true(model) for _ in 1:num_iters]
    # Concatenate the vector representations and create a `Chains` from it.
    vals_arr = reduce(hcat, mapreduce(DynamicPPL.tovec, vcat, values(nt)) for nt in vals)
    chain = Chains(permutedims(vals_arr), map(Symbol, vns))

    # Compute the different pointwise logdensities.
    logjoints_pointwise = pointwise_logdensities(model, chain)
    logpriors_pointwise = pointwise_prior_logdensities(model, chain)
    loglikelihoods_pointwise = pointwise_loglikelihoods(model, chain)

    # Check that they contain the correct variables.
    @test all(string(vn) in keys(logjoints_pointwise) for vn in vns)
    @test all(string(vn) in keys(logpriors_pointwise) for vn in vns)
    @test !any(Base.Fix2(startswith, "x"), keys(logpriors_pointwise))
    @test !any(string(vn) in keys(loglikelihoods_pointwise) for vn in vns)
    @test all(Base.Fix2(startswith, "x"), keys(loglikelihoods_pointwise))

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
        sum(loglikelihoods_pointwise[vn][idx] for vn in keys(loglikelihoods_pointwise)) for
        idx in 1:num_iters
    ]

    for (val, logjoint, logprior, loglikelihood) in
        zip(vals, logjoints, logpriors, loglikelihoods)
        # Compare true logjoint with the one obtained from `pointwise_logdensities`.
        logjoint_true = TU.logjoint_true(model, val...)
        logprior_true = TU.logprior_true(model, val...)
        loglikelihood_true = TU.loglikelihood_true(model, val...)

        @test logjoint ≈ logjoint_true
        @test logprior ≈ logprior_true
        @test loglikelihood ≈ loglikelihood_true
    end
end
