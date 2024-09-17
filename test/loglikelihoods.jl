@testset "loglikelihoods.jl" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        example_values = DynamicPPL.TestUtils.rand_prior_true(m)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(m)
        for vn in DynamicPPL.TestUtils.varnames(m)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        # Compute the pointwise loglikelihoods.
        lls = pointwise_loglikelihoods(m, vi)

        if isempty(lls)
            # One of the models with literal observations, so we just skip.
            continue
        end

        loglikelihood = sum(sum, values(lls))
        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(m, example_values...)

        @test loglikelihood ≈ loglikelihood_true
    end
end

@testset "logpriors_var.jl" begin
    mod_ctx = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.2, PriorContext())
    mod_ctx2 = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.4, mod_ctx)
    #m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    # m = DynamicPPL.TestUtils.demo_assume_index_observe() # logp at i-level?
    @testset "$(m.f)" for (i, m) in enumerate(DynamicPPL.TestUtils.DEMO_MODELS)
        #@show i
        example_values = DynamicPPL.TestUtils.rand_prior_true(m)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(m)
        () -> begin
            for vn in DynamicPPL.TestUtils.varnames(m)
                global vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
            end
        end
        for vn in DynamicPPL.TestUtils.varnames(m)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        #chains = sample(m, SampleFromPrior(), 2; progress=false)

        # Compute the pointwise loglikelihoods.
        logpriors = DynamicPPL.varwise_logpriors(m, vi)
        logp1 = getlogp(vi)
        logp = logprior(m, vi)
        @test !isfinite(logp) || sum(x -> sum(x), values(logpriors)) ≈ logp
        #
        # test on modifying child-context
        logpriors_mod = DynamicPPL.varwise_logpriors(m, vi, mod_ctx2)
        logp1 = getlogp(vi)
        # Following line assumes no Likelihood contributions 
        #   requires lowest Context to be PriorContext
        @test !isfinite(logp1) || sum(x -> sum(x), values(logpriors_mod)) ≈ logp1 #
        @test all(values(logpriors_mod) .≈ values(logpriors) .* 1.2 .* 1.4)
    end
end;

@testset "logpriors_var chain" begin
    @model function demo(xs, y)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(xs)
            xs[i] ~ Normal(m, √s)
        end
        y ~ Normal(m, √s)
    end
    xs_true, y_true = ([0.3290767977680923, 0.038972110187911684, -0.5797496780649221], -0.7321425592768186)#randn(3), randn()
    model = demo(xs_true, y_true)
    () -> begin
        # generate the sample used below
        chain = sample(model, MH(), 10)
        arr0 = Array(chain)
    end
    arr0 = [1.8585322626573435 -0.05900855284939967; 1.7304068220366808 -0.6386249100228161; 1.7304068220366808 -0.6386249100228161; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039; 0.8732539292509538 -0.004885395480653039]; # generated in function above
    # split into two chains for testing
    arr1 = permutedims(reshape(arr0, 5,2,:),(1,3,2))
    chain = Chains(arr1, [:s, :m]);
    tmp1 = varwise_logpriors(model, chain)
    tmp = Chains(tmp1...); # can be used to create a Chains object
    vi = VarInfo(model)
    i_sample, i_chain = (1,2)
    DynamicPPL.setval!(vi, chain, i_sample, i_chain)
    lp1 =  DynamicPPL.varwise_logpriors(model, vi)
    @test all(tmp1[1][i_sample,:,i_chain] .≈ values(lp1))
end;