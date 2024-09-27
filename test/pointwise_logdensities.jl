@testset "logdensities_likelihoods.jl" begin
    likelihood_context = LikelihoodContext()
    prior_context = PriorContext()
    mod_ctx = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.2)
    mod_ctx2 = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.4, mod_ctx)
    #m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    @testset "$(m.f)" for (i, m) in enumerate(DynamicPPL.TestUtils.DEMO_MODELS)
        #@show i
        example_values = DynamicPPL.TestUtils.rand_prior_true(m)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(m)
        () -> begin # when interactively debugging, need the global keyword
            for vn in DynamicPPL.TestUtils.varnames(m)
                global vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
            end
        end
        for vn in DynamicPPL.TestUtils.varnames(m)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(m, example_values...)        
        logp_true = logprior(m, vi)

        # Compute the pointwise loglikelihoods.
        lls = pointwise_logdensities(m, vi, likelihood_context)
        #lls2 = pointwise_loglikelihoods(m, vi)
        loglikelihood = sum(sum, values(lls))
        if loglikelihood ≈ 0.0 #isempty(lls)
            # One of the models with literal observations, so we just skip.
            # TODO: Think of better way to detect this special case 
            loglikelihood_true = 0.0
        end
        @test loglikelihood ≈ loglikelihood_true

        # Compute the pointwise logdensities of the priors.
        lps_prior = pointwise_logdensities(m, vi, prior_context)
        logp = sum(sum, values(lps_prior))
        if false # isempty(lps_prior)
            # One of the models with only observations so we just skip.
        else
            logp1 = getlogp(vi)
            @test !isfinite(logp_true) || logp ≈ logp_true
        end

        # Compute both likelihood and logdensity of prior
        # using the default DefaultContex        
        lps = pointwise_logdensities(m, vi)
        logp = sum(sum, values(lps))
        @test logp ≈ (logp_true + loglikelihood_true)

        # Test that modifications of Setup are picked up
        lps = pointwise_logdensities(m, vi, mod_ctx2)
        logp = sum(sum, values(lps))
        @test logp ≈ (logp_true + loglikelihood_true) * 1.2 * 1.4
    end
end


@testset "pointwise_logdensities chain" begin
    @model function demo(x, ::Type{TV}=Vector{Float64}) where {TV}
        s ~ InverseGamma(2, 3)
        m = TV(undef, length(x))
        for i in eachindex(x)
            m[i] ~ Normal(0, √s)
        end
        x ~ MvNormal(m, √s)        
    end    
    x_true = [0.3290767977680923, 0.038972110187911684, -0.5797496780649221]
    model = demo(x_true)
    () -> begin
        # generate the sample used below
        chain = sample(model, MH(), MCMCThreads(), 10, 2)
        arr0 = stack(Array(chain, append_chains=false))
        @show(arr0);
    end
    arr0 = [5.590726417006858 -3.3407908212996493 -3.5126580698975687 -0.02830755634462317; 5.590726417006858 -3.3407908212996493 -3.5126580698975687 -0.02830755634462317; 0.9199555480151707 -0.1304320097505629 1.0669120062696917 -0.05253734412139093; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183; 1.0982766276744311 0.9593277181079177 0.005558174156359029 -0.45842032209694183;;; 3.5612802961176797 -5.167692608117693 1.3768066487740864 -0.9154694769223497; 3.5612802961176797 -5.167692608117693 1.3768066487740864 -0.9154694769223497; 2.5409470583244933 1.7838744695696407 0.7013562890105632 -3.0843947804314658; 0.8296370582311665 1.5360702767879642 -1.5964695255693102 0.16928084806166913; 2.6246697053824954 0.8096845024785173 -1.2621822861663752 1.1414885535466166; 1.1304261861894538 0.7325784741344005 -1.1866016911837542 -0.1639319562090826; 2.5669872989791473 -0.43642462460747317 0.07057300935786101 0.5168578624259272; 2.5669872989791473 -0.43642462460747317 0.07057300935786101 0.5168578624259272; 2.5669872989791473 -0.43642462460747317 0.07057300935786101 0.5168578624259272; 0.9838526141898173 -0.20198797220982412 2.0569535882007006 -1.1560724118010939]
    chain = Chains(arr0, [:s, Symbol("m[1]"), Symbol("m[2]"), Symbol("m[3]")]);
    tmp1 = pointwise_logdensities(model, chain)
    vi = VarInfo(model)
    i_sample, i_chain = (1,2)
    DynamicPPL.setval!(vi, chain, i_sample, i_chain)
    lp1 =  DynamicPPL.pointwise_logdensities(model, vi)
    # k = first(keys(lp1))
    for k in keys(lp1)
        @test tmp1[string(k)][i_sample,i_chain] .≈ lp1[k][1]
    end
end;