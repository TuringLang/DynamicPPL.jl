@testset "logdensities_likelihoods.jl" begin
    mod_ctx = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.2)
    mod_ctx2 = DynamicPPL.TestUtils.TestLogModifyingChildContext(1.4, mod_ctx)
    #model = DynamicPPL.TestUtils.DEMO_MODELS[1]
    #model = model = DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix2()
    demo_models = (
        DynamicPPL.TestUtils.DEMO_MODELS..., 
        DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix2())
    @testset "$(model.f)" for (i, model) in enumerate(demo_models)
        #@show i
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)

        # Instantiate a `VarInfo` with the example values.
        vi = VarInfo(model)
        () -> begin # when interactively debugging, need the global keyword
            for vn in DynamicPPL.TestUtils.varnames(model)
                global vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
            end
        end
        for vn in DynamicPPL.TestUtils.varnames(model)
            vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
        end

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(model, example_values...)        
        logp_true = logprior(model, vi)

        # Compute the pointwise loglikelihoods.
        lls = pointwise_loglikelihoods(model, vi)
        @test :s ∉ getsym.(keys(lls))
        if isempty(lls)
            # One of the models with literal observations, so we just skip.
            loglikelihood_true = 0.0
        else
            loglikelihood_sum = sum(sum, values(lls))
            @test loglikelihood_sum ≈ loglikelihood_true
        end

        # Compute the pointwise logdensities of the priors.
        lps_prior = pointwise_prior_logdensities(model, vi)
        @test :x ∉ getsym.(keys(lps_prior))
        logp = sum(sum, values(lps_prior))
        @test !isfinite(logp_true) || logp ≈ logp_true

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
        @show(arr0[1:2,:,:]);
    end
    arr0 = [5.590726417006858 -3.3407908212996493 -3.5126580698975687 -0.02830755634462317; 5.590726417006858 -3.3407908212996493 -3.5126580698975687 -0.02830755634462317;;; 3.5612802961176797 -5.167692608117693 1.3768066487740864 -0.9154694769223497; 3.5612802961176797 -5.167692608117693 1.3768066487740864 -0.9154694769223497]
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