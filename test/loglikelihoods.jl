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

() -> begin
    m = DynamicPPL.TestUtils.demo_assume_index_observe()
    example_values = DynamicPPL.TestUtils.rand_prior_true(m)
    vi = VarInfo(m)
    for vn in DynamicPPL.TestUtils.varnames(m)
        vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
    end
    ret_m = first(evaluate!!(m, vi, SamplingContext()))
    @test sum(map(k -> sum(ret_m.ld[k]), eachindex(ret_m.ld))) ≈ ret_m.logp
    () -> begin
        #by_generated_quantities
        s = vcat(example_values...)
        vnames = ["s[1]", "s[2]", "m[1]", "m[2]"]
        stup = (; zip(Symbol.(vnames), s)...)
        ret_m = generated_quantities(m, stup)
        @test sum(map(k -> sum(ret_m.ld[k]), eachindex(ret_m.ld))) ≈ ret_m.logp
        #chn = Chains(reshape(s, 1, : , 1), vnames);
        chn = Chains(reshape(s, 1, :, 1)) # causes warning but works
        ret_m = @test_logs (:warn,) generated_quantities(m, chn)[1, 1]
        @test sum(map(k -> sum(ret_m.ld[k]), eachindex(ret_m.ld))) ≈ ret_m.logp
    end
end

@testset "logpriors.jl" begin
    #m = DynamicPPL.TestUtils.DEMO_MODELS[1]
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
        tmp = DynamicPPL.pointwise_logpriors(m, vi)
        logp1 = getlogp(vi)
        logp = logprior(m, vi)
        @test !isfinite(getlogp(vi)) || sum(x -> sum(x), values(tmp)) ≈ logp
    end;
end;
