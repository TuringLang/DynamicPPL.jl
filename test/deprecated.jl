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
        loglikelihood = sum(sum, values(lls))

        #if isempty(lls)
        if loglikelihood ≈ 0.0 #isempty(lls)
            # One of the models with literal observations, so we just skip.
            # TODO: Think of better way to detect this special case 
            continue
        end

        loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(m, example_values...)

        #priors = 

        @test loglikelihood ≈ loglikelihood_true
    end
end

