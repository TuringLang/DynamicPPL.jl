@testset "logp.jl" begin
    Test.@testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        # generate a chain of sample parameter values.
        N = 100
        chain = Vector(undef, N)
        vi_vector = Dict()
        logpriors_true = Vector(undef, N)
        loglikelihoods_true = Vector(undef, N)
        logposteriors_true = Vector(undef, N)
        for i in 1:N
            # generate samples and extrac vi
            example_values = rand(NamedTuple, m)
            print(example_values)
            chain[i] = example_values
            # append!(chain, [example_values])
            # Instantiate a `VarInfo` with the example values.
            vi = VarInfo(m)
            for vn in DynamicPPL.TestUtils.varnames(m)
                vi = DynamicPPL.setindex!!(vi, get(example_values, vn), vn)
            end
            vi_vector["$i"] = vi

            # calculate the true pointwise likelihood
            logprior_true = DynamicPPL.TestUtils.logprior_true(m, example_values...)
            logpriors_true[i] = logprior_true
            loglikelihood_true = DynamicPPL.TestUtils.loglikelihood_true(
                m, example_values...
            )
            loglikelihoods_true[i] = loglikelihood_true
            logposterior_true = logprior_true + loglikelihood_true
            logposteriors_true[i] = logposterior_true
        end
        # calculate the pointwise loglikelihoods for the whole chain using custom logprior.
        logpriors_new = logprior(m, chain)
        loglikelihoods_new = loglikelihoods(m, chain)
        logposteriors_new = logjoint(m, chain)
        # compare the likelihoods
        @test logpriors_new ≈ logpriors_true
        @test loglikelihoods_new ≈ loglikelihoods_true
        @test logposteriors_new ≈ logposteriors_true
    end
end
