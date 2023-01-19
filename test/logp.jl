@testset "logp.jl" begin
    Test.@testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        # generate a chain of sample parameter values.
        N = 200
        start_idx = 100

        logpriors_true = Vector{Float64}[undef, N-start_idx]
        loglikelihoods_true = Vector{Float64}[undef, N-start_idx]
        logposteriors_true = Vector{Float64}[undef, N-start_idx]

        chain = sample(m, NUTS(), N)

        map(start_idx:N) do i
            val = get_params(chain[i, :, :])
            example_values = (
                m = collect(Iterators.flatten(val.m)),
                s = collect(Iterators.flatten(val.s)),
            )
            logpriors_true[i] = DynamicPPL.TestUtils.logprior_true(m, example_values...)
            loglikelihoods_true[i] = DynamicPPL.TestUtils.loglikelihood_true(
                m, example_values...
            )
            logposteriors_true[i] = logpriors_true[i] + loglikelihoods_true[i]
        end
        # calculate the pointwise loglikelihoods for the whole chain using custom logprior.
        logpriors_new = logprior(m, chain, start_idx)
        loglikelihoods_new = loglikelihoods(m, chain, start_idx)
        logposteriors_new = logjoint(m, chain, start_idx)
        # compare the likelihoods
        @test logpriors_new ≈ logpriors_true
        @test loglikelihoods_new ≈ loglikelihoods_true
        @test logposteriors_new ≈ logposteriors_true
    end
end
