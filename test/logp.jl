
## Test 
@model function demo_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# generate data
using StableRNGs, Turing
rng = StableRNG(111)
using Random, Distributions
Random.seed!(111)
x = rand(Normal(1.0, 1.0), 1000)

# MCMC sampling 
demo_model_instance = demo_model(x)
chain = sample(demo_model_instance, NUTS(0.65), 5_00) # chain: 1st index is the iteration no, 3rd index is the chain no.

# evaluate logp
lls = logprior(demo_model_instance, chain)
lls = loglikelihood(demo_model_instance, chain)
lls = logjoint(demo_model_instance, chain)