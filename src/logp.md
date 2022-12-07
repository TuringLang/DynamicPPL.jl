"""
In many Bayesian modelling scenarios, we would like to evaluate the `log prior` and `log posterior`.

In previous versions, we have the `prob` macro for querying these likelihoods, e.g.: 
`prob"s = 1.0, m = 1.0 | model = gdemo, x = nothing, y = nothing`, where gdemo is a model instance, see the [Turing guide](https://turing.ml/v0.22/docs/using-turing/guide).

This script aims to replace the `prob` macro whose use will be gradually depreciated in the future.
In particular, the functions built here aim to evaluate the likelihoods for MCMC chains. 
We use the same names `logjoint` and `logprior` to multi-dispatch the already existing `DynamicPPL.logjoint` and `DynamicPPL.logprior` methods.

In building these functions, we make use of the `StatsBase.loglikelihood` for evaluating likelihoods, and `VarInfo` to extract variable information from a probabilistic model. 
"""

```julia
# functions for evaluating logp: log posterior and log prior
using Turing, DynamicPPL, MCMCChains, StatsBase

## 1. evaluate log posterior at sample parameter positions
using DynamicPPL
function DynamicPPL.logjoint(model, data, chain)
    """
    This function evaluates the `log posterior`.
    -- Inputs 
        model: the probabilistic model structure (not an model instance);
        data: the data, can be a single data point or an array;
        chain: a MCMCChain object.
    -- Outputs
        lls_dict: a dictionary with the data point as its key, and the `log posterior` as its value.
    """
    lls_dict = Dict{Float64, Matrix}(data[k] => zeros(size(chain, 1),size(chain, 3)) for k = 1:size(data, 1))
    for data_idx = 1:size(data, 1)
        model_instance = model([data[data_idx]]) # initantiate a model, given the data
        varinfo = VarInfo(model_instance) # extract variables info from the model
        lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
        for chain_idx = 1:size(chain, 3)
            for iteration_idx = 1:size(chain, 1) 
                # Extract sample parameter values using `varinfo` from the chain.
                # TODO: This does not work for cases where the model has dynamic support, i.e. some of the iterations might have differently sized parameter space.
                argvals_dict = OrderedDict(
                    vn => chain[iteration_idx, Symbol(vn), chain_idx]
                    for vn_parent in keys(varinfo)
                    for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
                )
                # Compute and store `loglikelihood`.
                lls[iteration_idx, chain_idx] = StatsBase.loglikelihood(model_instance, argvals_dict)
            end
        end
        lls_dict[data[data_idx]] = lls
    end
    return lls_dict
end

## 2. evaluate log prior at sample parameter positions
function DynamicPPL.logprior(model, data, chain)
    """
    This function evaluates the `log prior`.
    -- Inputs 
        model: the probabilistic model structure (not an model instance);
        data: the data, can be a single data point or an array;
        chain: a MCMCChain object.
    -- Outputs
        lls_dict: a dictionary with the data point as its key, and the `log prior` as its value.
    """
    lls_dict = Dict{Float64, Matrix}(data[k] => zeros(size(chain, 1),size(chain, 3)) for k = 1:size(data, 1))
    for data_idx = 1:size(data, 1)
        model_instance = model([data[data_idx]]) # initantiate a model, given the data
        varinfo = VarInfo(model_instance) # extract variables info from the model
        lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
        for chain_idx = 1:size(chain, 3)
            for iteration_idx = 1:size(chain, 1)
                # Extract sample parameter values using `varinfo` from the chain.
                # TODO: This does not work for cases where the model has dynamic support, i.e. some of the iterations might have differently sized parameter space.
                argvals_dict = OrderedDict(
                    vn => chain[iteration_idx, Symbol(vn), chain_idx]
                    for vn_parent in keys(varinfo)
                    for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
                )
                # Compute and store `loglikelihood`.
                lls[iteration_idx, chain_idx] = DynamicPPL.logprior(model_instance, argvals_dict)
            end
        end
        lls_dict[data[data_idx]] = lls
    end
    return lls_dict
end

## Test 
@model function demo_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# generate data
using StableRNGs
rng = StableRNG(111)
using Random, Distributions
Random.seed!(111)
x = rand(Normal(1.0, 1.0), 1000)

# MCMC sampling 
chain = sample(demo_model(x), NUTS(0.65), 3_000) # chain: 1st index is the iteration no, 3rd index is the chain no.
using StatsPlots
StatsPlots.plot(group(chain, :m)[:,1,1])

# evaluate logp
lls = DynamicPPL.logjoint(demo_model, x[1:10], chain) # a Dict with each key-value pair of size 3000x1
lls = DynamicPPL.logprior(demo_model, x[1:10], chain) # a Dict with each key-value pair of size 3000x1
```
