"""
In many Bayesian modelling scenarios, we would like to evaluate the `log prior`, `log likelihood` and `log posterior`, particularly for MCMC chains.

In previous versions, we have the `prob` macro for querying these likelihoods, e.g.: 
`prob"s = 1.0, m = 1.0 | model = gdemo, x = nothing, y = nothing`, where gdemo is a model instance, see the [Turing guide](https://turing.ml/v0.22/docs/using-turing/guide). This script aims to replace the `prob` macro whose use will be gradually depreciated in the future.

In particular, the functions built here aim to evaluate 3 `logp`s for MCMC chains: log-prior, log-likelihood and log-posterior.

We use the same names `logprior` and `logjoint` to multi-dispatch the already existing `DynamicPPL.logprior` and `DynamicPPL.logjoint` methods.

For all 3 functions, the inputs are a model instance and a chain; the output is a likelihood matrix with dimension (no. of iterations, no. of chains).

"""

```julia
# functions for evaluating logp: log posterior and log prior
# functions for evaluating logp: log posterior, log likelihood and log prior
using Turing, DynamicPPL, MCMCChains, StatsBase

## 1. evaluate log prior at sample parameter positions
function DynamicPPL.logprior(model_instance::Model, chain::Chains)
    """
    This function evaluates the `log prior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        chain: an MCMC chain.
    -- Outputs
        lls: a Vector of log prior values.
    """
    varinfo = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx = 1:size(chain, 3)
        for iteration_idx = 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx]
                for vn_parent in keys(varinfo)
                for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = DynamicPPL.logprior(model_instance, argvals_dict)
        end
    end
    return lls
end

## 2. evaluate log likelihood at sample parameter positions
function DynamicPPL.loglikelihood(model_instance::Model, chain::Chains)
    """
    This function evaluates the `log likelihood` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        chain: an MCMC chain 
    -- Outputs
        lls: a Vector of log likelihood values.
    """
    varinfo = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx = 1:size(chain, 3)
        for iteration_idx = 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx]
                for vn_parent in keys(varinfo)
                for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = StatsBase.loglikelihood(model_instance, argvals_dict)
        end
    end
    return lls
end

## 3. evaluate log posterior at sample parameter positions
function DynamicPPL.logjoint(model_instance::Model, chain::Chains)
    """
    This function evaluates the `log posterior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        chain: an MCMC chain object.
    -- Outputs
        lls: a Vector of log posterior values.
    """
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
            # Compute and store.
            lls[iteration_idx, chain_idx] = StatsBase.loglikelihood(model_instance, argvals_dict) + DynamicPPL.logprior(model_instance, argvals_dict)
        end
    end
    return lls
 end

 function DynamicPPL.logjoint(model_instance::Model, arr::AbstractArray)
    """
    This function evaluates the `log posterior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        arr: an un-named array of sample parameter values.
    -- Outputs
        lls: a Vector of log posterior values.
    """
    varinfo = VarInfo(model_instance) # extract variables info from the model
    lls = Array{Float64}(undef, size(arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(arr, 1)
        # Extract sample parameter values using `varinfo` from the chain.
        # TODO: This does not work for cases where the model has dynamic support, i.e. some of the iterations might have differently sized parameter space.
        argvals_dict = OrderedDict(
            vn => arr[param_idx]
            for vn_parent in keys(varinfo)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
        )
        # Compute and store.
        lls[param_idx] = StatsBase.loglikelihood(model_instance, argvals_dict) + DynamicPPL.logprior(model_instance, argvals_dict)
    end
    return lls
 end

 function DynamicPPL.logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    """
    This function evaluates the `log posterior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        nt_array: an array of NamedTuple of sample parameter values.
    -- Outputs
        lls: a Vector of log posterior values.
    """
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = StatsBase.loglikelihood(model_instance, nt_arr[param_idx]) + DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
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

# evaluate logp
demo_model_instance = demo_model(x[1:10])
lls = DynamicPPL.logprior(demo_model_instance, chain) 
lls = DynamicPPL.loglikelihood(demo_model_instance, chain) 
lls = DynamicPPL.logjoint(demo_model_instance, chain)  
```