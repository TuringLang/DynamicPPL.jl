# functions for evaluating logp: log posterior, log likelihood and log prior

using DynamicPPL, AbstractMCMC, StatsBase

## 1. evaluate log prior at sample parameter positions
function DynamicPPL.logprior(model_instance::Model, chain::AbstractChains)
    """
    This function evaluates the `log prior` for chain.
    -- Inputs 
        model_instance: the probabilistic model instance;
        chain: an MCMC chain.
    -- Outputs
        lls: a Vector of log prior values.
    """
    varinfo = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(varinfo) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = DynamicPPL.logprior(
                model_instance, argvals_dict
            )
        end
    end
    return lls
end

## 2. evaluate log likelihood at sample parameter positions
function DynamicPPL.loglikelihood(model_instance::Model, chain::AbstractChains)
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
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(varinfo) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = StatsBase.loglikelihood(
                model_instance, argvals_dict
            )
        end
    end
    return lls
end

## 3. evaluate log posterior at sample parameter positions
function DynamicPPL.logjoint(model_instance::Model, chain::AbstractChains)
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
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            # Extract sample parameter values using `varinfo` from the chain.
            # TODO: This does not work for cases where the model has dynamic support, i.e. some of the iterations might have differently sized parameter space.
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(varinfo) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, varinfo[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] =
                StatsBase.loglikelihood(model_instance, argvals_dict) +
                DynamicPPL.logprior(model_instance, argvals_dict)
        end
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
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            StatsBase.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
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

# final comments:
# 1. this script is doing similar to `pointwise_loglikelihoods`: https://beta.turing.ml/DynamicPPL.jl/stable/api/#DynamicPPL.pointwise_loglikelihoods
# 2. if the probabilistic model has a return statement for the log likelihood you would like to calculate, you can use `generated_quantities(model, chain)` to evaluate the likelihoods at sample positions.