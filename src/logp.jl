# functions for evaluating logp: log posterior, log likelihood and log prior

#### 1. logprior ####
"""
This function evaluates the `log prior` for chain.
-- Inputs 
    model_instance: the probabilistic model instance;
    chain: an MCMC chain.
-- Outputs
    lls: a Vector of log prior values.
"""
function chain_logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(vi) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = DynamicPPL.logprior(
                model_instance, argvals_dict
            )
        end
    end
    return lls
end

"""
    This function evaluates the `log prior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        nt_array: an array of NamedTuple of sample parameter values.
    -- Outputs
        lls: a Vector of log prior values.
"""
function chain_logprior(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end

"""
    This function evaluates the `log prior` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        nt_array: an array of NamedTuple of sample parameter values.
    -- Outputs
        lls: a Vector of log prior values.
"""
function chain_logprior(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end


#### 2. loglikelihood ####
"""
    This function evaluates the `log likelihood` for chain.
    -- Inputs 
        model: the probabilistic model instance;
        chain: an MCMC chain 
    -- Outputs
        lls: a Vector of log likelihood values.
"""
function chain_loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(vi) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] = Distributions.loglikelihood(
                model_instance, argvals_dict
            )
        end
    end
    return lls
end

"""
This function evaluates the `log likelihood` for chain.
-- Inputs 
    model: the probabilistic model instance;
    nt_array: an array of NamedTuple of sample parameter values.
-- Outputs
    lls: a Vector of log likelihood values.
"""
function chain_loglikelihoods(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = Distributions.loglikelihood(model_instance, nt_arr[param_idx])
    end
    return lls
end

"""
This function evaluates the `log likelihood` for chain.
-- Inputs 
    model: the probabilistic model instance;
    nt_array: an array of NamedTuple of sample parameter values.
-- Outputs
    lls: a Vector of log likelihood values.
"""
function chain_loglikelihoods(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = Distributions.loglikelihood(model_instance, nt_arr[param_idx])
    end
    return lls
end

#### 3. logjoint ####
"""
This function evaluates the `log posterior` for chain.
-- Inputs 
    model: the probabilistic model instance;
    chain: an MCMC chain object.
-- Outputs
    lls: a Vector of log posterior values.
"""
## evaluate log posterior at sample parameter positions
function chain_logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    lls = Matrix{Float64}(undef, size(chain, 1), size(chain, 3)) # initialize a matrix to store the evaluated log posterior
    for chain_idx in 1:size(chain, 3)
        for iteration_idx in 1:size(chain, 1)
            # Extract sample parameter values using `varinfo` from the chain.
            # TODO: This does not work for cases where the model has dynamic support, i.e. some of the iterations might have differently sized parameter space.
            argvals_dict = OrderedDict(
                vn => chain[iteration_idx, Symbol(vn), chain_idx] for
                vn_parent in keys(vi) for
                vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
            )
            # Compute and store.
            lls[iteration_idx, chain_idx] =
                Distributions.loglikelihood(model_instance, argvals_dict) +
                DynamicPPL.logprior(model_instance, argvals_dict)
        end
    end
    return lls
end

"""
This function evaluates the `log posterior` for chain.
-- Inputs 
    model: the probabilistic model instance;
    nt_array: an array of NamedTuple of sample parameter values.
-- Outputs
    lls: a Vector of log posterior values.
"""
function chain_logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end

"""
This function evaluates the `log posterior` for chain.
-- Inputs 
    model: the probabilistic model instance;
    nt_array: an array of NamedTuple of sample parameter values.
-- Outputs
    lls: a Vector of log posterior values.
"""
function chain_logjoint(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end