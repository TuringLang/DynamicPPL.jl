# functions for evaluating logp: log posterior, log likelihood and log prior

#### 1. logprior ####
"""
    logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    logprior(model_instance::Model, nt_arr::Vector{NamedTuple})
    logprior(model_instance::Model, nt_arr::Vector{Any})

Return an array of log priors evaluated at each sample in an MCMC chain or sample array.

Example 1:
    
    ```julia
    @model function demo_model(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        for i in 1:length(x)
            x[i] ~ Normal(m, sqrt(s))
        end
    end
    # generate data
    using Random, Distributions, Turing
    Random.seed!(111)
    x = rand(Normal(1.0, 1.0), 1000)
    # construct a chain of samples via MCMC sampling 
    using Turing
    chain = sample(demo_model(x), NUTS(0.65), 3_000) # chain: 1st index is the iteration no, 3rd index is the chain no.
    logprior(demo_model(x), chain) 
    
    Outputs:
    julia> 3000×1 Matrix{Float64}:
    -2.2188656502119937
    -2.233332271475687
     ⋮
    ```   

Example 2:
    
    ```julia
    # generate data
    sample_array = Vector(undef, 100)
    m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    for i in 1:100
        example_values = rand(NamedTuple, m)
        sample_array[i] = example_values
    end
    # calculate the pointwise loglikelihoods for the whole array.
    logprior(m, sample_array)

    Outputs:
    julia> 100-element Vector{Float64}:
    -3.8653211194703863
    -6.727832990780729
    ⋮
    ```  
"""
function logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end
function logprior(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
function logprior(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end


#### 2. loglikelihood ####
"""
    loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    loglikelihoods(model_instance::Model, nt_arr::Vector{NamedTuple})
    loglikelihoods(model_instance::Model, nt_arr::Vector{Any})

Return an array of log likelihoods evaluated at each sample in an MCMC chain or sample array.

Example 1:
    
    ```julia
    @model function demo_model(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        for i in 1:length(x)
            x[i] ~ Normal(m, sqrt(s))
        end
    end
    # generate data
    using Random, Distributions, Turing
    Random.seed!(111)
    x = rand(Normal(1.0, 1.0), 1000)
    # construct a chain of samples via MCMC sampling 
    using Turing
    chain = sample(demo_model(x), NUTS(0.65), 3_000) # chain: 1st index is the iteration no, 3rd index is the chain no.
    loglikelihoods(demo_model(x), chain)

    Outputs:
    julia> 3000×1 Matrix{Float64}:
    -1460.969366266108
    -1460.115380195131
    ⋮
    ```   

Example 2:
    
    ```julia
    # generate data
    sample_array = Vector(undef, 100)
    m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    for i in 1:100
        example_values = rand(NamedTuple, m)
        sample_array[i] = example_values
    end
    # calculate the pointwise loglikelihoods for the whole array.
    loglikelihoods(m, sample_array)

    Outputs:
    julia> 100-element Vector{Float64}:
    -7.31146824079265
    -3.006725528921822
    ⋮
    ```  
"""
function loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict)
    end
end
function loglikelihoods(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = Distributions.loglikelihood(model_instance, nt_arr[param_idx])
    end
    return lls
end
function loglikelihoods(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = Distributions.loglikelihood(model_instance, nt_arr[param_idx])
    end
    return lls
end

#### 3. logjoint ####
"""
    logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    logjoint(model_instance::Model, nt_arr::Vector{Any})

Return an array of log posteriors evaluated at each sample in an MCMC chain or sample array.

Example 1:
    
    ```julia
    @model function demo_model(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        for i in 1:length(x)
            x[i] ~ Normal(m, sqrt(s))
        end
    end
    # generate data
    using Random, Distributions, Turing
    Random.seed!(111)
    x = rand(Normal(1.0, 1.0), 1000)
    # construct a chain of samples via MCMC sampling 
    using Turing
    chain = sample(demo_model(x), NUTS(0.65), 3_000) # chain: 1st index is the iteration no, 3rd index is the chain no.
    logjoint(demo_model(x), chain)

    Outputs:
    julia> 3000×1 Matrix{Float64}:
    -1463.1882319163199
    -1462.3487124666067
    ⋮
    ```   

Example 2:
    
    ```julia
    # generate data
    sample_array = Vector(undef, 100)
    m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    for i in 1:100
        example_values = rand(NamedTuple, m)
        sample_array[i] = example_values
    end
    # calculate the pointwise loglikelihoods for the whole array.
    logjoint(m, sample_array)

    Outputs:
    julia> 100-element Vector{Float64}:
    -11.176789360263037
     -9.734558519702551
     ⋮
    ```  
"""
## evaluate log posterior at sample parameter positions
function logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict) +
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end
function logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
function logjoint(model_instance::Model, nt_arr::Vector{Any})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx = 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
