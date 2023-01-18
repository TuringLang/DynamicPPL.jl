"""
    logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)

Return an array of log priors evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
# generate data
using Random, MCMCChains
Random.seed!(111)
# construct a chain of samples using MCMCChains
val = rand(500, 2, 3)
chain = Chains(val, [:s, :m])
logprior(demo_model(x), chain) 

# output
julia> logprior(demo_model(x), chain)
500×3 Matrix{Float64}:
   -61.6682     -1.70769   -2.11987
    -2.19736    -5.04622   -1.835
    -2.15858    -1.69405   -2.31419
    -1.74928   -66.3069    -1.76168
 -6367.73       -4.74417   -2.21238
    -8.8449     -1.70692   -1.73748
    -5.65338    -2.04857   -4.46512
     ⋮
    -9.09991    -4.5134    -2.5894
    -5.45221  -170.779     -1.97001
    -2.04826   -11.3178   -93.6076
   -13.68       -8.1437    -2.35059
    -4.45329    -1.70161   -1.88288
    -1.94955    -2.53816   -2.84721
   -21.7945     -1.99002   -3.27705
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


"""
    loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)

Return an array of log likelihoods evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest

# generate data
using Random, MCMCChains
Random.seed!(111)
val = rand(500, 2, 3)
chain = Chains(val, [:s, :m])
loglikelihoods(demo_model(x), chain)

# output
julia> 3000×1 Matrix{Float64}:
-1460.969366266108
-1460.115380195131
⋮
```  
"""
function loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict)
    end
end


"""
    logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    logjoint(model_instance::Model, nt_arr::Vector{Any})

Return an array of log posteriors evaluated at each sample in an MCMC chain or sample array.

Example 1:
    
```jldoctest
# generate data
using Random, MCMCChains
Random.seed!(111)
# construct a chain of samples using MCMCChains
val = rand(500, 2, 3)
chain = Chains(val, [:s, :m])
logjoint(demo_model(x), chain)

# output
julia> 3000×1 Matrix{Float64}:
-1463.1882319163199
-1462.3487124666067
⋮
```   

Example 2:
    
<<<<<<< HEAD
```jldoctest
# generate data
m = DynamicPPL.TestUtils.DEMO_MODELS[1]
samples = map(1:100) do _
    return rand(NamedTuple, m)
end
# calculate the pointwise loglikelihoods for the whole array.
logjoint(m, samples)
=======
    ```julia
    # generate data
    m = DynamicPPL.TestUtils.DEMO_MODELS[1]
    samples = map(1:100) do _
        return rand(NamedTuple, m)
    end
    # calculate the pointwise loglikelihoods for the whole array.
    logjoint(m, sample_array)
>>>>>>> 7ebcf10044f3bf5bf60fea03d283abcfd5511363

# output
julia> 100-element Vector{Float64}:
-11.176789360263037
    -9.734558519702551
    ⋮
```  
"""
function logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict) +
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end
<<<<<<< HEAD
function logjoint(model_instance::Model, nt_arr::Vector{NamedTuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
function logjoint(::Model, ::Vector{NamedTuple{Any, T} where T<:Tuple})
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
=======
>>>>>>> 7ebcf10044f3bf5bf60fea03d283abcfd5511363
