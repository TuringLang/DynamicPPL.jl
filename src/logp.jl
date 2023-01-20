"""
	logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log priors evaluated at each sample in an MCMC chain or sample array.

Example:
	
```jldoctest; setup = :(using MCMCChains, StableRNGs), strict=false
julia> #
	   @model function demo_model(x)
		   s ~ InverseGamma(2, 3)
		   m ~ Normal(0, sqrt(s))
		   for i in 1:length(x)
		x[i] ~ Normal(m, sqrt(s))
		   end
	   end
demo_model (generic function with 2 methods)

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3);

julia> chain = Chains(val, [:s, :m]); # construct a chain of samples using MCMCChains

julia> logprior(demo_model([1., 2.]), chain);
```   
"""
function logprior(
    model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int=1
)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(
        Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3))
    ) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end


"""
	loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log likelihoods evaluated at each sample in an MCMC chain or sample array.

Example:
	
```jldoctest; setup = :(using MCMCChains, StableRNGs), strict=false
julia> #
	   @model function demo_model(x)
		   s ~ InverseGamma(2, 3)
		   m ~ Normal(0, sqrt(s))
		   for i in 1:length(x)
		x[i] ~ Normal(m, sqrt(s))
		   end
	   end
demo_model (generic function with 2 methods)

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3);

julia> chain = Chains(val, [:s, :m]); # construct a chain of samples using MCMCChains

julia> DynamicPPL.loglikelihoods(demo_model([1., 2.]), chain);
```  
"""
function loglikelihoods(
    model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int=1
)
	vi = VarInfo(model_instance) # extract variables info from the model
	map(
		Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3)),
	) do (iteration_idx, chain_idx)
		argvals_dict = OrderedDict(
			vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
			for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
		)
		Distributions.loglikelihood(model_instance, argvals_dict)
	end
end


"""
	logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log posteriors evaluated at each sample in an MCMC chain or sample array.

Example:
	
```jldoctest; setup = :(using MCMCChains, StableRNGs), strict=false
julia> #
	   @model function demo_model(x)
		   s ~ InverseGamma(2, 3)
		   m ~ Normal(0, sqrt(s))
		   for i in 1:length(x)
		x[i] ~ Normal(m, sqrt(s))
		   end
	   end
demo_model (generic function with 2 methods)

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3);

julia> chain = Chains(val, [:s, :m]); # construct a chain of samples using MCMCChains

julia> logjoint(demo_model([1., 2.]), chain, 2);
```   
"""
function logjoint(
	model_instance::Model,
	chain::AbstractMCMC.AbstractChains,
	start_idx::Int = 1,
)
	vi = VarInfo(model_instance) # extract variables info from the model
	map(
		Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3)),
	) do (iteration_idx, chain_idx)
		argvals_dict = OrderedDict(
			vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
			for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
		)
		Distributions.loglikelihood(model_instance, argvals_dict) +
		DynamicPPL.logprior(model_instance, argvals_dict)
	end
end
