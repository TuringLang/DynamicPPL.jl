module DynamicPPLMCMCChainsExt

using DynamicPPL: DynamicPPL, AbstractPPL
using MCMCChains: MCMCChains

# Load state from a `Chains`: By convention, it is stored in `:samplerstate` metadata
function DynamicPPL.loadstate(chain::MCMCChains.Chains)
    if !haskey(chain.info, :samplerstate)
        throw(
            ArgumentError(
                "The chain object does not contain the final state of the sampler: Metadata `:samplerstate` missing.",
            ),
        )
    end
    return chain.info[:samplerstate]
end

_has_varname_to_symbol(info::NamedTuple{names}) where {names} = :varname_to_symbol in names

function DynamicPPL.supports_varname_indexing(chain::MCMCChains.Chains)
    return _has_varname_to_symbol(chain.info)
end

function _check_varname_indexing(c::MCMCChains.Chains)
    return DynamicPPL.supports_varname_indexing(c) ||
           error("This `Chains` object does not support indexing using `VarName`s.")
end

function DynamicPPL.getindex_varname(
    c::MCMCChains.Chains, sample_idx, vn::DynamicPPL.VarName, chain_idx
)
    _check_varname_indexing(c)
    return c[sample_idx, c.info.varname_to_symbol[vn], chain_idx]
end
function DynamicPPL.varnames(c::MCMCChains.Chains)
    _check_varname_indexing(c)
    return keys(c.info.varname_to_symbol)
end

function chain_sample_to_varname_dict(
    c::MCMCChains.Chains{Tval}, sample_idx, chain_idx
) where {Tval}
    _check_varname_indexing(c)
    d = Dict{DynamicPPL.VarName,Tval}()
    for vn in DynamicPPL.varnames(c)
        d[vn] = DynamicPPL.getindex_varname(c, sample_idx, vn, chain_idx)
    end
    return d
end

"""
    predict([rng::AbstractRNG,] model::Model, chain::MCMCChains.Chains; include_all=false)

Sample from the posterior predictive distribution by executing `model` with parameters fixed to each sample
in `chain`, and return the resulting `Chains`.

The `model` passed to `predict` is often different from the one used to generate `chain`.
Typically, the model from which `chain` originated treats certain variables as observed (i.e.,
data points), while the model you pass to `predict` may mark these same variables as missing
or unobserved. Calling `predict` then leverages the previously inferred parameter values to
simulate what new, unobserved data might look like, given your posterior beliefs.

For each parameter configuration in `chain`:
1. All random variables present in `chain` are fixed to their sampled values.
2. Any variables not included in `chain` are sampled from their prior distributions.

If `include_all` is `false`, the returned `Chains` will contain only those variables that were not fixed by
the samples in `chain`. This is useful when you want to sample only new variables from the posterior
predictive distribution.

# Examples
```jldoctest
using AbstractMCMC, Distributions, DynamicPPL, Random

@model function linear_reg(x, y, σ = 0.1)
    β ~ Normal(0, 1)
    for i in eachindex(y)
        y[i] ~ Normal(β * x[i], σ)
    end
end

# Generate synthetic chain using known ground truth parameter
ground_truth_β = 2.0

# Create chain of samples from a normal distribution centered on ground truth
β_chain = MCMCChains.Chains(
    rand(Normal(ground_truth_β, 0.002), 1000), [:β,]
)

# Generate predictions for two test points
xs_test = [10.1, 10.2]

m_train = linear_reg(xs_test, fill(missing, length(xs_test)))

predictions = DynamicPPL.AbstractPPL.predict(
    Random.default_rng(), m_train, β_chain
)

ys_pred = vec(mean(Array(predictions); dims=1))

# Check if predictions match expected values within tolerance
(
    isapprox(ys_pred[1], ground_truth_β * xs_test[1], atol = 0.01),
    isapprox(ys_pred[2], ground_truth_β * xs_test[2], atol = 0.01)
)

# output

(true, true)
```
"""
function DynamicPPL.predict(
    rng::DynamicPPL.Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    include_all=false,
)
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)

    # Set up a VarInfo with the right accumulators
    varinfo = DynamicPPL.setaccs!!(
        DynamicPPL.VarInfo(),
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogJacobianAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.ValuesAsInModelAccumulator(false),
        ),
    )
    _, varinfo = DynamicPPL.init!!(model, varinfo)
    varinfo = DynamicPPL.typed_varinfo(varinfo)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    predictive_samples = map(iters) do (sample_idx, chain_idx)
        # Extract values from the chain
        values_dict = chain_sample_to_varname_dict(parameter_only_chain, sample_idx, chain_idx)
        # Resample any variables that are not present in `values_dict`
        _, varinfo = DynamicPPL.init!!(
            rng,
            model,
            varinfo,
            DynamicPPL.InitFromParams(values_dict, DynamicPPL.InitFromPrior()),
        )
        vals = DynamicPPL.getacc(varinfo, Val(:ValuesAsInModel)).values
        varname_vals = mapreduce(
            collect,
            vcat,
            map(AbstractPPL.varname_and_value_leaves, keys(vals), values(vals)),
        )

        return (varname_and_values=varname_vals, logp=DynamicPPL.getlogjoint(varinfo))
    end

    chain_result = reduce(
        MCMCChains.chainscat,
        [
            _predictive_samples_to_chains(predictive_samples[:, chain_idx]) for
            chain_idx in 1:size(predictive_samples, 2)
        ],
    )
    parameter_names = if include_all
        MCMCChains.names(chain_result, :parameters)
    else
        filter(
            k -> !(k in MCMCChains.names(parameter_only_chain, :parameters)),
            names(chain_result, :parameters),
        )
    end
    return chain_result[parameter_names]
end
function DynamicPPL.predict(
    model::DynamicPPL.Model, chain::MCMCChains.Chains; include_all=false
)
    return DynamicPPL.predict(
        DynamicPPL.Random.default_rng(), model, chain; include_all=include_all
    )
end

function _predictive_samples_to_arrays(predictive_samples)
    variable_names_set = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()

    sample_dicts = map(predictive_samples) do sample
        varname_value_pairs = sample.varname_and_values
        varnames = map(first, varname_value_pairs)
        values = map(last, varname_value_pairs)
        for varname in varnames
            push!(variable_names_set, varname)
        end

        return DynamicPPL.OrderedCollections.OrderedDict(zip(varnames, values))
    end

    variable_names = collect(variable_names_set)
    variable_values = [
        get(sample_dicts[i], key, missing) for i in eachindex(sample_dicts),
        key in variable_names
    ]

    return variable_names, variable_values
end

function _predictive_samples_to_chains(predictive_samples)
    variable_names, variable_values = _predictive_samples_to_arrays(predictive_samples)
    variable_names_symbols = map(Symbol, variable_names)

    internal_parameters = [:lp]
    log_probabilities = reshape([sample.logp for sample in predictive_samples], :, 1)

    parameter_names = [variable_names_symbols; internal_parameters]
    parameter_values = hcat(variable_values, log_probabilities)
    parameter_values = MCMCChains.concretize(parameter_values)

    return MCMCChains.Chains(
        parameter_values, parameter_names, (internals=internal_parameters,)
    )
end

"""
    returned(model::Model, chain::MCMCChains.Chains)

Execute `model` for each of the samples in `chain` and return an array of the values
returned by the `model` for each sample.

# Examples
## General
Often you might have additional quantities computed inside the model that you want to
inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()
    return interesting_quantity(θ, x)
end
m = demo(data)
chain = sample(m, alg, n)
# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples
# from the posterior/`chain`:
returned(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using Turing

julia> @model function demo(xs)
           s ~ InverseGamma(2, 3)
           m_shifted ~ Normal(10, √s)
           m = m_shifted - 10

           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end

           return (m, )
       end
demo (generic function with 1 method)

julia> model = demo(randn(10));

julia> chain = sample(model, MH(), 10);

julia> returned(model, chain)
10×1 Array{Tuple{Float64},2}:
 (2.1964758025119338,)
 (2.1964758025119338,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.043088571494005024,)
 (-0.16489786710222099,)
 (-0.16489786710222099,)
```
"""
function DynamicPPL.returned(model::DynamicPPL.Model, chain_full::MCMCChains.Chains)
    chain = MCMCChains.get_sections(chain_full, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        # Extract values from the chain
        values_dict = chain_sample_to_varname_dict(chain, sample_idx, chain_idx)
        # Resample any variables that are not present in `values_dict`, and
        # return the model's retval.
        retval, _ = DynamicPPL.init!!(
            model,
            varinfo,
            DynamicPPL.InitFromParams(values_dict, DynamicPPL.InitFromPrior()),
        )
        retval
    end
end

"""
    DynamicPPL.pointwise_logdensities(
        model::DynamicPPL.Model,
        chain::MCMCChains.Chains,
        ::Type{Tout}=MCMCChains.Chains
        ::Val{whichlogprob}=Val(:both),
    )

Runs `model` on each sample in `chain`, returning a new `MCMCChains.Chains` object where
the log-density of each variable at each sample is stored (rather than its value).

`whichlogprob` specifies which log-probabilities to compute. It can be `:both`, `:prior`, or
`:likelihood`.

You can pass `Tout=OrderedDict` to get the result as an `OrderedDict{VarName,
Matrix{Float64}}` instead.

See also: [`DynamicPPL.pointwise_loglikelihoods`](@ref),
[`DynamicPPL.pointwise_prior_logdensities`](@ref).

# Examples

```jldoctest pointwise-logdensities-chains; setup=:(using Distributions)
julia> using MCMCChains

julia> @model function demo(xs, y)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end
           y ~ Normal(m, √s)
       end
demo (generic function with 2 methods)

julia> # Example observations.
       model = demo([1.0, 2.0, 3.0], [4.0]);

julia> # A chain with 3 iterations.
       chain = Chains(
           reshape(1.:6., 3, 2),
           [:s, :m];
           info=(varname_to_symbol=Dict(
               @varname(s) => :s,
               @varname(m) => :m,
           ),),
       );

julia> plds = pointwise_logdensities(model, chain)
Chains MCMC chain (3×6×1 Array{Float64, 3}):

Iterations        = 1:1:3
Number of chains  = 1
Samples per chain = 3
parameters        = s, m, xs[1], xs[2], xs[3], y
[...]

julia> plds[:s]
2-dimensional AxisArray{Float64,2,...} with axes:
    :iter, 1:1:3
    :chain, 1:1
And data, a 3×1 Matrix{Float64}:
 -0.8027754226637804
 -1.3822169643436162
 -2.0986122886681096

julia> # The above is the same as:
       logpdf.(InverseGamma(2, 3), chain[:s])
3×1 Matrix{Float64}:
 -0.8027754226637804
 -1.3822169643436162
 -2.0986122886681096
```

julia> # Alternatively:
       plds_dict = pointwise_logdensities(model, chain, OrderedDict)
OrderedDict{VarName, Matrix{Float64}} with 6 entries:
  s     => [-0.802775; -1.38222; -2.09861;;]
  m     => [-8.91894; -7.51551; -7.46824;;]
  xs[1] => [-5.41894; -5.26551; -5.63491;;]
  xs[2] => [-2.91894; -3.51551; -4.13491;;]
  xs[3] => [-1.41894; -2.26551; -2.96824;;]
  y     => [-0.918939; -1.51551; -2.13491;;]
"""
function DynamicPPL.pointwise_logdensities(
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains,
    ::Type{Tout}=MCMCChains.Chains,
    ::Val{whichlogprob}=Val(:both),
) where {whichlogprob,Tout}
    vi = DynamicPPL.VarInfo(model)
    acc = DynamicPPL.PointwiseLogProbAccumulator{whichlogprob}()
    accname = DynamicPPL.accumulator_name(acc)
    vi = DynamicPPL.setaccs!!(vi, (acc,))
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    pointwise_logps = map(iters) do (sample_idx, chain_idx)
        # Extract values from the chain
        values_dict = chain_sample_to_varname_dict(parameter_only_chain, sample_idx, chain_idx)
        # Re-evaluate the model
        _, vi = DynamicPPL.init!!(
            model,
            vi,
            DynamicPPL.InitFromParams(values_dict, DynamicPPL.InitFromPrior()),
        )
        DynamicPPL.getacc(vi, Val(accname)).logps
    end

    # pointwise_logps is a matrix of OrderedDicts
    all_keys = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()
    for d in pointwise_logps
        union!(all_keys, DynamicPPL.OrderedCollections.OrderedSet(keys(d)))
    end
    # this is a 3D array: (iterations, variables, chains)
    new_data = [
        get(pointwise_logps[iter, chain], k, missing) for
        iter in 1:size(pointwise_logps, 1), k in all_keys,
        chain in 1:size(pointwise_logps, 2)
    ]

    if Tout == MCMCChains.Chains
        return MCMCChains.Chains(new_data, Symbol.(collect(all_keys)))
    elseif Tout <: AbstractDict
        return Tout{DynamicPPL.VarName,Matrix{Float64}}(
            k => new_data[:, i, :] for (i, k) in enumerate(all_keys)
        )
    end
end

"""
    DynamicPPL.pointwise_loglikelihoods(
        model::DynamicPPL.Model,
        chain::MCMCChains.Chains,
        ::Type{Tout}=MCMCChains.Chains
    )

Compute the pointwise log-likelihoods of the model given the chain. This is the same as
`pointwise_logdensities(model, chain)`, but only including the likelihood terms.

See also: [`DynamicPPL.pointwise_logdensities`](@ref), [`DynamicPPL.pointwise_prior_logdensities`](@ref).
"""
function DynamicPPL.pointwise_loglikelihoods(
    model::DynamicPPL.Model, chain::MCMCChains.Chains, ::Type{Tout}=MCMCChains.Chains
) where {Tout}
    return DynamicPPL.pointwise_logdensities(model, chain, Tout, Val(:likelihood))
end

"""
    DynamicPPL.pointwise_prior_logdensities(
        model::DynamicPPL.Model,
        chain::MCMCChains.Chains
    )

Compute the pointwise log-prior-densities of the model given the chain. This is the same as
`pointwise_logdensities(model, chain)`, but only including the prior terms.

See also: [`DynamicPPL.pointwise_logdensities`](@ref), [`DynamicPPL.pointwise_loglikelihoods`](@ref).
"""
function DynamicPPL.pointwise_prior_logdensities(
    model::DynamicPPL.Model, chain::MCMCChains.Chains, ::Type{Tout}=MCMCChains.Chains
) where {Tout}
    return DynamicPPL.pointwise_logdensities(model, chain, Tout, Val(:prior))
end

"""
    logjoint(model::Model, chain::MCMCChains.Chains)

Return an array of log joint probabilities evaluated at each sample in an MCMC `chain`.

# Examples

```jldoctest
julia> using MCMCChains, Distributions

julia> @model function demo_model(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           for i in eachindex(x)
               x[i] ~ Normal(m, sqrt(s))
           end
       end;

julia> # Construct a chain of samples using MCMCChains.
       # This sets s = 0.5 and m = 1.0 for all three samples.
       chain = Chains(repeat([0.5 1.0;;;], 3, 1, 1), [:s, :m]);

julia> logjoint(demo_model([1., 2.]), chain)
3×1 Matrix{Float64}:
 -5.440428709758045
 -5.440428709758045
 -5.440428709758045
```
"""
function DynamicPPL.logjoint(model::DynamicPPL.Model, chain::MCMCChains.Chains)
    var_info = DynamicPPL.VarInfo(model) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = DynamicPPL.OrderedCollections.OrderedDict{DynamicPPL.VarName,Any}(
            vn_parent => DynamicPPL.values_from_chain(
                var_info, vn_parent, chain, chain_idx, iteration_idx
            ) for vn_parent in keys(var_info)
        )
        DynamicPPL.logjoint(model, argvals_dict)
    end
end

"""
    loglikelihood(model::DynamicPPL.Model, chain::MCMCChains.Chains)

Return an array of log likelihoods evaluated at each sample in an MCMC `chain`.
# Examples

```jldoctest
julia> using MCMCChains, Distributions

julia> @model function demo_model(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           for i in eachindex(x)
               x[i] ~ Normal(m, sqrt(s))
           end
       end;

julia> # Construct a chain of samples using MCMCChains.
       # This sets s = 0.5 and m = 1.0 for all three samples.
       chain = Chains(repeat([0.5 1.0;;;], 3, 1, 1), [:s, :m]);

julia> loglikelihood(demo_model([1., 2.]), chain)
3×1 Matrix{Float64}:
 -2.1447298858494
 -2.1447298858494
 -2.1447298858494
```
"""
function DynamicPPL.loglikelihood(model::DynamicPPL.Model, chain::MCMCChains.Chains)
    var_info = DynamicPPL.VarInfo(model) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = DynamicPPL.OrderedCollections.OrderedDict{DynamicPPL.VarName,Any}(
            vn_parent => DynamicPPL.values_from_chain(
                var_info, vn_parent, chain, chain_idx, iteration_idx
            ) for vn_parent in keys(var_info)
        )
        DynamicPPL.loglikelihood(model, argvals_dict)
    end
end

"""
    logprior(model::DynamicPPL.Model, chain::MCMCChains.Chains)

Return an array of log prior probabilities evaluated at each sample in an MCMC `chain`.

# Examples

```jldoctest
julia> using MCMCChains, Distributions

julia> @model function demo_model(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           for i in eachindex(x)
               x[i] ~ Normal(m, sqrt(s))
           end
       end;

julia> # Construct a chain of samples using MCMCChains.
       # This sets s = 0.5 and m = 1.0 for all three samples.
       chain = Chains(repeat([0.5 1.0;;;], 3, 1, 1), [:s, :m]);

julia> logprior(demo_model([1., 2.]), chain)
3×1 Matrix{Float64}:
 -3.2956988239086447
 -3.2956988239086447
 -3.2956988239086447
```
"""
function DynamicPPL.logprior(model::DynamicPPL.Model, chain::MCMCChains.Chains)
    var_info = DynamicPPL.VarInfo(model) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = DynamicPPL.OrderedCollections.OrderedDict{DynamicPPL.VarName,Any}(
            vn_parent => DynamicPPL.values_from_chain(
                var_info, vn_parent, chain, chain_idx, iteration_idx
            ) for vn_parent in keys(var_info)
        )
        DynamicPPL.logprior(model, argvals_dict)
    end
end

end
