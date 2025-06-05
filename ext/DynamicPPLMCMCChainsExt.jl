module DynamicPPLMCMCChainsExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using MCMCChains: MCMCChains
else
    using ..DynamicPPL: DynamicPPL
    using ..MCMCChains: MCMCChains
end

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
           error("Chains do not support indexing using `VarName`s.")
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
    varinfo = DynamicPPL.VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    predictive_samples = map(iters) do (sample_idx, chain_idx)
        DynamicPPL.setval_and_resample!(varinfo, parameter_only_chain, sample_idx, chain_idx)
        model(rng, varinfo, DynamicPPL.SampleFromPrior())

        vals = DynamicPPL.values_as_in_model(model, false, varinfo)
        varname_vals = mapreduce(
            collect,
            vcat,
            map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals)),
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
        # TODO: Use `fix` once we've addressed https://github.com/TuringLang/DynamicPPL.jl/issues/702.
        # Update the varinfo with the current sample and make variables not present in `chain`
        # to be sampled.
        DynamicPPL.setval_and_resample!(varinfo, chain, sample_idx, chain_idx)
        # NOTE: Some of the varialbes can be a view into the `varinfo`, so we need to
        # `deepcopy` the `varinfo` before passing it to the `model`.
        model(deepcopy(varinfo))
    end
end

end
