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

If `include_all` is `false`, the returned `Chains` will contain only those variables that were not fixed by
the samples in `chain`. This is useful when you want to sample only new variables from the posterior 
predictive distribution.

# Examples
```jldoctest
julia> using DynamicPPL, AbstractMCMC, AdvancedHMC, ForwardDiff;

julia> @model function linear_reg(x, y, σ = 0.1)
           β ~ Normal(0, 1)
           for i ∈ eachindex(y)
               y[i] ~ Normal(β * x[i], σ)
           end
       end;

julia> σ = 0.1; f(x) = 2 * x + 0.1 * randn();

julia> Δ = 0.1; xs_train = 0:Δ:10; ys_train = f.(xs_train);

julia> xs_test = [10 + Δ, 10 + 2 * Δ]; ys_test = f.(xs_test);

julia> m_train = linear_reg(xs_train, ys_train, σ);

julia> n_train_logdensity_function = DynamicPPL.LogDensityFunction(m_train, DynamicPPL.VarInfo(m_train));

julia> chain_lin_reg = AbstractMCMC.sample(n_train_logdensity_function, NUTS(0.65), 200; chain_type=MCMCChains.Chains, param_names=[:β], discard_initial=100)
┌ Info: Found initial step size
└   ϵ = 0.003125

julia> m_test = linear_reg(xs_test, Vector{Union{Missing, Float64}}(undef, length(ys_test)), σ);

julia> predictions = predict(m_test, chain_lin_reg)
Object of type Chains, with data of type 100×2×1 Array{Float64,3}

Iterations        = 1:100
Thinning interval = 1
Chains            = 1
Samples per chain = 100
parameters        = y[1], y[2]

2-element Array{ChainDataFrame,1}

Summary Statistics
  parameters     mean     std  naive_se     mcse       ess   r_hat
  ──────────  ───────  ──────  ────────  ───────  ────────  ──────
        y[1]  20.1974  0.1007    0.0101  missing  101.0711  0.9922
        y[2]  20.3867  0.1062    0.0106  missing  101.4889  0.9903

Quantiles
  parameters     2.5%    25.0%    50.0%    75.0%    97.5%
  ──────────  ───────  ───────  ───────  ───────  ───────
        y[1]  20.0342  20.1188  20.2135  20.2588  20.4188
        y[2]  20.1870  20.3178  20.3839  20.4466  20.5895

julia> ys_pred = vec(mean(Array(group(predictions, :y)); dims = 1));

julia> sum(abs2, ys_test - ys_pred) ≤ 0.1
true
```
"""
function DynamicPPL.predict(
    rng::DynamicPPL.Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    include_all=false,
)
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)
    prototypical_varinfo = DynamicPPL.VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    predictive_samples = map(iters) do (sample_idx, chain_idx)
        varinfo = deepcopy(prototypical_varinfo)
        DynamicPPL.setval_and_resample!(
            varinfo, parameter_only_chain, sample_idx, chain_idx
        )
        model(rng, varinfo, DynamicPPL.SampleFromPrior())

        vals = DynamicPPL.values_as_in_model(model, varinfo)
        varname_vals = mapreduce(
            collect,
            vcat,
            map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals)),
        )

        return (varname_and_values=varname_vals, logp=DynamicPPL.getlogp(varinfo))
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
