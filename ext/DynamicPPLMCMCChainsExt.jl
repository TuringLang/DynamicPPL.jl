module DynamicPPLMCMCChainsExt

using DynamicPPL: DynamicPPL, AbstractPPL, AbstractMCMC, Random
using BangBang: setindex!!
using MCMCChains: MCMCChains

function getindex_varname(
    c::MCMCChains.Chains, sample_idx, vn::DynamicPPL.VarName, chain_idx
)
    return c[sample_idx, c.info.varname_to_symbol[vn], chain_idx]
end
function get_varnames(c::MCMCChains.Chains)
    haskey(c.info, :varname_to_symbol) ||
        error("This `Chains` object does not support indexing using `VarName`s.")
    return keys(c.info.varname_to_symbol)
end

function get_stat_with_fallback(stats::NamedTuple, key::Symbol)
    # MCMCChains can't handle any contents that are not Real or Missing. That means
    # that if we have other types of statistics (e.g., arrays), we need to just silently
    # drop them.
    stat = get(stats, key, missing)
    return if stat isa Union{Real,Missing}
        stat
    else
        missing
    end
end

"""
    AbstractMCMC.from_samples(
        ::Type{MCMCChains.Chains},
        params_and_stats::AbstractMatrix{<:ParamsWithStats}
    )

Convert an array of `DynamicPPL.ParamsWithStats` to an `MCMCChains.Chains` object.
"""
function AbstractMCMC.from_samples(
    ::Type{MCMCChains.Chains},
    params_and_stats::AbstractMatrix{<:DynamicPPL.ParamsWithStats},
)
    # Handle parameters
    all_vn_leaves = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()
    split_dicts = map(params_and_stats) do ps
        # Separate into individual VarNames.
        vn_leaves_and_vals = if isempty(ps.params)
            Tuple{DynamicPPL.VarName,Any}[]
        else
            iters = map(
                AbstractPPL.varname_and_value_leaves,
                keys(ps.params),
                values(ps.params),
            )
            mapreduce(collect, vcat, iters)
        end
        vn_leaves = map(first, vn_leaves_and_vals)
        vals = map(last, vn_leaves_and_vals)
        for vn_leaf in vn_leaves
            push!(all_vn_leaves, vn_leaf)
        end
        DynamicPPL.OrderedCollections.OrderedDict(zip(vn_leaves, vals))
    end
    vn_leaves = collect(all_vn_leaves)
    param_vals = [
        get(split_dicts[i, j], key, missing) for i in eachindex(axes(split_dicts, 1)),
        key in vn_leaves, j in eachindex(axes(split_dicts, 2))
    ]
    param_symbols = map(Symbol, vn_leaves)
    # Handle statistics
    stat_keys = DynamicPPL.OrderedCollections.OrderedSet{Symbol}()
    for ps in params_and_stats
        for k in keys(ps.stats)
            push!(stat_keys, k)
        end
    end
    stat_keys = collect(stat_keys)
    stat_vals = [
        get_stat_with_fallback(params_and_stats[i, j].stats, key) for
        i in eachindex(axes(params_and_stats, 1)), key in stat_keys,
        j in eachindex(axes(params_and_stats, 2))
    ]
    # Construct name map and info
    name_map = (internals=stat_keys,)
    info = (
        varname_to_symbol=DynamicPPL.OrderedCollections.OrderedDict(
            zip(all_vn_leaves, param_symbols)
        ),
    )
    # Concatenate parameter and statistic values
    vals = cat(param_vals, stat_vals; dims=2)
    symbols = vcat(param_symbols, stat_keys)
    return MCMCChains.Chains(MCMCChains.concretize(vals), symbols, name_map; info=info)
end

"""
    AbstractMCMC.to_samples(
        ::Type{DynamicPPL.ParamsWithStats},
        chain::MCMCChains.Chains,
    )

Convert an `MCMCChains.Chains` object to an array of `DynamicPPL.ParamsWithStats`.

For this to work, `chain` must contain the `varname_to_symbol` mapping in its `info` field.
"""
function AbstractMCMC.to_samples(
    ::Type{DynamicPPL.ParamsWithStats}, chain::MCMCChains.Chains
)
    idxs = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    # Get parameters
    params_matrix = map(idxs) do (sample_idx, chain_idx)
        vnt = DynamicPPL.VarNamedTuple()
        for vn in get_varnames(chain)
            vnt = setindex!!(vnt, getindex_varname(chain, sample_idx, vn, chain_idx), vn)
        end
        vnt
    end
    # Statistics
    stats_matrix = if :internals in MCMCChains.sections(chain)
        internals_chain = MCMCChains.get_sections(chain, :internals)
        map(idxs) do (sample_idx, chain_idx)
            get(internals_chain[sample_idx, :, chain_idx], keys(internals_chain); flatten=true)
        end
    else
        fill(NamedTuple(), size(idxs))
    end
    # Bundle them together
    return map(idxs) do (sample_idx, chain_idx)
        DynamicPPL.ParamsWithStats(
            params_matrix[sample_idx, chain_idx], stats_matrix[sample_idx, chain_idx]
        )
    end
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:DynamicPPL.ParamsWithStats},
    model::DynamicPPL.Model,
    spl::AbstractMCMC.AbstractSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state=false,
    stats=missing,
    sort_chain=false,
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    bare_chain = AbstractMCMC.from_samples(MCMCChains.Chains, reshape(ts, :, 1))

    # Add additional MCMC-specific info
    info = bare_chain.info
    if save_state
        info = merge(info, (model=model, sampler=spl, samplerstate=state))
    end
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Reconstruct the chain with the extra information
    # Yeah, this is quite ugly. Blame MCMCChains.
    chain = MCMCChains.Chains(
        bare_chain.value.data,
        names(bare_chain),
        bare_chain.name_map;
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )
    return sort_chain ? sort(chain) : chain
end

"""
    reevaluate_with_chain(
        rng::AbstractRNG,
        model::Model,
        chain::MCMCChains.Chains
        accs::NTuple{N,AbstractAccumulator};
        fallback=nothing,
    )

Re-evaluate `model` for each sample in `chain` using the accumulators provided in `accs`,
returning a matrix of `(retval, updated_at)` tuples.

This loops over all entries in the chain and uses `DynamicPPL.InitFromParams` as the
initialisation strategy when re-evaluating the model. For many usecases the fallback should
not be provided (as we expect the chain to contain all necessary variables); but for
`predict` this has to be `InitFromPrior()` to allow sampling new variables (i.e. generating
the posterior predictions).
"""
function reevaluate_with_chain(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains,
    accs::NTuple{N,DynamicPPL.AbstractAccumulator},
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
) where {N}
    params_with_stats = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, chain)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.AccumulatorTuple(accs))
    return map(params_with_stats) do ps
        DynamicPPL.init!!(rng, model, vi, DynamicPPL.InitFromParams(ps.params, fallback))
    end
end
function reevaluate_with_chain(
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains,
    accs::NTuple{N,DynamicPPL.AbstractAccumulator},
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
) where {N}
    return reevaluate_with_chain(Random.default_rng(), model, chain, accs, fallback)
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
    accs = (
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
        DynamicPPL.ValuesAsInModelAccumulator(false),
    )
    predictions = map(
        DynamicPPL.ParamsWithStats ∘ last,
        reevaluate_with_chain(
            rng, model, parameter_only_chain, accs, DynamicPPL.InitFromPrior()
        ),
    )
    chain_result = AbstractMCMC.from_samples(MCMCChains.Chains, predictions)
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
    return map(first, reevaluate_with_chain(model, chain, (), nothing))
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
    acc = DynamicPPL.PointwiseLogProbAccumulator{whichlogprob}()
    accname = DynamicPPL.accumulator_name(acc)
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)
    pointwise_logps =
        map(reevaluate_with_chain(model, parameter_only_chain, (acc,), nothing)) do (_, vi)
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
    return map(
        DynamicPPL.getlogjoint ∘ last,
        reevaluate_with_chain(
            model,
            chain,
            (DynamicPPL.LogPriorAccumulator(), DynamicPPL.LogLikelihoodAccumulator()),
            nothing,
        ),
    )
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
    return map(
        DynamicPPL.getloglikelihood ∘ last,
        reevaluate_with_chain(
            model, chain, (DynamicPPL.LogLikelihoodAccumulator(),), nothing
        ),
    )
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
    return map(
        DynamicPPL.getlogprior ∘ last,
        reevaluate_with_chain(model, chain, (DynamicPPL.LogPriorAccumulator(),), nothing),
    )
end

end
