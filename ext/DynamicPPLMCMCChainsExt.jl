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

# this is copied from Turing.jl, `stats` field is omitted as it is never used
struct Transition{T,F}
    θ::T
    lp::F
end

function Transition(model::DynamicPPL.Model, vi::DynamicPPL.VarInfo)
    return Transition(getparams(model, vi), DynamicPPL.getlogp(vi))
end

# a copy of Turing.Inference.getparams
getparams(model, t) = t.θ
function getparams(model::DynamicPPL.Model, vi::DynamicPPL.VarInfo)
    # NOTE: In the past, `invlink(vi, model)` + `values_as(vi, OrderedDict)` was used.
    # Unfortunately, using `invlink` can cause issues in scenarios where the constraints
    # of the parameters change depending on the realizations. Hence we have to use
    # `values_as_in_model`, which re-runs the model and extracts the parameters
    # as they are seen in the model, i.e. in the constrained space. Moreover,
    # this means that the code below will work both of linked and invlinked `vi`.
    # Ref: https://github.com/TuringLang/Turing.jl/issues/2195
    # NOTE: We need to `deepcopy` here to avoid modifying the original `vi`.
    vals = DynamicPPL.values_as_in_model(model, deepcopy(vi))

    # Obtain an iterator over the flattened parameter names and values.
    iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))

    # Materialize the iterators and concatenate.
    return mapreduce(collect, vcat, iters)
end

function _params_to_array(model::DynamicPPL.Model, ts::Vector)
    names_set = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        nms_and_vs = getparams(model, t)
        nms = map(first, nms_and_vs)
        vs = map(last, nms_and_vs)
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return DynamicPPL.OrderedCollections.OrderedDict(zip(nms, vs))
    end
    names = collect(names_set)
    vals = [
        get(dicts[i], key, missing) for i in eachindex(dicts), (j, key) in enumerate(names)
    ]

    return names, vals
end

"""

    predict([rng::AbstractRNG,] model::Model, chain::MCMCChains.Chains; include_all=false)

Execute `model` conditioned on each sample in `chain`, and return the resulting `Chains`.

If `include_all` is `false`, the returned `Chains` will contain only those variables
sampled/not present in `chain`.

# Details
Internally calls `Turing.Inference.transitions_from_chain` to obtained the samples
and then converts these into a `Chains` object using `AbstractMCMC.bundle_samples`.

# Example
```jldoctest
julia> using AbstractMCMC, AdvancedHMC, DynamicPPL, ForwardDiff;
[ Info: [Turing]: progress logging is disabled globally

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

julia> chain_lin_reg = AbstractMCMC.sample(n_train_logdensity_function, NUTS(0.65), 200; chain_type=MCMCChains.Chains, param_names=[:β]);
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
    # Don't need all the diagnostics
    chain_parameters = MCMCChains.get_sections(chain, :parameters)

    spl = DynamicPPL.SampleFromPrior()

    # Sample transitions using `spl` conditioned on values in `chain`
    transitions = transitions_from_chain(rng, model, chain_parameters; sampler=spl)

    # Let the Turing internals handle everything else for you
    chain_result = reduce(
        MCMCChains.chainscat,
        [
            _bundle_samples(transitions[:, chain_idx], model, spl) for
            chain_idx in 1:size(transitions, 2)
        ],
    )

    parameter_names = if include_all
        MCMCChains.names(chain_result, :parameters)
    else
        filter(
            k -> !(k in MCMCChains.names(chain_parameters, :parameters)),
            names(chain_result, :parameters),
        )
    end

    return chain_result[parameter_names]
end

getlogp(t::Transition) = t.lp

function get_transition_extras(ts::AbstractVector{<:Transition})
    valmat = reshape([getlogp(t) for t in ts], :, 1)
    return [:lp], valmat
end

function names_values(extra_data::AbstractVector{<:NamedTuple{names}}) where {names}
    values = [getfield(data, name) for data in extra_data, name in names]
    return collect(names), values
end

function names_values(xs::AbstractVector{<:NamedTuple})
    # Obtain all parameter names.
    names_set = Set{Symbol}()
    for x in xs
        for k in keys(x)
            push!(names_set, k)
        end
    end
    names_unique = collect(names_set)

    # Extract all values as matrix.
    values = [haskey(x, name) ? x[name] : missing for x in xs, name in names_unique]

    return names_unique, values
end

getlogevidence(transitions, sampler, state) = missing

# this is copied from Turing.jl/src/mcmc/Inference.jl, types are more restrictive (removed types that are defined in Turing)
# the function is simplified, so that unused arguments are removed
function _bundle_samples(
    ts::Vector{<:Transition}, model::DynamicPPL.Model, spl::DynamicPPL.SampleFromPrior
)
    # Convert transitions to array format.
    # Also retrieve the variable names.
    varnames, vals = _params_to_array(model, ts)
    varnames_symbol = map(Symbol, varnames)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(ts)

    # Extract names & construct param array.
    nms = [varnames_symbol; extra_params]
    parray = hcat(vals, extra_values)

    # Set up the info tuple.
    info = NamedTuple()

    info = merge(
        info,
        (
            varname_to_symbol=DynamicPPL.OrderedCollections.OrderedDict(
                zip(varnames, varnames_symbol)
            ),
        ),
    )

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    chain = MCMCChains.Chains(parray, nms, (internals=extra_params,))

    return chain
end

"""
    transitions_from_chain(
        [rng::AbstractRNG,]
        model::Model,
        chain::MCMCChains.Chains;
        sampler = DynamicPPL.SampleFromPrior()
    )

Execute `model` conditioned on each sample in `chain`, and return resulting transitions.

The returned transitions are represented in a `Vector{<:Turing.Inference.Transition}`.

# Details

In a bit more detail, the process is as follows:
1. For every `sample` in `chain`
   1. For every `variable` in `sample`
      1. Set `variable` in `model` to its value in `sample`
   2. Execute `model` with variables fixed as above, sampling variables NOT present
      in `chain` using `SampleFromPrior`
   3. Return sampled variables and log-joint

# Example
```julia-repl
julia> using Turing

julia> @model function demo()
           m ~ Normal(0, 1)
           x ~ Normal(m, 1)
       end;

julia> m = demo();

julia> chain = Chains(randn(2, 1, 1), ["m"]); # 2 samples of `m`

julia> transitions = Turing.Inference.transitions_from_chain(m, chain);

julia> [Turing.Inference.getlogp(t) for t in transitions] # extract the logjoints
2-element Array{Float64,1}:
 -3.6294991938628374
 -2.5697948166987845

julia> [first(t.θ.x) for t in transitions] # extract samples for `x`
2-element Array{Array{Float64,1},1}:
 [-2.0844148956440796]
 [-1.704630494695469]
```
"""
function transitions_from_chain(
    model::DynamicPPL.Model, chain::MCMCChains.Chains; kwargs...
)
    return transitions_from_chain(Random.default_rng(), model, chain; kwargs...)
end

function transitions_from_chain(
    rng::DynamicPPL.Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    sampler=DynamicPPL.SampleFromPrior(),
)
    vi = DynamicPPL.VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    transitions = map(iters) do (sample_idx, chain_idx)
        # Set variables present in `chain` and mark those NOT present in chain to be resampled.
        DynamicPPL.setval_and_resample!(vi, chain, sample_idx, chain_idx)
        model(rng, vi, sampler)

        # Convert `VarInfo` into `NamedTuple` and save.
        Transition(model, vi)
    end

    return transitions
end

"""
    generated_quantities(model::Model, chain::MCMCChains.Chains)

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
generated_quantities(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using DynamicPPL, Turing

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

julia> generated_quantities(model, chain)
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
function DynamicPPL.generated_quantities(
    model::DynamicPPL.Model, chain_full::MCMCChains.Chains
)
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
