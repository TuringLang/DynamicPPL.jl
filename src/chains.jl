"""
chains.jl
---------
This file defines a (somewhat loose) API for converting MCMC chain types to and from
DynamicPPL samples. This functionality is then used to implement various utilities for model
re-evaluation.

To 'work with' DynamicPPL, an MCMC chain type `Tchain` must subtype
`AbstractMCMC.AbstractChains` and implement the following methods:

- `AbstractMCMC.to_samples(::Type{DynamicPPL.ParamsWithStats}, chain::Tchain)`
- `AbstractMCMC.from_samples(::Type{Tchain}, samples::AbstractMatrix{<:DynamicPPL.ParamsWithStats})`
- `DynamicPPL.predict(rng, model, chain::Tchain; include_all=false)`
- `DynamicPPL.pointwise_logdensities(model, chain::Tchain; whichlogprob)`

Optionally, `Tchain` can also implement:

- `DynamicPPL.reevaluate_with_chain(rng, model, chain::Tchain, accs, fallback)`

There is a default implementation of the above, but it can often be made more efficient by
implementing it directly for `Tchain`.

Implementing these methods allows the following extra methods to work for `Tchain` out of
the box:

- `DynamicPPL.logjoint(model, chain::Tchain)`
- `DynamicPPL.logprior(model, chain::Tchain)`
- `DynamicPPL.loglikelihood(model, chain::Tchain)`
- `DynamicPPL.returned(model, chain::Tchain)`
- `DynamicPPL.make_prior_chain(model, n_iters, Tchain)`
- `DynamicPPL.pointwise_loglikelihoods(model, chain::Tchain)`
- `DynamicPPL.pointwise_prior_logdensities(model, chain::Tchain)`
"""

"""
    ParamsWithStats

A struct which contains parameter values extracted from a `VarInfo`, along with any
statistics associated with the VarInfo. The statistics are provided as a NamedTuple and are
optional.
"""
struct ParamsWithStats{P<:OrderedDict{<:VarName,<:Any},S<:NamedTuple}
    params::P
    stats::S
end

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        model::Model,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `model` with the provided `varinfo`.
Re-evaluation of the model is often necessary to obtain correct parameter values as well as
log probabilities. This is especially true when using linked VarInfos, i.e., when variables
have been transformed to unconstrained space, and if this is not done, subtle correctness
bugs may arise: see, e.g., https://github.com/TuringLang/Turing.jl/issues/2195.

`include_colon_eq` controls whether variables on the left-hand side of `:=` are included in
the resulting parameters.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo,
    model::DynamicPPL.Model,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    varinfo = maybe_to_typed_varinfo(varinfo)
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),)
    end
    varinfo = DynamicPPL.setaccs!!(varinfo, accs)
    varinfo = last(DynamicPPL.evaluate!!(model, varinfo))
    params = DynamicPPL.getacc(varinfo, Val(:ValuesAsInModel)).values
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(varinfo),
                loglikelihood=DynamicPPL.getloglikelihood(varinfo),
                logjoint=DynamicPPL.getlogjoint(varinfo),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end

# Re-evaluating the model is unconscionably slow for untyped VarInfo. It's much faster to
# convert it to a typed varinfo first, hence this method.
# https://github.com/TuringLang/Turing.jl/issues/2604
maybe_to_typed_varinfo(vi::UntypedVarInfo) = typed_varinfo(vi)
maybe_to_typed_varinfo(vi::UntypedVectorVarInfo) = typed_vector_varinfo(vi)
maybe_to_typed_varinfo(vi::AbstractVarInfo) = vi

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

There is one case where re-evaluation is not necessary, which is when the VarInfos all
already contain `DynamicPPL.ValuesAsInModelAccumulator`. This accumulator stores values
as seen during the model evaluation, so the values can be simply read off. In this case,
the `model` argument can be omitted, and no re-evaluation will be performed. However, it is
the caller's responsibility to ensure that `ValuesAsInModelAccumulator` is indeed present
inside `varinfo`.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo, stats::NamedTuple=NamedTuple(); include_log_probs::Bool=true
)
    params = DynamicPPL.getacc(varinfo, Val(:ValuesAsInModel)).values
    if include_log_probs
        has_prior_acc = DynamicPPL.hasacc(varinfo, Val(:LogPrior))
        has_likelihood_acc = DynamicPPL.hasacc(varinfo, Val(:LogLikelihood))
        if has_prior_acc
            stats = merge(stats, (logprior=DynamicPPL.getlogprior(varinfo),))
        end
        if has_likelihood_acc
            stats = merge(stats, (loglikelihood=DynamicPPL.getloglikelihood(varinfo),))
        end
        if has_prior_acc && has_likelihood_acc
            stats = merge(stats, (logjoint=DynamicPPL.getlogjoint(varinfo),))
        end
    end
    return ParamsWithStats(params, stats)
end

"""
    ParamsWithStats(
        param_vector::AbstractVector,
        ldf::DynamicPPL.LogDensityFunction,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `ldf` with the provided
`param_vector`.

This method is intended to replace the old method of obtaining parameters and statistics
via `unflatten` plus re-evaluation. It is faster for two reasons:

1. It does not rely on `deepcopy`-ing the VarInfo object (this used to be mandatory as
   otherwise re-evaluation would mutate the VarInfo, rendering it unusable for subsequent
   MCMC iterations).
2. The re-evaluation is faster as it uses `OnlyAccsVarInfo`.
"""
function ParamsWithStats(
    param_vector::AbstractVector,
    ldf::DynamicPPL.LogDensityFunction{Tlink},
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
) where {Tlink}
    strategy = InitFromParams(
        VectorWithRanges{Tlink}(
            ldf._iden_varname_ranges, ldf._varname_ranges, param_vector
        ),
        nothing,
    )
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),)
    end
    _, vi = DynamicPPL.init!!(ldf.model, OnlyAccsVarInfo(AccumulatorTuple(accs)), strategy)
    params = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(vi),
                loglikelihood=DynamicPPL.getloglikelihood(vi),
                logjoint=DynamicPPL.getlogjoint(vi),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end

"""
    DynamicPPL.InitFromParams(
        params::DynamicPPL.ParamsWithStats,
        fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
    )

Initialise a VarInfo from the parameters stored in `params`.
"""
function DynamicPPL.InitFromParams(
    params::ParamsWithStats,
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
)
    return DynamicPPL.InitFromParams(params.params, fallback)
end

"""
    DynamicPPL.make_prior_chain(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        n_iters::Int,
        chain_type::Type{<:AbstractMCMC.AbstractChains}
    )

Construct an AbstractChains object by sampling from the prior of `model` for
`n_iters` iterations.
"""
function make_prior_chain(
    rng::Random.AbstractRNG, model::Model, n_iters::Int, ::Type{T}
) where {T<:AbstractMCMC.AbstractChains}
    vi = DynamicPPL.OnlyAccsVarInfo((DynamicPPL.ValuesAsInModelAccumulator(false),))
    ps = hcat([ParamsWithStats(last(DynamicPPL.init!!(rng, model, vi))) for _ in 1:n_iters])
    return AbstractMCMC.from_samples(T, ps)
end
function make_prior_chain(
    model::Model, n_iters::Int, ::Type{T}
) where {T<:AbstractMCMC.AbstractChains}
    return make_prior_chain(Random.default_rng(), model, n_iters, T)
end

"""
    DynamicPPL.reevaluate_with_chain(
        [rng::AbstractRNG,]
        model::DynamicPPL.Model,
        chain::AbstractChains,
        accs::NTuple{N,DynamicPPL.AbstractAccumulator},
        fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
    ) where {N}

Re-evaluate `model` using the parameters stored in each sample of `chain`, using the
accumulators provided in `at`.

If `rng` is not provided, `Random.default_rng()` is used. `AbstractChains` subtypes should
only implement the method with the `rng` argument.

This function should return a matrix of `(retval, updated_vi)` tuples, where `retval` is the
return value of the model re-evaluation, and `updated_vi` is the updated VarInfo after
re-evaluation that contains the accumulators specified in `accs`. The metadata field of the
VarInfo is unspecified and may be empty (i.e., `updated_vi` may be an `OnlyAccsVarInfo`).
"""
function reevaluate_with_chain(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::AbstractMCMC.AbstractChains,
    accs::NTuple{N,DynamicPPL.AbstractAccumulator},
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
) where {N}
    params_with_stats = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, chain)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.AccumulatorTuple(accs))
    return map(params_with_stats) do ps
        DynamicPPL.init!!(rng, model, vi, DynamicPPL.InitFromParams(ps, fallback))
    end
end
function reevaluate_with_chain(
    model::DynamicPPL.Model,
    chain::AbstractMCMC.AbstractChains,
    accs::NTuple{N,DynamicPPL.AbstractAccumulator},
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
) where {N}
    return reevaluate_with_chain(Random.default_rng(), model, chain, accs, fallback)
end

"""
    AbstractPPL.predict(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        chain::AbstractMCMC.AbstractChains;
        include_all=false
    )

Sample from a predictive distribution by executing `model` with parameters fixed to each
sample in `chain`, and return the resulting `AbstractChains`.

See https://turinglang.org/docs/usage/predictive-distributions/ for a detailed user guide.

The `model` passed to `predict` is often different from the one used to generate `chain`.
Typically, the model from which `chain` originated treats certain variables as observed
(i.e., data points), while the model you pass to `predict` may mark these same variables as
missing or unobserved. Calling `predict` then leverages the previously inferred parameter
values to simulate what new, unobserved data might look like, given your beliefs as encoded
in the chain.

For each set of parameters in `chain`:

1. All random variables present in `chain` are fixed to their sampled values.
2. Any variables not included in `chain` are sampled from their prior distributions.

If `include_all` is `false`, the returned `AbstractChains` will contain only those variables
that were not fixed by the samples in `chain`. This is useful when you want to sample only
new variables from the predictive distribution. Otherwise, all variables (both fixed and
newly sampled) are included in the output chain.

The returned chain will also contain log-probabilities corresponding to the re-evaluation of
the model. In particular, the log probability for the newly predicted variables are now
considered as prior terms. However, note that the log-prior of the newly returned chain will
also contain the log-prior terms of the parameters already present in the input `chain`.
Thus, if you want to obtain the log-probability of the predicted variables *only*, you can
subtract the two log-prior terms. The `include_all` keyword argument has no effect on the
log-probability fields.

# Examples

```jldoctest
julia> using Distributions, DynamicPPL, MCMCChains

julia> @model function demo()
           x ~ Normal()
           y ~ Normal(x)
       end;

julia> # Generate synthetic chain using known ground truth parameter.
       ground_truth_x = 2.0;

julia> # Construct a chain where `x` is sampled from the prior (i.e. 0).
       # Ordinarily, this chain would be obtained via MCMC sampling:
       # e.g. using `sample(demo() | (; y = observed_y), NUTS(), 1000)`.
       x_chain = make_prior_chain(demo() | (; y = 2.0), 100, MCMCChains.Chains);

julia> # Create a model for prediction where `y` is missing. Note that we don't
       # provide any observed data for `y`.
       prediction_model = demo();

julia> chain = predict(prediction_model, x_chain);

julia> keys(chain)
1-element Vector{Symbol}:
 :y
```
"""
function predict end

"""
    returned(model::DynamicPPL>Model, chain::AbstractChains)

Often there are situations where we are interested in extra quantities that are not
themselves random variables (but which may depend on them).

`returned(model, chain)` will execute `model` for each of the samples in `chain` and return
an array of the values returned by the `model` for each sample.

# Example

```jldoctest
julia> using DynamicPPL, Distributions, MCMCChains

julia> @model function demo()
           x ~ Normal()
           return x^2
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> chain = DynamicPPL.make_prior_chain(model, 5, MCMCChains.Chains);

julia> returned(model, chain) ≈ (chain[:x] .^ 2)
true
```
"""
function DynamicPPL.returned(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)
    return map(first, reevaluate_with_chain(model, chain, (), nothing))
end

"""
    DynamicPPL.pointwise_logdensities(
        model::Model,
        chain::AbstractChains,
        ::Val{whichlogprob}=Val(:both),
    ) where {whichlogprob}

Calculate the log probability density associated with each variable in the model, for each
iteration in the chain.

The `whichlogprob` argument controls which log probabilities are calculated and stored. It
can take the values `:prior`, `:likelihood`, or `:both` (the default).

Returns a new chain with the same structure as the input `chain`, mapping the variables to
their log probabilities.
"""
function pointwise_logdensities end

"""
    DynamicPPL.pointwise_loglikelihoods(
        model::DynamicPPL.Model,
        chain::AbstractChains
    )

Compute the pointwise log-likelihoods of the model given the chain. This is the same as
`pointwise_logdensities(model, chain)`, but only including the likelihood terms.

See also: [`DynamicPPL.pointwise_logdensities`](@ref),
[`DynamicPPL.pointwise_prior_logdensities`](@ref).
"""
function DynamicPPL.pointwise_loglikelihoods(model::DynamicPPL.Model, chain::AbstractChains)
    return DynamicPPL.pointwise_logdensities(model, chain, Val(:likelihood))
end

"""
    DynamicPPL.pointwise_prior_logdensities(
        model::DynamicPPL.Model,
        chain::AbstractChains,
    )

Compute the pointwise log-prior-densities of the model given the chain. This is the same as
`pointwise_logdensities(model, chain)`, but only including the prior terms.

See also: [`DynamicPPL.pointwise_logdensities`](@ref),
[`DynamicPPL.pointwise_loglikelihoods`](@ref).
"""
function DynamicPPL.pointwise_prior_logdensities(
    model::DynamicPPL.Model, chain::AbstractChains
)
    return DynamicPPL.pointwise_logdensities(model, chain, Val(:prior))
end

"""
    logjoint(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)

Return an array of log joint probabilities evaluated at each sample in an MCMC `chain`.

# Examples

```jldoctest
julia> using MCMCChains, Distributions, DynamicPPL

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
function DynamicPPL.logjoint(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)
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
    loglikelihood(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)

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
function DynamicPPL.loglikelihood(
    model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains
)
    return map(
        DynamicPPL.getloglikelihood ∘ last,
        reevaluate_with_chain(
            model, chain, (DynamicPPL.LogLikelihoodAccumulator(),), nothing
        ),
    )
end

"""
    logprior(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)

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
function DynamicPPL.logprior(model::DynamicPPL.Model, chain::AbstractMCMC.AbstractChains)
    return map(
        DynamicPPL.getlogprior ∘ last,
        reevaluate_with_chain(model, chain, (DynamicPPL.LogPriorAccumulator(),), nothing),
    )
end
