# Chains extensions

DynamicPPL provides extensions for working with posterior chains from two packages:

  - [`MCMCChains.Chains`](https://turinglang.org/MCMCChains.jl/stable/) (via `DynamicPPLMCMCChainsExt`)
  - [`FlexiChains.FlexiChain{VarName}`](https://turinglang.org/FlexiChains.jl/stable/) (via `DynamicPPLFlexiChainsExt`)

## Converting to and from chains

### `from_samples`

```@docs
AbstractMCMC.from_samples(::Type{MCMCChains.Chains}, ::AbstractMatrix{<:DynamicPPL.ParamsWithStats})
AbstractMCMC.from_samples(::Type{MCMCChains.Chains}, ::AbstractMatrix{<:DynamicPPL.VarNamedTuple})
AbstractMCMC.from_samples(::Type{<:FlexiChains.VNChain}, ::AbstractMatrix{<:DynamicPPL.ParamsWithStats})
AbstractMCMC.from_samples(::Type{<:FlexiChains.VNChain}, ::AbstractMatrix{<:DynamicPPL.VarNamedTuple})
```

### `to_samples`

```@docs
AbstractMCMC.to_samples(::Type{DynamicPPL.ParamsWithStats}, ::MCMCChains.Chains, ::DynamicPPL.Model)
AbstractMCMC.to_samples(::Type{DynamicPPL.ParamsWithStats}, ::FlexiChains.FlexiChain{T}, ::DynamicPPL.Model) where {T<:AbstractPPL.VarName}
AbstractMCMC.to_samples(::Type{DynamicPPL.ParamsWithStats}, ::FlexiChains.FlexiChain{T}) where {T<:AbstractPPL.VarName}
```

## Initialising from a chain

```@docs
DynamicPPL.InitFromParams(::FlexiChains.FlexiChain{<:AbstractPPL.VarName}, ::Union{Int,FlexiChains.At}, ::Union{Int,FlexiChains.At}, ::Union{DynamicPPL.AbstractInitStrategy,Nothing})
```

## Log probabilities

```@docs
DynamicPPL.logjoint(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.logjoint(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
DynamicPPL.loglikelihood(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.loglikelihood(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
DynamicPPL.logprior(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.logprior(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
```

## Return values

```@docs
DynamicPPL.returned(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.returned(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
```

## Predictions

```@docs
DynamicPPL.predict(::Random.AbstractRNG, ::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.predict(::Random.AbstractRNG, ::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
```

## Pointwise log-probabilities

```@docs
DynamicPPL.pointwise_logdensities(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.pointwise_logdensities(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
DynamicPPL.pointwise_loglikelihoods(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.pointwise_loglikelihoods(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
DynamicPPL.pointwise_prior_logdensities(::DynamicPPL.Model, ::MCMCChains.Chains)
DynamicPPL.pointwise_prior_logdensities(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
```

## LOO-CV

Leave-one-out cross-validation can be performed with FlexiChains (note that you must import PosteriorStats separately):

```@docs
PosteriorStats.loo(::DynamicPPL.Model, ::FlexiChains.FlexiChain{<:AbstractPPL.VarName})
```
