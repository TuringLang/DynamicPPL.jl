# `to_distribution` and `to_submodel`

Both functions embed one probabilistic program in another. [`to_distribution`](@ref)
converts a supported model representation into a distribution whose value contains its
latent variables. [`to_submodel`](@ref) instead evaluates a DynamicPPL model and yields its
return value while recording its latent variables separately.

Currently, `to_distribution` supports only Stan through BridgeStan; `to_submodel` accepts only
DynamicPPL models.

## Stan example

Load BridgeStan and pass a Stan program to `to_distribution`. The first call requires a
BridgeStan toolchain; identical calls reuse the cached distribution.

```@example to-distribution
using AbstractPPL
using ADTypes: AutoForwardDiff
using BridgeStan, Distributions, DynamicPPL, LogDensityProblems
using DifferentiationInterface, ForwardDiff # Load AbstractPPL's ForwardDiff backend.

const STAN = raw"""
parameters {
  real theta;
}
model {
  theta ~ normal(0, 1);
}
"""
const STAN_DISTRIBUTION = to_distribution(STAN)

@model function demo(stan, y)
    theta ~ stan
    y ~ Normal(theta[1], 1)
end

ldf = LogDensityFunction(demo(STAN_DISTRIBUTION, 0.4))
u = [0.0]
prepared = AbstractPPL.prepare(
    AutoForwardDiff(), u -> LogDensityProblems.logdensity(ldf, u), u
)
logdensity, gradient = AbstractPPL.value_and_gradient!!(prepared, u)
```

The left-hand side receives Stan's flattened constrained parameters. The `data` and `seed`
keywords configure model construction; `stanc_args` and `make_args` configure the build.
BridgeStan supplies no sampler, so `InitFromPrior` uses DynamicPPL's uniform initializer in
unconstrained space.

## API

```@docs
to_distribution
to_submodel
DynamicPPL.prefix
```
