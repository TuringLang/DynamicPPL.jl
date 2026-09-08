# `to_distribution` and `to_submodel`

Both functions embed one probabilistic program in another. [`to_distribution`](@ref)
converts a supported model representation into a distribution over its latent variables.
[`to_submodel`](@ref) instead evaluates a DynamicPPL model and yields its return value while
recording its latent variables separately.

Currently, `to_distribution` supports only Stan through BridgeStan; `to_submodel` accepts only
DynamicPPL models.

## Stan example

Load BridgeStan and pass Stan source to `to_distribution`. The first call for a given program
and set of options requires a BridgeStan toolchain; identical calls reuse the cached
distribution.

```@example to-distribution
using AbstractPPL
using ADTypes: AutoForwardDiff
using BridgeStan, Distributions, DynamicPPL, LogDensityProblems
using ForwardDiff

const STAN = raw"""
parameters {
  real location;
  real<lower=0> scale;
  simplex[3] weights;
  ordered[2] cutpoints;
}
model {
  location ~ normal(0, 1);
  scale ~ lognormal(0, 1);
  weights ~ dirichlet(rep_vector(1, 3));
  cutpoints ~ normal(0, 2);
}
"""

@model function demo(stan, y)
    params ~ to_distribution(stan)
    return y ~ Normal(params[1], params[2])
end

ldf = LogDensityFunction(demo(STAN, 0.4))
u = zeros(LogDensityProblems.dimension(ldf))
prepared = AbstractPPL.prepare(
    AutoForwardDiff(), u -> LogDensityProblems.logdensity(ldf, u), u
)
logdensity, gradient = AbstractPPL.value_and_gradient!!(prepared, u)
```

The left-hand side receives Stan's constrained parameters as a flat vector in declaration
order. In this example, `params[1]` is `location` and `params[2]` is `scale`. The `data` and
`seed` keywords configure model construction; `stanc_args` and `make_args` configure the
build. BridgeStan supplies no sampler, so `InitFromPrior` uses DynamicPPL's uniform
initializer in unconstrained space.

## API

```@docs
to_distribution
to_submodel
DynamicPPL.prefix
```
