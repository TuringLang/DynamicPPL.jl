# `to_distribution` and `to_submodel`

Both functions embed one probabilistic program in another. [`to_distribution`](@ref)
converts a supported model representation into a distribution over its latent variables.
[`to_submodel`](@ref) instead evaluates a DynamicPPL model and yields its return value while
recording its latent variables separately.

Currently, `to_distribution` supports only Stan through BridgeStan; `to_submodel` accepts only
DynamicPPL models.

Each DynamicPPL submodel contributes its tilde address: `a ~ to_submodel(child)` records
its variable `x` as `a.x`. `to_submodel(child, true)` remains accepted;
`to_submodel(child, false)` throws `ArgumentError`. To migrate a manually prefixed child,
replace `unused ~ to_submodel(prefix(child, @varname(a)), false)` with
`a ~ to_submodel(child)`. Update parameter and binding addresses when adding a child name.
An explicit prefix composes with that name: `a ~ to_submodel(prefix(child, @varname(b)))`
records `a.b.x`.

Submodels take observations through `condition`, not arguments. If `y` is observed,
replace `a ~ to_submodel(inner(y, x))` with
`a ~ to_submodel(condition(inner(missing, x), (y=y,)))`.
Using `fix` instead would omit the likelihood term for `y`.
Argument observations remain supported on the model called directly by the user.

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
