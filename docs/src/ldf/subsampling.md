# Independent subsampling

`subsample` constructs a `LogDensityFunction` for one fixed minibatch and scales its
likelihood by `N / n`, where `N` is the dataset size and `n` is the minibatch size. This is
useful when implementing stochastic objectives in inference packages such as AdvancedVI.

Write the observation with `independent_distribution` and supply its complete dataset by
conditioning, rather than as a model argument:

```@example subsampling
using Distributions, DynamicPPL, ForwardDiff, LogDensityProblems

@model function location_model(scales)
    μ ~ Normal(0, 1)
    return x ~ independent_distribution(i -> Normal(μ, scales[i]), length(scales))
end

scales = [0.5, 1.0, 1.5, 2.0]
data = [-1.0, 0.0, 1.0, 2.0]
model = location_model(scales) | (x=data,)

select_first_and_third(rng, N) = [1, 3]
ldf = subsample(model, select_first_and_third, length(data))

θ = [0.25]
logdensity = LogDensityProblems.logdensity(ldf, θ)
gradient = ForwardDiff.gradient(p -> LogDensityProblems.logdensity(ldf, p), θ)

(; logdensity, gradient)
```

`select_first_and_third` is deterministic so that the example is reproducible. It accepts
the required RNG and dataset-size arguments but does not use them.

The distribution factory gives each observation a different known scale. Subsampling
constructs only `O(n)` distributions.

## Scaling and differentiation

In model coordinates, the full log density for latent parameters ``\theta`` is

```math
\ell(\theta) = \log p(\theta) + \sum_{i=1}^{N} \log p(x_i \mid \theta).
```

For returned indices ``I_1, \ldots, I_n``, including any duplicates, `subsample`
evaluates

```math
\widehat{\ell}(\theta) = \log p(\theta) + \frac{N}{n} \sum_{j=1}^{n} \log p(x_{I_j} \mid \theta).
```

The prior is included once; only the likelihood is scaled. Automatic differentiation
therefore computes

```math
\nabla \widehat{\ell}(\theta) = \nabla \log p(\theta) + \frac{N}{n} \sum_{j=1}^{n} \nabla \log p(x_{I_j} \mid \theta).
```

This estimator is unbiased under the resampling condition below. With `LinkAll()`,
``\theta`` instead denotes unconstrained coordinates, and DynamicPPL includes the
change-of-variables term and its gradient.

The parameter vector uses the same latent names, ranges, transforms, and dimensions as the
full model. The default transform strategy is `UnlinkAll()`; pass
`transform_strategy=LinkAll()` when an inference algorithm requires unconstrained
parameters.

The integer form samples uniformly without replacement using exactly `n` caller-RNG draws:

```julia
ldf = subsample(model, n, N)
```

A custom resampler is called once during construction and returns integer indices:

```julia
ldf = subsample(model, resampler, N)
```

An inference package that selects batches externally can pass the indices directly:

```julia
ldf = subsample(model, indices, N)
```

The returned batch remains fixed across log-density and gradient evaluations. Construct a
new `ldf` when the inference objective should draw another minibatch. If ``C_i`` is the
multiplicity of element ``i`` in the returned indices, unbiased scaling requires
``\mathbb{E}[C_i / n] = 1 / N``. For fixed ``n``, this reduces to
``\mathbb{E}[C_i] = n / N``.

Construction neither scores nor reads the full conditioned dataset; a shape-only structural
probe fails if the model attempts to inspect it. DynamicPPL copies only the selected
observations, so their likelihood requires `O(n)` work. Model arguments are retained
unchanged and may therefore still contain dataset-sized covariates. The model must have
exactly one conditioned observation using `independent_distribution`, with no later
probability-bearing statement. Additional likelihood contributions through `@addlogprob!`
are rejected.
