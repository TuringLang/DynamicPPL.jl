# The Probability Interface

The easiest way to manipulate and query DynamicPPL models is via the DynamicPPL probability
interface.

Let's use a simple model of normally-distributed data as an example.
```@example probinterface
using DynamicPPL
using Distributions
using FillArrays
using LinearAlgebra
using Random

Random.seed!(1776) # Set seed for reproducibility

@model function gdemo(n)
   μ ~ Normal(0, 1)
   x ~ MvNormal(Fill(μ, n), I)
   return nothing
end
nothing # hide
```

We generate some data using `μ = 0` and `σ = 1`:

```@example probinterface
dataset = randn(100)
nothing # hide
```

## Conditioning and Deconditioning

Bayesian models can be transformed with two main operations, conditioning and deconditioning (also known as marginalization).
Conditioning takes a variable and fixes its value as known.
We do this by passing a model and a named tuple of conditioned variables to `|`:
```@example probinterface
model = gdemo(length(dataset)) | (x=dataset, μ=0, σ=1)
nothing # hide
```

This operation can be reversed by applying `decondition`:
```@example probinterface
decondition(model)
nothing # hide
```

We can also decondition only some of the variables:
```@example probinterface
decondition(model, :μ)
nothing # hide
```

Sometimes it is helpful to define convenience functions for conditioning on some variable(s).
For instance, in this example we might want to define a version of `gdemo` that conditions on some observations of `x`:

```@example probinterface
gdemo(x::AbstractVector{<:Real}) = gdemo(length(x)) | (; x)
```

## Probabilities and Densities

We often want to calculate the (unnormalized) probability density for an event.
This probability might be a prior, a likelihood, or a posterior (joint) density.
DynamicPPL provides convenient functions for this.
For example, if we wanted to calculate the probability of a draw from the prior:
```@example probinterface
model = gdemo(dataset)
x1 = rand(model)
logjoint(model, x1)
```

For convenience, we provide the functions `loglikelihood` and `logjoint` to calculate probabilities for a named tuple, given a model:
```@example probinterface
@assert logjoint(model, x1) ≈ loglikelihood(model, x1) + logprior(model, x1)
```

## Example: Cross-validation

To give an example of the probability interface in use, we can use it to estimate the performance of our model using cross-validation.
In cross-validation, we split the dataset into several equal parts.
Then, we choose one of these sets to serve as the validation set.
Here, we measure fit using the cross entropy (Bayes loss).[^1]
```@example probinterface
using MLUtils

function cross_val(
    dataset::AbstractVector{<:Real};
    nfolds::Int=5,
    nsamples::Int=1_000,
    rng::Random.AbstractRNG=Random.default_rng(),
)
   # Initialize `loss` in a way such that the loop below does not change its type
   model = gdemo([first(dataset)])
   loss = zero(logjoint(model, rand(rng, model)))

   for (train, validation) in kfolds(dataset, nfolds)
      # First, we train the model on the training set, i.e., we obtain samples from the posterior.
      # For normally-distributed data, the posterior can be computed in closed form.
      # For general models, however, typically samples will be generated using MCMC with Turing.
      posterior = Normal(mean(train), 1)
      samples = rand(rng, posterior, nsamples)

      # Evaluation on the validation set.
      validation_model = gdemo(validation)
      loss += sum(samples) do sample
         logjoint(validation_model, (μ = sample,))
      end
   end

   return loss
end

cross_val(dataset)
```

[^1]: See [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) for a faster and more accurate implementation of cross-validation than the one provided here.
