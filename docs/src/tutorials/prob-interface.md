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

## Probabilities and Densities

We often want to calculate the (unnormalized) probability density for an event.
This probability might be a prior, a likelihood, or a posterior (joint) density.
DynamicPPL provides convenient functions for this.
For example, if we wanted to calculate the probability of a draw from the prior:
```@example probinterface
model = gdemo(length(dataset)) | (x=dataset,)
x1 = rand(model)
logjoint(model, x1)
```

For convenience, we provide the functions `loglikelihood` and `logjoint` to calculate probabilities for a named tuple, given a model:
```@example probinterface
@assert logjoint(model, x1) ≈ loglikelihood(model, x1) + logprior(model, x1)
```

## Example: Cross-validation

To give an example of the probability interface in use, we can use it to estimate the performance of our model using cross-validation. In cross-validation, we split the dataset into several equal parts. Then, we choose one of these sets to serve as the validation set. Here, we measure fit using the cross entropy (Bayes loss).¹
``` @example probinterface
function cross_val(model, dataset)
   training_loss = zero(logjoint(model, rand(model)))

   # Partition our dataset into 5 folds with 20 observations:
   test_folds = collect(Iterators.partition(dataset, 20))
   train_folds = setdiff.((dataset,), test_folds)

   for (train, test) in zip(train_folds, test_folds)
      # First, we train the model on the training set.
      # For normally-distributed data, the posterior can be solved in closed form:
      posterior = Normal(mean(train), 1)
      # Sample from the posterior
      samples = NamedTuple{(:μ,)}.(rand(posterior, 1000))
      # Test
      testing_model = gdemo(length(test)) | (x = test,)
      training_loss += sum(samples) do sample
         logjoint(testing_model, sample)
      end
   end
   return training_loss
end
cross_val(model, dataset)
```

¹See [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) for a faster and more accurate implementation of cross-validation than the one provided here.
