# The Probability Interface

The easiest way to manipulate and query DynamicPPL models is via the DynamicPPL probability
interface.

Let's use a simple model of normally-distributed data as an example.
```julia
using DynamicPPL, Distributions, Random
rng = Xoshiro(1776);  # Set seed for reproducibility
n = 100

DynamicPPL.@model function gdemo(n)
   μ ~ Normal(0, 1)
   σ ~ Exponential(1)
   x = Vector{Float64}(undef, n)
   x .~ Normal(μ, σ)
   return nothing
end

dataset = randn(rng, n)
```


## Conditioning and Deconditioning

Bayesian models can be transformed with two main operations, conditioning and deconditioning (also known as marginalization). Conditioning takes a variable and fixes its value as known. We do this by passing a model and a named tuple of conditioned variables to `|`:
```julia
model = gdemo(n) | (x=dataset, μ=0, σ=1)
```

This operation can be reversed by applying `decondition`:
```julia
decondition(model)
```

We can also decondition only some of the variables:
```julia
decondition(model, :μ, :σ)
```


## Probabilities and Densities

We often want to calculate the (unnormalized) probability density for an event. This
probability might be a prior, a likelihood, or a posterior (joint) density. DynamicPPL
provides convenient functions for this.
For example, if we wanted to calculate the probability of a draw from the prior:
```julia
model = gdemo(n) | (x=dataset,)
x1 = rand(rng, model)
logjoint(model, x1)
```

For convenience, we provide the functions `loglikelihood` and `logjoint` to calculate probabilities for a named tuple, given a model:
```julia
logjoint(model, x1) ≈ loglikelihood(model, x1) + logprior(model, x1)
```


## Example: Cross-validation

To give an example of the probability interface in use, we can write a function to test the performance of our model using cross-validation. In cross-validation, we split the dataset into several equal parts. Then, we choose one of these sets to serve as the test set. Here, we measure fit using the cross entropy (Bayes loss). (See [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) for a faster and more accurate implementation of cross-validation.)
```julia
using MLUtils, Turing
training_loss = zero(logjoint(model, x1))
for (train, test) in kfolds(dataset, 5)
   # First, we train the model on the training set using Turing.jl
   trained_posterior = sample(
      rng,
      gdemo(length(train)) | (x = train,),
      NUTS(),
      1000;
      chain_type=StructArray
   )
   # Extract posterior samples
   samples = map.(only∘first, getproperty.(trained_posterior, :θ))
   training_loss += sum(samples) do sample
      model = gdemo(length(test))
      logjoint(model | (x = test,), sample)
   end
end
training_loss
```
