# The Probability Interface

The easiest way to manipulate and query DynamicPPL models is via the DynamicPPL probability interface.

Let's use a simple model of normally-distributed data as an example.
```@example probinterface
using DynamicPPL
using Distributions
using FillArrays

using LinearAlgebra
using Random

@model function gdemo(n)
    μ ~ Normal(0, 1)
    x ~ MvNormal(Fill(μ, n), I)
    return nothing
end
nothing # hide
```

We generate some data using `μ = 0` and `σ = 1`:

```@example probinterface
Random.seed!(1776)
dataset = randn(100)
nothing # hide
```

## Conditioning and Deconditioning

Bayesian models can be transformed with two main operations, conditioning and deconditioning (also known as marginalization).
Conditioning takes a variable and fixes its value as known.
We do this by passing a model and a collection of conditioned variables to [`|`](@ref) or its alias [`condition`](@ref):
```@example probinterface
model = gdemo(length(dataset)) | (x=dataset, μ=0, σ=1)
nothing # hide
```

This operation can be reversed by applying [`decondition`](@ref):
```@example probinterface
decondition(model)
nothing # hide
```

We can also decondition only some of the variables:
```@example probinterface
decondition(model, :μ)
nothing # hide
```

!!! note
    Sometimes it is helpful to define convenience functions for conditioning on some variable(s).
    For instance, in this example we might want to define a version of `gdemo` that conditions on some observations of `x`:

    ```julia
    gdemo(x::AbstractVector{<:Real}) = gdemo(length(x)) | (; x)
    ```

    For illustrative purposes, however, we do not use this function in the examples below.

## Probabilities and Densities

We often want to calculate the (unnormalized) probability density for an event.
This probability might be a prior, a likelihood, or a posterior (joint) density.
DynamicPPL provides convenient functions for this.
For example, we can calculate the joint probability of a set of samples (here drawn from the prior) with [`logjoint`](@ref):

```@example probinterface
model = gdemo(length(dataset)) | (x=dataset,)

Random.seed!(124)
sample = rand(model)
logjoint(model, sample)
```

For models with many variables `rand(model)` can be prohibitively slow since it returns a `NamedTuple` of samples from the prior distribution of the unconditioned variables.
We recommend working with samples of type `DataStructures.DOrderedDict` in this case:
```@example probinterface
using DataStructures

Random.seed!(124)
sample_dict = rand(OrderedDict, model)
logjoint(model, sample_dict)
```

The prior probability and the likelihood of a set of samples can be calculated with the functions [`loglikelihood`](@ref) and [`logjoint`](@ref), respectively:

```@example probinterface
logjoint(model, sample) ≈ loglikelihood(model, sample) + logprior(model, sample)
```

```@example probinterface
logjoint(model, sample_dict) ≈ loglikelihood(model, sample_dict) + logprior(model, sample_dict)
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
    model = gdemo(1) | (x=[first(dataset)],)
    loss = zero(logjoint(model, rand(rng, model)))

    for (train, validation) in kfolds(dataset, nfolds)
        # First, we train the model on the training set, i.e., we obtain samples from the posterior.
        # For normally-distributed data, the posterior can be computed in closed form.
        # For general models, however, typically samples will be generated using MCMC with Turing.
        posterior = Normal(mean(train), 1)
        samples = rand(rng, posterior, nsamples)

        # Evaluation on the validation set.
        validation_model = gdemo(length(validation)) | (x=validation,)
        loss += sum(samples) do sample
            logjoint(validation_model, (μ=sample,))
        end
    end

    return loss
end

cross_val(dataset)
```

[^1]: See [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) for a faster and more accurate implementation of cross-validation than the one provided here.
