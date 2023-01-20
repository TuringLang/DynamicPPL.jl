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
    σ = 1
    x ~ MvNormal(Fill(μ, n), σ * I)
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
We first instantiate a model and sample from the prior:

```@example probinterface
model = gdemo(length(dataset)) | (x=dataset,)

Random.seed!(124)
sample_nt = rand(model)
```

For models with many variables `rand(model)` can be prohibitively slow since it returns a `NamedTuple` of samples from the prior distribution of the unconditioned variables.
Alternatively, we can work with samples of type `DataStructures.OrderedDict`:

```@example probinterface
using DataStructures

Random.seed!(124)
sample_dict = rand(OrderedDict, model)
```

Here we work with sample in the format of `NamedTuple`.
The prior probability and the likelihood of a set of samples (in the format `NamedTuple`) can be calculated with the following helper functions:

```julia
# Here we build two loosen/temporary helper functions which:
    # accept: a model and a vector of named tuples (therefore a single NamedTuple needs to be square bracketed to be made a vector) as arguments, and 
    # output: a vector of log posteriors.
function logjoint(model_instance, nt_arr)
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] =
            Distributions.loglikelihood(model_instance, nt_arr[param_idx]) +
            DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
logprior(model, [x1])

function logprior(model_instance, nt_arr)
    lls = Array{Float64}(undef, size(nt_arr, 1)) # initialize a matrix to store the evaluated log posterior
    for param_idx in 1:size(nt_arr, 1)
        # Compute and store.
        lls[param_idx] = DynamicPPL.logprior(model_instance, nt_arr[param_idx])
    end
    return lls
end
logjoint(model, [sample_nt])
```

We can build similar interfaces for the two temporary functions to accept sample in the format of `OrderedDict`.

```@example probinterface
logjoint(model, [sample_nt])[1] ≈
loglikelihood(model, sample_nt) + logprior(model, [sample_nt])[1]
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
    loss = zero(logjoint(model, [rand(rng, model)]))

    for (train, validation) in MLUtils.kfolds(dataset, nfolds)
        # First, we train the model on the training set, i.e., we obtain samples from the posterior.
        # For normally-distributed data, the posterior can be computed in closed form.
        # For general models, however, typically samples will be generated using MCMC with Turing.
        posterior = Normal(mean(train), 1)
        samples = rand(rng, posterior, nsamples)

        # Evaluation on the validation set.
        validation_model = gdemo(length(validation)) | (x=validation,)
        loss += sum(samples) do sample
            logjoint(validation_model, [(μ=sample,)])
        end
    end

    return loss
end

cross_val(dataset)
```

[^1]: See [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) for a faster and more accurate implementation of cross-validation than the one provided here.
