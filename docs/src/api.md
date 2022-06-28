# API

Part of the API of DynamicPPL is defined in the more lightweight interface package [AbstractPPL.jl](https://github.com/TuringLang/AbstractPPL.jl) and reexported here.

## Model

### Macros

A core component of DynamicPPL is the [`@model`](@ref) macro.
It can be used to define probabilistic models in an intuitive way by specifying random variables and their distributions with `~` statements.
These statements are rewritten by `@model` as calls of [internal functions](@ref model_internal) for sampling the variables and computing their log densities.

```@docs
@model
```

One can nest models and call another model inside the model function with [`@submodel`](@ref).

```@docs
@submodel
```

### Type

A [`Model`](@ref) can be created by calling the model function, as defined by [`@model`](@ref).

```@docs
Model
```

[`Model`](@ref)s are callable structs.

```@docs
Model()
```

Basic properties of a model can be accessed with [`getargnames`](@ref), [`getmissings`](@ref), and [`nameof`](@ref).

```@docs
nameof(::Model)
getargnames
getmissings
```

## Evaluation

With [`rand`](@ref) one can draw samples from the prior distribution of a [`Model`](@ref).

```@docs
rand
```

One can also evaluate the log prior, log likelihood, and log joint probability.

```@docs
logprior
loglikelihood
logjoint
```

### LogDensityProblems.jl interface

The [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface is also supported by simply wrapping a [`Model`](@ref) in a `DynamicPPL.LogDensityFunction`:

```@docs
DynamicPPL.LogDensityFunction
```

## Condition and decondition

A [`Model`](@ref) can be conditioned on a set of observations with [`AbstractPPL.condition`](@ref) or its alias [`|`](@ref).

```@docs
|(::Model, ::Any)
condition
DynamicPPL.conditioned
```

Similarly, one can specify with [`AbstractPPL.decondition`](@ref) that certain, or all, random variables are not observed.

```@docs
decondition
```

## Utilities

It is possible to manually increase (or decrease) the accumulated log density from within a model function.

```@docs
@addlogprob!
```

Return values of the model function for a collection of samples can be obtained with [`generated_quantities`](@ref).

```@docs
generated_quantities
```

For a chain of samples, one can compute the pointwise log-likelihoods of each observed random variable with [`pointwise_loglikelihoods`](@ref).

```@docs
pointwise_loglikelihoods
```

```@docs
WrappedDistribution
NamedDist
NoDist
```

## Testing Utilities

DynamicPPL provides several demo models and helpers for testing samplers in the `DynamicPPL.TestUtils` submodule.

```@docs
DynamicPPL.TestUtils.test_sampler
DynamicPPL.TestUtils.test_sampler_on_demo_models
DynamicPPL.TestUtils.test_sampler_continuous
DynamicPPL.TestUtils.marginal_mean_of_samples
```

```@docs
DynamicPPL.TestUtils.DEMO_MODELS
```

For every demo model, one can define the true log prior, log likelihood, and log joint probabilities.

```@docs
DynamicPPL.TestUtils.logprior_true
DynamicPPL.TestUtils.loglikelihood_true
DynamicPPL.TestUtils.logjoint_true
```

And in the case where the model includes constrained variables, it can also be useful to define

```@docs
DynamicPPL.TestUtils.logprior_true_with_logabsdet_jacobian
DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian
```

Finally, the following methods can also be of use:

```@docs
DynamicPPL.TestUtils.varnames
DynamicPPL.TestUtils.posterior_mean
DynamicPPL.TestUtils.setup_varinfos
DynamicPPL.TestUtils.update_values!!
DynamicPPL.TestUtils.test_values
```

## Advanced

### Variable names

Names and possibly nested indices of variables are described with `AbstractPPL.VarName`.
They can be defined with `AbstractPPL.@varname`.
Please see the documentation of [AbstractPPL.jl](https://github.com/TuringLang/AbstractPPL.jl) for further information.

### Data Structures of Variables

DynamicPPL provides different data structures for samples from the model and their log density.
All of them are subtypes of [`AbstractVarInfo`](@ref).

```@docs
AbstractVarInfo
```

### Common API

#### Accumulation of log-probabilities

```@docs
getlogp
setlogp!!
acclogp!!
resetlogp!!
```

#### Variables and their realizations

```@docs
keys
getindex
DynamicPPL.getindex_raw
push!!
empty!!
isempty
```

```@docs
values_as
```

#### Transformations

```@docs
DynamicPPL.AbstractTransformation
DynamicPPL.NoTransformation
DynamicPPL.DynamicTransformation
DynamicPPL.StaticTransformation
```

```@docs
DynamicPPL.istrans
DynamicPPL.settrans!!
DynamicPPL.transformation
DynamicPPL.link!!
DynamicPPL.invlink!!
DynamicPPL.default_transformation
DynamicPPL.maybe_invlink_before_eval!!
``` 

#### Utils

```@docs
DynamicPPL.unflatten
DynamicPPL.tonamedtuple
```

#### `SimpleVarInfo`

```@docs
SimpleVarInfo
```

#### `VarInfo`

Another data structure is [`VarInfo`](@ref).

```@docs
VarInfo
TypedVarInfo
```

One main characteristic of [`VarInfo`](@ref) is that samples are stored in a linearized form.

```@docs
link!
invlink!
```

```@docs
set_flag!
unset_flag!
is_flagged
```

For Gibbs sampling the following functions were added.

```@docs
setgid!
updategid!
```

The following functions were used for sequential Monte Carlo methods.

```@docs
get_num_produce
set_num_produce!
increment_num_produce!
reset_num_produce!
setorder!
set_retained_vns_del_by_spl!
```

```@docs
Base.empty!
```

### Evaluation Contexts

Internally, both sampling and evaluation of log densities are performed with [`AbstractPPL.evaluate!!`](@ref).

```@docs
AbstractPPL.evaluate!!
```

The behaviour of a model execution can be changed with evaluation contexts that are passed as additional argument to the model function.
Contexts are subtypes of `AbstractPPL.AbstractContext`.

```@docs
SamplingContext
DefaultContext
LikelihoodContext
PriorContext
MiniBatchContext
PrefixContext
```

### Samplers

In DynamicPPL two samplers are defined that are used to initialize unobserved random variables:
[`SampleFromPrior`](@ref) which samples from the prior distribution, and [`SampleFromUniform`](@ref) which samples from a uniform distribution.

```@docs
SampleFromPrior
SampleFromUniform
```

Additionally, a generic sampler for inference is implemented.

```@docs
Sampler
```

The default implementation of [`Sampler`](@ref) uses the following unexported functions.

```@docs
DynamicPPL.initialstep
DynamicPPL.loadstate
DynamicPPL.initialsampler
```

### [Model-Internal Functions](@id model_internal)

```@docs
tilde_assume
dot_tilde_assume
```

```@docs
tilde_observe
dot_tilde_observe
```

