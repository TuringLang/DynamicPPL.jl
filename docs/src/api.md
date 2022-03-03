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
NamedDist
```

## Testing Utilities

DynamicPPL provides several demo models and helpers for testing samplers in the `DynamicPPL.TestUtils` submodule.

```@docs
DynamicPPL.TestUtils.test_sampler_demo_models
DynamicPPL.TestUtils.test_sampler_continuous
```

For every demo model, one can define the true log prior, log likelihood, and log joint probabilities.

```@docs
DynamicPPL.TestUtils.logprior_true
DynamicPPL.TestUtils.loglikelihood_true
DynamicPPL.TestUtils.logjoint_true
```

## Advanced

### Variable names

Names and possibly nested indices of variables are described with `AbstractPPL.VarName`.
They can be defined with `AbstractPPL.@varname`.
Please see the documentation of AbstractPPL for further information.

### Data Structures of Variables

DynamicPPL provides different data structures for samples from the model and their log density.
All of them are subtypes of [`AbstractVarInfo`](@ref).

```@docs
AbstractVarInfo
```

### Common API

```@docs
getlogp
setlogp!!
acclogp!!
resetlogp!!
```

```@docs
getindex
push!!
empty!!
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
tonamedtuple
link!
invlink!
istrans
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

