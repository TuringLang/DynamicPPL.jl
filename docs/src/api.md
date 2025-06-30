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

The context of a model can be set using [`contextualize`](@ref):

```@docs
contextualize
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

The [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface is also supported by wrapping a [`Model`](@ref) in a `DynamicPPL.LogDensityFunction`.

```@docs
LogDensityFunction
```

## Condition and decondition

A [`Model`](@ref) can be conditioned on a set of observations with [`AbstractPPL.condition`](@ref) or its alias [`|`](@ref).

```@docs
|(::Model, ::Union{Tuple,NamedTuple,AbstractDict{<:VarName}})
condition
DynamicPPL.conditioned
```

Similarly, one can specify with [`AbstractPPL.decondition`](@ref) that certain, or all, random variables are not observed.

```@docs
decondition
```

## Fixing and unfixing

We can also _fix_ a collection of variables in a [`Model`](@ref) to certain values using [`DynamicPPL.fix`](@ref).

This is quite similar to the aforementioned [`condition`](@ref) and its siblings,
but they are indeed different operations:

  - `condition`ed variables are considered to be _observations_, and are thus
    included in the computation [`logjoint`](@ref) and [`loglikelihood`](@ref),
    but not in [`logprior`](@ref).
  - `fix`ed variables are considered to be _constant_, and are thus not included
    in any log-probability computations.

The differences are more clearly spelled out in the docstring of [`DynamicPPL.fix`](@ref) below.

```@docs
DynamicPPL.fix
DynamicPPL.fixed
```

The difference between [`DynamicPPL.fix`](@ref) and [`DynamicPPL.condition`](@ref) is described in the docstring of [`DynamicPPL.fix`](@ref) above.

Similarly, we can revert this with [`DynamicPPL.unfix`](@ref), i.e. return the variables to their original meaning:

```@docs
DynamicPPL.unfix
```

## Predicting

DynamicPPL provides functionality for generating samples from the posterior predictive distribution through the `predict` function. This allows you to use posterior parameter samples to generate predictions for unobserved data points.

The `predict` function has two main methods:

 1. For `AbstractVector{<:AbstractVarInfo}` - useful when you have a collection of `VarInfo` objects representing posterior samples.
 2. For `MCMCChains.Chains` (only available when `MCMCChains.jl` is loaded) - useful when you have posterior samples in the form of an `MCMCChains.Chains` object.

```@docs
predict
```

### Basic Usage

The typical workflow for posterior prediction involves:

 1. Fitting a model to observed data to obtain posterior samples
 2. Creating a new model instance with some variables marked as missing (unobserved)
 3. Using `predict` to generate samples for these missing variables based on the posterior parameter samples

When using `predict` with `MCMCChains.Chains`, you can control which variables are included in the output with the `include_all` parameter:

  - `include_all=false` (default): Include only newly predicted variables
  - `include_all=true`: Include both parameters from the original chain and predicted variables

## Models within models

One can include models and call another model inside the model function with `left ~ to_submodel(model)`.

```@docs
to_submodel
```

Note that a `[to_submodel](@ref)` is only sampleable; one cannot compute `logpdf` for its realizations.

In the context of including models within models, it's also useful to prefix the variables in sub-models to avoid variable names clashing:

```@docs
DynamicPPL.prefix
```

Under the hood, [`to_submodel`](@ref) makes use of the following method to indicate that the model it's wrapping is a model over its return-values rather than something else

```@docs
returned(::Model)
```

## Utilities

It is possible to manually increase (or decrease) the accumulated log likelihood or prior from within a model function.

```@docs
@addlogprob!
```

Return values of the model function for a collection of samples can be obtained with [`returned(model, chain)`](@ref).

```@docs
returned(::DynamicPPL.Model, ::NamedTuple)
```

For a chain of samples, one can compute the pointwise log-likelihoods of each observed random variable with [`pointwise_loglikelihoods`](@ref). Similarly, the log-densities of the priors using
[`pointwise_prior_logdensities`](@ref) or both, i.e. all variables, using
[`pointwise_logdensities`](@ref).

```@docs
pointwise_logdensities
pointwise_loglikelihoods
pointwise_prior_logdensities
```

For converting a chain into a format that can more easily be fed into a `Model` again, for example using `condition`, you can use [`value_iterator_from_chain`](@ref).

```@docs
value_iterator_from_chain

```

Sometimes it can be useful to extract the priors of a model. This is the possible using [`extract_priors`](@ref).

```@docs
extract_priors
```

Safe extraction of values from a given [`AbstractVarInfo`](@ref) as they are seen in the model can be done using [`values_as_in_model`](@ref).

```@docs
values_as_in_model
```

```@docs
NamedDist
```

## AD testing and benchmarking utilities

To test and/or benchmark the performance of an AD backend on a model, DynamicPPL provides the following utilities:

```@docs
DynamicPPL.TestUtils.AD.run_ad
```

THe default test setting is to compare against ForwardDiff.
You can have more fine-grained control over how to test the AD backend using the following types:

```@docs
DynamicPPL.TestUtils.AD.AbstractADCorrectnessTestSetting
DynamicPPL.TestUtils.AD.WithBackend
DynamicPPL.TestUtils.AD.WithExpectedResult
DynamicPPL.TestUtils.AD.NoTest
```

These are returned / thrown by the `run_ad` function:

```@docs
DynamicPPL.TestUtils.AD.ADResult
DynamicPPL.TestUtils.AD.ADIncorrectException
```

## Demo models

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

## Debugging Utilities

DynamicPPL provides a few methods for checking validity of a model-definition.

```@docs
check_model
check_model_and_trace
```

And some which might be useful to determine certain properties of the model based on the debug trace.

```@docs
DynamicPPL.has_static_constraints
```

For determining whether one might have type instabilities in the model, the following can be useful

```@docs
DynamicPPL.DebugUtils.model_warntype
DynamicPPL.DebugUtils.model_typed
```

Interally, the type-checking methods make use of the following method for construction of the call with the argument types:

```@docs
DynamicPPL.DebugUtils.gen_evaluator_call_with_types
```

## Advanced

### Variable names

Names and possibly nested indices of variables are described with `AbstractPPL.VarName`.
They can be defined with `AbstractPPL.@varname`.
Please see the documentation of [AbstractPPL.jl](https://github.com/TuringLang/AbstractPPL.jl) for further information.

### Data Structures of Variables

DynamicPPL provides different data structures used in for storing samples and accumulation of the log-probabilities, all of which are subtypes of [`AbstractVarInfo`](@ref).

```@docs
AbstractVarInfo
```

But exactly how a [`AbstractVarInfo`](@ref) stores this information can vary.

#### `VarInfo`

```@docs
VarInfo
```

```@docs
DynamicPPL.untyped_varinfo
DynamicPPL.typed_varinfo
DynamicPPL.untyped_vector_varinfo
DynamicPPL.typed_vector_varinfo
```

One main characteristic of [`VarInfo`](@ref) is that samples are transformed to unconstrained Euclidean space and stored in a linearized form, as described in the [main Turing documentation](https://turinglang.org/docs/developers/transforms/dynamicppl/).
The [Transformations section below](#Transformations) describes the methods used for this.
In the specific case of `VarInfo`, it keeps track of whether samples have been transformed by setting flags on them, using the following functions.

```@docs
set_flag!
unset_flag!
is_flagged
```

The following functions were used for sequential Monte Carlo methods.

```@docs
get_num_produce
set_num_produce!!
increment_num_produce!!
reset_num_produce!!
setorder!
set_retained_vns_del!
```

```@docs
Base.empty!
```

#### `SimpleVarInfo`

```@docs
SimpleVarInfo
```

### Accumulators

The subtypes of [`AbstractVarInfo`](@ref) store the cumulative log prior and log likelihood, and sometimes other variables that change during executing, in what are called accumulators.

```@docs
AbstractAccumulator
```

DynamicPPL provides the following default accumulators.

```@docs
LogPriorAccumulator
LogLikelihoodAccumulator
NumProduceAccumulator
```

### Common API

#### Accumulation of log-probabilities

```@docs
getlogp
setlogp!!
acclogp!!
getlogjoint
getlogprior
setlogprior!!
acclogprior!!
getloglikelihood
setloglikelihood!!
accloglikelihood!!
resetlogp!!
```

#### Variables and their realizations

```@docs
keys
getindex
push!!
empty!!
isempty
DynamicPPL.getindex_internal
DynamicPPL.setindex_internal!
DynamicPPL.update_internal!
DynamicPPL.insert_internal!
DynamicPPL.length_internal
DynamicPPL.reset!
DynamicPPL.update!
DynamicPPL.insert!
DynamicPPL.loosen_types!!
DynamicPPL.tighten_types
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
DynamicPPL.link
DynamicPPL.invlink
DynamicPPL.link!!
DynamicPPL.invlink!!
DynamicPPL.default_transformation
DynamicPPL.link_transform
DynamicPPL.invlink_transform
DynamicPPL.maybe_invlink_before_eval!!
```

#### Utils

```@docs
Base.merge(::AbstractVarInfo)
DynamicPPL.subset
DynamicPPL.unflatten
DynamicPPL.varname_leaves
DynamicPPL.varname_and_value_leaves
```

### Evaluation Contexts

Internally, model evaluation is performed with [`AbstractPPL.evaluate!!`](@ref).

```@docs
AbstractPPL.evaluate!!
```

This method mutates the `varinfo` used for execution.
By default, it does not perform any actual sampling: it only evaluates the model using the values of the variables that are already in the `varinfo`.
To perform sampling, you can either wrap `model.context` in a `SamplingContext`, or use this convenience method:

```@docs
DynamicPPL.evaluate_and_sample!!
```

The behaviour of a model execution can be changed with evaluation contexts, which are a field of the model.
Contexts are subtypes of `AbstractPPL.AbstractContext`.

```@docs
SamplingContext
DefaultContext
PrefixContext
ConditionContext
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

Finally, to specify which varinfo type a [`Sampler`](@ref) should use for a given [`Model`](@ref), this is specified by [`DynamicPPL.default_varinfo`](@ref) and can thus be overloaded for each  `model`-`sampler` combination. This can be useful in cases where one has explicit knowledge that one type of varinfo will be more performant for the given `model` and `sampler`.

```@docs
DynamicPPL.default_varinfo
```

There is also the _experimental_ [`DynamicPPL.Experimental.determine_suitable_varinfo`](@ref), which uses static checking via [JET.jl](https://github.com/aviatesk/JET.jl) to determine whether one should use [`DynamicPPL.typed_varinfo`](@ref) or [`DynamicPPL.untyped_varinfo`](@ref), depending on which supports the model:

```@docs
DynamicPPL.Experimental.determine_suitable_varinfo
DynamicPPL.Experimental.is_suitable_varinfo
```

### [Model-Internal Functions](@id model_internal)

```@docs
tilde_assume
```
