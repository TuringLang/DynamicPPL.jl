# API

Part of the API of DynamicPPL is defined in the more lightweight interface package [AbstractPPL.jl](https://github.com/TuringLang/AbstractPPL.jl) and reexported here.

## Model

### Macros

A core component of DynamicPPL is the [`@model`](@ref) macro.
It can be used to define probabilistic models in an intuitive way by specifying random variables and their distributions with `~` statements.
These statements are rewritten by `@model` as calls of internal functions for sampling the variables and computing their log densities.

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

Some models require threadsafe evaluation (see [the Turing docs](https://turinglang.org/docs/usage/threadsafe-evaluation/) for more information on when this is necessary).
If this is the case, one must enable threadsafe evaluation for a model:

```@docs
setthreadsafe
requires_threadsafe
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
LogDensityFunction(::Model, ::Any, ::VarNamedTuple)
LogDensityFunction
RangeAndTransform
get_range_and_transform
get_all_ranges_and_transforms
get_logdensity_callable
get_input_vector_type
get_sample_input_vector
subsample
```

Internally, this is accomplished using [`init!!`](@ref) with [`VarInfo`](@ref).

```@docs
to_vector_params
```

You can also draw vectorised samples from a `LogDensityFunction` via

```@docs
Base.rand(::Random.AbstractRNG, ::LogDensityFunction, ::AbstractInitStrategy)
```

(although note that this is a limited interface as it only generates parameters; please see [the documentation](@ref ldf-model) for more information on how to combine `LogDensityFunction` with `init!!` more generally.)

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
 2. Creating a new model instance with the prediction sites left unconditioned
 3. Using `predict` to sample these sites based on the posterior parameter samples

When using `predict` with `MCMCChains.Chains`, you can control which variables are included in the output with the `include_all` parameter:

  - `include_all=false` (default): Include only newly predicted variables
  - `include_all=true`: Include both parameters from the original chain and predicted variables

## Marginalisation

DynamicPPL provides the `marginalize` function to marginalise out variables from a model.
This requires `MarginalLogDensities.jl` to be loaded in your environment.

```@docs
marginalize
```

A `MarginalLogDensity` object acts as a function which maps non-marginalised parameter values to a marginal log-probability.
To retrieve a VarInfo object from it, you can use [`InitFromVector`](@ref).

## Utilities

It is possible to manually increase (or decrease) the accumulated log likelihood or prior from within a model function.

```@docs
@addlogprob!
```

Return values of the model function can be obtained with [`returned(model, sample)`](@ref), where `sample` is either a `MCMCChains.Chains` object (which represents a collection of samples), or a single sample represented as a `NamedTuple` or a dictionary of VarNames.

```@docs
returned(::DynamicPPL.Model, ::MCMCChains.Chains)
returned(::DynamicPPL.Model, ::Union{NamedTuple,AbstractDict{<:VarName}})
```

For a chain of samples, one can compute the pointwise log-likelihoods of each observed random variable with [`pointwise_loglikelihoods`](@ref). Similarly, the log-densities of the priors using
[`pointwise_prior_logdensities`](@ref) or both, i.e. all variables, using
[`pointwise_logdensities`](@ref).

```@docs
pointwise_logdensities
pointwise_loglikelihoods
pointwise_prior_logdensities
```

Sometimes it can be useful to extract the priors of a model. This is the possible using [`extract_priors`](@ref).

```@docs
extract_priors
```

## Distribution wrappers

```@docs
filldist
arraydist
independent_distribution
```

## Distributions

These distributions are defined here, but not in Distributions.jl.

```@docs
Flat
FlatPos
BinomialLogit
OrderedLogistic
LogPoisson
```

## AD testing and benchmarking utilities

To test and/or benchmark the performance of an AD backend on a model, DynamicPPL provides the following utilities:

```@docs
DynamicPPL.TestUtils.AD.run_ad
```

The default test setting is to compare against ForwardDiff.
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

DynamicPPL provides several demo models in the `DynamicPPL.TestUtils` submodule.

```@docs
DynamicPPL.TestUtils.DEMO_MODELS
DynamicPPL.TestUtils.ALL_MODELS
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
DynamicPPL.TestUtils.test_values
```

## Debugging Utilities

DynamicPPL provides a few methods for checking validity of a model-definition.

```@docs
check_model
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

DynamicPPL provides a data structure for storing samples and accumulation of the log-probabilities, called [`VarInfo`](@ref).
The interface that `VarInfo` respects is described by the abstract type [`AbstractVarInfo`](@ref).
Internally DynamicPPL also uses a couple of other subtypes of `AbstractVarInfo`.

```@docs
AbstractVarInfo
```

```@docs
VarInfo
DynamicPPL.get_values
```

[`VarInfo`](@ref) stores only accumulators. A `VectorValueAccumulator` records vectorised samples, whose transforms are selected by the evaluation context.
The [Transformations section below](#Transformations) describes the methods used for this.
Inspect the transforms recorded by a `VectorValueAccumulator` with:

```@docs
is_transformed
```

#### `VarNamedTuple`s

Value accumulators use [`VarNamedTuple`](@ref), which stores data keyed by `VarName`s.
For more details on `VarNamedTuple`, see the Internals section of our documentation.

```@docs
DynamicPPL.VarNamedTuples.VarNamedTuple
DynamicPPL.VarNamedTuples.@vnt
DynamicPPL.VarNamedTuples.apply!!
DynamicPPL.VarNamedTuples.densify!!
DynamicPPL.VarNamedTuples.skeleton
DynamicPPL.VarNamedTuples.map_pairs!!
DynamicPPL.VarNamedTuples.map_values!!
DynamicPPL.VarNamedTuples.PartialArray
DynamicPPL.VarNamedTuples.templated_setindex!!
DynamicPPL.VarNamedTuples.NoTemplate
DynamicPPL.VarNamedTuples.SkipTemplate
```

VarNamedTuple provides a Dict-like interface, so you can iterate over `keys(vnt)`, `values(vnt)`, and `pairs(vnt)`.
You can also use `getindex(vnt, key)`, but `setindex!` is not allowed: all changes to a `VarNamedTuple` must be done via `setindex!!` or `templated_setindex!!`.
Please see the VarNamedTuple documentation for more details.

You can convert a `VarNamedTuple` to a NamedTuple in the case where all keys are VarNames with identity optics.

```@docs
NamedTuple(::VarNamedTuple)
```

### Accumulators

The subtypes of [`AbstractVarInfo`](@ref) store the cumulative log prior and log likelihood, and sometimes other variables that change during executing, in what are called accumulators.

```@docs
AbstractAccumulator
accumulate_assume!!
accumulate_observe!!
accumulator_name
DynamicPPL.reset
DynamicPPL.split
DynamicPPL.combine
```

The float type used for accumulation of log-probabilities is defined by a compile-time preference:

```@docs
DynamicPPL.LogProbType
DynamicPPL.set_logprob_type!
DynamicPPL.NoLogProb
```

```@docs
VNTAccumulator
DoNotAccumulate
```

To manipulate the accumulators in a `VarInfo`, one can use:

```@docs
getacc
setacc!!
setaccs!!
deleteacc!!
```

### Common API

#### Accumulation of log-probabilities

```@docs
getlogp
setlogp!!
acclogp!!
getlogjoint
getlogjoint_internal
getlogjac
setlogjac!!
acclogjac!!
getlogprior
getlogprior_internal
setlogprior!!
acclogprior!!
getloglikelihood
setloglikelihood!!
accloglikelihood!!
```

#### Variables and their realizations

```@docs
keys
empty!!
isempty
DynamicPPL.getindex_internal
DynamicPPL.setindex_internal!!
```

#### Transformations

```@docs
DynamicPPL.link
DynamicPPL.invlink
DynamicPPL.link!!
DynamicPPL.invlink!!
DynamicPPL.update_transform_status!!
```

```@docs
DynamicPPL.AbstractTransformStrategy
DynamicPPL.LinkAll
DynamicPPL.UnlinkAll
DynamicPPL.LinkSome
DynamicPPL.UnlinkSome
DynamicPPL.WithTransforms
```

```@docs
DynamicPPL.AbstractTransform
DynamicPPL.DynamicLink
DynamicPPL.Unlink
DynamicPPL.FixedTransform
DynamicPPL.NoTransform
DynamicPPL.target_transform
DynamicPPL.apply_transform_strategy
```

#### Utils

```@docs
Base.merge(::AbstractVarInfo)
DynamicPPL.subset
unflatten!!
internal_values_as_vector
```

### Evaluation contexts

Internally, model evaluation is performed with [`AbstractPPL.evaluate!!`](@ref).

```@docs
AbstractPPL.evaluate!!
```

Call `evaluate!!(model, context, varinfo)` to evaluate with an explicit context and collect
outputs in `varinfo`. Accumulators are reset before evaluation.

The context is an evaluation input; it is not stored in the model.
Prefixes are stored separately from values. Conditioned and fixed values share one store, with each value carrying its role. Only latent sites reach the context; observations and tracked values go directly to accumulators.

`Context` is the sole evaluation context. It supplies an RNG, an initialisation strategy,
and a transform strategy. The output `varinfo` never supplies latent inputs.

```@docs
DynamicPPL.Context
```

Customise value selection through `init` methods on initialisation strategies, and
observation handling through `accumulate_observe!!` methods on accumulators.

```@docs
tilde_assume!!
tilde_observe!!
DynamicPPL.store_coloneq_value!!
```

Downstream evaluators that control execution directly can prepare arguments for `model.f`:

```@docs
DynamicPPL.make_evaluate_args_and_kwargs
```

### VarInfo initialisation

The function `init!!` constructs a `Context` and evaluates the model, resetting the output accumulators.

```@docs
init!!
```

To accomplish this, an initialisation _strategy_ is required, which defines how new values are to be obtained.
There are several concrete strategies provided in DynamicPPL: see the [initialisation strategies page](@ref init-strategies) for more information.

If you wish to write your own, you have to subtype [`DynamicPPL.AbstractInitStrategy`](@ref) and implement the `init` method.
In very rare situations, you may also need to implement `get_param_eltype`, which defines the element type of the parameters generated by the strategy.

```@docs
AbstractInitStrategy
init
get_param_eltype
```

The function [`DynamicPPL.init`](@ref) should return a `TransformedValue`.

```@docs
DynamicPPL.TransformedValue
```

The interface for working with transformed values consists of:

```@docs
DynamicPPL.get_transform
DynamicPPL.get_internal_value
DynamicPPL.get_raw_value
DynamicPPL.set_internal_value
```

### Converting VarInfos to/from chains

It is a fairly common operation to want to convert a collection of `VarInfo` objects into a chains object for downstream analysis.

This can be accomplished by first converting each `VarInfo` into a `ParamsWithStats` object:

```@docs
ParamsWithStats
ParamsWithStats(::AbstractInitStrategy, ::Model)
ParamsWithStats(::AbstractVarInfo)
ParamsWithStats(::AbstractVector, ::LogDensityFunction)
```

Once you have a **matrix** of these, you can convert them into a chains object using:

```@docs
AbstractMCMC.from_samples(::Type{MCMCChains.Chains}, ::AbstractMatrix{<:DynamicPPL.ParamsWithStats})
```

If you only have a vector you can use `hcat` to convert it into an `N×1` matrix first.

Furthermore, one can convert chains back into a collection of parameter dictionaries and/or stats with:

```@docs
AbstractMCMC.to_samples(::Type{DynamicPPL.ParamsWithStats}, ::MCMCChains.Chains, ::DynamicPPL.Model)
```

(Note that the model argument is mandatory as it provides templating information for the variables in the chains.)
With these, you can (for example) extract the parameter dictionaries and use `InitFromParams` to re-evaluate a model at each point in the chain.
