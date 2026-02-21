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
LogDensityFunction
```

Internally, this is accomplished using [`init!!`](@ref) on:

```@docs
OnlyAccsVarInfo
to_vector_params
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

## Marginalisation

DynamicPPL provides the `marginalize` function to marginalise out variables from a model.
This requires `MarginalLogDensities.jl` to be loaded in your environment.

```@docs
marginalize
```

A `MarginalLogDensity` object acts as a function which maps non-marginalised parameter values to a marginal log-probability.
To retrieve a VarInfo object from it, you can use:

```@docs
VarInfo(::MarginalLogDensities.MarginalLogDensity{<:DPPLMLDExt.LogDensityFunctionWrapper}, ::Union{AbstractVector,Nothing})
```

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

## Utilities

`typed_identity` is the same as `identity`, but with an overload for `with_logabsdet_jacobian` that ensures that it never errors.

```@docs
typed_identity
```

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

```@docs
NamedDist
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
DynamicPPL.setindex_with_dist!!
```

One main characteristic of [`VarInfo`](@ref) is that samples are transformed to unconstrained Euclidean space and stored in a linearized form, as described in the [main Turing documentation](https://turinglang.org/docs/developers/transforms/dynamicppl/).
The [Transformations section below](#Transformations) describes the methods used for this.
In the specific case of `VarInfo`, it keeps track of whether samples have been transformed by setting flags on them, using the following functions.

```@docs
is_transformed
set_transformed!!
```

#### `VarNamedTuple`s

`VarInfo` is only a thin wrapper around [`VarNamedTuple`](@ref), which stores arbitrary data keyed by `VarName`s.
For more details on `VarNamedTuple`, see the Internals section of our documentation.

```@docs
DynamicPPL.VarNamedTuples.VarNamedTuple
DynamicPPL.VarNamedTuples.@vnt
DynamicPPL.VarNamedTuples.apply!!
DynamicPPL.VarNamedTuples.densify!!
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
getindex
empty!!
isempty
DynamicPPL.getindex_internal
DynamicPPL.setindex_internal!!
```

#### Transformations

```@docs
DynamicPPL.AbstractTransformation
DynamicPPL.NoTransformation
DynamicPPL.DynamicTransformation
DynamicPPL.StaticTransformation
```

```@docs
DynamicPPL.link
DynamicPPL.invlink
DynamicPPL.link!!
DynamicPPL.invlink!!
DynamicPPL.update_link_status!!
```

```@docs
DynamicPPL.AbstractTransformStrategy
DynamicPPL.LinkAll
DynamicPPL.UnlinkAll
DynamicPPL.LinkSome
DynamicPPL.UnlinkSome
```

```@docs
DynamicPPL.AbstractTransform
DynamicPPL.DynamicLink
DynamicPPL.Unlink
DynamicPPL.target_transform
DynamicPPL.apply_transform_strategy
```

```@docs
DynamicPPL.transformation
DynamicPPL.default_transformation
DynamicPPL.link_transform
DynamicPPL.invlink_transform
```

#### Utils

```@docs
Base.merge(::AbstractVarInfo)
DynamicPPL.subset
unflatten!!
internal_values_as_vector
```

### Evaluation Contexts

Internally, model evaluation is performed with [`AbstractPPL.evaluate!!`](@ref).

```@docs
AbstractPPL.evaluate!!
```

This method mutates the `varinfo` used for execution.
By default, it does not perform any actual sampling: it only evaluates the model using the values of the variables that are already in the `varinfo`.
If you wish to sample new values, see the section on [VarInfo initialisation](#VarInfo-initialisation) just below this.

The behaviour of a model execution can be changed with evaluation contexts, which are a field of the model.

All contexts are subtypes of `AbstractPPL.AbstractContext`.

Contexts are split into two kinds:

**Leaf contexts**: These are the most important contexts as they ultimately decide how model evaluation proceeds.
For example, `DefaultContext` evaluates the model using values stored inside a VarInfo's metadata, whereas `InitContext` obtains new values either by sampling or from a known set of parameters.
DynamicPPL has more leaf contexts which are used for internal purposes, but these are the two that are exported.

```@docs
DefaultContext
InitContext
```

To implement a leaf context, you need to subtype `AbstractPPL.AbstractContext` and implement the `tilde_assume!!` and `tilde_observe!!` methods for your context.

```@docs
tilde_assume!!
tilde_observe!!
```

**Parent contexts**: These essentially act as 'modifiers' for leaf contexts.
For example, `PrefixContext` adds a prefix to all variable names during evaluation, while `CondFixContext` marks certain variables as being either conditioned or fixed.

To implement a parent context, you have to subtype `DynamicPPL.AbstractParentContext`, and implement the `childcontext` and `setchildcontext` methods.
If needed, you can also implement `tilde_assume!!` and `tilde_observe!!` for your context.
This is optional; the default implementation is to simply delegate to the child context.

```@docs
AbstractParentContext
childcontext
setchildcontext
```

Since contexts form a tree structure, these functions are automatically defined for manipulating context stacks.
They are mainly useful for modifying the fundamental behaviour (i.e. the leaf context), without affecting any of the modifiers (i.e. parent contexts).

```@docs
leafcontext
setleafcontext
```

### VarInfo initialisation

The function `init!!` is used to initialise, or overwrite, values in a VarInfo.
It is really a thin wrapper around using `evaluate!!` with an `InitContext`.

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

The function [`DynamicPPL.init`](@ref) should return an `AbstractTransformedValue`.
There are three subtypes currently available:

```@docs
DynamicPPL.AbstractTransformedValue
DynamicPPL.VectorValue
DynamicPPL.LinkedVectorValue
DynamicPPL.UntransformedValue
```

The interface for working with transformed values consists of:

```@docs
DynamicPPL.get_transform
DynamicPPL.get_internal_value
DynamicPPL.set_internal_value
```

### Converting VarInfos to/from chains

It is a fairly common operation to want to convert a collection of `VarInfo` objects into a chains object for downstream analysis.

This can be accomplished by first converting each `VarInfo` into a `ParamsWithStats` object:

```@docs
DynamicPPL.ParamsWithStats
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
