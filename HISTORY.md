# DynamicPPL Changelog

## 0.39.15

Fix AD performance with ReverseDiff (v0.39.9 inadvertently introduced a bug that did not cause any correctness issues, but did cause severe slowdowns with ReverseDiff -- this patch reverts that).

## 0.39.14

Optimise AD performance with ReverseDiff.

## 0.39.13

Add compatibility with Mooncake.jl 0.5.

## 0.39.12

When constructing an `MCMCChains.Chains`, sampler statistics that are not `Union{Real,Missing}` are dropped from the chain (previously this would cause chain construction to fail).
Note that MCMCChains in general cannot contain non-numeric statistics, so this is the only reasonable behaviour.

## 0.39.11

Allow passing `accs::Union{NTuple{N,AbstractAccumulator},AccumulatorTuple}` into the `LogDensityFunction` constructor to specify custom accumulators to use when evaluating the model.
Previously, this was hard-coded.

## 0.39.10

Rename the internal functions `matchingvalue` and `get_matching_type` to `convert_model_argument` and `promote_model_type_argument` respectively.
The behaviour of `promote_model_type_argument` has also been slightly changed in some edge cases: for example, `promote_model_type_argument(ForwardDiff.Dual{Nothing,Float64,0}, Vector{Real})` now returns `Vector{ForwardDiff.Dual{Nothing,Real,0}}` instead of `Vector{ForwardDiff.Dual{Nothing,Float64,0}}`.
In other words, abstract types in the type argument are now properly respected.

This should have almost no impact on end users (unless you were passing `::Type{T}=Vector{Real}` into the model, with an abstract eltype).

## 0.39.9

The internals of `LogDensityFunction` have been changed slightly so that you do not need to specify `function_annotation` when performing AD with Enzyme.jl.
There are also some small performance improvements with other AD backends.

## 0.39.8

Allow the `getlogdensity` argument of `LogDensityFunction` to accept callable structs as well as functions.

## 0.39.7

Improve concreteness when merging two `Metadata` structs.

## 0.39.6

Mark `haskey(varinfo, varname)` as having zero derivative to make life easier for AD.

## 0.39.5

Fixed a bug which prevented passing immutable data (such as NamedTuples or ordinary structs) as arguments to DynamicPPL models, or fixing the model on such data.

## 0.39.4

Removed the internal functions `DynamicPPL.getranges`, `DynamicPPL.vector_getrange`, and `DynamicPPL.vector_getranges` (the new LogDensityFunction construction does exactly the same thing, so this specialised function was not needed).

## 0.39.3

`DynamicPPL.TestUtils.AD.run_ad` now generates much prettier output.
In particular, when a test fails, it also tells you the tolerances needed to make it pass.

## 0.39.2

`returned(model, parameters...)` now accepts any arguments that can be wrapped in `InitFromParams` (previously it would only accept `NamedTuple`, `AbstractDict{<:VarName}`, or a chain).

There should also be some minor performance improvements (maybe 10%) on AD with ForwardDiff / Mooncake.

## 0.39.1

`LogDensityFunction` now allows you to call `logdensity_and_gradient(ldf, x)` with `AbstractVector`s `x` that are not plain Vectors (they will be converted internally before calculating the gradient).

## 0.39.0

### Breaking changes

#### Fast Log Density Functions

This version provides a reimplementation of `LogDensityFunction` that provides performance improvements on the order of 2–10× for both model evaluation as well as automatic differentiation.
Exact speedups depend on the model size: larger models have less significant speedups because the bulk of the work is done in calls to `logpdf`.

For more information about how this is accomplished, please see https://github.com/TuringLang/DynamicPPL.jl/pull/1113 as well as the `src/logdensityfunction.jl` file, which contains extensive comments.

As a result of this change, `LogDensityFunction` no longer stores a VarInfo inside it.
In general, if `ldf` is a `LogDensityFunction`, it is now only valid to access `ldf.model` and `ldf.adtype`.
If you were previously relying on this behaviour, you will need to store a VarInfo separately.

#### Threadsafe evaluation

DynamicPPL models have traditionally supported running some probabilistic statements (e.g. tilde-statements, or `@addlogprob!`) in parallel.
Prior to DynamicPPL 0.39, thread safety for such models used to be enabled by default if Julia was launched with more than one thread.

In DynamicPPL 0.39, **thread-safe evaluation is now disabled by default**.
If you need it (see below for more discussion of when you _do_ need it), you **must** now manually mark it as so, using:

```julia
@model f() = ...
model = f()
model = setthreadsafe(model, true)
```

The problem with the previous on-by-default is that it can sacrifice a huge amount of performance when thread safety is not needed.
This is especially true when running Julia in a notebook, where multiple threads are often enabled by default.
Furthermore, it is not actually the correct approach: just because Julia has multiple threads does not mean that a particular model actually requires threadsafe evaluation.

**A model requires threadsafe evaluation if, and only if, the VarInfo object used inside the model is manipulated in parallel.**
This can occur if any of the following are inside `Threads.@threads` or other concurrency functions / macros:

  - tilde-statements
  - calls to `@addlogprob!`
  - any direct manipulation of the special `__varinfo__` variable

If you have none of these inside threaded blocks, then you do not need to mark your model as threadsafe.
**Notably, the following do not require threadsafe evaluation:**

  - Using threading for any computation that does not involve VarInfo. For example, you can calculate a log-probability in parallel, and then add it using `@addlogprob!` outside of the threaded block. This does not require threadsafe evaluation.
  - Sampling with `AbstractMCMC.MCMCThreads()`.

For more information about threadsafe evaluation, please see [the Turing docs](https://turinglang.org/docs/usage/threadsafe-evaluation/).

When threadsafe evaluation is enabled for a model, an internal flag is set on the model.
The value of this flag can be queried using `DynamicPPL.requires_threadsafe(model)`, which returns a boolean.
This function is newly exported in this version of DynamicPPL.

#### Parent and leaf contexts

The `DynamicPPL.NodeTrait` function has been removed.
Instead of implementing this, parent contexts should subtype `DynamicPPL.AbstractParentContext`.
This is an abstract type which requires you to overload two functions, `DynamicPPL.childcontext` and `DynamicPPL.setchildcontext`.

There should generally be few reasons to define your own parent contexts (the only one we are aware of, outside of DynamicPPL itself, is `Turing.Inference.GibbsContext`), so this change should not really affect users.

Leaf contexts require no changes, apart from a removal of the `NodeTrait` function.

`ConditionContext` and `PrefixContext` are no longer exported.
You should not need to use these directly, please use `AbstractPPL.condition` and `DynamicPPL.prefix` instead.

#### ParamsWithStats

In the 'stats' part of `DynamicPPL.ParamsWithStats`, the log-joint is now consistently represented with the key `logjoint` instead of `lp`.

#### Miscellaneous

Removed the method `returned(::Model, values, keys)`; please use `returned(::Model, ::AbstractDict{<:VarName})` instead.

The unexported functions `supports_varname_indexing(chain)`, `getindex_varname(chain)`, and `varnames(chain)` have been removed.

The method `DynamicPPL.init` (for implementing `AbstractInitStrategy`) now has a different signature: it must return a tuple of the generated value, plus a transform function that maps it back to unlinked space.
This is a generalisation of the previous behaviour, where `init` would always return an unlinked value (in effect forcing the transform to be the identity function).

The family of functions `returned(model, chain)`, along with the same signatures of `pointwise_logdensities`, `logjoint`, `loglikelihood`, and `logprior`, have been changed such that if the chain does not contain all variables in the model, an error is thrown.
Previously the behaviour would have been to sample missing variables.

## 0.38.10

`returned(model, chain)` and `pointwise_logdensities(model, chain)` will now error if a value for a random variable cannot be found in the chain.
(Previously, they would instead resample such variables, which could lead to silent mistakes.)

If you encounter this error and it is accompanied by a warning about `hasvalue` not being implemented, you should be able to fix this by [using FlexiChains instead of MCMCChains](https://github.com/penelopeysm/FlexiChains.jl).
(Alternatively, implementations of `hasvalue` for unsupported distributions are more than welcome; these must be provided in the Distributions extension of AbstractPPL.jl.)

## 0.38.9

Remove warning when using Enzyme as the AD backend.

## 0.38.8

Added a new exported struct, `DynamicPPL.ParamsWithStats`.
This can broadly be used to represent the output of a model: it consists of an `OrderedDict` of `VarName` parameters and their values, along with a `stats` NamedTuple which can hold arbitrary data, such as (but not limited to) log-probabilities.

Implemented the functions `AbstractMCMC.to_samples` and `AbstractMCMC.from_samples`, which convert between an `MCMCChains.Chains` object and a matrix of `DynamicPPL.ParamsWithStats` objects.

## 0.38.7

Made a small tweak to DynamicPPL's compiler output to avoid potential undefined variables when resuming model functions midway through (e.g. with Libtask in Turing's SMC/PG samplers).

## 0.38.6

Renamed keyword argument `only_ddpl` to `only_dppl` for `Experimental.is_suitable_varinfo`.

## 0.38.5

Improve performance of VarNamedVector, mostly by changing how it handles contiguification.

## 0.38.4

Improve performance of VarNamedVector. It should now be very nearly on par with Metadata for all models we've benchmarked on.

## 0.38.3

Add an implementation of `returned(::Model, ::AbstractDict{<:VarName})`.
Please note we generally recommend using Dict, as NamedTuples cannot correctly represent variables with indices / fields on the left-hand side of tildes, like `x[1]` or `x.a`.

The generic method `returned(::Model, values, keys)` is deprecated and will be removed in the next minor version.

## 0.38.2

Added a compatibility entry for JET@0.11.

## 0.38.1

Added `from_linked_vec_transform` and `from_vec_transform` methods for `ProductNamedTupleDistribution`.
This patch allows sampling from `ProductNamedTupleDistribution` in DynamicPPL models.

## 0.38.0

### Breaking changes

#### Introduction of `InitContext`

DynamicPPL 0.38 introduces a new evaluation context, `InitContext`.
It is used to generate fresh values for random variables in a model.

Evaluation contexts are stored inside a `DynamicPPL.Model` object, and control what happens with tilde-statements when a model is run.
The two major leaf (basic) contexts are `DefaultContext` and, now, `InitContext`.
`DefaultContext` is the default context, and it simply uses the values that are already stored in the `VarInfo` object passed to the model evaluation function.
On the other hand, `InitContext` ignores values in the VarInfo object and inserts new values obtained from a specified source.
(It follows also that the VarInfo being used may be empty, which means that `InitContext` is now also the way to obtain a fresh VarInfo for a model.)

DynamicPPL 0.38 provides three flavours of _initialisation strategies_, which are specified as the second argument to `InitContext`:

  - `InitContext(rng, InitFromPrior())`: New values are sampled from the prior distribution (on the right-hand side of the tilde).
  - `InitContext(rng, InitFromUniform(a, b))`: New values are sampled uniformly from the interval `[a, b]`, and then invlinked to the support of the distribution on the right-hand side of the tilde.
  - `InitContext(rng, InitFromParams(p, fallback))`: New values are obtained by indexing into the `p` object, which can be a `NamedTuple` or `Dict{<:VarName}`. If a variable is not found in `p`, then the `fallback` strategy is used, which is simply another of these strategies. In particular, `InitFromParams` enables the case where different variables are to be initialised from different sources.

(It is possible to define your own initialisation strategy; users who wish to do so are referred to the DynamicPPL API documentation and source code.)

**The main impact on the upcoming Turing.jl release** is that, instead of providing initial values for sampling, the user will be expected to provide an initialisation strategy instead.
This is a more flexible approach, and not only solves a number of pre-existing issues with initialisation of Turing models, but also improves the clarity of user code.
In particular:

  - When providing a set of fixed parameters (i.e. `InitFromParams(p)`), `p` must now either be a NamedTuple or a Dict. Previously Vectors were allowed, which is error-prone because the ordering of variables in a VarInfo is not obvious.
  - The parameters in `p` must now always be provided in unlinked space (i.e., in the space of the distribution on the right-hand side of the tilde). Previously, whether a parameter was expected to be in linked or unlinked space depended on whether the VarInfo was linked or not, which was confusing.

#### Removal of `SamplingContext`

For developers working on DynamicPPL, `InitContext` now completely replaces what used to be `SamplingContext`, `SampleFromPrior`, and `SampleFromUniform`.
Evaluating a model with `SamplingContext(SampleFromPrior())` (e.g. with `DynamicPPL.evaluate_and_sample!!(model, VarInfo(), SampleFromPrior())` has a direct one-to-one replacement in `DynamicPPL.init!!(model, VarInfo(), InitFromPrior())`.
Please see the docstring of `init!!` for more details.
Likewise `SampleFromUniform()` can be replaced with `InitFromUniform()`.
`InitFromParams()` provides new functionality which was previously implemented in the roundabout way of manipulating the VarInfo (e.g. using `unflatten`, or even more hackily by directly modifying values in the VarInfo), and then evaluating using `DefaultContext`.

The main change that this is likely to create is for those who are implementing samplers or inference algorithms.
The exact way in which this happens will be detailed in the Turing.jl changelog when a new release is made.
Broadly speaking, though, `SamplingContext(MySampler())` will be removed so if your sampler needs custom behaviour with the tilde-pipeline you will likely have to define your own context.

#### Removal of `DynamicPPL.Sampler`

`DynamicPPL.Sampler` and **all associated interface functions** have also been removed entirely.
If you were using these, the corresponding replacements are:

  - `DynamicPPL.Sampler(S)`: just don't wrap `S`; but make sure `S` subtypes `AbstractMCMC.AbstractSampler`
  - `DynamicPPL.initialstep`: directly implement `AbstractMCMC.step` and `AbstractMCMC.step_warmup` as per the AbstractMCMC interface
  - `DynamicPPL.loadstate`: `Turing.loadstate` (will be introduced in the next version)
  - `DynamicPPL.default_chain_type`: removed, just use the `chain_type` keyword argument directly
  - `DynamicPPL.initialsampler`: `Turing.Inference.init_strategy` (will be introduced in the next version; note that this function must return an `AbstractInitStrategy`, see above for explanation)
  - `DynamicPPL.default_varinfo`: `Turing.Inference.default_varinfo` (will be introduced in the next version)
  - `DynamicPPL.TestUtils.test_sampler` and related methods: removed, please implement your own testing utilities as needed

#### Simplification of the tilde-pipeline

There are now only two functions in the tilde-pipeline that need to be overloaded to change the behaviour of tilde-statements, namely, `tilde_assume!!` and `tilde_observe!!`.
Other functions such as `tilde_assume` and `assume` (and their `observe` counterparts) have been removed.

Note that this was effectively already the case in DynamicPPL 0.37 (where they were just wrappers around each other).
The separation of these functions was primarily implemented to avoid performing extra work where unneeded (e.g. to not calculate the log-likelihood when `PriorContext` was being used). This functionality has since been replaced with accumulators (see the 0.37 changelog for more details).

#### Removal of the `"del"` flag

Previously `VarInfo` (or more correctly, the `Metadata` object within a `VarInfo`), had a flag called `"del"` for all variables. If it was set to `true` the variable was to be overwritten with a new value at the next evaluation. The new `InitContext` and related changes above make this flag unnecessary, and it has been removed.

The only flag other than `"del"` that `Metadata` ever used was `"trans"`. Thus the generic functions `set_flag!`, `unset_flag!` and `is_flagged!` have also been removed in favour of more specific ones. We've also used this opportunity to name the `"trans"` flag and the corresponding `istrans` function to be more explicit. The new, exported interface consists of the `is_transformed` and `set_transformed!!` functions.

#### Removal of `resume_from`

The `resume_from=chn` keyword argument to `sample` has been removed; please use the `initial_state` argument instead.
`loadstate` will be exported from Turing in the next release of Turing.

#### Change of output type for `pointwise_logdensities`

The functions `pointwise_prior_logdensities`, `pointwise_logdensities`, and `pointwise_loglikelihoods` when called on `MCMCChains.Chains` objects, now return new `MCMCChains.Chains` objects by default, instead of dictionaries of matrices.

If you want the old behaviour, you can pass `OrderedDict` as the third argument, i.e., `pointwise_logdensities(model, chain, OrderedDict)`.

### Other changes

#### `predict(model, chain; include_all)`

The `include_all` keyword argument for `predict` now works even when no RNG is specified (previously it would only work when an RNG was explicitly passed).

#### `DynamicPPL.setleafcontext(model, context)`

This convenience method has been added to quickly modify the leaf context of a model.

#### Reimplementation of functions using `InitContext`

A number of functions have been reimplemented and unified with the help of `InitContext`.
In particular, this release brings substantial performance improvements for `returned` and `predict`.
Their APIs are the same.

#### Upstreaming of VarName functionality

The implementation of the `varname_leaves` and `varname_and_value_leaves` functions have been moved to AbstractPPL.jl.
Their behaviour is otherwise identical, and they are still accessible from the DynamicPPL module (though still not exported).

## 0.37.5

A minor optimisation for Enzyme AD on DynamicPPL models.

## 0.37.4

An extension for MarginalLogDensities.jl has been added.

Loading DynamicPPL and MarginalLogDensities now provides the `DynamicPPL.marginalize` function to marginalise out variables from a model.
This is useful for averaging out random effects or nuisance parameters while improving inference on fixed effects/parameters of interest.
The `marginalize` function returns a `MarginalLogDensities.MarginalLogDensity`, a function-like callable struct that returns the approximate log-density of a subset of the parameters after integrating out the rest of them.
By default, this uses the Laplace approximation and sparse AD, making the marginalisation computationally very efficient.
Note that the Laplace approximation relies on the model being differentiable with respect to the marginalised variables, and that their posteriors are unimodal and approximately Gaussian.

Please see [the MarginalLogDensities documentation](https://eloceanografo.github.io/MarginalLogDensities.jl/stable) and the [new Marginalisation section of the DynamicPPL documentation](https://turinglang.org/DynamicPPL.jl/v0.37/api/#Marginalisation) for further information.

## 0.37.3

Prevents inlining of `DynamicPPL.istrans` with Enzyme, which allows Enzyme to differentiate models where `VarName`s have the same symbol but different types.

## 0.37.2

Make the `resume_from` keyword work for multiple-chain (parallel) sampling as well.
Prior to this version, it was silently ignored.
Note that to get the correct behaviour you also need to have a recent version of MCMCChains (v7.2.1).

## 0.37.1

Update DynamicPPLMooncakeExt to work with Mooncake 0.4.147.

## 0.37.0

DynamicPPL 0.37 comes with a substantial reworking of its internals.
Fundamentally, there is no change to the actual modelling syntax: if you are a Turing.jl user, for example, this release will not affect you too much (apart from the changes to `@addlogprob!`).
Any such changes will be covered separately in the Turing.jl changelog when a release is made.
However, if you are a package developer or someone who uses DynamicPPL's functionality directly, you will notice a number of changes.

To avoid overwhelming the reader, we begin by listing the most important, user-facing changes, before explaining the changes to the internals in more detail.

Note that virtually all changes listed here are breaking.

**Public-facing changes**

### Submodel macro

The `@submodel` macro is fully removed; please use `to_submodel` instead.

### `DynamicPPL.TestUtils.AD.run_ad`

The three keyword arguments, `test`, `reference_backend`, and `expected_value_and_grad` have been merged into a single `test` keyword argument.
Please see the API documentation for more details.
(The old `test=true` and `test=false` values are still valid, and you only need to adjust the invocation if you were explicitly passing the `reference_backend` or `expected_value_and_grad` arguments.)

There is now also an `rng` keyword argument to help seed parameter generation.

Instead of specifying `value_atol` and `grad_atol`, you can now specify `atol` and `rtol` which are used for both value and gradient.
Their semantics are the same as in Julia's `isapprox`; two values are equal if they satisfy either `atol` or `rtol`.

Finally, the `ADResult` object returned by `run_ad` now has both `grad_time` and `primal_time` fields, which contain (respectively) the time it took to calculate the gradient of logp, and the time taken to calculate logp itself.
Times are reported in seconds.
Previously there was only a single `time_vs_primal` field which represented the ratio of these two.
You can of course access the same quantity by dividing `grad_time` by `primal_time`.

### `DynamicPPL.TestUtils.check_model`

You now need to explicitly pass a `VarInfo` argument to `check_model` and `check_model_and_trace`.
Previously, these functions would generate a new VarInfo for you (using an optionally provided `rng`).

### Evaluating model log-probabilities in more detail

Previously, during evaluation of a model, DynamicPPL only had the capability to store a _single_ log probability (`logp`) field.
`DefaultContext`, `PriorContext`, and `LikelihoodContext` were used to control what this field represented: they would accumulate the log joint, log prior, or log likelihood, respectively.

In this version, we have overhauled this quite substantially.
The technical details of exactly _how_ this is done is covered in the 'Accumulators' section below, but the upshot is that the log prior, log likelihood, and log Jacobian terms (for any linked variables) are separately tracked.

Specifically, you will want to use the following functions to access these log probabilities:

  - `getlogprior(varinfo)` to get the log prior. **Note:** This version introduces new, more consistent behaviour for this function, in that it always returns the log-prior of the values in the original, untransformed space, even if the `varinfo` has been linked.
  - `getloglikelihood(varinfo)` to get the log likelihood.
  - `getlogjoint(varinfo)` to get the log joint probability. **Note:** Similar to `getlogprior`, this function now always returns the log joint of the values in the original, untransformed space, even if the `varinfo` has been linked.

If you are using linked VarInfos (e.g. if you are writing a sampler), you may find that you need to obtain the log probability of the variables in the transformed space.
To this end, you can use:

  - `getlogjac(varinfo)` to get the log Jacobian of the link transforms for any linked variables.
  - `getlogprior_internal(varinfo)` to get the log prior of the variables in the transformed space.
  - `getlogjoint_internal(varinfo)` to get the log joint probability of the variables in the transformed space.

Since transformations only apply to random variables, the likelihood is unaffected by linking.

### Removal of `PriorContext` and `LikelihoodContext`

Following on from the above, a number of DynamicPPL's contexts have been removed, most notably `PriorContext` and `LikelihoodContext`.
Although these are not the only _exported_ contexts, we consider unlikely that anyone was using _other_ contexts manually: if you have a question about contexts _other_ than these, please continue reading the 'Internals' section below.

If you were evaluating a model with `PriorContext`, you can now just evaluate it with `DefaultContext`, and instead of calling `getlogp(varinfo)`, you can call `getlogprior(varinfo)` (and similarly for the likelihood).

If you were constructing a `LogDensityFunction` with `PriorContext`, you can now stick to `DefaultContext`.
`LogDensityFunction` now has an extra field, called `getlogdensity`, which represents a function that takes a `VarInfo` and returns the log density you want.
Thus, if you pass `getlogprior_internal` as the value of this parameter, you will get the same behaviour as with `PriorContext`.
(You should consider whether your use case needs the log prior in the transformed space, or the original space, and use (respectively) `getlogprior_internal` or `getlogprior` as needed.)

The other case where one might use `PriorContext` was to use `@addlogprob!` to add to the log prior.
Previously, this was accomplished by manually checking `__context__ isa DynamicPPL.PriorContext`.
Now, you can write `@addlogprob (; logprior=x, loglikelihood=y)` to add `x` to the log-prior and `y` to the log-likelihood.

### Removal of `order` and `num_produce`

The `VarInfo` type used to carry with it:

  - `num_produce`, an integer which recorded the number of observe tilde-statements that had been evaluated so far; and
  - `order`, an integer per `VarName` which recorded the value of `num_produce` at the time that the variable was seen.

These fields were used in particle samplers in Turing.jl.
In DynamicPPL 0.37, these fields and the associated functions have been removed:

  - `get_num_produce`
  - `set_num_produce!!`
  - `reset_num_produce!!`
  - `increment_num_produce!!`
  - `set_retained_vns_del!`
  - `setorder!!`

Because this is one of the more arcane features of DynamicPPL, some extra explanation is warranted.

`num_produce` and `order`, along with the `del` flag in `VarInfo`, were used to control whether new values for variables were sampled during model execution.
For example, the particle Gibbs method has a _reference particle_, for which variables are never resampled.
However, if the reference particle is _forked_ (i.e., if the reference particle is selected by a resampling step multiple times and thereby copied), then the variables that have not yet been evaluated must be sampled anew to ensure that the new particle is independent of the reference particle.

Previously, this was accomplished by setting the `del` flag in the `VarInfo` object for all variables with `order` greater or equal to than `num_produce`.
Note that setting the `del` flag does not itself trigger a new value to be sampled; rather, it indicates that a new value should be sampled _if the variable is encountered again_.
[This Turing.jl PR](https://github.com/TuringLang/Turing.jl/pull/2629) changes the implementation to set the `del` flag for _all_ variables in the `VarInfo`.
Since the `del` flag only makes a difference when encountering a variable, this approach is entirely equivalent as long as the same variable is not seen multiple times in the model.
The interested reader is referred to that PR for more details.

**Internals**

### Accumulators

This release overhauls how VarInfo objects track variables such as the log joint probability. The new approach is to use what we call accumulators: Objects that the VarInfo carries on it that may change their state at each `tilde_assume!!` and `tilde_observe!!` call based on the value of the variable in question. They replace both variables that were previously hard-coded in the `VarInfo` object (`logp` and `num_produce`) and some contexts. This brings with it a number of breaking changes:

  - `PriorContext` and `LikelihoodContext` no longer exist. By default, a `VarInfo` tracks both the log prior and the log likelihood separately, and they can be accessed with `getlogprior` and `getloglikelihood`. If you want to execute a model while only accumulating one of the two (to save clock cycles), you can do so by creating a `VarInfo` that only has one accumulator in it, e.g. `varinfo = setaccs!!(varinfo, (LogPriorAccumulator(),))`.
  - `MiniBatchContext` does not exist anymore. It can be replaced by creating and using a custom accumulator that replaces the default `LikelihoodContext`. We may introduce such an accumulator in DynamicPPL in the future, but for now you'll need to do it yourself.
  - `tilde_observe` and `observe` have been removed. `tilde_observe!!` still exists, and any contexts should modify its behaviour. We may further rework the call stack under `tilde_observe!!` in the near future.
  - `tilde_assume` no longer returns the log density of the current assumption as its second return value. We may further rework the `tilde_assume!!` call stack as well.
  - For literal observation statements like `0.0 ~ Normal(blahblah)` we used to call `tilde_observe!!` without the `vn` argument. This method no longer exists. Rather we call `tilde_observe!!` with `vn` set to `nothing`.
  - `@addlogprob!` now _always_ adds to the log likelihood. Previously it added to the log probability that the execution context specified, e.g. the log prior when using `PriorContext`.
  - `getlogp` now returns a `NamedTuple` with keys `logprior` and `loglikelihood`. If you want the log joint probability, which is what `getlogp` used to return, use `getlogjoint`.
  - Correspondingly `setlogp!!` and `acclogp!!` should now be called with a `NamedTuple` with keys `logprior` and `loglikelihood`. The `acclogp!!` method with a single scalar value has been deprecated and falls back on `accloglikelihood!!`, and the single scalar version of `setlogp!!` has been removed. Corresponding setter/accumulator functions exist for the log prior as well.

### Evaluation contexts

Historically, evaluating a DynamicPPL model has required three arguments: a model, some kind of VarInfo, and a context.
It's less known, though, that since DynamicPPL 0.14.0 the _model_ itself actually contains a context as well.
This version therefore excises the context argument, and instead uses `model.context` as the evaluation context.

The upshot of this is that many functions that previously took a context argument now no longer do.
There were very few such functions where the context argument was actually used (most of them simply took `DefaultContext()` as the default value).

`evaluate!!(model, varinfo, ext_context)` is removed, and broadly speaking you should replace calls to that with `new_model = contextualize(model, ext_context); evaluate!!(new_model, varinfo)`.
If the 'external context' `ext_context` is a parent context, then you should wrap `model.context` appropriately to ensure that its information content is not lost.
If, on the other hand, `ext_context` is a `DefaultContext`, then you can just drop the argument entirely.

**To aid with this process, `contextualize` is now exported from DynamicPPL.**

The main situation where one _did_ want to specify an additional evaluation context was when that context was a `SamplingContext`.
Doing this would allow you to run the model and sample fresh values, instead of just using the values that existed in the VarInfo object.
Thus, this release also introduces the **unexported** function `evaluate_and_sample!!`.
Essentially, `evaluate_and_sample!!(rng, model, varinfo, sampler)` is a drop-in replacement for `evaluate!!(model, varinfo, SamplingContext(rng, sampler))`.
**Do note that this is an internal method**, and its name or semantics are liable to change in the future without warning.

There are many methods that no longer take a context argument, and listing them all would be too much.
However, here are the more user-facing ones:

  - `LogDensityFunction` no longer has a context field (or type parameter)
  - `DynamicPPL.TestUtils.AD.run_ad` no longer uses a context (and the returned `ADResult` object no longer has a context field)
  - `VarInfo(rng, model, sampler)` and other VarInfo constructors / functions that made VarInfos (e.g. `typed_varinfo`) from a model
  - `(::Model)(args...)`: specifically, this now only takes `rng` and `varinfo` arguments (with both being optional)
  - If you are using the `__context__` special variable inside a model, you will now have to use `__model__.context` instead

And a couple of more internal changes:

  - Just like `evaluate!!`, the other functions `_evaluate!!`, `evaluate_threadsafe!!`, and `evaluate_threadunsafe!!` now no longer accept context arguments
  - `evaluate!!` no longer takes rng and sampler (if you used this, you should use `evaluate_and_sample!!` instead, or construct your own `SamplingContext`)
  - The model evaluation function, `model.f` for some `model::Model`, no longer takes a context as an argument
  - The internal representation and API dealing with submodels (i.e., `ReturnedModelWrapper`, `Sampleable`, `should_auto_prefix`, `is_rhs_model`) has been simplified. If you need to check whether something is a submodel, just use `x isa DynamicPPL.Submodel`. Note that the public API i.e. `to_submodel` remains completely untouched.

## 0.36.15

Bumped minimum Julia version to 1.10.8 to avoid potential crashes with `Core.Compiler.widenconst` (which Mooncake uses).

## 0.36.14

Added compatibility with AbstractPPL@0.12.

## 0.36.13

Added documentation for the `returned(::Model, ::MCMCChains.Chains)` method.

## 0.36.12

Removed several unexported functions.
The only notable one is `DynamicPPL.alg_str`, which was used in old versions of AdvancedVI and the Turing test suite.

## 0.36.11

Make `ThreadSafeVarInfo` hold a total of `Threads.nthreads() * 2` logp values, instead of just `Threads.nthreads()`.
This fix helps to paper over the cracks in using `threadid()` to index into the `ThreadSafeVarInfo` object.

## 0.36.10

Added compatibility with ForwardDiff 1.0.

## 0.36.9

Fixed a failure when sampling from `ProductNamedTupleDistribution` due to
missing `tovec` methods for `NamedTuple` and `Tuple`.

## 0.36.8

Made `LogDensityFunction` a subtype of `AbstractMCMC.AbstractModel`.

## 0.36.7

Added compatibility with MCMCChains 7.0.

## 0.36.6

`DynamicPPL.TestUtils.run_ad` now takes an extra `context` keyword argument, which is passed to the `LogDensityFunction` constructor.

## 0.36.5

`varinfo[:]` now returns an empty vector if `varinfo::DynamicPPL.NTVarInfo` is empty, rather than erroring.

In its place, `check_model` now issues a warning if the model is empty.

## 0.36.4

Added compatibility with DifferentiationInterface.jl 0.7, and also with JET.jl 0.10.

The JET compatibility entry should only affect you if you are using DynamicPPL on the Julia 1.12 pre-release.

## 0.36.3

Moved the `bijector(model)`, where `model` is a `DynamicPPL.Model`, function from the Turing main repo.

## 0.36.2

Improved docstrings for AD testing utilities.

## 0.36.1

Fixed a missing method for `tilde_assume`.

## 0.36.0

**Breaking changes**

### Submodels: conditioning

Variables in a submodel can now be conditioned and fixed in a correct way.
See https://github.com/TuringLang/DynamicPPL.jl/issues/857 for a full illustration, but essentially it means you can now do this:

```julia
@model function inner()
    x ~ Normal()
    return y ~ Normal()
end
@model function outer()
    return a ~ to_submodel(inner() | (x=1.0,))
end
```

and the `a.x` variable will be correctly conditioned.
(Previously, you would have to condition `inner()` with the variable `a.x`, meaning that you would need to know what prefix to use before you had actually prefixed it.)

### Submodel prefixing

The way in which VarNames in submodels are prefixed has been changed.
This is best explained through an example.
Consider this model and submodel:

```julia
using DynamicPPL, Distributions
@model inner() = x ~ Normal()
@model outer() = a ~ to_submodel(inner())
```

In previous versions, the inner variable `x` would be saved as `a.x`.
However, this was represented as a single symbol `Symbol("a.x")`:

```julia
julia> dump(keys(VarInfo(outer()))[1])
VarName{Symbol("a.x"), typeof(identity)}
  optic: identity (function of type typeof(identity))
```

Now, the inner variable is stored as a field `x` on the VarName `a`:

```julia
julia> dump(keys(VarInfo(outer()))[1])
VarName{:a, Accessors.PropertyLens{:x}}
  optic: Accessors.PropertyLens{:x} (@o _.x)
```

In practice, this means that if you are trying to condition a variable in the submodel, you now need to use

```julia
outer() | (@varname(a.x) => 1.0,)
```

instead of either of these (which would have worked previously)

```julia
outer() | (@varname(var"a.x") => 1.0,)
outer() | (a.x=1.0,)
```

In a similar way, if the variable on the left-hand side of your tilde statement is not just a single identifier, any fields or indices it accesses are now properly respected.
Consider the following setup:

```julia
using DynamicPPL, Distributions
@model inner() = x ~ Normal()
@model function outer()
    a = Vector{Float64}(undef, 1)
    a[1] ~ to_submodel(inner())
    return a
end
```

In this case, the variable sampled is actually the `x` field of the first element of `a`:

```julia
julia> only(keys(VarInfo(outer()))) == @varname(a[1].x)
true
```

Before this version, it used to be a single variable called `var"a[1].x"`.

Note that if you are sampling from a model with submodels, this doesn't affect the way you interact with the `MCMCChains.Chains` object, because VarNames are converted into Symbols when stored in the chain.
(This behaviour will likely be changed in the future, in that Chains should be indexable by VarNames and not just Symbols, but that has not been implemented yet.)

### AD testing utilities

`DynamicPPL.TestUtils.AD.run_ad` now links the VarInfo by default.
To disable this, pass the `linked=false` keyword argument.
If the calculated value or gradient is incorrect, it also throws a `DynamicPPL.TestUtils.AD.ADIncorrectException` rather than a test failure.
This exception contains the actual and expected gradient so you can inspect it if needed; see the documentation for more information.
From a practical perspective, this means that if you need to add this to a test suite, you need to use `@test run_ad(...) isa Any` rather than just `run_ad(...)`.

### SimpleVarInfo linking / invlinking

Linking a linked SimpleVarInfo, or invlinking an unlinked SimpleVarInfo, now displays a warning instead of an error.

### VarInfo constructors

`VarInfo(vi::VarInfo, values)` has been removed. You can replace this directly with `unflatten(vi, values)` instead.

The `metadata` argument to `VarInfo([rng, ]model[, sampler, context, metadata])` has been removed.
If you were not using this argument (most likely), then there is no change needed.
If you were using the `metadata` argument to specify a blank `VarNamedVector`, then you should replace calls to `VarInfo` with `DynamicPPL.typed_vector_varinfo` instead (see 'Other changes' below).

The `UntypedVarInfo` constructor and type is no longer exported.
If you needed to construct one, you should now use `DynamicPPL.untyped_varinfo` instead.

The `TypedVarInfo` constructor and type is no longer exported.
The _type_ has been replaced with `DynamicPPL.NTVarInfo`.
The _constructor_ has been replaced with `DynamicPPL.typed_varinfo`.

Note that the exact kind of VarInfo returned by `VarInfo(rng, model, ...)` is an implementation detail.
Previously, it was guaranteed that this would always be a VarInfo whose metadata was a `NamedTuple` containing `Metadata` structs.
Going forward, this is no longer the case, and you should only assume that the returned object obeys the `AbstractVarInfo` interface.

**Other changes**

While these are technically breaking, they are only internal changes and do not affect the public API.
The following four functions have been added and/or reworked to make it easier to construct VarInfos with different types of metadata:

 1. `DynamicPPL.untyped_varinfo([rng, ]model[, sampler, context])`
 2. `DynamicPPL.typed_varinfo([rng, ]model[, sampler, context])`
 3. `DynamicPPL.untyped_vector_varinfo([rng, ]model[, sampler, context])`
 4. `DynamicPPL.typed_vector_varinfo([rng, ]model[, sampler, context])`

The reason for this change is that there were several flavours of VarInfo.
Some, like `typed_varinfo`, were easy to construct because we had convenience methods for them; however, the others were more difficult.
This change makes it easier to access different VarInfo types, and also makes it more explicit which one you are constructing.

## 0.35.9

Fixed the `isnan` check introduced in 0.35.7 for distributions which returned NamedTuple.

## 0.35.8

Added the `DynamicPPL.TestUtils.AD.run_ad` function to test the correctness and/or benchmark the performance of an automatic differentiation backend on DynamicPPL models.
Please see [the docstring](https://turinglang.org/DynamicPPL.jl/api/#DynamicPPL.TestUtils.AD.run_ad) for more information.

## 0.35.7

`check_model_and_trace` now errors if any NaN's are encountered when evaluating the model.

## 0.35.6

Fixed the implementation of `.~`, such that running a model with it no longer requires DynamicPPL itself to be loaded.

## 0.35.5

Several internal methods have been removed:

  - `DynamicPPL.getall(vi::AbstractVarInfo)` has been removed. You can directly replace this with `getindex_internal(vi, Colon())`.
  - `DynamicPPL.setall!(vi::AbstractVarInfo, values)` has been removed. Rewrite the calling function to not assume mutation and use `unflatten(vi, values)` instead.
  - `DynamicPPL.replace_values(md::Metadata, values)` and `DynamicPPL.replace_values(nt::NamedTuple, values)` (where the `nt` is a NamedTuple of Metadatas) have been removed. Use `DynamicPPL.unflatten_metadata` as a direct replacement.
  - `DynamicPPL.set_values!!(vi::AbstractVarInfo, values)` has been renamed to `DynamicPPL.set_initial_values(vi::AbstractVarInfo, values)`; it also no longer mutates the varinfo argument.

The **exported** method `VarInfo(vi::VarInfo, values)` has been deprecated, and will be removed in the next minor version. You can replace this directly with `unflatten(vi, values)` instead.

## 0.35.4

Fixed a type instability in an implementation of `with_logabsdet_jacobian`, which resulted in the log-jacobian returned being an Int in some cases and a Float in others.
This resolves an Enzyme.jl error on a number of models.
More generally, this version also changes the type of various log probabilities to be more consistent with one another.
Although we aren't fully there yet, our eventual aim is that log probabilities will generally default to Float64 on 64-bit systems, and Float32 on 32-bit systems.
If you run into any issues with these types, please get in touch.

## 0.35.3

`model | (@varname(x) => 1.0, @varname(y) => 2.0)` now works.
Previously, this would throw a `MethodError` if the tuple had more than one element.

## 0.35.2

`unflatten(::VarInfo, params)` now works with params that have non-float types (such as Int or Bool).

## 0.35.1

`subset(::AbstractVarInfo, ::AbstractVector{<:VarName})` now preserves the ordering of the varnames in the original varinfo argument.
Previously, this would select the varnames according to their order in the second argument.
This fixes an upstream Turing.jl issue with Gibbs sampling when a component sampler was assigned multiple variables.

## 0.35.0

**Breaking changes**

### `.~` right hand side must be a univariate distribution

Previously we allowed statements like

```julia
x .~ [Normal(), Gamma()]
```

where the right hand side of a `.~` was an array of distributions, and ones like

```julia
x .~ MvNormal(fill(0.0, 2), I)
```

where the right hand side was a multivariate distribution.

These are no longer allowed. The only things allowed on the right hand side of a `.~` statement are univariate distributions, such as

```julia
x = Array{Float64,3}(undef, 2, 3, 4)
x .~ Normal()
```

The reasons for this are internal code simplification and the fact that broadcasting where both sides are multidimensional but of different dimensions is typically confusing to read.

If the right hand side and the left hand side have the same dimension, one can simply use `~`. Arrays of distributions can be replaced with `product_distribution`. So instead of

```julia
x .~ [Normal(), Gamma()]
x .~ Normal.(y)
x .~ MvNormal(fill(0.0, 2), I)
```

do

```julia
x ~ product_distribution([Normal(), Gamma()])
x ~ product_distribution(Normal.(y))
x ~ MvNormal(fill(0.0, 2), I)
```

This is often more performant as well. Note that using `~` rather than `.~` does change the internal storage format a bit: With `.~` `x[i]` are stored as separate variables, with `~` as a single multivariate variable `x`. In most cases this does not change anything for the user, but if it does cause issues, e.g. if you are dealing with `VarInfo` objects directly and need to keep the old behavior, you can always expand into a loop, such as

```julia
dists = Normal.(y)
for i in 1:length(dists)
    x[i] ~ dists[i]
end
```

Cases where the right hand side is of a different dimension than the left hand side, and neither is a scalar, must be replaced with a loop. For example,

```julia
x = Array{Float64,3}(undef, 2, 3, 4)
x .~ MvNormal(fill(0, 2), I)
```

should be replaced with something like

```julia
x = Array{Float64,3}(2, 3, 4)
for i in 1:3, j in 1:4
    x[:, i, j] ~ MvNormal(fill(0, 2), I)
end
```

This release also completely rewrites the internal implementation of `.~`, where from now on all `.~` statements are turned into loops over `~` statements at macro time. However, the only breaking aspect of this change is the above change to what's allowed on the right hand side.

### Remove indexing by samplers

This release removes the feature of `VarInfo` where it kept track of which variable was associated with which sampler. This means removing all user-facing methods where `VarInfo`s where being indexed with samplers. In particular,

  - `link` and `invlink`, and their `!!` versions, no longer accept a sampler as an argument to specify which variables to (inv)link. The `link(varinfo, model)` methods remain in place, and as a new addition one can give a `Tuple` of `VarName`s to (inv)link only select variables, as in `link(varinfo, varname_tuple, model)`.
  - `set_retained_vns_del_by_spl!` has been replaced by `set_retained_vns_del!` which applies to all variables.
  - `getindex`, `setindex!`, and `setindex!!` no longer accept samplers as arguments
  - `unflatten` no longer accepts a sampler as an argument
  - `eltype(::VarInfo)` no longer accepts a sampler as an argument
  - `keys(::VarInfo)` no longer accepts a sampler as an argument
  - `VarInfo(::VarInfo, ::Sampler, ::AbstractVector)` no longer accepts the sampler argument.
  - `push!!` and `push!` no longer accept samplers or `Selector`s as arguments
  - `getgid`, `setgid!`, `updategid!`, `getspace`, and `inspace` no longer exist

### Reverse prefixing order

  - For submodels constructed using `to_submodel`, the order in which nested prefixes are applied has been changed.
    Previously, the order was that outer prefixes were applied first, then inner ones.
    This version reverses that.
    To illustrate:
    
    ```julia
    using DynamicPPL, Distributions
    
    @model function subsubmodel()
        return x ~ Normal()
    end
    
    @model function submodel()
        x ~ to_submodel(prefix(subsubmodel(), :c), false)
        return x
    end
    
    @model function parentmodel()
        x1 ~ to_submodel(prefix(submodel(), :a), false)
        return x2 ~ to_submodel(prefix(submodel(), :b), false)
    end
    
    keys(VarInfo(parentmodel()))
    ```
    
    Previously, the final line would return the variable names `c.a.x` and `c.b.x`.
    With this version, it will return `a.c.x` and `b.c.x`, which is more intuitive.
    (Note that this change brings `to_submodel`'s behaviour in line with the now-deprecated `@submodel` macro.)
    
    This change also affects sampling in Turing.jl.

### `LogDensityFunction` argument order

  - The method `LogDensityFunction(varinfo, model, sampler)` has been removed.
    The only accepted order is `LogDensityFunction(model, varinfo, context; adtype)`.
    (For an explanation of `adtype`, see below.)
    The varinfo and context arguments are both still optional.

**Other changes**

### New exports

`LogDensityFunction` and `predict` are now exported from DynamicPPL.

### `LogDensityProblems` interface

LogDensityProblemsAD is now removed as a dependency.
Instead of constructing a `LogDensityProblemAD.ADgradient` object, we now directly use `DifferentiationInterface` to calculate the gradient of the log density with respect to model parameters.

Note that if you wish, you can still construct an `ADgradient` out of a `LogDensityFunction` object (there is nothing preventing this).

However, in this version, `LogDensityFunction` now takes an extra AD type argument.
If this argument is not provided, the behaviour is exactly the same as before, i.e. you can calculate `logdensity` but not its gradient.
However, if you do pass an AD type, that will allow you to calculate the gradient as well.
You may thus find that it is easier to instead do this:

```julia
@model f() = ...

ldf = LogDensityFunction(f(); adtype=AutoForwardDiff())
```

This will return an object which satisfies the `LogDensityProblems` interface to first-order, i.e. you can now directly call both

```
LogDensityProblems.logdensity(ldf, params)
LogDensityProblems.logdensity_and_gradient(ldf, params)
```

without having to construct a separate `ADgradient` object.

If you prefer, you can also construct a new `LogDensityFunction` with a new AD type afterwards.
The model, varinfo, and context will be taken from the original `LogDensityFunction`:

```julia
@model f() = ...

ldf = LogDensityFunction(f())  # by default, no adtype set
ldf_with_ad = LogDensityFunction(ldf, AutoForwardDiff())
```

## 0.34.2

  - Fixed bugs in ValuesAsInModelContext as well as DebugContext where underlying PrefixContexts were not being applied.
    From a user-facing perspective, this means that for models which use manually prefixed submodels, e.g.
    
    ```julia
    using DynamicPPL, Distributions
    
    @model inner() = x ~ Normal()
    
    @model function outer()
        x1 ~ to_submodel(prefix(inner(), :a), false)
        return x2 ~ to_submodel(prefix(inner(), :b), false)
    end
    ```
    
    will: (1) no longer error when sampling due to `check_model_and_trace`; and (2) contain both submodel's variables in the resulting chain (the behaviour before this patch was that the second `x` would override the first `x`).

  - More broadly, implemented a general `prefix(ctx::AbstractContext, ::VarName)` which traverses the context tree in `ctx` to apply all necessary prefixes. This was a necessary step in fixing the above issues, but it also means that `prefix` is now capable of handling context trees with e.g. multiple prefixes at different levels of nesting.

## 0.34.1

  - Fix an issue that prevented merging two VarInfos if they had different dimensions for a variable.

  - Upper bound the compat version of KernelAbstractions to work around an issue in determining the right VarInfo type to use.

## 0.34.0

**Breaking**

  - `rng` argument removed from `values_as_in_model`, and `varinfo` made non-optional. This means that the only signatures allowed are
    
    ```
    values_as_in_model(::Model, ::Bool, ::AbstractVarInfo)
    values_as_in_model(::Model, ::Bool, ::AbstractVarInfo, ::AbstractContext)
    ```
    
    If you aren't using this function (it's probably only used in Turing.jl) then this won't affect you.

## 0.33.1

Reworked internals of `condition` and `decondition`.
There are no changes to the public-facing API, but internally you can no longer use `condition` and `decondition` on an `AbstractContext`, you can only use it on a `DynamicPPL.Model`. If you want to modify a context, use `ConditionContext` and `decondition_context`.

## 0.33.0

**Breaking**

  - `values_as_in_model()` now requires an extra boolean parameter, specifying whether variables on the lhs of `:=` statements are to be included in the resulting `OrderedDict` of values.
    The type signature is now `values_as_in_model([rng,] model, include_colon_eq::Bool [, varinfo, context])`

**Other**

  - Moved the implementation of `predict` from Turing.jl to DynamicPPL.jl; the user-facing behaviour is otherwise the same
  - Improved error message when a user tries to initialise a model with parameters that don't correspond strictly to the underlying VarInfo used
