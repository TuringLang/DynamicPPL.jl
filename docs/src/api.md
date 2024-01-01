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

## Fixing and unfixing

We can also _fix_ a collection of variables in a [`Model`](@ref) to certain using [`fix`](@ref).

This might seem quite similar to the aforementioned [`condition`](@ref) and its siblings,
but they are indeed different operations:

  - `condition`ed variables are considered to be _observations_, and are thus
    included in the computation [`logjoint`](@ref) and [`loglikelihood`](@ref),
    but not in [`logprior`](@ref).
  - `fix`ed variables are considered to be _constant_, and are thus not included
    in any log-probability computations.

The differences are more clearly spelled out in the docstring of [`fix`](@ref) below.

```@docs
fix
DynamicPPL.fixed
```

The difference between [`fix`](@ref) and [`condition`](@ref) is described in the docstring of [`fix`](@ref) above.

Similarly, we can [`unfix`](@ref) variables, i.e. return them to their original meaning:

```@docs
unfix
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

For converting a chain into a format that can more easily be fed into a `Model` again, for example using `condition`, you can use [`value_iterator_from_chain`](@ref).

```@docs
value_iterator_from_chain

```

Sometimes it can be useful to extract the priors of a model. This is the possible using [`extract_priors`](@ref).

```@docs
extract_priors
```

```@docs
NamedDist
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

DynamicPPL provides different data structures used in for storing samples and accumulation of the log-probabilities, all of which are subtypes of [`AbstractVarInfo`](@ref).

```@docs
AbstractVarInfo
```

But exactly how a [`AbstractVarInfo`](@ref) stores this information can vary.

#### `VarInfo`

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

#### `SimpleVarInfo`

```@docs
SimpleVarInfo
```

#### `VarNameVector`

```@docs
VarNameVector
```

#### Design

[`VarInfo`](@ref) is a fairly simple structure; it contains

  - a `logp` field for accumulation of the log-density evaluation, and
  - a `metadata` field for storing information about the realizations of the different variables.

Representing `logp` is fairly straight-forward: we'll just use a `Real` or an array of `Real`, depending on the context.

**Representing `metadata` is a bit trickier**. This is supposed to contain all the necessary information for each `VarName` to enable the different executions of the model + extraction of different properties of interest after execution, e.g. the realization / value corresponding used for, say, `@varname(x)`.

!!! note
    
    We want to work with `VarName` rather than something like `Symbol` or `String` as `VarName` contains additional structural information, e.g. a `Symbol("x[1]")` can be a result of either `var"x[1]" ~ Normal()` or `x[1] ~ Normal()`; these scenarios are disambiguated by `VarName`.

To ensure that `varinfo` is simple and intuitive to work with, we need the underlying `metadata` to replicate the following functionality of `Dict`:

  - `keys(::Dict)`: return all the `VarName`s present in `metadata`.
  - `haskey(::Dict)`: check if a particular `VarName` is present in `metadata`.
  - `getindex(::Dict, ::VarName)`: return the realization corresponding to a particular `VarName`.
  - `setindex!(::Dict, val, ::VarName)`: set the realization corresponding to a particular `VarName`.
  - `delete!(::Dict, ::VarName)`: delete the realization corresponding to a particular `VarName`.
  - `empty!(::Dict)`: delete all realizations in `metadata`.
  - `merge(::Dict, ::Dict)`: merge two `metadata` structures according to similar rules as `Dict`.

*But* for general-purpose samplers, we often want to work with a simple flattened structure, typically a `Vector{<:Real}`. Therefore we also want `varinfo` to be able to replicate the following functionality of `Vector{<:Real}`:

  - `getindex(::Vector{<:Real}, ::Int)`: return the i-th value in the flat representation of `metadata`.
    
      + For example, if `metadata` contains a realization of `x ~ MvNormal(zeros(3), I)`, then `getindex(varinfo, 1)` should return the realization of `x[1]`, `getindex(varinfo, 2)` should return the realization of `x[2]`, etc.

  - `setindex!(::Vector{<:Real}, val, ::Int)`: set the i-th value in the flat representation of `metadata`.
  - `length(::Vector{<:Real})`: return the length of the flat representation of `metadata`.
  - `similar(::Vector{<:Real})`: return a new

Moreover, we want also want the underlying representation used in `metadata` to have a few performance-related properties:

 1. Type-stable when possible, but still functional when not.
 2. Efficient storage and iteration.

In the following sections, we'll outline how we achieve this in [`VarInfo`](@ref).

##### Type-stability

This is somewhat non-trivial to address since we want to achieve this for both continuous (typically `Float64`) and discrete (typically `Int`) variables.

Suppose we have an implementation of `metadata` which implements the functionality outlined in the previous section. The way we approach this in `VarInfo` is to use a `NamedTuple` with a separate `metadata` *for each distinct `Symbol` used*. For example, if we have a model of the form

```@example varinfo-design
using DynamicPPL, Distributions

@model function demo()
    x ~ Bernoulli(0.5)
    return y ~ Normal(0, 1)
end
```

then we construct a type-stable representation by using a `NamedTuple{(:x, :y), Tuple{Vx, Vy}}` where

  - `Vx` is a container with `eltype` `Bool`, and
  - `Vy` is a container with `eltype` `Float64`.

Since `VarName` contains the `Symbol` used in its type, something like `getindex(varinfo, @varname(x))` can be resolved to `getindex(varinfo.metadata.x, @varname(x))` at compile-time.

For example, with the model above we have

```@example varinfo-design
# Type-unstable `VarInfo`
varinfo_untyped = DynamicPPL.untyped_varinfo(demo())
typeof(varinfo_untyped.metadata)
```

```@example varinfo-design
# Type-stable `VarInfo`
varinfo_typed = DynamicPPL.typed_varinfo(demo())
typeof(varinfo_typed.metadata)
```

But they both work as expected:

```@example varinfo-design
varinfo_untyped[@varname(x)], varinfo_untyped[@varname(y)]
```

```@example varinfo-design
varinfo_typed[@varname(x)], varinfo_typed[@varname(y)]
```

!!! warning
    
    Of course, this `NamedTuple` approach is *not* necessarily going to help us in scenarios where the `Symbol` does not correspond to a unique type, e.g.
    
    ```julia
    x[1] ~ Bernoulli(0.5)
    x[2] ~ Normal(0, 1)
    ```
    
    In this case we'll end up with a `NamedTuple((:x,), Tuple{Vx})` where `Vx` is a container with `eltype` `Union{Bool, Float64}` or something worse. This is *not* type-stable but will still be functional.
    
    In practice, we see that such mixing of types is not very common, and so in DynamicPPL and more widely in Turing.jl, we use a `NamedTuple` approach for type-stability with great success.

!!! warning
    
    Another downside with this approach is that if we have a model with lots of tilde-statements, e.g. `a ~ Normal()`, `b ~ Normal()`, ..., `z ~ Normal()` will result in a `NamedTuple` with 27 entries, potentially leading to long compilation times.

Hence we obtain a "type-stable when possible"-representation by wrapping it in a `NamedTuple` and partially resolving the `getindex`, `setindex!`, etc. methods at compile-time. When type-stability is *not* desired, we can simply use a `VarNameVector` for all `VarName`s instead of a `NamedTuple` wrapping `VarNameVector`s.

##### Efficient storage and iteration

In [`VarNameVector{<:VarName,Vector{T}}`](@ref), we achieve this by storing the values for different `VarName`s contiguously in a `Vector{T}` and keeping track of which ranges correspond to which `VarName`s.

This does require a bit of book-keeping, in particular when it comes to insertions and deletions. Internally, this is handled by assigning each `VarName` a unique `Int` index in the `varname_to_index` field, which is then used to index into the following fields:

  - `varnames::Vector{VarName}`: the `VarName`s in the order they appear in the `Vector{T}`.
  - `ranges::Vector{UnitRange{Int}}`: the ranges of indices in the `Vector{T}` that correspond to each `VarName`.
  - `transforms::Vector`: the transforms associated with each `VarName`.

Mutating functions, e.g. `setindex!`, are then treated according to the following rules:

 1. If `VarName` is not already present: add it to the end of `varnames`, add the value to the underlying `Vector{T}`, etc.

 2. If `VarName` is already present in the `VarNameVector`:
    
     1. If `value` has the *same length* as the existing value for `VarName`: replace existing value.
     2. If `value` has a *smaller length* than the existing value for `VarName`: replace existing value and mark the remaining indices as "inactive" by adding the range to the `inactive_ranges` field.
     3. If `value` has a *larger length* than the existing value for `VarName`: mark the entire current range for `VarName` as "inactive", expand the underlying `Vector{T}` to accommodate the new value, and update the `ranges` to point to the new range for this `VarName`.

This "delete-by-mark" instead of having to actually delete elements from the underlying `Vector{T}` ensures that `setindex!` will be fairly efficient; if we instead tried to insert a new larger value at the same location as the old value, then we would have to shift all the elements after the insertion point, potentially requiring a lot of memory allocations. This also means that the underlying `Vector{T}` can grow without bound, so we have the following methods to interact with the inactive ranges:

```@docs
DynamicPPL.has_inactive_ranges
DynamicPPL.inactive_ranges_sweep!
```

!!! note
    
    Higher-dimensional arrays, e.g. `Matrix`, are handled by simply vectorizing them before storing them in the `Vector{T}`, and composing he `VarName`'s transformation with a `DynamicPPL.FromVec`.

##### Additional methods

We also want some additional methods that are not part of the `Dict` or `Vector` interface:

  - `push!(container, ::VarName, value[, transform])`:  add a new element to the container, _but_ for this we also need the `VarName` to associate to the new `value`, so the semantics are different from `push!` for a `Vector`.

  - `update!(container, ::VarName, value[, transform])`: similar to `push!` but if the `VarName` is already present in the container, then we update the corresponding value instead of adding a new element.

In addition, we want to be able to access the transformed / "unconstrained" realization for a particular `VarName` and so we also need corresponding methods for this:

  - `getindex_raw` and `setindex_raw!` for extracting and mutating the, possibly unconstrained / transformed, realization for a particular `VarName`.

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
DynamicPPL.link
DynamicPPL.invlink
DynamicPPL.link!!
DynamicPPL.invlink!!
DynamicPPL.default_transformation
DynamicPPL.maybe_invlink_before_eval!!
DynamicPPL.reconstruct
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
