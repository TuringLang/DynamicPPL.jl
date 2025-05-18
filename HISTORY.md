# DynamicPPL Changelog

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
