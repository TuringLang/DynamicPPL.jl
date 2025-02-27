# DynamicPPL Changelog

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

This is often more performant as well.

The new implementation of `x .~ ...` is just a short-hand for `x ~ filldist(...)`, which means that `x` will be seen as a single multivariate variable. In most cases this does not change anything for the user, with the one notable exception being `pointwise_loglikelihoods`, which previously treated `.~` assignments as assigning multiple univariate variables. If you _do_ want a variable to be seen as an array of univariate variables rather than a single multivariate variable, you can always expand into a loop, such as

```julia
dists = Normal.(y)
for i in 1:length(dists)
    x[i] ~ dists[i]
end
```

Cases where the right hand side is of a different dimension than the left hand side, and neither is a scalar, must always be replaced with a loop. For example,

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

  - The method `LogDensityFunction(varinfo, model, context)` has been removed.
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
