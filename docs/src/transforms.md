# [Transform strategies](@id transform-strategies)

Often it is useful to evaluate the log-probability of a model in a different space to the original one that it is defined in.

!!! note
    
    The [main Turing documentation site](https://turinglang.org/docs/developers/transforms/distributions/) has a more detailed introduction to variable transformations in MCMC sampling.
    This page only describes the implementation in DynamicPPL.

Consider the following model:

```@example 1
using DynamicPPL, Distributions

@model function f()
    x ~ LogNormal()
    y ~ LogNormal()
    return (x, y)
end
```

There are several ways in which we might want to evaluate this model:

  - **In the original ('untransformed') space:** we provide values for `x` and `y` which are both positive, and we evaluate the log-probability directly.
    This corresponds directly to
    
    ```@example 1
    x, y = 1.5, 2.0
    logp = logpdf(LogNormal(), x) + logpdf(LogNormal(), y)
    ```

  - **In unconstrained ('transformed') space**: we provide values for `x` and `y` in unconstrained space (i.e. they are real numbers), and evaluate the log-probability in transformed space.
    This corresponds to something like:
    
    ```@example 1
    x, y = 1.5, 2.0
    
    # To calculate the correct log-probability, we need to transform back to the
    # original space, but also account for the log-absolute-determinant of the
    # transformation Jacobian.
    trf_x, trf_y = log(x), log(y)
    logdetJx = -trf_x
    logdetJy = -trf_y
    
    logp = logpdf(LogNormal(), x) + logpdf(LogNormal(), y) - logdetJx - logdetJy
    ```
  - We might also want to have **a mix of transformed and untransformed variables**, for example, `x` in untransformed space and `y` in transformed space.
    This could be useful for example when using Gibbs sampling, or Metropolis–Hastings with different proposal distributions for different variables.

## `AbstractTransformStrategy`

DynamicPPL allows you to specify which variables you want to evaluate in transformed space using **transform strategies**.
All transform strategies are subtypes of `AbstractTransformStrategy`.
Currently, DynamicPPL provides the transform strategies [`LinkAll`](@ref), [`UnlinkAll`](@ref), [`LinkSome`](@ref), and [`UnlinkSome`](@ref).
Their meanings should be fairly self-explanatory; here is a brief demonstration:

```@example 1
params = @vnt begin
    x := 1.5
    y := 2.0
end
_, vi_unlinked = init!!(f(), OnlyAccsVarInfo(), InitFromParams(params), UnlinkAll())
vi_unlinked.accs
```

```@example 1
_, vi_linked = init!!(f(), OnlyAccsVarInfo(), InitFromParams(params), LinkAll())
vi_linked.accs
```

!!! note "Initialisation strategy does not determine log-Jacobian"
    
    In the above examples, we used `InitFromParams` to provide variable values.
    `InitFromParams` is an [initialisation strategy](./init.md) and when given a VarNamedTuple of values as we did above, it always interprets those values as being in untransformed space.
    
    This does not however mean that the log-Jacobian is disregarded!
    As we see in the second example above, when using `LinkAll()`, the log-Jacobian is still applied even though the values were provided in untransformed space.
    The *transform strategy* is what determines whether the log-Jacobian is applied or not when evaluating the log-probability.
    One could think of the transform strategy as being a *re-interpretation* of the value provided by the initialisation strategy.
    
    This frees up the initialisation strategy to return whatever kind of `AbstractTransformedValue` is most convenient for it.

## Making your own transform strategy

The only requirement for a subtype of an `AbstractTransformStrategy` is that it must implement [`target_transform(::AbstractTransformStrategy, vn::VarName)`](@ref DynamicPPL.target_transform), where `vn` is the variable on the left-hand side of a tilde-statement.

`target_transform` must in turn return an [`AbstractTransform`](@ref) specifying whether the variable should be transformed or not.

For example, the following would cause `x` to be transformed but not `y`.

```@example 1
struct LookupTransformsInVNT{V<:VarNamedTuple} <: AbstractTransformStrategy
    transforms::V
end

function DynamicPPL.target_transform(l::LookupTransformsInVNT, vn::VarName)
    return l.transforms[vn]
end

link_x_only = LookupTransformsInVNT(@vnt begin
    x := DynamicLink()
    y := Unlink()
end)

_, vi_link_x_only = init!!(f(), OnlyAccsVarInfo(), InitFromParams(params), link_x_only)
vi_link_x_only.accs
```

## `DynamicLink` in more detail

Looking at the two transform types used above, `Unlink()` is probably more intuitive: it just means 'do not interpret this variable as being in transformed space'.
However, `DynamicLink()` is a bit more subtle.

In particular, for `DynamicLink()`, the actual transformation used is obtained at runtime from the distribution on the right-hand side of the tilde-statement, using:

```@example 1
DynamicPPL.from_linked_vec_transform(LogNormal())
```

(which ultimately calls functions from Bijectors.jl).
This means that the transformation is recalculated on every model evaluation.

One might question why this is necessary: for example, in this simple model, we know that `x` and `y` are always `LogNormal`, so why not just store use the log-transform directly?

The answer is to do with distributions whose support (and hence transformation) can vary.
For example, consider:

```@example 1
@model function g()
    x ~ Normal()
    return y ~ truncated(Normal(); lower=x)
end
```

In this example, the support of `y` depends on what value `x` takes, and that can vary from one evaluation to the next.
Consequently, the transformation used for `y` must be determined at runtime; if we cache a fixed transformation, it is possible that this transformation will be invalid (e.g. by mapping unconstrained values to values outside the support of `y`).

For correctness, DynamicPPL therefore always prefers to determine the transformation at runtime when using `DynamicLink()`.
This behaviour is encoded in (for example) Turing's HMC samplers, which use `LinkAll()` as the default transform strategy, and hence every `VarName` will have a `target_transform` of `DynamicLink()`.

## Fixed transformations

For some models, it may be known that the support of a variable does not change, and that the transformations should be fixed.
This allows us to avoid the overhead of recomputing the transformation at every model evaluation.

This is currently not implemented, but there is a plan for it; see [this DynamicPPL issue](https://github.com/TuringLang/DynamicPPL.jl/issues/1249) for details.

## Why not let `init()` determine the transform?

!!! warning
    
    This section is mainly for developers and advanced users interested in the design decisions behind DynamicPPL; it has no real implications for everyday usage.

An alternative to having an explicit link strategy would be to simply allow the initialisation strategy to determine whether variables are transformed or not.
In this world, if `init(rng, vn, dist, strategy)` returned a transformed value, then we could treat it as being in transformed space, and vice versa.

The reason why we do this is to allow more flexibility in how models are evaluated, which can in turn save us from having to rerun the model multiple times.

Consider, for example, `DynamicPPL.InitFromUniform`.
This is an initialisation strategy which samples uniformly from `[-2, 2]` in transformed space, and is used by Turing's HMC samplers to generate initial values.
This used to be a standard workflow in Turing:

```@example 1
using Random
_, vi = init!!(Xoshiro(468), f(), VarInfo(), InitFromUniform())

# We'll run this later.
# vi_linked = link!!(vi, f())
```

In the first line, we are populating an empty `VarInfo` with values.
This initialisation always causes the `VarInfo` to be unlinked, because the behaviour of `init!!` on a `VarInfo` is to always retain the transform state of the `VarInfo` (which is empty at this point, and hence unlinked).

If we did not have a separate transform strategy, then we would have to make sure that `init()` always returned *untransformed* values when using `InitFromUniform` (otherwise we would be filling the `VarInfo` with transformed values, which is not what we want here).
So we have this extra step where `InitFromUniform` has to generate transformed values, untransform them, and then pass them on to the `VarInfo` so that it can store untransformed values.
(This was indeed [the way `InitFromUniform` was implemented in DynamicPPL <= v0.39](https://github.com/TuringLang/DynamicPPL.jl/blob/eea8d01c5fb217c1a0f4df170bf1ca16ee879c10/src/contexts/init.jl#L108-L119)).

If we then wanted to link the VarInfo again

```@example 1
vi_linked = link!!(vi, f())
```

then we would have to recompute the forward transform, which frustratingly enough is exactly the same as what `InitFromUniform` had to _undo_.
So we are calculating the same transformation twice, and evaluating the model twice, to make sure that our `VarInfo` ends up in the right state.

Instead, now with a separate transform strategy, we can immediately do:

```@example 1
_, vi_linked = init!!(Xoshiro(468), f(), VarInfo(), InitFromUniform(), LinkAll())
vi_linked
```

We can see that we have gotten exactly the same result, but with only running the model once, and only calculating the transformation once.
Furthermore, this allows us to remove the inverse transform step inside `InitFromUniform`: it can simply return a `LinkedVectorValue` directly, and the transform strategy is then responsible for performing the inverse transform a single time.

Essentially, having a separate transform strategy allows us to:

 1. Free up the initialisation strategy to return whatever kind of `AbstractTransformedValue` is most convenient for it, without worrying about whether it needs to perform some transform.

 2. Consolidate all the actual transformation in a single function (`DynamicPPL.apply_transform_strategy`), which allows us to ensure that each tilde-statement involves at most *one* transformation.
