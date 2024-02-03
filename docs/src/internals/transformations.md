# Transforming variables

## Motivation

In a probabilistic programming language (PPL) such as DynamicPPL.jl, one crucial functionality for enabling a large number of inference algorithms to be implemented, in particular gradient-based ones, is the ability to work with "unconstrained" variables.

For example, consider the following model:

```julia
@model function demo()
    s ~ InverseGamma(2, 3)
    return m ~ Normal(0, √s)
end
```

Here we have two variables `s` and `m`, where `s` is constrained to be positive, while `m` can be any real number.

For certain inference methods, it's necessary / much more convenient to work with an equivalent model to `demo` but where all the variables can take any real values (they're "unconstrained").

!!! note
    
    We write "unconstrained" with quotes because there are many ways to transform a constrained variable to an unconstrained one, *and* DynamicPPL can work with a much broader class of bijective transformations of variables, not just ones that go to the entire real line. But for MCMC, unconstraining is the most common transformation so we'll stick with that terminology.

For a large family of constraints encoucntered in practice, it is indeed possible to transform a (partially) contrained model to a completely unconstrained one in such a way that sampling in the unconstrained space is equivalent to sampling in the constrained space.

In DynamicPPL.jl, this is often referred to as *linking* (a term originating in the statistics literature) and is done using transformations from [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl).

For example, the above model could be transformed into (the following psuedo-code; it's not working code):

```julia
@model function demo()
    log_s ~ log(InverseGamma(2, 3))
    s = exp(log_s)
    return m ~ Normal(0, √s)
end
```

Here `log_s` is an unconstrained variable, and `s` is a constrained variable that is a deterministic function of `log_s`.

But to ensure that we stay consistent with what the user expects, DynamicPPL.jl does not actually transform the model as above, but can instead makes use of transformed variables internally to achieve the same effect, when desired.

In the end, we'll end up with something that looks like this:

```@raw html
<div style="flex-direction: row; display: flex; justify-content: space-around; margin: 1em;">
<img style="border-radius: 5px;" src="../../assets/images/transformations.dot.png" />
</div>
```

Below we'll see how this is done.

## What do we need?

There are two aspects to transforming from the internal representation of a variable in a `varinfo` to the representation wanted in the model:

 1. Different implementations of [`AbstractVarInfo`](@ref) represent realizations of a model in different ways internally, so we need to transform from this internal representation to the desired representation in the model. For example,
    
      + [`VarInfo`](@ref) represents a realization of a model as in a "flattened" / vector representation, regardless of form of the variable in the model.
      + [`SimpleVarInfo`](@ref) represents a realization of a model exactly as in the model (unless it has been transformed; we'll get to that later).

 2. We need the ability to transform from "constrained space" to "unconstrained space", as we saw in the previous section.

## Working example

A good and non-trivial example to keep in mind throughout is the following model:

```@example transformations-internal
using DynamicPPL, Distributions
@model demo_lkj() = x ~ LKJCholesky(2, 1.0)
```

`LKJCholesky` is a `LKJ(2, 1.0)` distribution, a distribution over correlation matrices (covariance matrices but with unit diagonal), but working directly with the Cholesky factorization of the correlation matrix rather than the correlation matrix itself (this is more numerically stable and computationally efficient).

!!! note
    
    This is a particularly "annoying" case because the return-value is not a simple `Real` or `AbstractArray{<:Real}`, but rather a `LineraAlgebra.Cholesky` object which wraps a triangular matrix (whether it's upper- or lower-triangular depends on the instance).

As mentioned, some implementations of `AbstractVarInfo`, e.g. [`VarInfo`](@ref), works with a "flattened" / vector representation of a variable, and so in this case we need two transformations:

 1. From the `Cholesky` object to a vector representation.
 2. From the `Cholesky` object to an "unconstrained" / linked vector representation.

And similarly, we'll need the inverses of these transformations.

## From internal representation to model representation

To go from the internal variable representation of an `AbstractVarInfo` to the variable representation wanted in the model, e.g. from a `Vector{Float64}` to `Cholesky` in the case of [`VarInfo`](@ref) in `demo_lkj`, we have the following methods:

```@docs
DynamicPPL.to_internal_transform
DynamicPPL.from_internal_transform
```

These methods allows us to extract the internal-to-model transformation function depending on the `varinfo`, the variable, and the distribution of the variable:

  - `varinfo` + `vn` defines the internal representation of the variable.
  - `dist` defines the representation expected within the model scope.

!!! note
    
    If `vn` is not present in `varinfo`, then the internal representation is fully determined by `varinfo` alone. This is used when we're about to add a new variable to the `varinfo` and need to know how to represent it internally.

Continuing from the example above, we can inspect the internal representation of `x` in `demo_lkj` with [`VarInfo`](@ref) using [`DynamicPPL.getindex_internal`](@ref):

```@example transformations-internal
model = demo_lkj()
varinfo = VarInfo(model)
x_internal = DynamicPPL.getindex_internal(varinfo, @varname(x))
```

```@example transformations-internal
f_from_internal = DynamicPPL.from_internal_transform(
    varinfo, @varname(x), LKJCholesky(2, 1.0)
)
f_from_internal(x_internal)
```

Let's confirm that this is the same as `varinfo[@varname(x)]`:

```@example transformations-internal
x_model = varinfo[@varname(x)]
```

Similarly, we can go from the model representation to the internal representation:

```@example transformations-internal
f_to_internal = DynamicPPL.to_internal_transform(varinfo, @varname(x), LKJCholesky(2, 1.0))

f_to_internal(x_model)
```

It's also useful to see how this is done in [`SimpleVarInfo`](@ref):

```@example transformations-internal
simple_varinfo = SimpleVarInfo(varinfo)
DynamicPPL.getindex_internal(simple_varinfo, @varname(x))
```

Here see that the internal representation is exactly the same as the model representation, and so we'd expect `from_internal_transform` to be the `identity` function:

```@example transformations-internal
DynamicPPL.from_internal_transform(simple_varinfo, @varname(x), LKJCholesky(2, 1.0))
```

Great!

## From *unconstrained* internal representation to model representation

In addition to going from internal representation to model representation of a variable, we also need to be able to go from the *unconstrained* internal representation to the model representation.

For this, we have the following methods:

```@docs
DynamicPPL.to_linked_internal_transform
DynamicPPL.from_linked_internal_transform
```

These are very similar to [`DynamicPPL.to_internal_transform`](@ref) and [`DynamicPPL.from_internal_transform`](@ref), but here the internal representation is also linked / "unconstrained".

Continuing from the example above:

```@example transformations-internal
f_to_linked_internal = DynamicPPL.to_linked_internal_transform(
    varinfo, @varname(x), LKJCholesky(2, 1.0)
)

x_linked_internal = f_to_linked_internal(x_model)
```

```@example transformations-internal
f_from_linked_internal = DynamicPPL.from_linked_internal_transform(
    varinfo, @varname(x), LKJCholesky(2, 1.0)
)

f_from_linked_internal(x_linked_internal)
```

Here we see a significant difference between the linked representation and the non-linked representation: the linked representation is only of length 1, whereas the non-linked representation is of length 4. This is because we actually only need a single element to represent a 2x2 correlation matrix, as the diagonal elements are always 1 *and* it's symmetric.

We can also inspect the transforms themselves:

```@example transformations-internal
f_from_internal
```

vs.

```@example transformations-internal
f_from_linked_internal
```

Here we see that `f_from_linked_internal` is a single function taking us directly from the linked representation to the model representation, whereas `f_from_internal` is a composition of a few functions: one reshaping the underlying length 4 array into 2x2 matrix, and the other converting this matrix into a `Cholesky`, as required to be compatible with `LKJCholesky(2, 1.0)`.

## Why do we need both `to_internal_transform` and `to_linked_internal_transform`?

One might wonder why we need both `to_internal_transform` and `to_linked_internal_transform` instead of just a single `to_internal_transform` which returns the "standard" internal representation if the variable is not linked / "unconstrained" and the linked / "unconstrained" internal representation if it is.

That is, why can't we just do

```@raw html
<div style="flex-direction: row; display: flex; justify-content: space-around; margin: 1em;">
<img style="border-radius: 5px;" src="../../assets/images/transformations-assume-without-istrans.dot.png" />
</div>
```

Unfortunately, this is not possible in general. Consider for example the following model:

```@example transformations-internal
@model function demo_dynamic_constraint()
    m ~ Normal()
    x ~ truncated(Normal(); lower=m)

    return (m=m, x=x)
end
```

Here the variable `x` has is constrained to be on the domain `(m, Inf)`, where `m` is sampled according to a `Normal`.

```@example transformations-internal
model = demo_dynamic_constraint()
varinfo = VarInfo(model)
varinfo[@varname(m)], varinfo[@varname(x)]
```

We see that the realization of `x` is indeed greater than `m`, as expected.

But what if we [`link`](@ref) this `varinfo` so that we end up working on an "unconstrained" space, i.e. both `m` and `x` can take on any values in `(-Inf, Inf)`:

```@example transformations-internal
varinfo_linked = link(varinfo, model)
varinfo_linked[@varname(m)], varinfo_linked[@varname(x)]
```

Still get the same values, as expected, since internally `varinfo` transforms from the linked internal representation to the model representation.

But what if we change the value of `m`, to, say, a bit larger than `x`?

```@example transformations-internal
# Update realization for `m` in `varinfo_linked`.
varinfo_linked[@varname(m)] = varinfo_linked[@varname(x)] + 1
varinfo_linked[@varname(m)], varinfo_linked[@varname(x)]
```

Now we see that the constraint `m < x` is no longer satisfied!

Hence one might expect that if we try to compute, say, the [`logjoint`](@ref) using `varinfo_linked` with this "invalid" realization, we'll get an error:

```@example transformations-internal
logjoint(model, varinfo_linked)
```

But we don't! In fact, if we look at the actual value used within the model

```@example transformations-internal
first(DynamicPPL.evaluate!!(model, varinfo_linked, DefaultContext()))
```

we see that we indeed satisfy the constraint `m < x`, as desired.

!!! warning
    
    One shouldn't be setting variables in a linked `varinfo` willy-nilly directly like this unless one knows that the value will be compatible with the constraints of the model.

The reason for this is that internally in a model evaluation, we construct the transformation from the internal to the model representation based on the *current* realizations in the model! That is, we take the `dist` in a `x ~ dist` expression _at model evaluation time_ and use that to construct the transformation, thus allowing it to change between model evaluations without invalidating the transformation.

But to be able to do this, we need to know whether the variable is linked / "unconstrained" or not, since the transformation is different in the two cases. Hence we need to be able to determine this at model evaluation time. Hence the the internals end up looking something like this:

```julia
if istrans(varinfo, varname)
    from_linked_internal_transform(varinfo, varname, dist)
else
    from_internal_transform(varinfo, varname, dist)
end
```

That is, if the variable is linked / "unconstrained", we use the [`DynamicPPL.from_linked_internal_transform`](@ref), otherwise we use [`DynamicPPL.from_internal_transform`](@ref).

And so the earlier diagram becomes:

```@raw html
<div style="flex-direction: row; display: flex; justify-content: space-around; margin: 1em;">
<img style="border-radius: 5px;" src="../../assets/images/transformations-assume.dot.png" />
</div>
```

!!! note
    
    If the support of `dist` was constant, this would not be necessary since we could just determine the transformation at the time of `varinfo_linked = link(varinfo, model)` and define this as the `from_internal_transform` for all subsequent evaluations. However, since the support of `dist` is *not* constant in general, we need to be able to determine the transformation at the time of the evaluation *and* thus whether we should construct the transformation from the linked internal representation or the non-linked internal representation. This is annoying, but necessary.

This is also the reason why we have two definitions of `getindex`:

  - [`getindex(::AbstractVarInfo, ::VarName, ::Distribution)`](@ref): used internally in model evaluations with the `dist` in a `x ~ dist` expression.
  - [`getindex(::AbstractVarInfo, ::VarName)`](@ref): used externally by the user to get the realization of a variable.

For `getindex` we have the following diagram:

```@raw html
<div style="flex-direction: row; display: flex; justify-content: space-around; margin: 1em;">
<img style="border-radius: 5px;" src="../../assets/images/transformations-getindex-with-dist.dot.png" />
</div>
```

While if `dist` is not provided, we have:

```@raw html
<div style="flex-direction: row; display: flex; justify-content: space-around; margin: 1em;">
<img style="border-radius: 5px;" src="../../assets/images/transformations-getindex-without-dist.dot.png" />
</div>
```

Notice that `dist` is not present here, but otherwise the diagrams are the same.

!!! warning
    
    This does mean that the `getindex(varinfo, varname)` might not be the same as the `getindex(varinfo, varname, dist)` that occurs within a model evaluation! This can be confusing, but as outlined above, we do want to allow the `dist` in a `x ~ dist` expression to "override" whatever transformation `varinfo` might have.

## Other functionalities

There are also some additional methods for transforming between representations that are all automatically implemented from [`DynamicPPL.from_internal_transform`](@ref), [`DynamicPPL.from_linked_internal_transform`](@ref) and their siblings, and thus don't need to be implemented manually.

Convenience methods for constructing transformations:

```@docs
DynamicPPL.from_maybe_linked_internal_transform
DynamicPPL.to_maybe_linked_internal_transform
DynamicPPL.internal_to_linked_internal_transform
DynamicPPL.linked_internal_to_internal_transform
```

Convenience methods for transforming between representations without having to explicitly construct the transformation:

```@docs
DynamicPPL.to_maybe_linked_internal
DynamicPPL.from_maybe_linked_internal
```

# Supporting a new distribution

To support a new distribution, one needs to implement for the desired `AbstractVarInfo` the following methods:

  - [`DynamicPPL.from_internal_transform`](@ref)
  - [`DynamicPPL.from_linked_internal_transform`](@ref)

At the time of writing, [`VarInfo`](@ref) is the one that is most commonly used, whose internal representation is always a `Vector`. In this scenario, one can just implement the following methods instead:

```@docs
DynamicPPL.from_vec_transform(::Distribution)
DynamicPPL.from_linked_vec_transform(::Distribution)
```

These are used internally by [`VarInfo`](@ref).

Optionally, if `inverse` of the above is expensive to compute, one can also implement:

  - [`DynamicPPL.to_internal_transform`](@ref)
  - [`DynamicPPL.to_linked_internal_transform`](@ref)

And similarly, there are corresponding to-methods for the `from_*_vec_transform` variants too

```@docs
DynamicPPL.to_vec_transform
DynamicPPL.to_linked_vec_transform
```

!!! warning
    
    Whatever the resulting transformation is, it should be invertible, i.e. implement `InverseFunctions.inverse`, and have a well-defined log-abs-det Jacobian, i.e. implement `ChangesOfVariables.with_logabsdet_jacobian`.

# TL;DR

  - DynamicPPL.jl has three representations of a variable: the **model representation**, the **internal representation**, and the **linked internal representation**.
    
      + The **model representation** is the representation of the variable as it appears in the model code / is expected by the `dist` on the right-hand-side of the `~` in the model code.
      + The **internal representation** is the representation of the variable as it appears in the `varinfo`, which varies between implementations of [`AbstractVarInfo`](@ref), e.g. a `Vector` in [`VarInfo`](@ref). This can be converted to the model representation by [`DynamicPPL.from_internal_transform`](@ref).
      + The **linked internal representation** is the representation of the variable as it appears in the `varinfo` after [`link`](@ref)ing. This can be converted to the model representation by [`DynamicPPL.from_linked_internal_transform`](@ref).

  - Having separation between *internal* and *linked internal* is necessary because transformations might be constructed at the time of model evaluation, and thus we need to know whether to construct the transformation from the internal representation or the linked internal representation.
