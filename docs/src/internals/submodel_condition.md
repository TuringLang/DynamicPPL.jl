# How `PrefixContext` and `ConditionContext` interact

```@meta
ShareDefaultModule = true
```

## PrefixContext

`PrefixContext` is a context that, as the name suggests, prefixes all variables inside a model with a given symbol.
Thus, for example:

```@example
using DynamicPPL, Distributions

@model function f()
    x ~ Normal()
    return y ~ Normal()
end

@model function g()
    return a ~ to_submodel(f())
end
```

inside the submodel `f`, the variables `x` and `y` become `a.x` and `a.y` respectively.
This is easiest to observe by running the model:

```@example
vi = VarInfo(g())
keys(vi)
```

!!! note
    
    In this case, where `to_submodel` is called without any other arguments, the prefix to be used is automatically inferred from the name of the variable on the left-hand side of the tilde.
    We will return to the 'manual prefixing' case later.

The phrase 'becoming' a different variable is a little underspecified: it is useful to pinpoint the exact location where the prefixing occurs, which is `tilde_assume`.
The method responsible for it is `tilde_assume(::PrefixContext, right, vn, vi)`: this attaches the prefix in the context to the `VarName` argument, before recursively calling `tilde_assume` with the new prefixed `VarName`.
This means that even though a statement `x ~ dist` still enters the tilde pipeline at the top level as `x`, if the model evaluation context contains a `PrefixContext`, any function from `tilde_assume` onwards will see `a.x` instead.

## ConditionContext

`ConditionContext` is a context which stores values of variables that are to be conditioned on.
These values may be stored as a `Dict` which maps `VarName`s to values, or alternatively as a `NamedTuple`.
The latter only works correctly if all `VarName`s are 'basic', in that they have an identity optic (i.e., something like `a.x` or `a[1]` is forbidden).
Because of this limitation, we will only use `Dict` in this example.

!!! note
    
    If a `ConditionContext` with a `NamedTuple` encounters anything to do with a prefix, its internal `NamedTuple` is converted to a `Dict` anyway, so it is quite reasonable to ignore the `NamedTuple` case in this exposition.

One can inspect the conditioning values with, for example:

```@example
@model function d()
    x ~ Normal()
    return y ~ Normal()
end

cond_model = d() | (@varname(x) => 1.0)
cond_ctx = cond_model.context
```

There are several internal functions that are used to determine whether a variable is conditioned, and if so, what its value is.

```@example
DynamicPPL.hasconditioned_nested(cond_ctx, @varname(x))
```

```@example
DynamicPPL.getconditioned_nested(cond_ctx, @varname(x))
```

These functions are in turn used by the function `DynamicPPL.contextual_isassumption`, which is largely the same as `hasconditioned_nested`, but also checks whether the value is `missing` (in which case it isn't really conditioned).

```@example
DynamicPPL.contextual_isassumption(cond_ctx, @varname(x))
```

!!! note
    
    Notice that (neglecting `missing` values) the return value of `contextual_isassumption` is the _opposite_ of `hasconditioned_nested`, i.e. for a variable that _is_ conditioned on, `contextual_isassumption` returns `false`.

If a variable `x` is conditioned on, then the effect of this is to set the value of `x` to the given value (while still including its contribution to the log probability density).
Since `x` is no longer a random variable, if we were to evaluate the model, we would find only one key in the `VarInfo`:

```@example
keys(VarInfo(cond_model))
```

## Joint behaviour: desiderata at the model level

When paired together, these two contexts have the potential to cause substantial confusion: `PrefixContext` modifies the variable names that are seen, which may cause them to be out of sync with the values contained inside the `ConditionContext`.

We begin by mentioning some high-level desiderata for their joint behaviour.
Take these models, for example:

```@example
# We define a helper function to unwrap a layer of SamplingContext, to
# avoid cluttering the print statements.
unwrap_sampling_context(ctx::DynamicPPL.SamplingContext) = ctx.context
unwrap_sampling_context(ctx::DynamicPPL.AbstractContext) = ctx
@model function inner()
    println("inner context: $(unwrap_sampling_context(__context__))")
    x ~ Normal()
    return y ~ Normal()
end

@model function outer()
    println("outer context: $(unwrap_sampling_context(__context__))")
    return a ~ to_submodel(inner())
end

# 'Outer conditioning'
with_outer_cond = outer() | (@varname(a.x) => 1.0)

# 'Inner conditioning'
inner_cond = inner() | (@varname(x) => 1.0)
@model function outer2()
    println("outer context: $(unwrap_sampling_context(__context__))")
    return a ~ to_submodel(inner_cond)
end
with_inner_cond = outer2()
```

We want that:

 1. `keys(VarInfo(outer()))` should return `[a.x, a.y]`;
 2. `keys(VarInfo(with_outer_cond))` should return `[a.y]`;
 3. `keys(VarInfo(with_inner_cond))` should return `[a.y]`,

**In other words, we can condition submodels either from the outside (point (2)) or from the inside (point (3)), and the variable name we use to specify the conditioning should match the level at which we perform the conditioning.**

This is an incredibly salient point because it means that submodels can be treated as individual, opaque objects, and we can condition them without needing to know what it will be prefixed with, or the context in which that submodel is being used.
For example, this means we can reuse `inner_cond` in another model with a different prefix, and it will _still_ have its inner `x` value be conditioned, despite the prefix differing.

!!! info
    
    In the current version of DynamicPPL, these criteria are all fulfilled. However, this was not the case in the past: in particular, point (3) was not fulfilled, and users had to condition the internal submodel with the prefixes that were used outside. (See [this GitHub issue](https://github.com/TuringLang/DynamicPPL.jl/issues/857) for more information; this issue was the direct motivation for this documentation page.)

## Desiderata at the context level

The above section describes how we expect conditioning and prefixing to behave from a user's perpective.
We now turn to the question of how we implement this in terms of DynamicPPL contexts.
We do not specify the implementation details here, but we will sketch out something resembling an API that will allow us to achieve the target behaviour.

**Point (1)** does not involve any conditioning, only prefixing; it is therefore already satisfied by virtue of the `tilde_assume` method shown above.

**Points (2) and (3)** are more tricky.
As the reader may surmise, the difference between them is the order in which the contexts are stacked.

For the _outer_ conditioning case (point (2)), the `ConditionContext` will contain a `VarName` that is already prefixed.
When we enter the inner submodel, this `ConditionContext` has to be passed down and somehow combined with the `PrefixContext` that is created when we enter the submodel.
We make the claim here that the best way to do this is to nest the `PrefixContext` _inside_ the `ConditionContext`.
This is indeed what happens, as can be demonstrated by running the model.

```@example
with_outer_cond();
nothing;
```

!!! info
    
    The `; nothing` at the end is purely to circumvent a Documenter.jl quirk where stdout is only shown if the return value of the final statement is `nothing`.
    If these documentation pages are moved to Quarto, it will be possible to remove this.

For the _inner_ conditioning case (point (3)), the outer model is not run with any special context.
The inner model will itself contain a `ConditionContext` will contain a `VarName` that is not prefixed.
When we run the model, this `ConditionContext` should be then nested _inside_ a `PrefixContext` to form the final evaluation context.
Again, we can run the model to see this in action:

```@example
with_inner_cond();
nothing;
```

Putting all of the information so far together, what it means is that if we have these two inner contexts (taken from above):

```@example
using DynamicPPL: PrefixContext, ConditionContext, DefaultContext

inner_ctx_with_outer_cond = ConditionContext(
    Dict(@varname(a.x) => 1.0), PrefixContext(@varname(a))
)
inner_ctx_with_inner_cond = PrefixContext(
    @varname(a), ConditionContext(Dict(@varname(x) => 1.0))
)
```

then we want both of these to be `true` (and thankfully, they are!):

```@example
DynamicPPL.hasconditioned_nested(inner_ctx_with_outer_cond, @varname(a.x))
```

```@example
DynamicPPL.hasconditioned_nested(inner_ctx_with_inner_cond, @varname(a.x))
```

This allows us to finally specify our task as follows:

(1) Given the correct arguments, we need to make sure that `hasconditioned_nested` and `getconditioned_nested` behave correctly.

(2) We need to make sure that both the correct arguments are supplied. In order to do so:

  - (2a) We need to make sure that when evaluating a submodel, the context stack is arranged such that `PrefixContext` is applied _inside_ the parent model's context, but _outside_ the submodel's own context.

  - (2b) We also need to make sure that the `VarName` passed to it is prefixed correctly.

## How do we do it?

(1) `hasconditioned_nested` and `getconditioned_nested` accomplish this by first 'collapsing' the context stack, i.e. they go through the context stack, remove all `PrefixContext`s, and apply those prefixes to any conditioned variables below it in the stack.
Once the `PrefixContext`s have been removed, one can then iterate through the context stack and check if any of the `ConditionContext`s contain the variable, or get the value itself.
For more details the reader is encouraged to read the source code.

(2a) We ensure that the context stack is correctly arranged by relying on the behaviour of `make_evaluate_args_and_kwargs`.
This function is called whenever a model (which itself contains a context) is evaluated with a separate ('external') context, and makes sure to arrange both of these contexts such that _the model's context is nested inside the external context_.
Thus, as long as prefixing is implemented by applying a `PrefixContext` on the outermost layer of the _inner_ model context, this will be correctly combined with an external context to give the behaviour seen above.

(2b) At first glance, it seems like `tilde_assume` can take care of the `VarName` prefixing for us (as described in the first section).
However, this is not actually the case: `contextual_isassumption`, which is the function that calls `hasconditioned_nested`, is much higher in the call stack than `tilde_assume` is.
So, we need to explicitly prefix it before passing it to `contextual_isassumption`.
This is done inside the `@model` macro, or technically, its subsidiary function `isassumption`.

## Nested submodels

Just in case the above wasn't complicated enough, we need to also be very careful when dealing with nested submodels, which have multiple layers of `PrefixContext`s which may be interspersed with `ConditionContext`s.
For example, in this series of nested submodels,

```@example
@model function charlie()
    x ~ Normal()
    y ~ Normal()
    return z ~ Normal()
end
@model function bravo()
    return b ~ to_submodel(charlie() | (@varname(x) => 1.0))
end
@model function alpha()
    return a ~ to_submodel(bravo() | (@varname(b.y) => 1.0))
end
```

we expect that the only variable to be sampled should be `z` inside `charlie`, or rather, `a.b.z` once it has been through the prefixes.

```@example
keys(VarInfo(alpha()))
```

The general strategy that we adopt is similar to above.
Following the principle that `PrefixContext` should be nested inside the outer context, but outside the inner submodel's context, we can infer that the correct context inside `charlie` should be:

```@example
big_ctx = PrefixContext(
    @varname(a),
    ConditionContext(
        Dict(@varname(b.y) => 1.0),
        PrefixContext(@varname(b), ConditionContext(Dict(@varname(x) => 1.0))),
    ),
)
```

We need several things to work correctly here: we need the `VarName` prefixing to behave correctly, and then we need to implement `hasconditioned_nested` and `getconditioned_nested` on the resulting prefixed `VarName`.
It turns out that the prefixing itself is enough to illustrate the most important point in this section, namely, the need to traverse the context stack in a _different direction_ to what most of DynamicPPL does.

Let's work with a function called `myprefix(::AbstractContext, ::VarName)` (to avoid confusion with any existing DynamicPPL function).
We should like `myprefix(big_ctx, @varname(x))` to return `@varname(a.b.x)`.
Consider the following naive implementation, which mirrors a lot of code in the tilde-pipeline:

```@example
using DynamicPPL: NodeTrait, IsLeaf, IsParent, childcontext, AbstractContext
using AbstractPPL: AbstractPPL

function myprefix(ctx::DynamicPPL.AbstractContext, vn::VarName)
    return myprefix(NodeTrait(ctx), ctx, vn)
end
function myprefix(::IsLeaf, ::AbstractContext, vn::VarName)
    return vn
end
function myprefix(::IsParent, ctx::AbstractContext, vn::VarName)
    return myprefix(childcontext(ctx), vn)
end
function myprefix(ctx::DynamicPPL.PrefixContext, vn::VarName)
    # The functionality to actually manipulate the VarNames is in AbstractPPL
    new_vn = AbstractPPL.prefix(vn, ctx.vn_prefix)
    # Then pass to the child context
    return myprefix(childcontext(ctx), new_vn)
end

myprefix(big_ctx, @varname(x))
```

This implementation clearly is not correct, because it applies the _inner_ `PrefixContext` before the outer one.

The right way to implement `myprefix` is to, essentially, reverse the order of two lines above:

```@example
function myprefix(ctx::DynamicPPL.PrefixContext, vn::VarName)
    # Pass to the child context first
    new_vn = myprefix(childcontext(ctx), vn)
    # Then apply this context's prefix
    return AbstractPPL.prefix(new_vn, ctx.vn_prefix)
end

myprefix(big_ctx, @varname(x))
```

This is a much better result!
The implementation of related functions such as `hasconditioned_nested` and `getconditioned_nested`, under the hood, use a similar recursion scheme, so you will find that this is a common pattern when reading the source code of various prefixing-related functions.
When editing this code, it is worth being mindful of this as a potential source of incorrectness.

!!! info
    
    If you have encountered left and right folds, the above discussion illustrates the difference between them: the wrong implementation of `myprefix` uses a left fold (which collects prefixes in the opposite order from which they are encountered), while the correct implementation uses a right fold.

## Loose ends 1: Manual prefixing

Sometimes users may want to manually prefix a model, for example:

```@example
@model function inner_manual()
    x ~ Normal()
    return y ~ Normal()
end

@model function outer_manual()
    return _unused ~ to_submodel(prefix(inner_manual(), :a), false)
end
```

In this case, the `VarName` on the left-hand side of the tilde is not used, and the prefix is instead specified using the `prefix` function.

The way to deal with this follows on from the previous discussion.
Specifically, we said that:

> [...] as long as prefixing is implemented by applying a `PrefixContext` on the outermost layer of the _inner_ model context, this will be correctly combined [...]

When automatic prefixing is used, this application of `PrefixContext` occurs inside the `tilde_assume!!` method.
In the manual prefixing case, we need to make sure that `prefix(submodel::Model, ::Symbol)` does the same thing, i.e. it inserts a `PrefixContext` at the outermost layer of `submodel`'s context.
We can see that this is precisely what happens:

```@example
@model f() = x ~ Normal()

model = f()
prefixed_model = prefix(model, :a)

(model.context, prefixed_model.context)
```

## Loose ends 2: FixedContext

Finally, note that all of the above also applies to the interaction between `PrefixContext` and `FixedContext`, except that the functions have different names.
(`FixedContext` behaves the same way as `ConditionContext`, except that unlike conditioned variables, fixed variables do not contribute to the log probability density.)
This generally results in a large amount of code duplication, but the concepts that underlie both contexts are exactly the same.
