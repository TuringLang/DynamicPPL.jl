# Nested Submodel Inference Note

This note explains why `_evaluate!!(::Submodel, ...)` calls `model.f` directly instead of
calling `_evaluate!!(model, vi)`.

## The Issue

The problem is not that `_evaluate!!(::Submodel, ...)` is recursive. It is recursive, but
Julia's recursion limiter does not widen that call.

The problem is the extra recursive call through `_evaluate!!(::Model, vi)`:

```julia
_evaluate!!(::Submodel, vi, parent_context, left_vn)
    -> _evaluate!!(::Model, vi)
```

At that call site, Julia infers an argument tuple type for `_evaluate!!(model, vi)`.
The second argument type is a concrete `Model{..., Ctx}`. With nested submodels, the
`Ctx` parameter grows at each level:

```julia
Model{..., PrefixContext{c, InitContext{...}}}
Model{..., PrefixContext{c, PrefixContext{b, InitContext{...}}}}
```

Julia sees the same `_evaluate!!(::Model, vi)` method recurring with a more complex call
signature, so it widens the signature to force convergence:

```julia
Tuple{typeof(_evaluate!!), Model, OnlyAccsVarInfo{...}}
```

Once `model` is only known as abstract `Model`, inference can no longer resolve
`model.f` precisely, and the return type collapses.

## Main vs Branch Call Chain

The old path was:

```text
_evaluate!!(::Model, vi)                         # top-level model
  -> model.f(...)
    -> tilde_assume!!(..., ::Submodel, ...)
      -> _evaluate!!(::Submodel, vi, context, vn)
        -> contextualize(submodel.model, eval_context)
        -> _evaluate!!(::Model, vi)              # problematic recursive call
          -> model.f(...)
            -> tilde_assume!!(..., ::Submodel, ...)
              -> _evaluate!!(::Submodel, ...)
```

The fixed path is:

```text
_evaluate!!(::Model, vi)                         # top-level model
  -> model.f(...)
    -> tilde_assume!!(..., ::Submodel, ...)
      -> _evaluate!!(::Submodel, vi, context, vn)
        -> contextualize(submodel.model, eval_context)
        -> make_evaluate_args_and_kwargs(model, vi)
        -> model.f(...)                          # direct call, no _evaluate!!(::Model)
          -> tilde_assume!!(..., ::Submodel, ...)
            -> _evaluate!!(::Submodel, ...)
```

The fix removes the nested `_evaluate!!(::Model, vi)` call edge. It does not remove
submodel recursion.

## Why The Submodel Recursion Is Fine

For a submodel tilde statement, the caller already has the concrete context and submodel
in its own signature:

```julia
tilde_assume!!(context, submodel, vn, NoTemplate, vi)
    -> _evaluate!!(submodel, vi, context, vn)
```

The callee signature mostly reorders values that were already present in the caller
signature. In Julia's `Compiler/src/typelimits.jl`, `type_more_complex` has an early exit
for this case:

```julia
elseif is_derived_type_from_any(unwrap_unionall(t), sources, depth)
    return false # t isn't something new
end
```

That is why a recursive `_evaluate!!(::Submodel, ...)` call can be eligible for limiting
but still not be widened: the growing context is not newly packaged into another dispatch
type. It is a call argument that was already visible in the caller.

## Why The Model Recursion Is Not Fine

The old implementation did this inside `_evaluate!!(::Submodel, ...)`:

```julia
model = contextualize(submodel.model, eval_context)
return _evaluate!!(model, vi)
```

That constructs a fresh `Model{..., Ctx}` type and then passes it to the shared
`_evaluate!!(::Model, vi)` method. The full contextualized model type was not already
present in the caller signature; it was synthesized inside the submodel evaluator.

So Julia compares recursive `_evaluate!!(::Model, vi)` signatures like:

```julia
Tuple{typeof(_evaluate!!), Model{..., PrefixContext{c, InitContext{...}}}, VI}
Tuple{typeof(_evaluate!!), Model{..., PrefixContext{c, PrefixContext{b, InitContext{...}}}}, VI}
```

The second call is more complex because the `Ctx` parameter inside `Model` is more deeply
nested. `limit_type_size` therefore widens the `Model` argument.

## Why Manual Inlining Works

The body of `_evaluate!!(::Model, vi)` is:

```julia
args, kwargs = make_evaluate_args_and_kwargs(model, vi)
return model.f(args...; kwargs...)
```

Doing those two lines directly inside `_evaluate!!(::Submodel, ...)` avoids creating the
recursive `_evaluate!!(::Model, vi)` call edge. The contextualized `model` value still
exists, but it is no longer passed through a recursive call to the shared
`_evaluate!!(::Model, vi)` method.

An `@inline` annotation on `_evaluate!!(::Model, vi)` is not enough. Julia applies this
recursion-limiting heuristic during abstract interpretation, before optimizer inlining.
