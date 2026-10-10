# Migrating old `VarInfo` code

`OnlyAccsVarInfo` is removed. Use `VarInfo`, which has the same accumulator constructor
forms: `VarInfo(accs...)`, `VarInfo(accs::Tuple)`, and `VarInfo(accs::AccumulatorTuple)`.
The old `VarInfo{Tfm,T,Accs}` is now `VarInfo{Accs}`; update dispatch that uses the old
type parameters. Transform strategies are evaluation inputs, not part of the output type.
`VarInfo()` records log densities, not parameter values; `keys`, `haskey`, `length`, `values`
and `isempty` throw `ArgumentError` without a `VectorValueAccumulator`.
Recording values is opt-in because most evaluations, such as
sampling and gradients, need only log densities. Add a `RawValueAccumulator`
or `VectorValueAccumulator` when those outputs are needed, for example
`VarInfo(VectorValueAccumulator(), DynamicPPL.default_accumulators()...)`; `VarInfo(model)` remains
a convenience constructor that records vectorised values and log densities.

To reuse previous values, extract them explicitly before evaluating:

```julia
context = InitContext(rng, InitFromParams(get_vector_values(previous), nothing), LinkAll())
retval, outputs = evaluate!!(model, context, VarInfo())
```

Use `get_raw_values(previous)` for a raw-value accumulator. Choose `UnlinkAll()`,
`LinkAll()`, or a partial/fixed transform strategy for the desired output representation.
`init!!` previously inferred its default transform strategy from the `VarInfo`.
It now defaults to `UnlinkAll()` regardless of the recorded values. Replace
`init!!(rng, model, vi, init)` with `init!!(rng, model, vi, init, strategy)` when the
outputs should use a different representation.
For custom strategies, implement `get_param_eltype(strategy)` when evaluation needs
to promote argument storage or thread-local accumulators for AD.

All output accumulators reset before evaluation. Previously recorded LHS variables that are
not executed again are absent from the new outputs. Replace `vi.values` with
`get_vector_values(vi)`; there is no separate parameter store. Previously, `get_values(vi)`
returned the always-present parameter store. It now returns the `VarNamedTuple` of vectorised
`TransformedValue`s from the optional `VectorValueAccumulator`, and throws `ArgumentError`
if that accumulator is absent. Replace unconditional `get_values(VarInfo())` calls with
evaluation into `VarInfo(VectorValueAccumulator())` followed by `get_vector_values(vi)`;
use `RawValueAccumulator` and `get_raw_values(vi)` for model-space values.

Please get in touch if you have some old code you're unsure how to migrate, and we will be happy to add it to this list.

```@example 1
using DynamicPPL, Distributions, Random
using BangBang: BangBang

@model function f()
    x ~ Normal()
    y ~ LogNormal()
    return 1.0 ~ Normal(x + y)
end

model = f()
```

## Sampling from the prior

Old:

```julia
vi = VarInfo(Xoshiro(468), model)
```

New:

```@example 1
accs = VarInfo()
_, vi = init!!(Xoshiro(468), model, accs, InitFromPrior(), UnlinkAll())
vi
```

## Getting parameter values

Old:

```julia
vi = VarInfo(Xoshiro(468), model)
vi.values[@varname(x)]
```

New:

```@example 1
# Set to true if you want to include results of `:=` statements.
accs = VarInfo(RawValueAccumulator(false))
_, vi = init!!(Xoshiro(468), model, accs, InitFromPrior(), UnlinkAll())
get_raw_values(vi)
```

## Generating vectorised parameters from linked VarInfo

Old:

```julia
vi = VarInfo(Xoshiro(468), model)
vi = DynamicPPL.link!!(vi, model)
vi[:]
```

The new pattern recognises that in practice you are likely using `vi[:]` [in conjunction with a `LogDensityFunction`](@ref ldf).
So we make one first:

```@example 1
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll())
nothing # hide
```

Then you can do:

```@example 1
rand(Xoshiro(468), ldf)
```

This gives you a set of parameters, but if you want to *also* obtain the log-density at the new parameters, you can do this in a single call to `init!!`; please see the [documentation on `LogDensityFunction`](@ref ldf-model) for more details on how to do this.

## Re-evaluating log density at new parameters

Old:

```julia
vi = VarInfo(Xoshiro(468), model)

vals = [1.0, 1.0]
vi = DynamicPPL.unflatten!!(vi, vals)
_, vi = DynamicPPL.evaluate!!(model, vi)
vi
```

The new path *also* assumes that you are using a `LogDensityFunction`:

```@example 1
# Note that we use `UnlinkAll()` here to match the VarInfo above.
# If your VarInfo was linked, you should use `LinkAll()` instead.

ldf = LogDensityFunction(model, getlogjoint_internal, UnlinkAll())
```

Then you can do:

```@example 1
vals = [1.0, 1.0]
init_strategy = InitFromVector(vals, ldf)

vi = VarInfo()
_, vi = init!!(Xoshiro(468), model, vi, init_strategy, ldf.transform_strategy)
vi
```

## Partial linking and unlinking

Whole-model `link!!(vi, model)` and `invlink!!(vi, model)` remain available.
The partial forms, including non-mutating `link` and `invlink`, are removed.
The second argument of `LinkSome` and `UnlinkSome`, `fallback`, is the strategy for
all variables outside `vns`. The old forms kept their current transforms; pass the
strategy that produced `vi` as `fallback` to do the same. For example, if `y` is
already linked, `LinkSome(Set([@varname(x)]), UnlinkAll())` links `x` but unlinks
`y`; use `LinkSome(Set([@varname(x), @varname(y)]), UnlinkAll())` to keep it linked.

Old:

```julia
vi = VarInfo(Xoshiro(468), model)
vns = (@varname(x),)
vi = DynamicPPL.link!!(vi, vns, model)
vi = DynamicPPL.invlink!!(vi, vns, model)
```

New:

```@example 1
rng = Xoshiro(468)
vi = VarInfo(rng, model)
vns = (@varname(x),)
fallback = UnlinkAll()
linked = LinkSome(Set(vns), fallback)
_, vi = init!!(rng, model, vi, InitFromParams(get_vector_values(vi), nothing), linked)
_, vi = init!!(
    rng,
    model,
    vi,
    InitFromParams(get_vector_values(vi), nothing),
    UnlinkSome(Set(vns), linked),
)
vi
```

To preserve the input `vi`, pass `copy(vi)` as the output argument to `init!!`.
Unlike the old partial-link helpers, `init!!` resets and recomputes all accumulators.

## Replacing individual values and transform state

`setindex_with_dist!!` and `update_transform_strategy` are removed.
Prepare named input values separately and pass the desired strategy to `init!!`.
Use `LinkSome`, `UnlinkSome`, or `WithTransforms` to specify partial or fixed transforms.

Old:

```julia
vi = VarInfo(Xoshiro(468), model)
vi = DynamicPPL.setindex_with_dist!!(
    vi, TransformedValue(2.0, NoTransform()), Normal(), @varname(x), nothing
)
_, vi = DynamicPPL.evaluate!!(model, vi)
```

New:

```@example 1
vi = VarInfo(Xoshiro(468), model)
params = get_vector_values(vi)
params = BangBang.setindex!!(params, TransformedValue(2.0, NoTransform()), @varname(x))
_, vi = init!!(Xoshiro(468), model, vi, InitFromParams(params, nothing), UnlinkAll())
vi
```

## Binding arguments and data

Arguments on the LHS are observed by default. A whole `missing` or `nothing` argument,
including `@model gdemo(x=missing)` called as `gdemo()`, and `missing` elements of a top-level
argument array still supply no observation. Other `missing` and `nothing` values, such as
`x[i][j]` or NamedTuple fields, throw where a tilde reads them; replace them with
`decondition(model, @varname(...))`. See [Missing data](@ref) for a complete example.

For placeholder values in explicit bindings, replace `condition(m; x=missing)` with
`decondition(m, @varname(x))` when `x` is observed, and `fix(m; x=nothing)` with
`unfix(m, @varname(x))` when `x` is fixed. Leave already latent LHS variables unbound.
Replace `InitFromParams((; x=missing))` with `InitFromParams((;))` to use the fallback.

Explicit observations of arguments now act through the argument before the body:
`condition(f(1); x=2)` observes as `f(2)` does. Replace attempts to observe the original
bound `x` after transforming it in the body with a separate LHS variable for the raw data.

Fixed bindings shadow observations. Replace reliance on `unfix` restoring construction-time
arguments with explicit `condition` or `decondition` calls to retain or remove the desired
observations. `unfix` uncovers only surviving observations; `decondition` removes them even
beneath a fixed binding. Enclosing explicit bindings override child bindings, including fixed
ones. To retain a child's fixed value, remove the enclosing observation with
`decondition(parent, @varname(a.x))` or the enclosing fixed binding with `unfix`.

Replace partial edits of tuple or struct owners with whole replacements, at any depth:
`fix(f((1.0, 2.0)), @varname(x[1]) => 9.0)` becomes `fix(f((1.0, 2.0)); x=(9.0, 2.0))`.
Partial removals likewise require removing the enclosing owner whole. Reading argument
observations and binding complete local LHS addresses, including produced `VarNamedTuple`s,
remain supported.

Replace `condition(m, Dict(@varname(x) => v))` with `condition(m, @varname(x) => v)`,
a NamedTuple, or keyword arguments. Likewise, replace `m | Dict(@varname(x) => v)` with
`m | (@varname(x) => v)` or `m | (x=v,)`; `|` no longer accepts `AbstractDict`.
`:x => v` remains shorthand for `@varname(x) => v`.
Replace `@vnt` or `@template` storage for partial local bindings with a positional binding
template, for example `condition(m, @varname(z[2]) => 1.0, @of(z = of(Array, 3)))`, importing
`of, @of` from AbstractPPL. Produced `VarNamedTuple` values remain accepted.

Whole bindings must satisfy declared argument types, local storage types, and shared
signature constraints. Replace incompatible replacements with compatible storage, or
reconstruct the model. Partial bindings require exact conversion to the element or field
type: replace `0.1` bound into `Float32` storage with `Float32(0.1)`. For runtime AD
bindings, replace fixed `Float64` storage with storage built from running values, such as
`fill(zero(m), n)` or `@of(z = of(Array, typeof(m), n))`.

Replace slice bindings that resize enclosing storage, such as
`condition(m, @varname(x[1:2]) => ones(3))`, with `condition(m; x=ones(3))`.
Replace resizing fixed arguments in the body with covering values of static size and shape.
Shape validation stops at the LHS variable's address; it does not inspect nested values
inside a whole structured LHS variable. Replace construction-time indices in later binding
edits with indices of the latest enclosing bound value: after `condition(f(zeros(2)); x=ones(3))`,
`decondition(m, @varname(x[3]))` preserves length three.

Known-invalid addresses now throw. Replace bindings of covariates with model reconstruction;
correct unknown LHS names, nonexistent fields, and indices outside storage. Replace
NamedTuple integer or Symbol addresses such as `x[1]` or `x[:a]` with field addresses such
as `x.a`, both in bindings and on the LHS. `decondition` and `unfix` reject the same
addresses when the model can decide: covariates, names with no LHS variable, nonexistent
fields, and indices outside storage. Previously such removals were silent. Removing a valid
address with no stored binding, including removing it twice, is a no-op. To remove a child's
argument-supplied observation from its parent, use `decondition(parent, DynamicPPL.Recursive(), @varname(a.x))`. Use `decondition(parent, DynamicPPL.Recursive())`
for prior prediction throughout the model; `unfix(parent, DynamicPPL.Recursive())` uncovers
observations at every depth. Replace partial removal from one multivariate LHS variable,
such as `decondition(m, @varname(x[1]))` for `x ~ MvNormal(...)`, with a model declaring
separate LHS variables when their roles must differ.

Submodel return values cannot be bound. For local `a ~ to_submodel(child())`, replace
`condition(m; a=value)` with `condition(m, @varname(a.x) => value)` to observe the child's
`x`. Submodel tildes rooted at a model argument now throw `ArgumentError` when they run,
including indexed and field addresses. Replace `a ~ to_submodel(child())` with
`result ~ to_submodel(child())` when `a` is an argument. Bind the child's LHS variables
with `condition`; if needed, copy its return value afterwards with `a = result`.

Argument arrays are no longer defensively copied unless they hold `missing` elements, which
are snapshotted at construction. Replace mutation of argument storage after construction
with constructing the model again. Partial bindings snapshot the
remaining storage: replace mutation of original arguments after binding with rebuilding the
partial binding. Replace manual merges of argument data into `conditioned(m)` with
`conditioned(m)` itself; replace inspection of shadowed observations there with
`conditioned(unfix(m))`.

Keyword splats in `model.defaults` are nested under their generated name. For
`@model f(; kw...)`, replace `model.defaults.z` with `model.defaults.var"#splat#kw".z`.
Replace individual bindings such as `condition(m, @varname(kw.y) => 2)` with
`condition(m; kw=(; y=2))`. Inner-function LHS variables cannot shadow model arguments:
if `x` is a model argument, replace `(x -> x ~ Normal())` with `(z -> z ~ Normal())`.

Replace construction or dispatch using `Model{Threaded,missings}` with `@model` and
`decondition` for latent arguments. For direct construction, replace
`Model{false}(f, args, defaults)` with `Model{false}(f, args, defaults; args_on_lhs=(:y,))`
to declare argument-supplied observations of `y`. Handwritten evaluators must replace direct
argument use with the argument-preparation and binding-aware tilde protocol of `@model`,
or reuse an `@model` evaluator. In debug introspection, replace the assumption that
`gen_evaluator_call_with_types(m)[1] === m.f` with use of the returned callable and argument
types. See [`Model`](@ref) for evaluator obligations and [Binding rules](@ref) for the
binding and argument contracts.

Replace `CondFixContext` with `condition(model, values)` or `fix(model, values)`.
Replace `conditioned(context)` and `fixed(context)` with the corresponding model calls;
replace `decondition_context` and `unfix_context` with `decondition(model, names...)` and
`unfix(model, names...)`. Replace `hasconditioned`, `getconditioned`, `hasfixed`, `getfixed`,
and their `_nested` variants with `haskey(conditioned(m), vn)`, `conditioned(m)[vn]`, and
the corresponding `fixed(m)` operations. Replace role inspection through `inargnames`,
`inmissings`, `getmissings`, `isassumption`, `isfixed`, `contextual_isassumption`,
`contextual_isfixed`, or `hasmissing` with inspection of `conditioned(m)` and `fixed(m)`.

To rebuild a model with new arguments while keeping its bindings, previously
`contextualize(new, old.context)`, transfer the bindings stored on `old`. For bindings
confined to the model's own LHS variables, whose partial bindings fit the new model's
storage:

```julia
new = decondition(new)                         # drop new's own observations
new = condition(new, conditioned(unfix(old)))  # old's observations, also those under fixes
new = fix(new, fixed(old))
```

If an earlier binding changed storage shape, replay that binding before its partial
removals. For example, replay `condition(new; x=ones(3))` before
`decondition(new, @varname(x[1]))` when the original argument had length two. Transferring
only the surviving listing cannot reconstruct the enlarged owner.

`decondition(new)` removes only the new model's own observations. Transfer a
binding group with `condition(new, values)` or the corresponding
`fix` call, including groups with child addresses.
Reapply recursive removals separately; the listings omit removal markers.
`conditioned` and `fixed` list only bindings stored on `old`, not those held by its
submodels; `new` rebuilds its submodels from its own arguments. `old`'s argument-supplied
observations become explicit observations of `new`, so they replace `new`'s arguments at those
addresses. The listings keep `old`'s prefixed addresses, so apply `old`'s prefix to `new`
before replaying them.

Replace context overloads of `tilde_observe!!` with `accumulate_observe!!` implementations.
For direct observation calls, replace `tilde_observe!!(ctx, dist, value, vn, template, vi)`
with the current signature `tilde_observe!!(prefix, prefix_template, dist, value, vn, template, vi)`.
For submodel latent calls, replace `tilde_assume!!(ctx, submodel, vn, template, vi)` with
`tilde_assume!!(parent, ctx, submodel, vn, template, vi)`.
Replace `store_coloneq_value!!(ctx, vn, value, template, vi)` with
`store_coloneq_value!!(model, vn, value, template, vi)`; context-based storage hooks are removed.
Replace `DynamicPPL.TestUtils.test_context`, `test_leaf_context`, and `test_parent_context`
with explicit evaluation and interface tests.
