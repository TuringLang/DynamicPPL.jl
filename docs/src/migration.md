# Migrating to DynamicPPL 0.43

`OnlyAccsVarInfo` is removed. Use `VarInfo`, which has the same accumulator constructor
forms: `VarInfo(accs...)`, `VarInfo(accs::Tuple)`, and `VarInfo(accs::AccumulatorTuple)`.
The old `VarInfo{Tfm,T,Accs}` is now `VarInfo{Accs}`; update dispatch that uses the old
type parameters. Transform strategies are evaluation inputs, not part of the output type.
`VarInfo()` records log densities, not parameter values; `keys`, `haskey`, `length`, `values`
and `isempty` throw `ArgumentError` without a `VectorValueAccumulator`.
Recording values is opt-in because most evaluations, such as
sampling and gradients, need only log densities. Add a `VectorValueAccumulator` for these
queries and vectorised values, or a `RawValueAccumulator` for model-space values, for example
`VarInfo(VectorValueAccumulator(), DynamicPPL.default_accumulators()...)`; `VarInfo(model)` remains
a convenience constructor that records vectorised values and log densities.

`PrefixContext` and `extract_prefixes` are removed. Use `prefix(model, vn; template)`
to set a prefix and `DynamicPPL.getprefix(model)` to read the combined prefix
(`nothing` when absent). Do not read `model.prefix` directly; its contents are internal.
Replace `prefix(ctx, vn)` with
`AbstractPPL.prefix(vn, DynamicPPL.getprefix(model))` for a prefixed model.

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
`get_vector_values(vi)`. `get_values(vi)` now returns the `VarNamedTuple` of vectorised
`TransformedValue`s from the optional `VectorValueAccumulator` and throws `ArgumentError`
without it. Replace unconditional `get_values(VarInfo())` calls with
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

For binding changes, see [Migrating from 0.42](@ref).

Replace context overloads of `tilde_observe!!` with `accumulate_observe!!` implementations.
For direct observation calls, replace `tilde_observe!!(ctx, dist, value, vn, template, vi)`
with `tilde_observe!!(prefix, dist, value, vn, template, vi)`, where `prefix` is the model's
prefix value (`nothing`, a `VarName`, or a `PrefixTemplate`).
For submodel latent calls, replace `tilde_assume!!(ctx, submodel, vn, template, vi)` with
`tilde_assume!!(parent, ctx, submodel, vn, template, vi)`.
Replace `store_coloneq_value!!(ctx, vn, value, template, vi)` with
`store_coloneq_value!!(model, vn, value, template, vi)`; context-based storage hooks are removed.
Replace `DynamicPPL.TestUtils.test_context`, `test_leaf_context`, and `test_parent_context`
with explicit evaluation and interface tests.
