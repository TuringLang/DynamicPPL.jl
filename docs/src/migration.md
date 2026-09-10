# Migrating old `VarInfo` code

`VarInfo` now means the accumulator-only container formerly called `OnlyAccsVarInfo`.
`VarInfo()` records log densities, not parameter values. Add a `RawValueAccumulator`
or `VectorValueAccumulator` when those outputs are needed; `VarInfo(model)` remains
a convenience constructor that records vectorised values and log densities.

Replace `InitContext` with `Context`. `DefaultContext` and context subtyping are removed:
custom value selection belongs in initialisation strategies. To reuse previous values,
extract them explicitly before evaluating:

```julia
context = Context(rng, InitFromParams(get_vector_values(previous), nothing), LinkAll())
retval, outputs = evaluate!!(model, context, VarInfo())
```

Use `get_raw_values(previous)` for a raw-value accumulator. Choose `UnlinkAll()`,
`LinkAll()`, or a partial/fixed transform strategy for the desired output representation.
`init!!` defaults to `UnlinkAll()`; it never infers transforms from output state.
For custom strategies, implement `get_param_eltype(strategy)` when evaluation needs
to promote argument buffers or thread-local accumulators for AD.

All output accumulators reset before evaluation. Previously recorded sites that are
not executed again are absent from the new outputs. Replace `vi.values` with
`get_vector_values(vi)`; there is no separate parameter store.

Please get in touch if you have some old code you're unsure how to migrate, and we will be happy to add it to this list.

```@example 1
using DynamicPPL, Distributions, Random

@model function f()
    x ~ Normal()
    y ~ LogNormal()
    return 1.0 ~ Normal(x + y)
end

model = f()
```

## Sampling from the prior

Old:

```@example 1
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

```@example 1
vi = VarInfo(Xoshiro(468), model)
# This no longer works, but you may have used it.
# vi[@varname(x)], vi[@varname(y)]

# This still works
DynamicPPL.getindex_internal(vi, @varname(x))
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

```@example 1
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

```@example 1
vi = VarInfo(Xoshiro(468), model)

vals = [1.0, 1.0]
# Note this was `unflatten` (no exclamation mark) in the old code
vi = DynamicPPL.unflatten!!(vi, vals)
# Current syntax requires an explicit context.
_, vi = DynamicPPL.evaluate!!(
    model, Context(InitFromParams(get_vector_values(vi), nothing), UnlinkAll()), vi
)
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
init_strategy = InitFromVector(vals, ldf)

vi = VarInfo()
_, vi = init!!(Xoshiro(468), model, vi, init_strategy, ldf.transform_strategy)
vi
```
