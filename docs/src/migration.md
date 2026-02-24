# Migrating old `VarInfo` code

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
accs = OnlyAccsVarInfo()
_, vi = init!!(Xoshiro(468), model, accs, InitFromPrior(), UnlinkAll())
vi
```

## Getting parameter values

Old:

```@example 1
vi = VarInfo(Xoshiro(468), model)
vi[@varname(x)], vi[@varname(y)]
```

New:

```@example 1
# Set to true if you want to include results of `:=` statements.
accs = OnlyAccsVarInfo(RawValueAccumulator(false))
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
accs = OnlyAccsVarInfo(VectorValueAccumulator())
_, vi = init!!(Xoshiro(468), model, accs, InitFromPrior(), LinkAll())
ldf = LogDensityFunction(model, getlogjoint_internal, vi)
nothing # hide
```

Then you can do:

```@example 1
vec = to_vector_params(get_vector_values(vi), ldf)
```

(Note that to generate unlinked parameters, just replace `LinkAll()` with `UnlinkAll()` in the call to `init!!` above.)

## Re-evaluating log density at new parameters

Old:

```@example 1
vi = VarInfo(Xoshiro(468), model)

vals = [1.0, 1.0]
# Note this was `unflatten` (no exclamation mark) in the old code
vi = DynamicPPL.unflatten!!(vi, vals)
_, vi = DynamicPPL.evaluate!!(model, vi)
vi
```

The new path *also* assumes that you are using a `LogDensityFunction`:

```@example 1
ldf = LogDensityFunction(model, getlogjoint_internal, vi)
```

Then you can do:

```@example 1
init_strategy = InitFromVector(vals, ldf)

vi = OnlyAccsVarInfo()
_, vi = init!!(Xoshiro(468), model, vi, init_strategy, ldf.transform_strategy)
vi
```
