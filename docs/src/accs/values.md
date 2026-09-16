# Storing vectorised and raw values

`VarInfo` contains only accumulators. Choose which parameter representation to record:
`RawValueAccumulator` stores model-space values, while `VectorValueAccumulator` stores
vectorised `TransformedValue`s with their transforms. Neither accumulator supplies inputs
during evaluation.

## Vectorised values

Vectorised values preserve stochastic-site boundaries, including sites whose linked
dimension differs from their model-space dimension.

```@example 1
using DynamicPPL, Distributions
using Random: Xoshiro

@model function dirichlet()
    x = zeros(3)
    return x[1:3] ~ Dirichlet(ones(3))
end
model = dirichlet()
context = Context(Xoshiro(1), InitFromPrior(), LinkAll())
_, vi = evaluate!!(model, context, VarInfo(VectorValueAccumulator()))
vector_values = get_vector_values(vi)
keys(vector_values)
```

The entry for `x[1:3]` is one block, even though a linked Dirichlet value has only two
coordinates. See [Array-like blocks](@ref array-like-blocks).

```@example 1
internal_values_as_vector(vector_values)
```

These values can initialise a `LogDensityFunction`, which derives the flat parameter
layout and transforms from them. There is no separate value store in `VarInfo`.

## Raw values

A `RawValueAccumulator` records untransformed values. It does not retain stochastic-site
block boundaries: indexed sites are represented by their individual indices.

```@example 1
context = Context(Xoshiro(1), InitFromPrior(), UnlinkAll())
_, vi = evaluate!!(model, context, VarInfo(RawValueAccumulator(false)))
raw_values = get_raw_values(vi)
keys(raw_values)
```

Raw values are used for chain construction. A whole variable such as
`x ~ Dirichlet(ones(3))` remains one value when the chain format supports it.

## Reusing outputs as inputs

Reuse requires an explicit conversion outside evaluation:

```@example 1
context = Context(Xoshiro(1), InitFromParams(raw_values, nothing), LinkAll())
retval, outputs = evaluate!!(model, context, VarInfo(VectorValueAccumulator()))
get_vector_values(outputs)
```

The context determines the new output transforms, independently of the input
representation. `InitFromParams(vector_values, nothing)` also accepts vectorised inputs,
including dynamically linked values. Dynamic transforms are reconstructed from each
site's current distribution, so parameter-dependent supports remain correct.

The `nothing` fallback makes an absent parameter an error. Specify another
initialisation strategy when new sites should instead receive generated values.
