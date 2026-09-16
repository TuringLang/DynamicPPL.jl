# Storing vectorised and raw values

`VarInfo` contains only accumulators. Two accumulators record parameter values, in
different representations:

  - *Raw values* (`RawValueAccumulator`) are model-space values, as the model body and
    `logpdf` see them. For `x ~ Dirichlet(ones(3))`, the raw value is `[0.2, 0.3, 0.5]`.
  - *Vector values* (`VectorValueAccumulator`) hold one `TransformedValue` per tilde
    statement: that variable's value flattened to a vector, together with its transform.
    Linked, the same `x` becomes `TransformedValue([-0.69, -0.51], DynamicLink())`.

Both are `VarNamedTuple`s keyed by variable name, not one vector for the whole model.
[`internal_values_as_vector`](@ref) concatenates the vector values into the flat vector
that samplers and `LogDensityFunction` use. Neither accumulator supplies inputs during
evaluation.

## Vectorised values

Vectorised values preserve LHS variable boundaries, including LHS variables whose linked
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

The flat vector concatenates the per-statement vectors in key order. These values can
initialise a `LogDensityFunction`, which derives from them the range of each variable
within the flat vector and its transform. There is no separate value store in `VarInfo`.

## Raw values

A `RawValueAccumulator` records untransformed values. It does not retain LHS variable
block boundaries: indexed LHS variables are represented by their individual indices.

```@example 1
context = Context(Xoshiro(1), InitFromPrior(), UnlinkAll())
_, vi = evaluate!!(model, context, VarInfo(RawValueAccumulator(false)))
raw_values = get_raw_values(vi)
keys(raw_values)
```

Raw values are used for chain construction. A whole variable such as
`x ~ Dirichlet(ones(3))` remains one value when the chain format supports it.

## Reusing outputs as inputs

To pass outputs of one evaluation, such as parameter values, as inputs to the next,
convert them explicitly outside evaluation; the context holds inputs and the `VarInfo`
holds only outputs:

```@example 1
context = Context(Xoshiro(1), InitFromParams(raw_values, nothing), LinkAll())
retval, outputs = evaluate!!(model, context, VarInfo(VectorValueAccumulator()))
get_vector_values(outputs)
```

The context determines the new output transforms, independently of the input
representation. `InitFromParams(vector_values, nothing)` also accepts vectorised inputs,
including dynamically linked values. Dynamic transforms are reconstructed from each
LHS variable's current distribution, so parameter-dependent supports remain correct.

The `nothing` fallback makes an absent parameter an error. To sample absent LHS variables from
their priors instead, pass `InitFromParams(raw_values, InitFromPrior())`.
