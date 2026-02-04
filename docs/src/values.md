# Storing vectorised vs. raw values

## The role of VarInfo

As described in the [model evaluation documentation page](./flow.md), each tilde-statement is split up into three parts:

 1. Initialisation;
 2. Computation; and
 3. Accumulation.

Unfortunately, not everything in DynamicPPL follows this clean structure yet.
In particular, there is a struct, called `VarInfo`:

```julia
struct VarInfo{linked,V<:VarNamedTuple,A<:AccumulatorTuple}
    values::V
    accs::A
end
```

The `values` field stores either [`LinkedVectorValue`](@ref)s or [`VectorValue`](@ref)s.
The `link` type parameter can either be `true` or `false`, which indicates that _all values stored_ are linked or unlinked, respectively; or it can be `nothing`, which indicates that it is not known whether the values are linked or unlinked, and must be checked on a case-by-case basis.

Here is an example:

```@example 1
using DynamicPPL, Distributions

@model function dirichlet()
    x = zeros(3)
    return x[1:3] ~ Dirichlet(ones(3))
end
dirichlet_model = dirichlet()
vi = VarInfo(dirichlet_model)
vi
```

In `VarInfo`, it is mandatory to store `LinkedVectorValue`s or `VectorValue`s as `ArrayLikeBlock`s (see the [Array-like blocks](@ref) documentation for information on this).
The reason is because, if the value is linked, it may have a different size than the number of indices in the `VarName`.
This means that when retrieving the keys, we obtain each block as a single key:

```@example 1
keys(vi.values)
```

## Towards a new framework

In a `VarInfo`, the `accs` field is responsible for the accumulation step, just like an ordinary `AccumulatorTuple`.

However, `values` serves three purposes in one:

  - it is sometimes used for initialisation (when the model's leaf context is `DefaultContext`, the `AbstractTransformedValue` to be used in the computation step is read from it)
  - it also determines whether the log-Jacobian term should be included or not (if the value is a `LinkedVectorValue`, the log-Jacobian is included)
  - it is sometimes also used for accumulation (when evaluating a model with a VarInfo, we will potentially store a new `AbstractTransformedValue` in it!).

The path to removing `VarInfo` is essentially to separate these three roles:

 1. The initialisation role of `varinfo.values` can be taken over by an initialisation strategy that wraps it.
    Recall that the only role of an initialisation strategy is to provide an `AbstractTransformedValue` via [`DynamicPPL.init`](@ref).
    This can be trivially done by indexing into the `VarNamedTuple` stored in the strategy.

 2. Whether the log-Jacobian term should be included or not can be determined by a transform strategy.
    Much like how we can have an initialisation strategy that takes values from a `VarInfo`, we can also have a transform strategy that is defined by the existing status of a `VarInfo`.
    This is implemented in the `DynamicPPL.get_link_strategy(::AbstractVarInfo)` function.
 3. The accumulation role of `varinfo.values` can be taken over by a new accumulator, which we call `VectorValueAccumulator`.
    This name is chosen because it does not store generic `AbstractTransformedValue`s, but only two subtypes of it, `LinkedVectorValue` and `VectorValue`.
    `VectorValueAccumulator` is implemented inside `src/accs/vector_value.jl`.

!!! note
    
    Decoupling all of these components also means that we can mix and match different initialisation strategies, link strategies, and accumulators more easily.
    
    For example, previously, to create a linked VarInfo, you would need to first generate an unlinked VarInfo and then link it.
    Now, you can directly create a linked VarInfo (i.e., accumulate `LinkedVectorValue`s) by sampling from the prior (i.e., initialise with `InitFromPrior`).

## ValuesAsInModelAccumulator

Earlier we said that `VectorValueAccumulator` stores only two subtypes of `AbstractTransformedValue`: `LinkedVectorValue` and `VectorValue`.
One might therefore ask about the third subtype, namely, `UntransformedValue`.

It turns out that it is very often useful to store the raw values, i.e., [`UntransformedValue`](@ref)s.
Additionally, since `UntransformedValue`s must always correspond exactly to the indices they are assigned to, we can unwrap them and do not need to store them as array-like blocks!

This is the role of `ValuesAsInModelAccumulator`.

!!! note
    
    The name is a historical artefact, and can definitely be improved. Suggestions are welcome!

```@example 1
oavi = DynamicPPL.OnlyAccsVarInfo()
oavi = DynamicPPL.setaccs!!(oavi, (DynamicPPL.ValuesAsInModelAccumulator(false),))
_, oavi = DynamicPPL.init!!(dirichlet_model, oavi, InitFromPrior(), UnlinkAll())
raw_vals = DynamicPPL.getacc(oavi, Val(:ValuesAsInModel)).values
```

Note that when we unwrap `UntransformedValue`s, we also lose the block structure that was present in the model.
That means that in `ValuesAsInModelAccumulator`, there is no longer any notion that `x[1:3]` was set together, so the keys correspond to the individual indices.

```@example 1
keys(raw_vals)
```

In particular, the outputs of `ValuesAsInModelAccumulator` are used for chain construction.
This is why indices of keys like `x[1:3] ~ dist` end up being split up in chains.

!!! note
    
    If you have an entire vector belonging to a top-level symbol, e.g. `x ~ Dirichlet(ones(3))`, it will not be broken up (as long as you use FlexiChains).

## Why do we still need to store `TransformedValue`s?

Given that `ValuesAsInModelAccumulator` exists, one may wonder why we still need to store the other `AbstractTransformedValue`s at all, i.e. what the purpose of `VectorValueAccumulator` is.

Currently, the only remaining reason for transformed values is the fact that we may sometimes need to perform [`DynamicPPL.unflatten!!`](@ref) on a `VarInfo`, to insert new values into it from a vector.

```@example 1
vi = VarInfo(dirichlet_model)
vi[@varname(x[1:3])]
```

```@example 1
vi = DynamicPPL.unflatten!!(vi, [0.2, 0.5, 0.3])
vi[@varname(x[1:3])]
```

If we do not store the vectorised form of the values, we will not know how many values to read from the input vector for each key.

Removing upstream usage of `unflatten!!` would allow us to completely get rid of `TransformedValueAccumulator` and only ever use `ValuesAsInModelAccumulator`.
See [this DynamicPPL issue](https://github.com/TuringLang/DynamicPPL.jl/issues/836) for more information.

One possibility for removing `unflatten!!` is to turn it into a function that, instead of generating a new VarInfo, instead generates a tuple of new initialisation and link strategies which returns `LinkedVectorValue`s or `VectorValue`s containing views into the input vector.
This would be conceptually very similar to how `LogDensityFunction` currently works.
