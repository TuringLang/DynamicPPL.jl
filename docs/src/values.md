# Storing values

## The role of VarInfo

As described in the [model evaluation documentation page](./flow.md), each tilde-statement is split up into three parts:

 1. Initialisation;
 2. Computation; and
 3. Accumulation.

Unfortunately, not everything in DynamicPPL follows this clean structure yet.
In particular, there is a struct, called `VarInfo`, which has a dual role in both initialisation and accumulation:

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

However, `values` serves a dual purpose: it is sometimes used for initialisation (when the model's leaf context is `DefaultContext`, the `AbstractTransformedValue` to be used in the computation step is read from it) and it is sometimes also used for accumulation (when linking a VarInfo, we will potentially store a new `AbstractTransformedValue` in it).

The path to removing `VarInfo` is essentially to separate these two roles:

 1. The initialisation role of `varinfo.values` can be taken over by an initialisation strategy that wraps it.
    Recall that the only role of an initialisation strategy is to provide an `AbstractTransformedValue` via [`DynamicPPL.init`](@ref).
    This can be trivially done by indexing into the `VarNamedTuple` stored in the strategy.

 2. The accumulation role of `varinfo.values` can be taken over by a new accumulator, which we call `TransformedValueAccumulator`.
    The nomenclature here is not especially precise: it does not store `AbstractTransformedValue`s per se, but only two subtypes of it, `LinkedVectorValue` and `VectorValue`.

`TransformedValueAccumulator` is implemented inside `src/accs/transformed_value.jl`, and additionally includes a link strategy as a parameter: the link strategy is responsible for deciding which values should be stored as `LinkedVectorValue`s and which as `VectorValue`s.

!!! note
    
    Decoupling the initialisation from the accumulation also means that we can pair different initialisation strategies with a `TransformedValueAccumulator`.
    Previously, to link a VarInfo, you would need to first generate an unlinked VarInfo and then link it.
    Now, you can directly create a linked VarInfo (i.e., accumulate `LinkedVectorValue`s) by sampling from the prior (i.e., initialise with `InitFromPrior`).

## ValuesAsInModelAccumulator

Earlier we said that `TransformedValueAccumulator` stores only two subtypes of `AbstractTransformedValue`: `LinkedVectorValue` and `VectorValue`.

It is often also useful to store the raw values, i.e., [`UntransformedValue`](@ref)s; but additionally, since `UntransformedValue`s must always correspond exactly to the indices they are assigned to, we can unwrap them and do not need to store them as array-like blocks.

This is the role of `ValuesAsInModelAccumulator`.

!!! note
    
    The name is a historical artefact, and can definitely be improved. Suggestions are welcome!

```@example 1
oavi = DynamicPPL.OnlyAccsVarInfo()
oavi = DynamicPPL.setaccs!!(oavi, (DynamicPPL.ValuesAsInModelAccumulator(false),))
_, oavi = DynamicPPL.init!!(dirichlet_model, oavi)
raw_vals = DynamicPPL.getacc(oavi, Val(:ValuesAsInModel)).values
```

Note that when we unwrap `UntransformedValue`s, we also lose the block structure that was present in the model.
That means that in `ValuesAsInModelAccumulator`, there is no longer any notion that `x[1:3]` was set together, so the keys correspond to the individual indices.

```@example 1
keys(raw_vals)
```

In particular, the outputs of `ValuesAsInModelAccumulator` are used for chain construction.
This is why indices are split up in chains.

!!! note
    
    If you have an entire vector stored as a top-level symbol, e.g. `x ~ Dirichlet(ones(3))`, it will not be broken up (as long as you use FlexiChains).

## Why do we still need to store `TransformedValue`s?

Given that `ValuesAsInModelAccumulator` exists, one may wonder why we still need to store `TransformedValue`s at all, i.e. what the purpose of `TransformedValueAccumulator` is.

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
