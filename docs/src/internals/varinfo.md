# Design of `VarInfo`

[`VarInfo`](@ref) is a fairly simple structure.

```@docs; canonical=false
VarInfo
```

It contains

  - a `VarNamedTuple` field called `values`,
  - an `AccumulatorTuple` called `accs`, to hold accumulators.

`values` takes care of storing information related to values of individual random variables, while `accs` keeps track of information that we keep accumulating in the course of evaluating through a model.

Variables are regonised by their `VarName`.
We want to work with `VarName`s rather than something like `Symbol` or `String` as `VarName` contains additional structural information.
For instance, a `Symbol("x[1]")` can be a result of either `var"x[1]" ~ Normal()` or `x[1] ~ Normal()`; these scenarios are disambiguated by `VarName`.
`VarName`s also allow things such as setting values for `x[1]` and `x[2]` and getting a value for `x` as a whole.

To ensure that `VarInfo` is simple and intuitive to work with we want it to replicate the following functionality of `Dict`:

  - `keys(::VarInfo)`: return all the `VarName`s present.
  - `haskey(::VarInfo)`: check if a particular `VarName` is present.
  - `getindex(::VarInfo, ::VarName)`: return the realization corresponding to a particular `VarName`.
  - `setindex!!(::VarInfo, val, ::VarName)`: set the realization corresponding to a particular `VarName`.
  - `empty!!(::VarInfo)`: delete all data.
  - `merge(::VarInfo, ::VarInfo)`: merge two containers according to similar rules as `Dict`.

Note that we only define the BangBang methods such as `setindex!!`, rather than the mutating ones likes `setindex!`.
This is due to the design of `VarNamedTuple`, which is explained on its own page in these docs.

*But* for general-purpose samplers, we often want to work with a simple flattened structure, typically a `Vector{<:Real}`.
One can access a vectorised version of a variable's value with the following vector-like functions:

  - `getindex_internal(::VarInfo, ::VarName)`: get the flattened value of a single variable.
  - `getindex_internal(::VarInfo, ::Colon)`: get the flattened values of all variables.
  - `setindex_internal!!(::VarInfo, ::AbstractVector, ::VarName)`: set the flattened value of a variable.

The functions have `_internal` in their name because internally `VarInfo` always stores values as vectorised.

Moreover, a link transformation can be applied to a `VarInfo` with `link!!` (and reversed with `invlink!!`), which applies a reversible transformation to the internal storage format of a variable that makes the range of the random variable cover all of Euclidean space.
`getindex_internal` and `setindex_internal!` give direct access to the vectorised value after such a transformation, which is what samplers often need to be able sample in unconstrained space.

Finally, we want want the underlying storage to have a few performance-related properties:

 1. Type-stable when possible, but functional when not.
 2. Efficient storage and iteration when possible, but functional when not.

The "but functional when not" is important as we want to support arbitrary models, which means that we can't always have these performance properties.

To understand how these are achieved, we refer the reader to the documentation on `VarNamedTuple`, which underpins `VarInfo`.
