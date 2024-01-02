## Design of `VarInfo`

[`VarInfo`](@ref) is a fairly simple structure; it contains

  - a `logp` field for accumulation of the log-density evaluation, and
  - a `metadata` field for storing information about the realizations of the different variables.

Representing `logp` is fairly straight-forward: we'll just use a `Real` or an array of `Real`, depending on the context.

**Representing `metadata` is a bit trickier**. This is supposed to contain all the necessary information for each `VarName` to enable the different executions of the model + extraction of different properties of interest after execution, e.g. the realization / value corresponding to a variable `@varname(x)`.

!!! note
    
    We want to work with `VarName` rather than something like `Symbol` or `String` as `VarName` contains additional structural information, e.g. a `Symbol("x[1]")` can be a result of either `var"x[1]" ~ Normal()` or `x[1] ~ Normal()`; these scenarios are disambiguated by `VarName`.

To ensure that `varinfo` is simple and intuitive to work with, we need the underlying `metadata` to replicate the following functionality of `Dict`:

  - `keys(::Dict)`: return all the `VarName`s present in `metadata`.
  - `haskey(::Dict)`: check if a particular `VarName` is present in `metadata`.
  - `getindex(::Dict, ::VarName)`: return the realization corresponding to a particular `VarName`.
  - `setindex!(::Dict, val, ::VarName)`: set the realization corresponding to a particular `VarName`.
  - `delete!(::Dict, ::VarName)`: delete the realization corresponding to a particular `VarName`.
  - `empty!(::Dict)`: delete all realizations in `metadata`.
  - `merge(::Dict, ::Dict)`: merge two `metadata` structures according to similar rules as `Dict`.

*But* for general-purpose samplers, we often want to work with a simple flattened structure, typically a `Vector{<:Real}`. Therefore we also want `varinfo` to be able to replicate the following functionality of `Vector{<:Real}`:

  - `getindex(::Vector{<:Real}, ::Int)`: return the i-th value in the flat representation of `metadata`.
    
      + For example, if `metadata` contains a realization of `x ~ MvNormal(zeros(3), I)`, then `getindex(varinfo, 1)` should return the realization of `x[1]`, `getindex(varinfo, 2)` should return the realization of `x[2]`, etc.

  - `setindex!(::Vector{<:Real}, val, ::Int)`: set the i-th value in the flat representation of `metadata`.
  - `length(::Vector{<:Real})`: return the length of the flat representation of `metadata`.
  - `similar(::Vector{<:Real})`: return a new instance with the same `eltype` as the input.

Moreover, we want also want the underlying representation used in `metadata` to have a few performance-related properties:

 1. Type-stable when possible, but functional when not.
 2. Efficient storage and iteration when possible, but functional when not.
 
The "but functional when not" is important as we want to support arbitrary models, which means that we can't always have these performance properties.

In the following sections, we'll outline how we achieve this in [`VarInfo`](@ref).

### Type-stability

Ensuring type-stability is somewhat non-trivial to address since we want this to be the case even when models mix continuous (typically `Float64`) and discrete (typically `Int`) variables.

Suppose we have an implementation of `metadata` which implements the functionality outlined in the previous section. The way we approach this in `VarInfo` is to use a `NamedTuple` with a separate `metadata` *for each distinct `Symbol` used*. For example, if we have a model of the form

```@example varinfo-design
using DynamicPPL, Distributions, FillArrays

@model function demo()
    x ~ product_distribution(Fill(Bernoulli(0.5), 2))
    y ~ Normal(0, 1)
    return nothing
end
```

then we construct a type-stable representation by using a `NamedTuple{(:x, :y), Tuple{Vx, Vy}}` where

  - `Vx` is a container with `eltype` `Bool`, and
  - `Vy` is a container with `eltype` `Float64`.

Since `VarName` contains the `Symbol` used in its type, something like `getindex(varinfo, @varname(x))` can be resolved to `getindex(varinfo.metadata.x, @varname(x))` at compile-time.

For example, with the model above we have

```@example varinfo-design
# Type-unstable `VarInfo`
varinfo_untyped = DynamicPPL.untyped_varinfo(demo())
typeof(varinfo_untyped.metadata)
```

```@example varinfo-design
# Type-stable `VarInfo`
varinfo_typed = DynamicPPL.typed_varinfo(demo())
typeof(varinfo_typed.metadata)
```

But they both work as expected but one results in concrete typing and the other does not:

```@example varinfo-design
varinfo_untyped[@varname(x)], varinfo_untyped[@varname(y)]
```

```@example varinfo-design
varinfo_typed[@varname(x)], varinfo_typed[@varname(y)]
```

Notice that the untyped `VarInfo` uses `Vector{Real}` to store the boolean entries while the typed uses `Vector{Bool}`. This is because the untyped version needs the underlying container to be able to handle both the `Bool` for `x` and the `Float64` for `y`, while the typed version can use a `Vector{Bool}` for `x` and a `Vector{Float64}` for `y` due to its usage of `NamedTuple`.

!!! warning
    
    Of course, this `NamedTuple` approach is *not* necessarily going to help us in scenarios where the `Symbol` does not correspond to a unique type, e.g.
    
    ```julia
    x[1] ~ Bernoulli(0.5)
    x[2] ~ Normal(0, 1)
    ```
    
    In this case we'll end up with a `NamedTuple((:x,), Tuple{Vx})` where `Vx` is a container with `eltype` `Union{Bool, Float64}` or something worse. This is *not* type-stable but will still be functional.
    
    In practice, rarely observe such mixing of types, therefore in DynamicPPL, and more widely in Turing.jl, we use a `NamedTuple` approach for type-stability with great success.

!!! warning
    
    Another downside with such a `NamedTuple` approach is that if we have a model with lots of tilde-statements, e.g. `a ~ Normal()`, `b ~ Normal()`, ..., `z ~ Normal()` will result in a `NamedTuple` with 27 entries, potentially leading to long compilation times.
    
    For these scenarios it can be useful to fall back to "untyped" representations.

Hence we obtain a "type-stable when possible"-representation by wrapping it in a `NamedTuple` and partially resolving the `getindex`, `setindex!`, etc. methods at compile-time. When type-stability is *not* desired, we can simply use a single `metadata` for all `VarName`s instead of a `NamedTuple` wrapping a collection of `metadata`s.

### Efficient storage and iteration

Efficient storage and iteration we achieve through implementation of the `metadata`. In particular, we do so with [`VarNameVector`](@ref):

```@docs
DynamicPPL.VarNameVector
```

In a [`VarNameVector{<:VarName,Vector{T}}`](@ref), we achieve the desirata by storing the values for different `VarName`s contiguously in a `Vector{T}` and keeping track of which ranges correspond to which `VarName`s.

This does require a bit of book-keeping, in particular when it comes to insertions and deletions. Internally, this is handled by assigning each `VarName` a unique `Int` index in the `varname_to_index` field, which is then used to index into the following fields:

  - `varnames::Vector{VarName}`: the `VarName`s in the order they appear in the `Vector{T}`.
  - `ranges::Vector{UnitRange{Int}}`: the ranges of indices in the `Vector{T}` that correspond to each `VarName`.
  - `transforms::Vector`: the transforms associated with each `VarName`.

Mutating functions, e.g. `setindex!(vnv::VarNameVector, val, vn::VarName)`, are then treated according to the following rules:

 1. If `vn` is not already present: add it to the end of `vnv.varnames`, add the `val` to the underlying `vnv.vals`, etc.

 2. If `vn` is already present in `vnv`:
    
     1. If `val` has the *same length* as the existing value for `vn`: replace existing value.
     2. If `val` has a *smaller length* than the existing value for `vn`: replace existing value and mark the remaining indices as "inactive" by increasing the entry in `vnv.num_inactive` field.
     3. If `val` has a *larger length* than the existing value for `vn`: expand the underlying `vnv.vals` to accommodate the new value, update all `VarName`s occuring after `vn`, and update the `vnv.ranges` to point to the new range for `vn`.

This means that `VarNameVector` is allowed to grow as needed, while "shrinking" (i.e. insertion of smaller elements) is handled by simply marking the redundant indices as "inactive". This turns out to be efficient for use-cases that we are generally interested in.

For example, we want to optimize code-paths which effectively boil down to inner-loop in the following example:

```julia
# Construct a `VarInfo` with types inferred from `model`.
varinfo = VarInfo(model)

# Repeatedly sample from `model`.
for _ = 1:num_samples
    rand!(rng, model, varinfo)

    # Do something with `varinfo`.
    # ...
end
```

There are typically a few scenarios where we encounter changing representation sizes of a random variable `x`:

 1. We're working with a transformed version `x` which is represented in a lower-dimensional space, e.g. transforming a `x ~ LKJ(2, 1)` to unconstrained `y = f(x)` takes us from 2-by-2 `Matrix{Float64}` to a 1-length `Vector{Float64}`.
 2. `x` has a random size, e.g. in a mixture model with a prior on the number of components. Here the size of `x` can vary widly between every realization of the `Model`.

In scenario (1), we're usually *shrinking* the representation of `x`, and so we end up not making any allocations for the underlying `Vector{T}` but instead just marking the redundant part as "inactive".

In scenario (2), we  end up increasing the allocated memory for the randomly sized `x`, eventually leading to a vector that is large enough to hold realizations without needing to reallocate. But this can still lead to unnecessary memory usage, which might be undesirable. Hence one has to make a decision regarding the trade-off between memory usage and performance for the use-case at hand.

To help with this, we have the following functions:

```@docs
DynamicPPL.has_inactive
DynamicPPL.num_inactive
DynamicPPL.num_allocated
DynamicPPL.is_contiguous
DynamicPPL.contiguify!
```

For example, one might encounter the following scenario:

```@example varinfo-design
vnv = DynamicPPL.VarNameVector(@varname(x) => [true])
println("Before insertion: number of allocated entries  $(DynamicPPL.num_allocated(vnv))")

for i in 1:5
    x = fill(true, rand(1:100))
    DynamicPPL.update!(vnv, @varname(x), x)
    println("After insertion #$(i) of length $(length(x)): number of allocated entries  $(DynamicPPL.num_allocated(vnv))")
end
```

We can then insert a call to [`DynamicPPL.contiguify!`](@ref) after every insertion whenever the allocation grows too large to reduce overall memory usage:

```@example varinfo-design
vnv = DynamicPPL.VarNameVector(@varname(x) => [true])
println("Before insertion: number of allocated entries  $(DynamicPPL.num_allocated(vnv))")

for i in 1:5
    x = fill(true, rand(1:100))
    DynamicPPL.update!(vnv, @varname(x), x)
    if DynamicPPL.num_allocated(vnv) > 10
        DynamicPPL.contiguify!(vnv)
    end
    println("After insertion #$(i) of length $(length(x)): number of allocated entries  $(DynamicPPL.num_allocated(vnv))")
end
```

This does incur a runtime cost as it requires re-allocation of the `ranges` in addition to a `resize!` of the underlying `Vector{T}`. However, this also ensures that the the underlying `Vector{T}` is contiguous, which is important for performance. Hence, if we're about to do a lot of work with the `VarNameVector` without insertions, etc., it can be worth it to do a sweep to ensure that the underlying `Vector{T}` is contiguous.

!!! note
    
    Higher-dimensional arrays, e.g. `Matrix`, are handled by simply vectorizing them before storing them in the `Vector{T}`, and composing he `VarName`'s transformation with a `DynamicPPL.FromVec`.

Continuing from the example from the previous section, we can use a `VarInfo` with a `VarNameVector` as the `metadata` field:

```@example varinfo-design
# Type-unstable
varinfo_untyped_vnv = DynamicPPL.VectorVarInfo(varinfo_untyped)
varinfo_untyped_vnv[@varname(x)], varinfo_untyped_vnv[@varname(y)]
```

```@example varinfo-design
# Type-stable
varinfo_typed_vnv = DynamicPPL.VectorVarInfo(varinfo_typed)
varinfo_typed_vnv[@varname(x)], varinfo_typed_vnv[@varname(y)]
```

If we now try to `delete!` `@varname(x)`

```@example varinfo-design
haskey(varinfo_untyped_vnv, @varname(x))
```

```@example varinfo-design
DynamicPPL.has_inactive(varinfo_untyped_vnv.metadata)
```

```@example varinfo-design
# `delete!`
DynamicPPL.delete!(varinfo_untyped_vnv.metadata, @varname(x))
DynamicPPL.has_inactive(varinfo_untyped_vnv.metadata)
```

```@example varinfo-design
haskey(varinfo_untyped_vnv, @varname(x))
```

Or insert a differently-sized value for `@varname(x)`

```@example varinfo-design
DynamicPPL.update!(varinfo_untyped_vnv.metadata, @varname(x), fill(true, 1))
varinfo_untyped_vnv[@varname(x)]
```

```@example varinfo-design
DynamicPPL.num_allocated(varinfo_untyped_vnv.metadata, @varname(x))
```

```@example varinfo-design
DynamicPPL.update!(varinfo_untyped_vnv.metadata, @varname(x), fill(true, 4))
varinfo_untyped_vnv[@varname(x)]
```

```@example varinfo-design
DynamicPPL.num_allocated(varinfo_untyped_vnv.metadata, @varname(x))
```

#### Performance summary

In the end, we have the following "rough" performance characteristics for `VarNameVector`:

| Method | Is blazingly fast? |
| :-: | :-: |
| `getindex` | ${\color{green} \checkmark}$ |
| `setindex!` | ${\color{green} \checkmark}$ |
| `push!` | ${\color{green} \checkmark}$ |
| `delete!` | ${\color{red} \times}$ |
| `update!` on existing `VarName` | ${\color{green} \checkmark}$ if smaller or same size / ${\color{red} \times}$ if larger size |
| `convert(Vector, ::VarNameVector)` | ${\color{green} \checkmark}$ if contiguous / ${\color{orange} \div}$ otherwise |

### Additional methods

We also want some additional methods that are not part of the `Dict` or `Vector` interface:

  - `push!(container, ::VarName, value[, transform])`:  add a new element to the container, _but_ for this we also need the `VarName` to associate to the new `value`, so the semantics are different from `push!` for a `Vector`.

  - `update!(container, ::VarName, value[, transform])`: similar to `push!` but if the `VarName` is already present in the container, then we update the corresponding value instead of adding a new element.

In addition, we want to be able to access the transformed / "unconstrained" realization for a particular `VarName` and so we also need corresponding methods for this:

  - `getindex_raw` and `setindex_raw!` for extracting and mutating the, possibly unconstrained / transformed, realization for a particular `VarName`.
