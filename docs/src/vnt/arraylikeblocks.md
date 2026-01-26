# Array-like blocks

In a number of VNT use cases, it is necessary to associate multiple indices in a `VarNamedTuple` with an object that is not necessarily the same number of elements.

Consider, for example, this model:

```julia
@model function f()
    x = zeros(3)
    return x[1:3] ~ Dirichlet(ones(3))
end
```

and suppose we want to store the prior distribution associated with each variable in a `VarNamedTuple`.
With a `Dict{VarName,Distribution}`, we can do this:

```julia
d = Dict{VarName,Distribution}(@varname(x[1:3]) => Dirichlet(ones(3)))
```

but we incur all the costs associated with the use of a `Dict`, as described before.

With a `VarNamedTuple`, we cannot store this directly:

```julia
vnt.data.x = ... # some array
vnt.data.x[1:3] = Dirichlet(ones(3))  # will error
```

because `Dirichlet` is not an array, and `setindex!` will fail.
Nor can we write

```julia
vnt.data.x = ... # some array
vnt.data.x[1:3] .= Dirichlet(ones(3))
```

because although this will not error, it is semantically different: this means that every element `x[1]`, `x[2]`, and `x[3]` will be assigned the same `Dirichlet(ones(3))` object, which is not what we want.

The current solution to this is to use `ArrayLikeBlock`s, which are thin wrappers around the actual value, but additionally also store the indices used to set the value.
The second and third arguments here are the indices (positional and keyword) used to set the value, and the fourth argument is the size of the block.

```@example 1
using DynamicPPL, Distributions
using DynamicPPL.VarNamedTuples: ArrayLikeBlock

alb = ArrayLikeBlock(Dirichlet(ones(3)), 1:3, (;), (3,))
```

We then set this `ArrayLikeBlock` in all the relevant indices.
The extra information in the `ArrayLikeBlock` allows us to forbid partial indexing into it later on.
In particular, we want to ensure that users can only retrieve the entire block at once, and not e.g. just `x[1]` or `x[2:3]`.

## Getting and setting: in practice

As a user you should not have to deal with `ArrayLikeBlock`s directly.
Under the hood, `templated_setindex!!` will automatically wrap values in `ArrayLikeBlock`s when necessary:

```@example 1
x = zeros(5)
vnt = DynamicPPL.templated_setindex!!(
    VarNamedTuple(), Dirichlet(ones(3)), @varname(x[1:3]), x
)
```

You can access the value again as long as you refer to the full range:

```@example 1
vnt[@varname(x[1:3])]
```

Because we provided template information, you can access this via any other combination of indexing, as long as it refers to all three indices:

```@example 1
vnt[@varname(x[begin:(end - 2)])]
```

However, if you try to access only part of the block, you will get an error:

```@repl 1
vnt[@varname(x[1])]
```

Furthermore, if you set a value into any of the indices covered by the block, the entire block is invalidated and thus removed:

```@example 1
vnt = DynamicPPL.templated_setindex!!(vnt, Normal(), @varname(x[2]), x)
```

## Size checks

Currently, when setting any object `val` as an `ArrayLikeBlock`, there is a size check: we make sure that the range of indices being set to has the same size as `DynamicPPL.VarNamedTuples.vnt_size(val)`.
By default, `vnt_size(x)` returns `Base.size(x)`.

```@example 1
DynamicPPL.VarNamedTuples.vnt_size(Dirichlet(ones(3)))
```

This is what allows us to set a `Dirichlet` distribution to three indices.
However, trying to set the same distribution to two indices will fail:

```@repl 1
vnt = DynamicPPL.templated_setindex!!(
    VarNamedTuple(), Dirichlet(ones(3)), @varname(x[1:2]), zeros(5)
)
```

!!! note
    
    In principle, these checks can be removed since if `Dirichlet(ones(3))` is set as the prior of `x[1:2]`, then model evaluation will error anyway.
    Furthermore, if at any point we need to know the size of the block, we can always retrieve it via `size(view(parent_array, alb.ix...; alb.kw...))`.
    However, the checks are still here for now.

## Which parts of DynamicPPL use array-like blocks?

Simply put, anywhere where we don't store raw values.
Some examples follow.
(This list is meant to only be illustrative, not exhaustive!)

### VarInfo

In `VarInfo`, we need to be able to store either linked or unlinked values (in general, `AbstractTransformedValue`s).
These are always vectorised values, and the linked and unlinked vectors may have different sizes (this is indeed the case for Dirichlet distributions).

```@example 1
@model function dirichlet()
    x = zeros(3)
    return x[1:3] ~ Dirichlet(ones(3))
end
dirichlet_model = dirichlet()
vi = VarInfo(dirichlet_model)
vi.values
```

Thus, in the actual `VarInfo` we do not have a notion of what `x[1]` is.

**Note**: this is in contrast to `ValuesAsInModelAccumulator`, where we do store raw values:

```@example 1
oavi = DynamicPPL.OnlyAccsVarInfo()
oavi = DynamicPPL.setaccs!!(oavi, (DynamicPPL.ValuesAsInModelAccumulator(false),))
_, oavi = DynamicPPL.init!!(dirichlet_model, oavi)
raw_vals = DynamicPPL.getacc(oavi, Val(:ValuesAsInModel)).values
```

This distinction is important to understand when working with downstream code that uses `VarInfo` and its outputs.
In particular, when constructing a chain, we use the raw values from `ValuesAsInModelAccumulator`, not the linked/unlinked values from `VarInfo`.

There is also a difference between the keys.
Because the `VarInfo` stores array-like blocks, the keys correspond to the entire blocks:

```@example 1
keys(vi.values)
```

On the other hand, in `ValuesAsInModelAccumulator`, there is no longer any notion that `x[1:3]` was set together, so the keys correspond to the individual indices.
This is why indices are split up in chains:

```@example 1
keys(raw_vals)
```

### Prior distributions

In `extract_priors` we use a VNT to store the prior distributions seen at each point in the model.
This is exactly the same use case as in the introduction.

```@example 1
DynamicPPL.extract_priors(dirichlet_model)
```

### LogDensityFunction

`DynamicPPL.LogDensityFunction` has to retain information about whether each `VarName`s is linked or not, and the indices in the vectorised parameters that correspond to each variable.

For example, consider creating a linked `LogDensityFunction` for (a slightly expanded version of) the Dirichlet model above:

```@example 1
@model function expanded_dirichlet()
    x = zeros(4)
    x[1] ~ Normal()
    return x[2:4] ~ Dirichlet(ones(3))
end
model = expanded_dirichlet()

vi = VarInfo(model)
linked_vi = DynamicPPL.link!!(vi, model)
ldf = LogDensityFunction(model, DynamicPPL.getlogjoint_internal, linked_vi)
nothing # hide
```

When linking a `Dirichlet(ones(3))`, the resulting vector will have length 2.
So, the `LogDensityFunction` takes in a vector of length 3, for which the first index belongs to `x[1]`, and the next two indices belong to the linked parameters for `x[2:4]`.
Furthermore, it needs to remember that all the variables have been linked.
We can see that this is exactly how the `LogDensityFunction` has stored the information:

```@example 1
ldf._varname_ranges
```
