# Design

There are two aspects to the design of `VarNamedTuple`s: property access, and indexing.
`VarNamedTuple` consists of recursively nested `NamedTuple`s and `PartialArray`s which together support both kinds of access.

## Property access

Let's first talk about the `NamedTuple` part.
In a `VarNamedTuple` each level of a `Property` optic corresponds to a level of nested `NamedTuple`s, with the `Symbol`s of the lenses as keys.
For instance, the `VarNamedTuple` mapping `@varname(x) => 1, @varname(y.z) => 2` would be stored as

```julia
VarNamedTuple(; x=1, y=VarNamedTuple(; z=2))
```

where `VarNamedTuple(; x=a, y=b)` is just a thin wrapper around the `NamedTuple` `(; x=a, y=b)`.
In fact, if `vnt` is a `VarNamedTuple`, then `vnt.data` is exactly that underlying `NamedTuple`.

It's often handy to think of this as a tree, with each node being a `VarNamedTuple`, like so:

```
   VNT
x /   \ y
 1     VNT
         \ z
          2
```

By virtue of these being `NamedTuple`, all variable access is completely type-stable.

If all `VarName`s consisted of only `Property`es we would be done designing the data structure.
Sadly, that isn't the case!

## Indexing and `PartialArray`s

Indexing is much more complicated to handle than property access, for a number of reasons.

Let's start by talking about the most obvious issue: only some parts of an array may be set.
For example, the user may set `x[1]` and `x[10]` but not the ones in the middle.
This is accomplished using a `PartialArray`, which is a wrapper around an `AbstractArray` that holds the data of interest, together with a mask saying which elements have been set.

The array type and size of the mask is always the same as the data array (this is done using `Base.similar`).

!!! info
    
    Currently this is not enforced in an inner constructor, because that sometimes leads to extra allocations. It would be nice to investigate if this invariant can be enforced.

If `pa.mask[i]` is `true`, then `pa.data[i]` has been set; otherwise, `pa.data[i]` may contain some value, but it is not valid to index into that part of the array.

Here is an example:

```@example 1
using DynamicPPL.VarNamedTuples: PartialArray

data = randn(3)
mask = similar(data, Bool)
fill!(mask, false)
pa = PartialArray(data, mask)
```

The main way of interacting with a `PartialArray` is to use `BangBang.setindex!!`.
This sets one or more elements of the `PartialArray`'s data, and marks the corresponding elements in the mask as `true`.

```@example 1
using BangBang: setindex!!

setindex!!(pa, 12.0, 2)
pa.data, pa.mask
```

When printed, `pa` shows only the elements that are set:

```@example 1
pa
```

It is invalid to index into the unset elements:

```@repl 1
pa[1]
```

but you can access the set elements:

```@example 1
pa[2]
```

It follows from this, that because the `PartialArray` _is_ actually backed by a regular `Array`, it is constructive.
The call to `getindex` will return the values stored in `pa.data`, as long as all the elements in `pa.mask` are set.
For example:

```@example 1
setindex!!(pa, 11.0, 1)
pa[1:2]
```

You can even get the entirety of the array, once you have set all three elements.
In this case, the `PartialArray` is (morally) equivalent to a single `Array`.

```@example 1
setindex!!(pa, 13.0, 3)
pa[:]
```

All `Index` optics in `VarName`s correspond to `PartialArray`s in `VarNamedTuple`s.
For example, let's say we want to store the mappings `@varname(x[1].a) => 1.0`, and `y.b[2,3] => 2.0`.
The corresponding `VarNamedTuple` would look like this:

```
         VNT
        /   \y  
      x/     \    b
      /     VNT------PA[[ _, _, _  ],
     /                  [ _, _, 2.0]]
    PA [VNT]
         |
        a|
         |
        1.0
```

where `_` indicates masked elements in a `PartialArray`.
To demonstrate:

```@example 1
using DynamicPPL

vnt = VarNamedTuple()
vnt = setindex!!(vnt, 1.0, @varname(x[1].a))
vnt = setindex!!(vnt, 2.0, @varname(y.b[2, 3]))
```

```@example 1
vnt.data.x  # This is a PartialArray
```

```@example 1
vnt.data.x.data[1]   # This is a VNT
```

```@example 1
vnt.data.x.data[1].data.a  # This is 1.0
```

```@example 1
vnt.data.y  # This is a VNT
```

```@example 1
vnt.data.y.data.b  # This is a PartialArray
```

```@example 1
vnt.data.y.data.b.data[2, 3]  # This is 2.0
```

This illustrates the fundamental structure of `VarNamedTuple`s.
From this example, one can see that getting data from a `VarNamedTuple` is as type-stable as possible:

  - All property accesses are `NamedTuple` accesses, which are type-stable.
  - All indexing is done into `PartialArray`s: as long as indexing into the underlying data is type-stable (i.e., the element type of the data array is concrete), indexing into the `PartialArray` is type-stable as well.

!!! info
    
    In fact, the underlying data need not all have the same (concrete) type: all that is needed is that the _unmasked_ elements have the same (concrete) type. The function `_concretise_eltype!!` is an attempt to force this to be the case: if the element type is abstract, but all the set elements have the same concrete type, the entire data array's element type will be changed to that concrete type (with junk in the unset elements).

**One immediate question here is: how do we know what kind of array the `PartialArray` should use for its data and mask?**

## `GrowableArray`s

It's not obvious in the code above, but in the example above, we are implicitly making an assumption based on the indices that we see in the `VarName`s.
For example, for `@varname(x[1].a)`, based on the index `1` we assume that `x` should be a vector with a length of at least 1.
Similarly, for `@varname(y.b[2,3])`, we assume that `y.b` should be a matrix with at least 2 rows and 3 columns.

We can inspect this by looking into the `PartialArray`s:

```@example 1
vnt.data.x.data
```

```@example 1
vnt.data.y.data.b.data
```

So, these `PartialArray`s are backed by something called `GrowableArray`.
A `GrowableArray` is an array type, defined in DynamicPPL, that can grow in size as needed when `setindex!!` is called with indices outside of its current bounds (with other arrays that would error).
The reason for such an array type is that you may want to do something like

```@example 1
begin
    local vnt = VarNamedTuple()
    for i in 1:5
        vnt = setindex!!(vnt, i, @varname(x[i]))
    end
    vnt
end
```

and we don't have the ability to know in advance that `x` will eventually need to be at least of size 5.
So, every call to `setindex!!` here will cause the underlying `GrowableArray` to grow in size as needed.

The problem with this is that it makes a huge number of implicit assumptions about what kind of array `x` _actually_ is, and consequently it forbids a huge number of indexing operations in Julia.

These include, for example, linear indexing.
In this example, the first call will create a `GrowableArray` with one dimension (i.e., a vector); and the second call will fail since we can't index a vector with two indices.

```@repl 1
vnt = setindex!!(VarNamedTuple(), 10.0, @varname(x[1]))
vnt = setindex!!(vnt, 20.0, @varname(x[2, 2]))
```

Colons also don't work.

```@repl 1
vnt = setindex!!(VarNamedTuple(), randn(2), @varname(x[:]))
```

Other things like `OffsetArray`s don't work (because `x[11]` would create a length-11 `GrowableArray`, which might not be appropriate; and `x[-1]` will just straight-up error).
`DimArray`s will also fail if you use an index that isn't an integer.

Finally, we don't know how large the final size of the array should be.
If you try to access the entire array after setting some elements, it will work, but it will warn you:

```@repl 1
vnt = setindex!!(VarNamedTuple(), 10.0, @varname(x[1]))
vnt[@varname(x)]
```

## Templated arrays

**The general solution to this problem is for the user to provide a template for the array `x` in advance, so that we know what kind of array to create when we see `@varname(x[...])`.**

At a low level, this is done using the [`DynamicPPL.templated_setindex!!`](@ref) function, which takes an extra argument that specifies the shape of the top-level symbol in the `VarName`.

For example, the linear-indexing example above now works if you tell the function that `x` is a 2-by-2 matrix.

```@example 1
using DynamicPPL: templated_setindex!!

x = zeros(2, 2)
vnt = VarNamedTuple()
vnt = templated_setindex!!(VarNamedTuple(), 10.0, @varname(x[1]), x)
vnt = setindex!!(vnt, 20.0, @varname(x[2, 2]))
```

(The second call can also be `templated_setindex!!`, but it isn't necessary since the first call already establishes the shape of `x`, and indeed in such a case the template will be ignored.)
Notice that the `PartialArray` is now backed by the correct Array:

```@example 1
vnt.data.x.data
```

It is no longer growable, so if you try to set an out-of-bounds index, it will error:

```@repl 1
vnt = setindex!!(vnt, 30.0, @varname(x[3, 1]))
```

This mechanism makes it far more flexible to work with arrays in DynamicPPL models.
*Fundamentally, this resolves the inconsistency between indexing semantics in the model, and indexing semantics inside the `VarNamedTuple`.*

For example, you can use `DimArray`s:

```@example 1
import DimensionalData as DD

x = DD.DimArray(zeros(2, 3), (DD.X, DD.Y))

vnt = VarNamedTuple()
vnt = templated_setindex!!(vnt, 1.0, @varname(x[DD.X(1), DD.Y(2)]), x)
```

```@example 1
vnt.data.x.data
```

You can access the data back again in any way you like, for example using linear indexing here:

```@example 1
getindex(vnt, @varname(x[3]))
```

!!! info
    
    Note that support for such arrays is contingent on the provider of the array type, as well as BangBang.jl. There may be bugs that prevent some array types from fully working correctly. For example, `BangBang.setindex!!` does not accept keyword arguments, which precludes the use of keyword indices in `DimArray`s. However, DynamicPPL itself does not inherently prevent you from using such arrays. We would definitely like to fix upstream issues like these, but we don't always have the time to do so: help is *very* greatly appreciated!

## How do we provide the template?

At this point, it would seem like a major faff for users to have to provide templates any time they wanted to use a `VarNamedTuple`.
The good news is, within a DynamicPPL model, **the template always exists**.
For example, consider:

```@example 1
using Distributions
@model function bad_index()
    return x[1] ~ Normal()
end
nothing # hide
```

If you were to attempt to execute this model, even without any interaction with `VarNamedTuple`s, this would error, because _there is no array `x` to set the first element in_.
It would be like writing a function that does `x[1] = 10` without ever defining `y`:

```@repl 1
function bad_index_not_model()
    return x[1] = 10
end
bad_index_not_model()
```

The model can **only** work if `x` is already provided, e.g. as an argument or a variable inside the model:

```@example 1
@model function ok_index1(x::AbstractArray)
    return x[1] ~ Normal()
end

@model function ok_index2()
    x = Vector{Float64}(undef, 10)
    return x[1] ~ Normal()
end
nothing # hide
```

In both cases, we **do** have access to the template for `x`.
Thus, this is just a matter of plumbing this information through to `DynamicPPL.tilde_assume!!` so that it can be used when setting values in the `VarNamedTuple`.
In the macro output below, you can see that `x` is passed as one of the arguments to `tilde_assume!!`.

```@example 1
@macroexpand @model function bad_index()
    return x[1] ~ Normal()
end
```

What this means is that in the core use case of VarNamedTuple (i.e., for storing random variables in DynamicPPL models), templates will always be provided.
There are, for the most part, only two places where templates are unavailable, and we have to fall back on the `GrowableArray` approach:

  - Loading data from chains.
  - Providing conditioned or fixed values.

The first of these is sadly unavoidable (unless we store template data in a chain).
However, the second one could be 'fixed' by allowing users to provide templates themselves when constructing the collection of conditioned values.
Right now, the only way to do this is by manually calling `templated_setindex!!`.
See https://github.com/TuringLang/DynamicPPL.jl/issues/1217 for a possible solution to this, though.
