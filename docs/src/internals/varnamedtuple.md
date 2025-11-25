# VarNamedTuple

In DynamicPPL there is often a need to store data keyed by `VarName`s.
This comes up when getting conditioned variable values from the user, when tracking values of random variables in the model outputs or inputs, etc.
Historically we've had several different approaches to this: Dictionaries, NamedTuples, vectors with subranges corresponding to different `VarName`s, and various combinations thereof.

To unify the treatment of these use cases, and handle them all in a robust and performant way, is the purpose of `VarNamedTuple`, aka VNT.
It's a data structure that can store arbitrary data, indexed by (nearly) arbitrary `VarName`s, in a type stable and performant manner.

`VarNamedTuple` consists of nested `NamedTuple`s and `PartialArray`.
Let's first talk about the `NamedTuple` part.
This is what is needed for handling `PropertyLens`es in `VarName`s, that is, `VarName`s consisting of nested symbols, like in `@varname(a.b.c)`.
In a `VarNamedTuple` each level of such nesting of `PropertyLens`es corresponds to a level of nested `NamedTuple`s, with the `Symbol`s of the lens as the keys.
For instance, the `VarNamedTuple` mapping `@varname(x) => 1, @varname(y.z) => 2` would be stored as

```
VarNamedTuple(; x=1, y=VarNamedTuple(; z=2))
```

where `VarNamedTuple(; x=a, y=b)` is just a thin wrapper around the `NamedTuple` `(; x=a, y=b)`.

It's often handy to think of this as a tree, with each node being a `VarNamedTuple`, like so:

```
   VNT
x /   \ y
 1     VNT
         \ z
          2
```

If all `VarName`s consisted of only `PropertyLens`es we would be done designing the data structure.
However, recall that VarNames allow three different kinds of lenses: `PropertyLens`es, `IndexLens`es, and `identity` (the trivial lens).
The `identity` lens presents no complications, and in fact in the above example there was an implicit identity lens in e.g. `@varname(x) => 1`.
It is the `IndexLenses` that require more structure.

An `IndexLens` is the indexing layer in `VarName`s like `@varname(x[1])`, `@varname(x[1].a.b[2:3])` and `@varname(x[:].b[1,2,3].c[1:5,:])`.
`VarNamedTuple` can not deal with `IndexLens`es in their full generality, for reasons we'll discuss below.
Instead we restrict ourselves to `IndexLens`es where the indices are integers, explicit ranges with end points, like `1:5`, or tuples thereof.

When storing data in a `VarNamedTuple`, we recursively go through the nested lenses in the `VarName`, inserting a new `VarNamedTuple` for every `PropertyLens`.
When we meet an `IndexLens`, we instead instert into the tree something called a `PartialArray`.

A `PartialArray` is like a regular `Base.Array`, but with some elements possibly unset.
It is a data structure we define ourselves for use within `VarNamedTuple`s.
A `PartialArray` has an element type and a number of dimensions, and they are known at compile time, but it does not have a size, and this thus not an `AbstractArray`.
This is because if we set the elements `x[1,2]` and `x[14,10]` in a `PartialArray` called `x`, this does not mean that 14 and 10 are the ends of their respective dimensions.
The typical use of this structure in DynamicPPL is that the user may define values for elements in an array-like structure one by one, and we do not always know how large these arrays are.

This is also the reason why `PartialArray`, and by extension `VarNamedTuple`, do not support indexing by `Colon()`, i.e. `:`, as in `x[:]`.
A `Colon()` says that we should get or set all the values along that dimension, but a `PartialArray` does not know how many values there may be.
If `x[1]` and `x[4]` have been set, asking for `x[:]` is not a well-posed question.

`PartialArray`s have other restrictions, compared to the full indexing syntax of Julia, as well:
They do not support linearly indexing into multidimemensional arrays (as in `rand(3,3)[8]`), nor indexing with arrays of indices (as in `rand(4)[[1,3]]`), nor indexing with boolean mask arrays as in `rand(4)[[true, false, true, false]]`).
This is mostly because we haven't seen a need to support them, and implementing would complicate the codebase for little gain.
We may add support for them later if needed.

`PartialArray`s can hold any values, just like `Base.Array`s, and in particular they can hold `VarNamedTuple`s.
Thus we nest them with `VarNamedTuple`s to support storing `VarName`s with arbitrary combinations of `PropertyLens`es and `IndexLens`es.
A code example illustrates this the best:

```julia
julia> vnt = VarNamedTuple();

julia> vnt = setindex!!(vnt, 1.0, @varname(a));

julia> vnt = setindex!!(vnt, [2.0, 3.0], @varname(b.c));

julia> vnt = setindex!!(vnt, [:hip, :hop], @varname(d.e[2].f[3:4]));

julia> print(vnt)
VarNamedTuple(; a=1.0, b=VarNamedTuple(; c=[2.0, 3.0]), d=VarNamedTuple(; e=PartialArray{VarNamedTuple{(:f,), Tuple{DynamicPPL.VarNamedTuples.PartialArray{Symbol, 1}}},1}((2,) => VarNamedTuple(; f=PartialArray{Symbol,1}((3,) => hip, (4,) => hop)))))
```

The output there may be a bit hard bit hard to parse, so to illustrate:

```julia
julia> vnt[@varname(b)]
VarNamedTuple(; c=[2.0, 3.0])

julia> vnt[@varname(b.c[1])]
2.0

julia> vnt[@varname(d.e)]
PartialArray{VarNamedTuple{(:f,), Tuple{DynamicPPL.VarNamedTuples.PartialArray{Symbol, 1}}},1}((2,) => VarNamedTuple(; f=PartialArray{Symbol,1}((3,) => hip, (4,) => hop)))

julia> vnt[@varname(d.e[2].f)]
PartialArray{Symbol,1}((3,) => hip, (4,) => hop)
```

The above example also highlights how setting indices in a `VarNamedTuple` is done using `BangBang.setindex!!`.
We do not define a method for `Base.setindex!` at all, the `setindex!!` is the only way.
This is because `VarNamedTuple` mixes mutable an immutable data structures.
It is also for user convenience:
One does not ever have to think about whether the value that one is inserting into a `VarNamedTuple` is of the right type to fit in.
Rather the containers will flex to fit it, keeping element types concrete when possible, but making them abstract if needed.
`VarNamedTuple`, or more precisely `PartialArray`, even explicitly concretises element types whenever possible.
For instance, one can make an abstractly typed `VarNamedTuple` like so:

```julia
julia> vnt = VarNamedTuple();

julia> vnt = setindex!!(vnt, 1.0, @varname(a[1]));

julia> vnt = setindex!!(vnt, "hello", @varname(a[2]));

julia> print(vnt)
VarNamedTuple(; a=PartialArray{Any,1}((1,) => 1.0, (2,) => hello))
```

Note the element type of `PartialArray{Any}`.
But if one changes the values to make them homogeneous, the element type is automatically made concrete again:

```julia
julia> vnt = setindex!!(vnt, "me here", @varname(a[1]));

julia> print(vnt)
VarNamedTuple(; a=PartialArray{String,1}((1,) => me here, (2,) => hello))
```

This approach is at the core of why `VarNamedTuple` is performant:
As long as one does not store inhomogeneous types within a single `PartialArray`, by assigning different types to `VarName`s like `@varname(a[1])` and `@varname(a[2])`, different variables in a `VarNamedTuple` can have different types, and all `getindex` and `setindex!!` operations remain type stable.
Note that assigning a value to `@varname(a[1].b)` but not to `@varname(a[2].b)` has the same effect as assigning values of different types to `@varname(a[1])` and `@varname(a[2])`, and also causes a loss of type stability for for `getindex` and `setindex!!`.
Although, this only affects `getindex` and `setindex!!` on sub-`VarName`s of `@varname(a)`, you can still use the same `VarNamedTuple` to store information about an unrelated `@varname(c)` with stability.

Some miscellaneous notes

## Limitations

This design has a several of benefits, for performance and generality, but it also has limitations:

 1. The lack of support for `Colon`s in `VarName`s.
 2. The lack of support for some other indexing syntaxes supported by Julia, such as linear indexing and boolean indexing.
 3. An assymmetry between storing arrays with `setindex!!(vnt, array, @varname(a))` and elements of arrays with `setindex!!(vnt, element, @varname(a[i]))`.
    The former stores the whole array, which can then be indexed with both `@varname(a)` and `@varname(a[i])`.
    The latter stores only individual elements, and even if all elements have been set, one still can't get the value associated with `@varname(a)` as a regular `Base.Array`.
