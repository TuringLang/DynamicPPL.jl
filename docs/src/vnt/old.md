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
Although, this only affects `getindex` and `setindex!!` on sub-`VarName`s of `@varname(a)`;
You can still use the same `VarNamedTuple` to store information about an unrelated `@varname(c)` with stability.

## Non-Array blocks with `IndexLens`es

The above is all that is needed for setting regular scalar values.
However, in DynamicPPL we also have a particular need for something slightly odd:
We sometimes need to do calls like `setindex!!(vnt, @varname(a[1:5]), val)` on a `val` that is _not_ an `AbstractArray`, or even iterable at all.
Normally this would error: As a scalar value with size `()`, `val` is the wrong size to be set with `@varname(a[1:5])`, which clearly wants something with size `(5,)`.
However, we want to allow this even if `val` is not an iterable, if it is some object for which `size` is well-defined, and `size(val) == (5,)`.
In DynamicPPL this comes up when storing e.g. the priors of a model, where a random variable like `@varname(a[1:5])` may be associated with a prior that is a 5-dimensional distribution.

Internally, a `PartialArray` is just a regular `Array` with a mask saying which elements have been set.
Hence we can't store `val` directly in the same `PartialArray`:
We need it to take up a sub-block of the array, in our example case a sub-block of length 5.
To this end, internally, `PartialArray` uses a wrapper type called `ArrayLikeWrapper`, that stores `val` together with the indices that are being used to set it.
The `PartialArray` has all its corresponding elements, in our example elements 1, 2, 3, 4, and, 5, point to the same wrapper object.

While such blocks can be stored using a wrapper like this, some care must be taken in indexing into these blocks.
For instance, after setting a block with `setindex!!(vnt, @varname(a[1:5]), val)`, we can't `getindex(vnt, @varname(a[1]))`, since we can't return "the first element of five in `val`", because `val` may not be indexable in any way.
Similarly, if next we set `setindex!!(vnt, @varname(a[1]), some_other_value)`, that should invalidate/delete the elements `@varname(a[2:5])`, since the block only makes sense as a whole.
Because of these reasons, setting and getting blocks of well-defined size like this is allowed with `VarNamedTuple`s, but _only by always using the full range_.
For instance, if `setindex!!(vnt, @varname(a[1:5]), val)` has been set, then the only valid `getindex` key to access `val` is `@varname(a[1:5])`;
Not `@varname(a[1:10])`, nor `@varname(a[3])`, nor for anything else that overlaps with `@varname(a[1:5])`.
`haskey` likewise only returns true for `@varname(a[1:5])`, and `keys(vnt)` only has that as an element.

The size of a value, for the purposes of inserting it into a `PartialArray`, is determined by a call to `vnt_size`.
`vnt_size` falls back to calling `Base.size`.
The reason we define a distinct function is to be able to control its behaviour, if necessary, without type piracy.
