# Implementation

Having discussed what a `VarNamedTuple` _should_ look like, we now turn our attention to how it is constructed.
The core entry point is a function called `_setindex_optic!!`, which essentially says: 'set the value that can be accessed using this optic, creating it if necessary'.
For example,

```@example 1
using AbstractPPL, DynamicPPL
using DynamicPPL.VarNamedTuples: NoTemplate, _setindex_optic!!

collection = VarNamedTuple()
_setindex_optic!!(collection, 1.0, @opticof(_.x[1].a), NoTemplate())
```

means: 'modify this collection such that the value at the path `_.x[1].a` is set to `1.0`, creating any missing structure along the way, and with no template provided.'

Notice that, if `collection` is the top-level `VarNamedTuple` being used to hold variable values, this is the same as saying "add the `VarName` `@varname(x[1].a)` with value `1.0` to the `VarNamedTuple` `vnt`".
Indeed, `BangBang.setindex!!(vnt, 1.0, @varname(x[1].a))` directly calls the above.

If a template is provided, it must be for the entire structure being created, that is, the template should be the shape of `vnt`.

!!! info
    
    When we called `templated_setindex!!`, we said that the template should be for the top-level symbol, i.e. `x`.
    It seems like we are introducing an inconsistency here, since the template above would be for the entire structure.
    That is why `templated_setindex!!` does not pass the template as-is; it wraps the template in one level of `SkipTemplate{1}(template)`, which effectively means 'don't use a template for the first level, then use it for the next'.

Because `VarName`s have a linked list structure, it should not be surprising to find that `VarNamedTuple`s are constructed by recursing into `_setindex_optic!!`.
For an optic like `_.x[1].a`, the strategy is as follows:

  - Look at the outermost layer of the optic (`.x` above).
  - If `collection` already has a value for that layer (i.e. a property called `x`), get that sub-value, and call `_setindex_optic!!` on that sub-value with the rest of the optic (`[1].a` above).
  - If not, call the function `make_leaf`, which constructs the necessary sub-structure, using any template provided.

`make_leaf` is itself of course also recursive, and is where the bulk of the complicated logic occurs.

## Making leaves

Following on from the explanation above, it follows that `make_leaf(value, optic, template)` is responsible for creating a new structure `s`, which already holds `value` at the path specified by `optic`.
If a template is provided, `s` should be constructed to match the shape and type of `template`.

Let's build this up using some simple examples to illustrate the idea, first ignoring any templates.

Here, this is an identity optic, which is the base case.
Of course, the value itself (1.0) can be indexed into with the identity optic to give itself.
So we can just return the value.

```@example 1
using DynamicPPL.VarNamedTuples: make_leaf

make_leaf(1.0, @opticof(_), NoTemplate())
```

Next, consider a field optic like `_.a`.
To create a structure that holds `1.0` at that path, we can create a `VarNamedTuple` with a single field `a` set to `1.0`.

```@example 1
make_leaf(1.0, @opticof(_.a), NoTemplate())
```

Finally, consider an index optic like `_[3]`.
Since no template is provided, we will need to make a guess: in this case it will be a 1-dimensional `GrowableArray`, with a size of at least 3.

```@example 1
l = make_leaf(1.0, @opticof(_[3]), NoTemplate())
# l isa PartialArray, which is quite boring to print, so let's peek inside it.
typeof(l.data), l.data, l.mask
```

Conversely, if a template is provided, we'll just use that directly.
(But we still have to make sure to set the value at the correct index!)

```@example 1
l = make_leaf(1.0, @opticof(_[3]), zeros(4))
l.data, l.mask
```

Now, consider a recursive case, like `@opticof(_.a.b)`.
`make_leaf` needs to create something which can hold `1.0` at the path `.a.b`.

To do so, it first has to create the sub-structure that will hold `1.0` at `.b`, and _then_ it can create an outer structure that holds that sub-structure at `.a`.

```@example 1
make_leaf(1.0, @opticof(_.a.b), NoTemplate())
```

This is conceptually fairly straightforward.
The unfortunate part of this is in dealing with type stability, especially with multi-indices.

## Multi-indices

Consider, for example, this case: here, we want to create a structure that holds the vector `[1.0, 2.0]` at indices `2:3`.
We would make a `GrowableArray` of length 3 -- but what element type should we create it with?

```@example 1
make_leaf([1.0, 2.0], @opticof(_[2:3]), NoTemplate())
```

We could say that we don't care about what type it should be, and just let `BangBang.setindex!!` handle it.
That actually does work.
However, it's always better to create the structure with the correct type from the start, as that makes the implementation much more type stable.
Since we're setting to multiple indices with a vector value, we need to use `eltype([1.0, 2.0])` as the base element type for the parent array.

On the other hand, if we were setting a single index with a vector value, like this:

```@example 1
make_leaf([1.0, 2.0], @opticof(_[2]), NoTemplate())
```

what we're _really_ saying is that the new structure should hold a vector at index `2`, i.e., it is a vector of vectors.
In this case, the base element type should then just be `typeof([1.0, 2.0])`.

This means that in general we need a good way of differentiating between single-element indexing and multi-element indexing.
One _could_ iterate over the indices and check which of them are ranges or colons, for example; however, that's not general and doesn't allow for user-defined indices or arbitrary index types.
(Of course, if there is no template, then we do fall back on that approach.)

The solution is this helper function, which is used liberally throughout the VarNamedTuple implementation:

```@example 1
function _is_multiindex(template::AbstractArray, ix...; kw...)
    return ndims(view(template, ix...; kw...)) > 0
end
```

which actually works surprisingly well, and is frequently (if not always?) constant propagated.

```@repl 1
using DimensionalData;
using InteractiveUtils: @code_warntype;

da = DimArray(randn(2, 3), (X(), :y));
@code_warntype _is_multiindex(da; a=X(Not(2)), y=2)
```
