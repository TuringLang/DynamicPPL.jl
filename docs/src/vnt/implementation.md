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

Because `VarName`s have a nested structure, it should not be surprising to find that `VarNamedTuple`s are constructed by recursing into `_setindex_optic!!`.
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

This should conceptually be fairly understandable.
Here is a simplified version of the recursive implementation:

```julia
# Let's say `optic` is _[1].a
function make_leaf(leaf_value, optic, template)
    this_optic, child_optic = split_optic(optic)
    # ^ _[1]    ^ _.a 

    sub_value = make_leaf(leaf_value, child_optic, child(template))
    # ^ VNT                           ^ _.a        ^ template[1]

    empty_value = create_empty_structure(this_optic, template)
    # ^ PA                               ^ _[1]

    value = setindex!!(empty_value, this_optic, sub_value)
    # ^ PA->VNT        ^ PA         ^ _.[1]     ^ VNT

    return value
end
```

Some explanatory notes:

 1. The whole purpose of the function is to ensure that `value` is something where you can index into with `optic` to get `leaf_value`.

 2. Since `sub_value` is also created with the same function, that means it must be something you can index into with `child_optic` to get `leaf_value`.
 3. `empty_value` needs to be something that can hold `sub_value` at `this_optic`. We don't yet insert the data. However, to ensure type stability, we should *instantiate* the PA with the correct element type: in this case that's just `typeof(sub_value)`. (If we don't use the correct element type, the subsequent call to `setindex!!` will have to change the element type of the PA.)
 4. `value` is then created by putting `sub_value` into `empty_value` at `this_optic`.

!!! info
    
    Regarding point (3): we haven't yet covered `ArrayLikeBlock`s (that will be on the next page). If `sub_value` is something that would be stored as an `ArrayLikeBlock`, we need to instantiate `empty_value` with `typeof(ArrayLikeBlock(sub_value))` instead of `typeof(sub_value)`, again for type stability reasons. If you are not familiar with this, don't worry about it for now.

## Multi-indices

Multi-indexing, or slices, is where things get a bit more complicated, as it enforces extra constraints that are not present above.
To illustrate this, we'll now consider the case where the optic is `_[2:3][1]`.

Logically speaking, this is exactly the same as `_[2]`.
So we should be creating a PartialArray that holds `leaf_value` at index `2`.
But on top of that, we also need to make sure that the created structure has at least three indices: otherwise, the index `2:3` would be invalid.

```julia
# Let's say `optic` is _[2:3][1]
function make_leaf(leaf_value, optic, template)
    this_optic, child_optic = split_optic(optic)
    # ^ _[2:3]  ^ _[1]

    sub_value = make_leaf(leaf_value, child_optic, child(template))
    # ^ PA (x)                        ^ _[1]       ^ template[2:3]

    empty_value = create_empty_structure(this_optic, template)
    # ^ PA (y)                           ^ _[2:3]

    value = setindex!!(empty_value, this_optic, sub_value)
    # ^ PA (z)         ^ PA (y)     ^ _[2:3]    ^ PA (x)   

    return value
end
```

There are three PartialArrays being created here, which are called `x`, `y`, and `z` in the comments above.

The first point of difference is when creating `empty_value`: previously, for type stability purposes, we would create the PA with an element type of `typeof(sub_value).`
However, in this case, `sub_value` is actually a _slice_ of `empty_value`, and so we can't use its type: we need to use `eltype(sub_value)` instead.

The other tricky part is about lengths.
Let's work backwards from the last line.
`z` and `y` will have the same length, since they are related by the `setindex!!` call.
They must be at least of length 3.
This is easy to guarantee: when creating `y`, we either have a template to work with (which must already be of more than length 3), or we can use a GrowableArray and deduce the minimum length from the index `2:3`.

**However, `x` must also _exactly_ have length 2;** otherwise, when setting it into the indices `2:3` of `y`, an error will occur.
This part is more problematic, because the recursive call to `make_leaf` only ever sees the index `1`.
If it _doesn't_ have a template to work with, this will error, because it will create a GrowableArray of length 1, which cannot then be set into indices `2:3` of `y`.

(If it _does_ have a template, we are all good, because the template will be created by indexing into the upper-level template with `2:3`, which will give a template of the right size.)

To solve this, we have two choices:

 1. Recursively pass down information about the expected size of the template; or
 2. After getting `sub_value`, check if `this_index` is a multi-index, and if so, expand `sub_value` to the correct length.

The current implementation uses the second approach.
Note that this is only needed when there is no template provided, and when there is a multi-index.

## Detecting multi-indices

This means that in general we need a good way of differentiating between single-element indexing and multi-element indexing.
One _could_ iterate over the indices and check which of them are ranges or colons, for example; however, that's not general and doesn't allow for user-defined indices or arbitrary index types.
(Of course, if there is no template, then we do fall back on that approach.)

The solution is this helper function, which is used liberally throughout the VarNamedTuple implementation:

```@example 1
function _is_multiindex(template::AbstractArray, ix...; kw...)
    return ndims(view(template, ix...; kw...)) > 0
end
```

which works really well across different array types, and is frequently (if not always?) constant propagated.

```@repl 1
using DimensionalData;
using InteractiveUtils: @code_warntype;

da = DimArray(randn(2, 3), (X(), :y));
@code_warntype _is_multiindex(da; a=X(Not(2)), y=2)
```
