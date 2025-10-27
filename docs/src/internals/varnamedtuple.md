# VarNamedTuple as the basis of VarInfo

This document collects thoughts and ideas for how to unify our multitude of AbstractVarInfo types using a VarNamedTuple type. It may eventually turn into a draft design document, but for now it is more raw than that.

## The current situation

We currently have the following AbstractVarInfo types:

  - A: VarInfo with Metadata
  - B: VarInfo with VarNamedVector
  - C: VarInfo with NamedTuple, with values being Metadata
  - D: VarInfo with NamedTuple, with values being VarNamedVector
  - E: SimpleVarInfo with NamedTuples
  - F: SimpleVarInfo with OrderedDict

A and C are the classic ones, and the defaults. C wraps groups the Metadata objects by the lead Symbol of the VarName of a variable, e.g. `x` in `@varname(x.y[1].z)`, which allows different lead Symbols to have different element types and for the VarInfo to still be type stable. B and D were created to simplify A and C, give them a nicer interface, and make them deal better with changing variable sizes, but according to recent (Oct 2025) benchmarks are quite a lot slower, which needs work.

E and F are entirely distinct in implementation from the others. E is simply a mapping from Symbols to values, with each VarName being converted to a single symbol, e.g. `Symbol("a[1]")`. F is a mapping from VarNames to values as an OrderedDict, with VarName as the key type.

A-D carry within them values for variables, but also their bijectors/distributions, and store all values vectorised, using the bijectors to map to the original values. They also store for each variable a flag for whether the variable has been linked. E-F store only the raw values, and a global flag for the whole SimpleVarInfo for whether it's linked. The link transform itself is implicit.

TODO: Write a better summary of pros and cons of each approach.

## VarNamedTuple

VarNamedTuple has been discussed as a possible data structure to generalise the structure used in VarInfo to achieve type stability, i.e. grouping VarNames by their lead Symbol. The same NamedTuple structure has been used elsewhere, too, e.g. in Turing.GibbsContext. The idea was to encapsulate this structure into its own type, reducing code duplication and making the design more robust and powerful. See https://github.com/TuringLang/DynamicPPL.jl/issues/900 for the discussion.

An AbstractVarInfo type could be only one application of VarNamedTuple, but here I'll focus on it exclusively. If we can make VarNamedTuple work for an AbstractVarInfo, I bet we can make it work for other purposes (condition, fix, Gibbs) as well.

Without going into full detail, here's @mhauru's current proposal for what it would look like. This proposal remains in constant flux as I develop the code.

A VarNamedTuple is a mapping of VarNames to values. Values can be anything. In the case of using VarNamedTuple to implement an AbstractVarInfo, the values would be random samples for random variables. However, they could hold with them extra information. For instance, we might use a value that is a tuple of a vectorised value, a bijector, and a flag for whether the variable is linked.

I sometimes shorten VarNamedTuple to VNT.

Internally, a VarNamedTuple consists of nested NamedTuples. For instance, the mapping `@varname(x) => 1, @varname(y.z) => 2` would be stored as

```
(; x=1, y=(; z=2))
```

(This is a slight simplification, really it would be nested VarNamedTuples rather than NamedTuples, but I omit this detail.)
This forms a tree, with each node being a NamedTuple, like so:

```
   NT
x /  \ y
 1    NT
       \ z
        2
```

Each `NT` marks a NamedTuple, and the labels on the edges its keys. Here the root node has the keys `x` and `y`. This is like with the type stable VarInfo in our current design, except with possibly more levels (our current one only has the root node). Each nested `PropertyLens`, i.e. each `.` in a VarName like `@varname(a.b.c.e)`, creates a new layer of the tree.

For simplicity, at least for now, we ban any VarNames where an `IndexLens` precedes a `PropertyLens`. That is, we ban any VarNames like `@varname(a.b[1].c)`. Recall that VarNames allow three different kinds of lenses: `PropertyLens`es, `IndexLens`es, and `identity` (the trivial lens). Thus the only allowed VarName types are `@varname(a.b.c.d)` and `@varname(a.b.c.d[i,j,k])`.

This means that we can add levels to the NamedTuple tree until all `PropertyLenses` have been covered. The leaves of the tree are then of two kinds: They are either the raw value itself if the last lens of the VarName is an `identity`, or otherwise they are something that can be indexed with an `IndexLens`, such as an `Array`.

To get a value from a VarNamedTuple is very simple: For `getindex(vnt::VNT, vn::VarName{S})` (`S` being the lead Symbol) you recurse into `getindex(vnt[S], unprefix(vn, S))`. If the last lens of `vn` is an `IndexLens`, we assume that the leaf of the NamedTuple tree we've reached contains something that can be indexed with it.

Setting values in a VNT is equally simple if there are no `IndexLenses`: For `setindex!!(vnt::VNT, value::Any, vn::VarName)` one simply finds the leaf of the `vnt` tree corresponding to `vn` and sets its value to `value`.

The tricky part is what to do when setting values with `IndexLenses`. There are three possible situations. Say one calls `setindex!!(vnt, 3.0, @varname(a.b[3]))`.

 1. If `getindex(vnt, @varname(a.b))` is already a vector of length at least 3, this is easy: Just set the third element.
 2. If `getindex(vnt, @varname(a.b))` is a vector of length less than 3, what should we do? Do we error? Do we extend that vector?
 3. If `getindex(vnt, @varname(a.b))` isn't even set, what do we do? Say for instance that `vnt` is currently empty. We should set `vnt` to be something like `(; a=(; b=x))`, where `x` is such that `x[3] = 3.0`, but what exactly should `x` be? Is it a dictionary? A vector of length 3? If the latter, what are `x[2]` and `x[1]`? Or should this `setindex!!` call simply error?

A note at this point: VarNamedTuples must always use `setindex!!`, the `!!` version that may or may not operate in place. The NamedTuples can't be modified in place, but the values at the leaves may be. Always using a `!!` function makes type stability easier, and makes structures like the type unstable old VarInfo with Metadata unnecessary: Any value can be set into any VarNamedTuple. The type parameters of the VNT will simply expand as necessary.

To solve the problem of points 2. and 3. above I propose expanding the definition of VNT a bit. This will also help make VNT more flexible, which may help performance or allow more use cases. The modification is this:

Unlike I said above, let's say that VNT isn't just nested NamedTuples with some values at the leaves. Let's say it also has a field called `make_leaf`. `make_leaf(value, lens)` is a function that takes any value, and a lens that is either `identity` or an `IndexLens`, and returns the value wrapped in some suitable struct that can be stored in the leaf of the NamedTuple tree. The values should always be such that `make_leaf(value, lens)[lens] == value`.

Our earlier example of `VarNamedTuple(@varname(x) => 1, @varname(y.z) => 2; make_leaf=f)` would be stored as a tree like

```
         --NT--
      x /      \ y
f(1, identity)  NT
                 \ z
            f(2, identity)
```

The above, first draft of VNT which did not include `make_leaf` is equivalent to the trivial choice `make_leaf(value, lens) = lens === identity ? value : error("Don't know how to deal IndexLenses")`. The problems 2. and 3. above are "solved" by making it `make_leaf`'s problem to figure out what to do. For instance, `make_leaf` can always return a `Dict` that maps lenses to values. This is probably slow, but works for any lens. Or it can initialise a vector type, that can grow as needed when indexed into.

The idea would be to use `make_leaf` to try out different ways of implementing a VarInfo, find a good default, and ,if necessary, leave the option for power users to customise behaviour. The first ones to implement would be

  - `make_leaf` that returns a Metadata object. This would be a direct replacement for type stable VarInfo that uses Metadata, except now with more nested levels of NamedTuple.
  - `make_leaf` that returns an `OrderedDict`. This would be a direct replacement for SimpleVarInfo with OrderedDict.

You may ask, have we simple gone from too many VarInfo types to too many `make_leaf` functions. Yes we have. But hopefully we have gained something in the process:

  - The leaf types can be simpler. They do not need to deal with VarNames any more, they only need to deal with `identity` lenses and `IndexLenses`.
  - All AbstactVarInfos are as type stable as their leaf types allow. There is no more notion of an untyped VarInfo being converted to a typed one.
  - Type stability is maintained even with nested `PropertyLenses` like `@varname(a.b)`, which happens a lot with submodels.
  - Many functions that are currently implemented individually for each AbstactVarInfo type would now have a single implementation for the VarNamedTuple-based AbstactVarInfo type, reducing code duplication. I would also hope to get ride of most of the generated functions for in `varinfo.jl`.

My guess is that the eventual One AbstractVarInfo To Rule Them All would have a `make_leaf` function that stores the raw values when the lens is an `identity`, and uses a flexible Vector, a lot like VarNamedVector, when the lens is an IndexLens. However, I could be wrong on that being the best option. Implementing and benchmarking is the only way to know.

I think the two big questions are:

  - Will we run into some big, unanticipated blockers when we start to implement this.
  - Will the nesting of NamedTuples cause performance regressions, if the compiler either chokes or gives up.

I'll try to derisk these early on in this PR.

## Questions / issues

* People might really need IndexLenses in the middle of VarNames. The one place this comes up is submodels within a loop. I'm still inclined to keep designing without allowing for that, for now, but should keep in mind that that needs to be relaxed eventually. If it makes it easier, we can require that users explicitly tell us the size of any arrays for which this is done.
* When storing values for nested NamedTuples, the actual variable may be a struct. Do we need to be able to reconstruct the struct from the NamedTuple? If so, how do we do that?
* Do `Colon` indices cause any extra trouble for the leafnodes?
