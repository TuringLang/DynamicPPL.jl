"""
    VarNamedTuple{names,Values}

A `NamedTuple`-like structure with `VarName` keys.

`VarNamedTuple` is a data structure for storing arbitrary data, keyed by `VarName`s, in an
efficient and type stable manner. It is mainly used through `getindex`, `setindex!!`, and
`haskey`, all of which accept `VarName`s and only `VarName`s as keys. Other notable methods
are `merge` and `subset`.

`VarNamedTuple` has an ordering to its elements, and two `VarNamedTuple`s with the same keys
and values but in different orders are considered different for equality and hashing.
Iterations such as `keys` and `values` respect this ordering. The ordering is dependent on
the order in which elements were inserted into the `VarNamedTuple`, though isn't always
equal to it. More specifically

* Any new keys that have a joint parent `VarName` with an existing key are inserted after
  that key. For instance, if one first inserts, in order, `@varname(a.x)`, `@varname(b)`,
  and `@varname(a.y)`, the resulting order will be
  `(@varname(a.x), @varname(a.y), @varname(b))`.
* `Index` keys`, like `@varname(a[3])` or `@varname(b[2,3,4:5])`, are always iterated
  in the same order an `Array` with the same indices would be iterated. For instance,
  if one first inserts, in order, `@varname(a[2])`, `@varname(b)`, and `@varname(a[1])`,
  the resulting order will be `(@varname(a[1]), @varname(a[2]), @varname(b))`.

Otherwise insertion order is respected.

The there are two major limitations to indexing by VarNamedTuples:

* `VarName`s with `Colon`s, (e.g. `a[:]`) are not supported. This is because the meaning of
  `a[:]` is ambiguous if only some elements of `a`, say `a[1]` and `a[3]`, are defined.
  However, _concretised_ `VarName`s with `Colon`s are supported.
* Any `VarNames` with `Index` lenses must have a consistent number of indices. That is, one
  cannot set `a[1]` and `a[1,2]` in the same `VarNamedTuple`.

`setindex!!` and `getindex` on `VarNamedTuple` are type stable as long as one does not store
heterogeneous data under different indices of the same symbol. That is, if either

* one sets `a[1]` and `a[2]` to be of different types, or
* if `a[1]` and `a[2]` both exist, one sets `a[1].b` without setting `a[2].b`,

then getting values for `a[1]` or `a[2]` will not be type stable.

`VarNamedTuple` is intrinsically linked to `PartialArray`, which it'll use to store data
related to `VarName`s with `Index` components.
"""
struct VarNamedTuple{Names,Values}
    data::NamedTuple{Names,Values}

    function VarNamedTuple(data::NamedTuple{Names,Values}) where {Names,Values}
        return new{Names,Values}(data)
    end
end

VarNamedTuple(; kwargs...) = VarNamedTuple((; kwargs...))

"""
    VarNamedTuple(d)
    VarNamedTuple(nt::NamedTuple)

Create a `VarNamedTuple` from a collection or a `NamedTuple`.

Any collection `d` is assumed to be an iterable of key-value pairs, where the keys are
`VarName`s. This could be a an `AbstractDict`, a vector of `Pair`s or `Tuple`s, etc. The
only exception is `NamedTuple`s, for which the `Symbol` keys are converted to `VarName`s.

Note that `VarNamedTuple` has an ordering to its elements, and two `VarNamedTuple`s with the
same keys and values but in different orders are considered different. If `d` does not
guarantee an iteration order, then the order of the elements in the resulting
`VarNamedTuple` is undefined.
"""
function VarNamedTuple(d)
    vnt = VarNamedTuple()
    for (k, v) in d
        vnt = setindex!!(vnt, v, k)
    end
    return vnt
end

Base.:(==)(vnt1::VarNamedTuple, vnt2::VarNamedTuple) = vnt1.data == vnt2.data
Base.isequal(vnt1::VarNamedTuple, vnt2::VarNamedTuple) = isequal(vnt1.data, vnt2.data)
Base.hash(vnt::VarNamedTuple, h::UInt) = hash(vnt.data, h)

function Base.show(io::IO, vnt::VarNamedTuple)
    if isempty(vnt.data)
        return print(io, "VarNamedTuple()")
    end
    print(io, "VarNamedTuple")
    show(io, vnt.data)
    return nothing
end

function Base.copy(vnt::VarNamedTuple{names}) where {names}
    # Make a shallow copy of vnt, except for any VarNamedTuple or PartialArray elements,
    # which we recursively copy.
    return VarNamedTuple(
        NamedTuple{names}(
            map(
                x -> x isa Union{VarNamedTuple,PartialArray} ? copy(x) : x, values(vnt.data)
            ),
        ),
    )
end

# PartialArrays are an implementation detail of VarNamedTuple, and should never be the
# return value of getindex. Thus, we automatically convert them to dense arrays if needed.
# TODO(mhauru) The below doesn't handle nested PartialArrays. Is that a problem?
as_array_if_needed(pa::PartialArray) = as_array(pa)
as_array_if_needed(x) = x
function Base.getindex(vnt::VarNamedTuple, vn::VarName)
    return as_array_if_needed(_getindex_optic(vnt, vn))
end

Base.haskey(vnt::VarNamedTuple, vn::VarName) = _haskey_optic(vnt, vn)

"""
    DynamicPPL.VarNamedTuples.templated_setindex!!(vnt, value, vn, template; allow_new=Val(true))

Assign `value` to the location in `vnt` specified by `vn`.

The argument `template` must be provided in order to guide the creation of `PartialArray`s,
as well as to concretise any dynamic indices in `vn`. It must be an object that has the
shape of the top-level symbol in `vn`. For example:

```julia
vnt = VarNamedTuple()
templated_setindex!!(vnt, 10, @varname(x[1]), rand(2, 2))
```

Here, `rand(2, 2)` is the template for the top-level symbol `x`, which tells `setindex!!`
that `x` should be a `PartialArray` that is backed by a matrix.

The actual data inside `template` is not needed, and `template` is never mutated by this
call.
"""
function templated_setindex!!(vnt::VarNamedTuple, value, vn::VarName, template)
    return _setindex_optic!!(vnt, value, AbstractPPL.varname_to_optic(vn), template, true)
end

"""
    BangBang.setindex!!(vnt::VarNamedTuple, value, vn::VarName)

This is similar to `templated_setindex!!`, but does not take a `template` argument. In
effect the `template` passed through is `NoTemplate()`. It is often the case that the
template is not actually needed (for example, if you are setting a top-level VarName:
`@varname(a)`, or if the VarName only contains Property optics, or if the arrays already
exist and you are merely updating a value inside it). In such cases, this method will work
fine, but may throw an error if the template is actually needed.

For example, this will error, since it is not known what `x` should be:

```julia
vnt = VarNamedTuple()
setindex!!(vnt, 10, @varname(x[1]))
```

TODO(penelopeysm): Fix this such that there is a 'default' template that can be used when
none is provided.
"""
function BangBang.setindex!!(vnt::VarNamedTuple, value, vn::VarName)
    return _setindex_optic!!(
        vnt, value, AbstractPPL.varname_to_optic(vn), NoTemplate(), true
    )
end

"""
    _has_partial_array(::Type{VarNamedTuple{Names,Values}}) where {Names,Values}

Check if any of the types in the `Values` tuple is or contains a `PartialArray`.

Recurses into any sub-`VarNamedTuple`s.
"""
@generated function _has_partial_array(
    ::Type{VarNamedTuple{Names,Values}}
) where {Names,Values}
    for T in Values.parameters
        if _has_partial_array(T)
            return :(return true)
        end
    end
    return :(return false)
end

_has_partial_array(::Type{T}) where {T} = false
_has_partial_array(::Type{<:PartialArray}) = true

Base.empty(::VarNamedTuple) = VarNamedTuple()

"""
    empty!!(vnt::VarNamedTuple)

Create an empty version of `vnt` in place.

This differs from `Base.empty` in that any `PartialArray`s contained within `vnt` are kept
but have their contents deleted, rather than being removed entirely. This means that

1) The result has a "memory" of how many dimensions different variables had, and you cannot,
   for example, set `a[1,2]` after emptying a `VarNamedTuple` that had only `a[1]` defined.
2) Memory allocations may be reduced when reusing `VarNamedTuple`s, since the internal
   `PartialArray`s do not need to be reallocated from scratch.
"""
@generated function BangBang.empty!!(vnt::VarNamedTuple{Names,Values}) where {Names,Values}
    if !_has_partial_array(VarNamedTuple{Names,Values})
        return :(return VarNamedTuple())
    end
    # Check all the fields of the NamedTuple, and keep the ones that contain PartialArrays,
    # calling empty!! on them recursively.
    new_names = ()
    new_values = ()
    for (name, ValType) in zip(Names, Values.parameters)
        if _has_partial_array(ValType)
            new_values = (new_values..., :(BangBang.empty!!(vnt.data.$name)))
            new_names = (new_names..., name)
        end
    end
    return quote
        return VarNamedTuple(NamedTuple{$new_names}(($(new_values...),)))
    end
end

@generated function Base.isempty(vnt::VarNamedTuple{Names,Values}) where {Names,Values}
    if isempty(Names)
        return :(return true)
    end
    if !_has_partial_array(VarNamedTuple{Names,Values})
        return :(return false)
    end
    exs = Expr[]
    for (name, ValType) in zip(Names, Values.parameters)
        if !_has_partial_array(ValType)
            return :(return false)
        end
        push!(
            exs,
            quote
                val = vnt.data.$name
                if val isa VarNamedTuple || val isa PartialArray
                    if !Base.isempty(val)
                        return false
                    end
                else
                    return false
                end
            end,
        )
    end
    push!(exs, :(return true))
    return Expr(:block, exs...)
end
