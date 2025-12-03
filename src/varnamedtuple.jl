# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using BangBang
using Accessors
using ..DynamicPPL: _compose_no_identity

export VarNamedTuple

# We define our own getindex, setindex!!, and haskey functions, which we use to
# get/set/check values in VarNamedTuple and PartialArray. We do this because we want to be
# able to override their behaviour for some types exported from elsewhere without type
# piracy. This is needed because
# 1. We would want to index into things with lenses (from Accessors.jl) using getindex and
# setindex!!, but Accessors does not define these methods.
# 2. We would want `haskey` to fall back onto `checkbounds` when called on Base.Arrays.
function _getindex end
function _haskey end
function _setindex!! end

_getindex(arr::AbstractArray, optic::IndexLens) = getindex(arr, optic.indices...)
_haskey(arr::AbstractArray, optic::IndexLens) = _haskey(arr, optic.indices)
_haskey(arr::AbstractArray, inds) = checkbounds(Bool, arr, inds...)
function _setindex!!(arr::AbstractArray, value, optic::IndexLens)
    return setindex!!(arr, value, optic.indices...)
end

# Some utilities for checking what sort of indices we are dealing with.
_has_colon(::T) where {T<:Tuple} = any(x <: Colon for x in T.parameters)
function _is_multiindex(::T) where {T<:Tuple}
    return any(x <: UnitRange || x <: Colon for x in T.parameters)
end

"""
    _merge_recursive(x1, x2)

Recursively merge two values `x1` and `x2`.

Unlike `Base.merge`, this function is defined for all types, and by default returns the
second argument. It is overridden for `PartialArray` and `VarNamedTuple`, since they are
nested containers, and calls itself recursively on all elements that are found in both
`x1` and `x2`.

In other words, if both `x` and `y` are collections with the key `a`, `Base.merge(x, y)[a]`
is `y[a]`, whereas `_merge_recursive(x, y)[a]` will be `_merge_recursive(x[a], y[a])`,
unless no specific method is defined for the type of `x` and `y`, in which case
`_merge_recursive(x, y) === y`
"""
_merge_recursive(_, x2) = x2

"""The factor by which we increase the dimensions of PartialArrays when resizing them."""
const PARTIAL_ARRAY_DIM_GROWTH_FACTOR = 4

"""A convenience for defining method argument type bounds."""
const INDEX_TYPES = Union{Integer,UnitRange,Colon}

"""
    PartialArray{ElType,numdims}

An array-like like structure that may only have some of its elements defined.

A `PartialArray` is like a `Base.Array,` except not all of its elements are necessarily
defined. That is to say, one can create an empty `PartialArray` `arr` and e.g. set
`arr[3,2] = 5`, but asking for `arr[1,1]` may throw a `BoundsError` if `[1, 1]` has not been
explicitly set yet.

`PartialArray`s can be indexed with integer indices and ranges. Indexing is always 1-based.
Other types of indexing allowed by `Base.Array` are not supported. Some of these are simply
because we haven't seen a need and haven't bothered to implement them, namely boolean
indexing, linear indexing into multidimensional arrays, and indexing with arrays. However,
notably, indexing with colons (i.e. `:`) is not supported for more fundamental reasons.

To understand this, note that a `PartialArray` has no well-defined size. For example, if one
creates an empty array and sets `arr[3,2]`, it is unclear if that should be taken to mean
that the array has size `(3,2)`: It could be larger, and saying that the size is `(3,2)`
would also misleadingly suggest that all elements within `1:3,1:2` are set. This is also why
colon indexing is ill-defined: If one would e.g. set `arr[2,:] = [1,2,3]`, we would have no
way of saying whether the right hand side is of an acceptable size or not.

The fact that its size is ill-defined also means that `PartialArray` is not a subtype of
`AbstractArray`.

All indexing into `PartialArray`s is done with `getindex` and `setindex!!`. `setindex!`,
`push!`, etc. are not defined. The element type of a `PartialArray` will change as needed
under `setindex!!` to accomoddate the new values.

Like `Base.Array`s, `PartialArray`s have a well-defined, compile-time-known element type
`ElType` and number of dimensions `numdims`. Indices into a `PartialArray` must have exactly
`numdims` elements.

If the element type of a `PartialArray` is not concrete, any call to `setindex!!` will check
if, after the new value has been set, the element type can be made more concrete. If so,
a new `PartialArray` with a more concrete element type is returned. Thus the element type
of any `PartialArray` should always be as concrete as is allowed by the elements in it.

The internal implementation of an `PartialArray` consists of two arrays: one holding the
data and the other one being a boolean mask indicating which elements are defined. These
internal arrays may need resizing when new elements are set that have index ranges larger
than the current internal arrays. To avoid resizing too often, the internal arrays are
resized in exponentially increasing steps. This means that most `setindex!!` calls are very
fast, but some may incur substantial overhead due to resizing and copying data. It also
means that the largest index set so far determines the memory usage of the `PartialArray`.
`PartialArray`s are thus well-suited when most values in it will eventually be set. If only
a few scattered values are set, a structure like `SparseArray` may be more appropriate.
"""
struct PartialArray{ElType,num_dims}
    data::Array{ElType,num_dims}
    mask::Array{Bool,num_dims}

    function PartialArray(
        data::Array{ElType,num_dims}, mask::Array{Bool,num_dims}
    ) where {ElType,num_dims}
        if size(data) != size(mask)
            throw(ArgumentError("Data and mask arrays must have the same size"))
        end
        return new{ElType,num_dims}(data, mask)
    end
end

"""
    PartialArray{ElType,num_dims}(args::Vararg{Pair}; min_size=nothing)

Create a new `PartialArray`.

The element type and number of dimensions have to be specified explicitly as type
parameters. The positional arguments can be `Pair`s of indices and values. For example,
```jldoctest
julia> using DynamicPPL.VarNamedTuples: PartialArray

julia> pa = PartialArray{Int,2}((1,2) => 5, (3,4) => 10)
PartialArray{Int64,2}((1, 2) => 5, (3, 4) => 10)
```

The optional keyword argument `min_size` can be used to specify the minimum initial size.
This is purely a performance optimisation, to avoid resizing if the eventual size is known
ahead of time.
"""
function PartialArray{ElType,num_dims}(
    args::Vararg{Pair}; min_size::Union{Tuple,Nothing}=nothing
) where {ElType,num_dims}
    dims = if min_size === nothing
        ntuple(_ -> PARTIAL_ARRAY_DIM_GROWTH_FACTOR, num_dims)
    else
        map(_partial_array_dim_size, min_size)
    end
    data = Array{ElType,num_dims}(undef, dims)
    mask = fill(false, dims)
    pa = PartialArray(data, mask)

    for (inds, value) in args
        pa = _setindex!!(pa, convert(ElType, value), inds...)
    end
    return pa
end

Base.ndims(::PartialArray{ElType,num_dims}) where {ElType,num_dims} = num_dims
Base.eltype(::PartialArray{ElType}) where {ElType} = ElType

function Base.show(io::IO, pa::PartialArray)
    print(io, "PartialArray{", eltype(pa), ",", ndims(pa), "}(")
    is_first = true
    for inds in CartesianIndices(pa.mask)
        if @inbounds(!pa.mask[inds])
            continue
        end
        if !is_first
            print(io, ", ")
        else
            is_first = false
        end
        val = @inbounds(pa.data[inds])
        # Note the distinction: The raw strings that form part of the structure of the print
        # out are `print`ed, whereas the keys and values are `show`n. The latter ensures
        # that strings are quoted, Symbols are prefixed with :, etc.
        show(io, Tuple(inds))
        print(io, " => ")
        show(io, val)
    end
    print(io, ")")
    return nothing
end

# We deliberately don't define Base.size for PartialArray, because it is ill-defined.
# The size of the .data field is an implementation detail.
_internal_size(pa::PartialArray, args...) = size(pa.data, args...)

function Base.copy(pa::PartialArray)
    # Make a shallow copy of pa, except for any VarNamedTuple elements, which we recursively
    # copy.
    pa_copy = PartialArray(copy(pa.data), copy(pa.mask))
    if VarNamedTuple <: eltype(pa) || eltype(pa) <: VarNamedTuple
        @inbounds for i in eachindex(pa.mask)
            if pa.mask[i] && pa_copy.data[i] isa VarNamedTuple
                pa_copy.data[i] = copy(pa.data[i])
            end
        end
    end
    return pa_copy
end

function Base.:(==)(pa1::PartialArray, pa2::PartialArray)
    if ndims(pa1) != ndims(pa2)
        return false
    end
    size1 = _internal_size(pa1)
    size2 = _internal_size(pa2)
    # TODO(mhauru) This could be optimised by not calling checkbounds on all elements
    # outside the size of an array, but not sure it's worth it.
    merge_size = ntuple(i -> max(size1[i], size2[i]), ndims(pa1))
    for i in CartesianIndices(merge_size)
        m1 = checkbounds(Bool, pa1.mask, Tuple(i)...) ? pa1.mask[i] : false
        m2 = checkbounds(Bool, pa2.mask, Tuple(i)...) ? pa2.mask[i] : false
        if m1 != m2
            return false
        end
        if m1 && (pa1.data[i] != pa2.data[i])
            return false
        end
    end
    return true
end

function Base.hash(pa::PartialArray, h::UInt)
    h = hash(ndims(pa), h)
    for i in eachindex(pa.mask)
        @inbounds if pa.mask[i]
            h = hash(i, h)
            h = hash(pa.data[i], h)
        end
    end
    return h
end

"""
    _concretise_eltype!!(pa::PartialArray)

Concretise the element type of a `PartialArray`.

Returns a new `PartialArray` with the same data and mask as `pa`, but with its element type
concretised to the most specific type that can hold all currently defined elements.

Note that this function is fundamentally type unstable if the current element type of `pa`
is not already concrete.

The name has a `!!` not because it mutates its argument, but because the return value
aliases memory with the argument, and is thus not independent of it.
"""
function _concretise_eltype!!(pa::PartialArray)
    if isconcretetype(eltype(pa))
        return pa
    end
    new_et = promote_type((typeof(pa.data[i]) for i in eachindex(pa.mask) if pa.mask[i])...)
    # TODO(mhauru) Should we check as below, or rather isconcretetype(new_et)?
    # In other words, does it help to be more concrete, even if we aren't fully concrete?
    if new_et === eltype(pa)
        # The types of the elements do not allow for concretisation.
        return pa
    end
    new_data = Array{new_et,ndims(pa)}(undef, _internal_size(pa))
    @inbounds for i in eachindex(pa.mask)
        if pa.mask[i]
            new_data[i] = pa.data[i]
        end
    end
    return PartialArray(new_data, pa.mask)
end

"""Return the length needed in a dimension given an index."""
_length_needed(i::Integer) = i
_length_needed(r::UnitRange) = last(r)

"""Take the minimum size that a dimension of a PartialArray needs to be, and return the size
we choose it to be. This size will be the smallest possible power of
PARTIAL_ARRAY_DIM_GROWTH_FACTOR. Growing PartialArrays in big jumps like this helps reduce
data copying, as resizes aren't needed as often.
"""
function _partial_array_dim_size(min_dim)
    factor = PARTIAL_ARRAY_DIM_GROWTH_FACTOR
    return factor^(Int(ceil(log(factor, min_dim))))
end

"""Return the minimum internal size needed for a `PartialArray` to be able set the value
at inds.
"""
function _min_size(pa::PartialArray, inds)
    return ntuple(i -> max(_internal_size(pa, i), _length_needed(inds[i])), length(inds))
end

"""Resize a PartialArray to be able to accommodate the index inds. This operates in place
for vectors, but makes a copy for higher-dimensional arrays, unless no resizing is
necessary, in which case this is a no-op."""
function _resize_partialarray!!(pa::PartialArray, inds)
    min_size = _min_size(pa, inds)
    new_size = map(_partial_array_dim_size, min_size)
    if new_size == _internal_size(pa)
        return pa
    end
    # Generic multidimensional Arrays can not be resized, so we need to make a new one.
    # See https://github.com/JuliaLang/julia/issues/37900
    new_data = Array{eltype(pa),ndims(pa)}(undef, new_size)
    new_mask = fill(false, new_size)
    # Note that we have to use CartesianIndices instead of eachindex, because the latter
    # may use a linear index that does not match between the old and the new arrays.
    @inbounds for i in CartesianIndices(pa.data)
        mask_val = pa.mask[i]
        if mask_val
            new_mask[i] = mask_val
            new_data[i] = pa.data[i]
        end
    end
    return PartialArray(new_data, new_mask)
end

# The below implements the same functionality as above, but more performantly for 1D arrays.
function _resize_partialarray!!(pa::PartialArray{Eltype,1}, (ind,)) where {Eltype}
    # Resize arrays to accommodate new indices.
    old_size = _internal_size(pa, 1)
    min_size = max(old_size, _length_needed(ind))
    new_size = _partial_array_dim_size(min_size)
    if new_size == old_size
        return pa
    end
    resize!(pa.data, new_size)
    resize!(pa.mask, new_size)
    @inbounds pa.mask[(old_size + 1):new_size] .= false
    return pa
end

_getindex(pa::PartialArray, optic::IndexLens) = _getindex(pa, optic.indices...)
_haskey(pa::PartialArray, optic::IndexLens) = _haskey(pa, optic.indices)
function _setindex!!(pa::PartialArray, value, optic::IndexLens)
    return _setindex!!(pa, value, optic.indices...)
end

"""Throw an appropriate error if the given indices are invalid for `pa`."""
function _check_index_validity(pa::PartialArray, inds::NTuple{N,INDEX_TYPES}) where {N}
    if length(inds) != ndims(pa)
        throw(BoundsError(pa, inds))
    end
    if _has_colon(inds)
        throw(ArgumentError("Indexing PartialArrays with Colon is not supported"))
    end
    return nothing
end

function _getindex(pa::PartialArray, inds::Vararg{INDEX_TYPES})
    _check_index_validity(pa, inds)
    if !_haskey(pa, inds)
        throw(BoundsError(pa, inds))
    end
    return getindex(pa.data, inds...)
end

function _haskey(pa::PartialArray, inds::NTuple{N,INDEX_TYPES}) where {N}
    _check_index_validity(pa, inds)
    return checkbounds(Bool, pa.mask, inds...) && all(@inbounds(getindex(pa.mask, inds...)))
end

function _setindex!!(pa::PartialArray, value, inds::Vararg{INDEX_TYPES})
    _check_index_validity(pa, inds)
    pa = if checkbounds(Bool, pa.mask, inds...)
        pa
    else
        _resize_partialarray!!(pa, inds)
    end
    new_data = setindex!!(pa.data, value, inds...)
    if _is_multiindex(inds)
        pa.mask[inds...] .= true
    else
        pa.mask[inds...] = true
    end
    return _concretise_eltype!!(PartialArray(new_data, pa.mask))
end

Base.merge(x1::PartialArray, x2::PartialArray) = _merge_recursive(x1, x2)

function _merge_element_recursive(x1::PartialArray, x2::PartialArray, ind::CartesianIndex)
    m1 = x1.mask[ind]
    m2 = x2.mask[ind]
    return if m1 && m2
        _merge_recursive(x1.data[ind], x2.data[ind])
    elseif m2
        x2.data[ind]
    else
        x1.data[ind]
    end
end

# TODO(mhauru) Would this benefit from a specialised method for 1D PartialArrays?
function _merge_recursive(pa1::PartialArray, pa2::PartialArray)
    if ndims(pa1) != ndims(pa2)
        throw(
            ArgumentError("Cannot merge PartialArrays with different numbers of dimensions")
        )
    end
    num_dims = ndims(pa1)
    merge_size = ntuple(i -> max(_internal_size(pa1, i), _internal_size(pa2, i)), num_dims)
    return if merge_size == _internal_size(pa2)
        # Either pa2 is strictly bigger than pa1 or they are equal in size.
        result = copy(pa2)
        for i in CartesianIndices(pa1.data)
            @inbounds if pa1.mask[i]
                result = setindex!!(
                    result, _merge_element_recursive(pa1, result, i), Tuple(i)...
                )
            end
        end
        result
    else
        if merge_size == _internal_size(pa1)
            # pa1 is bigger than pa2
            result = copy(pa1)
            for i in CartesianIndices(pa2.data)
                @inbounds if pa2.mask[i]
                    result = setindex!!(
                        result, _merge_element_recursive(result, pa2, i), Tuple(i)...
                    )
                end
            end
            result
        else
            # Neither is strictly bigger than the other.
            et = promote_type(eltype(pa1), eltype(pa2))
            new_data = Array{et,num_dims}(undef, merge_size)
            new_mask = fill(false, merge_size)
            result = PartialArray(new_data, new_mask)
            for i in CartesianIndices(pa2.data)
                @inbounds if pa2.mask[i]
                    result.mask[i] = true
                    result.data[i] = pa2.data[i]
                end
            end
            for i in CartesianIndices(pa1.data)
                @inbounds if pa1.mask[i]
                    result = setindex!!(
                        result, _merge_element_recursive(pa1, result, i), Tuple(i)...
                    )
                end
            end
            result
        end
    end
end

function Base.keys(pa::PartialArray)
    inds = findall(pa.mask)
    lenses = map(x -> IndexLens(Tuple(x)), inds)
    ks = Any[]
    for lens in lenses
        val = getindex(pa.data, lens.indices...)
        if val isa VarNamedTuple
            subkeys = keys(val)
            for vn in subkeys
                sublens = _varname_to_lens(vn)
                push!(ks, _compose_no_identity(sublens, lens))
            end
        else
            push!(ks, lens)
        end
    end
    return ks
end

"""
    VarNamedTuple{names,Values}

A `NamedTuple`-like structure with `VarName` keys.

`VarNamedTuple` is a data structure for storing arbitrary data, keyed by `VarName`s, in an
efficient and type stable manner. It is mainly used through `getindex`, `setindex!!`, and
`haskey`, all of which accept `VarName`s and only `VarName`s as keys. Anther notable methods
is `merge`, which recursively merges two `VarNamedTuple`s.

The there are two major limitations to indexing by VarNamedTuples:

* `VarName`s with `Colon`s, (e.g. `a[:]`) are not supported. This is because the meaning of `a[:]` is ambiguous if only some elements of `a`, say `a[1]` and `a[3]`, are defined.
* Any `VarNames` with IndexLenses` must have a consistent number of indices. That is, one cannot set `a[1]` and `a[1,2]` in the same `VarNamedTuple`.

`setindex!!` and `getindex` on `VarNamedTuple` are type stable as long as one does not store
heterogeneous data under different indices of the same symbol. That is, if one either

* sets `a[1]` and `a[2]` to be of different types, or
* sets `a[1].b` and `a[2].c`, without setting `a[1].c`. or `a[2].b`,

then getting values for `a[1]` or `a[2]` will not be type stable.

`VarNamedTuple` is intrinsically linked to `PartialArray`, which it'll use to store data
related to `VarName`s with `IndexLens` components.
"""
struct VarNamedTuple{Names,Values}
    data::NamedTuple{Names,Values}
end

VarNamedTuple(; kwargs...) = VarNamedTuple((; kwargs...))

Base.:(==)(vnt1::VarNamedTuple, vnt2::VarNamedTuple) = vnt1.data == vnt2.data
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

"""
    varname_to_lens(name::VarName{S}) where {S}

Convert a `VarName` to an `Accessor` lens, wrapping the first symbol in a `PropertyLens`.

This is used to simplify method dispatch for `_getindex`, `_setindex!!`, and `_haskey`, by
considering `VarName`s to just be a special case of lenses.
"""
function _varname_to_lens(name::VarName{S}) where {S}
    return _compose_no_identity(getoptic(name), PropertyLens{S}())
end

_getindex(vnt::VarNamedTuple, name::VarName) = _getindex(vnt, _varname_to_lens(name))
_getindex(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = getindex(vnt.data, S)
_getindex(vnt::VarNamedTuple, name::Symbol) = vnt.data[name]

_haskey(vnt::VarNamedTuple, name::VarName) = _haskey(vnt, _varname_to_lens(name))
_haskey(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = haskey(vnt.data, S)
_haskey(vnt::VarNamedTuple, ::typeof(identity)) = true
_haskey(::VarNamedTuple, ::IndexLens) = false

function _setindex!!(vnt::VarNamedTuple, value, name::VarName)
    return _setindex!!(vnt, value, _varname_to_lens(name))
end

function _setindex!!(vnt::VarNamedTuple, value, ::PropertyLens{S}) where {S}
    # I would like for this to just read
    # return VarNamedTuple(_setindex!!(vnt.data, value, S))
    # but that seems to be type unstable. Why? Shouldn't it obviously be the same as the
    # below?
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((value,))))
end

Base.merge(x1::VarNamedTuple, x2::VarNamedTuple) = _merge_recursive(x1, x2)

# This needs to be a generated function for type stability.
@generated function _merge_recursive(
    vnt1::VarNamedTuple{names1}, vnt2::VarNamedTuple{names2}
) where {names1,names2}
    all_names = union(names1, names2)
    exs = Expr[]
    push!(exs, :(data = (;)))
    for name in all_names
        val_expr = if name in names1 && name in names2
            :(_merge_recursive(vnt1.data[$(QuoteNode(name))], vnt2.data[$(QuoteNode(name))]))
        elseif name in names1
            :(vnt1.data[$(QuoteNode(name))])
        else
            :(vnt2.data[$(QuoteNode(name))])
        end
        push!(exs, :(data = merge(data, NamedTuple{($(QuoteNode(name)),)}(($val_expr,)))))
    end
    push!(exs, :(return VarNamedTuple(data)))
    return Expr(:block, exs...)
end

# TODO(mhauru) The below remains unfinished an undertested. I think it's incorrect for more
# complex VarNames. It is unexported though.
"""
    apply!!(func, vnt::VarNamedTuple, name::VarName)

Apply `func` to the subdata at `name` in `vnt`, and set the result back at `name`.

```jldoctest
julia> using DynamicPPL: VarNamedTuple, setindex!!

julia> using DynamicPPL.VarNamedTuples: apply!!

julia> vnt = VarNamedTuple()
VarNamedTuple()

julia> vnt = setindex!!(vnt, [1, 2, 3], @varname(a))
VarNamedTuple(a = [1, 2, 3],)

julia> apply!!(x -> x .+ 1, vnt, @varname(a))
VarNamedTuple(a = [2, 3, 4],)
```
"""
function apply!!(func, vnt::VarNamedTuple, name::VarName)
    if !haskey(vnt, name)
        throw(KeyError(repr(name)))
    end
    subdata = _getindex(vnt, name)
    new_subdata = func(subdata)
    return _setindex!!(vnt, new_subdata, name)
end

# TODO(mhauru) Should this return tuples, like it does now? That makes sense for
# VarNamedTuple itself, but if there is a nested PartialArray the tuple might get very big.
# Also, this is not very type stable, it fails even in basic cases. A generated function
# would help, but I failed to make one. Might be something to do with a recursive
# generated function.
function Base.keys(vnt::VarNamedTuple)
    result = ()
    for sym in keys(vnt.data)
        subdata = vnt.data[sym]
        if subdata isa VarNamedTuple
            subkeys = keys(subdata)
            result = (
                result..., (AbstractPPL.prefix(sk, VarName{sym}()) for sk in subkeys)...
            )
        elseif subdata isa PartialArray
            subkeys = keys(subdata)
            result = (result..., (VarName{sym}(lens) for lens in subkeys)...)
        else
            result = (result..., VarName{sym}())
        end
    end
    return result
end

# The following methods, indexing with ComposedFunction, are exactly the same for
# VarNamedTuple and PartialArray, since they just fall back on indexing with the outer and
# inner lenses.
const VNT_OR_PA = Union{VarNamedTuple,PartialArray}

function _getindex(x::VNT_OR_PA, optic::ComposedFunction)
    subdata = _getindex(x, optic.inner)
    return _getindex(subdata, optic.outer)
end

function _setindex!!(vnt::VNT_OR_PA, value, optic::ComposedFunction)
    sub = if _haskey(vnt, optic.inner)
        _setindex!!(_getindex(vnt, optic.inner), value, optic.outer)
    else
        make_leaf(value, optic.outer)
    end
    return _setindex!!(vnt, sub, optic.inner)
end

function _haskey(vnt::VNT_OR_PA, optic::ComposedFunction)
    return _haskey(vnt, optic.inner) && _haskey(_getindex(vnt, optic.inner), optic.outer)
end

# The entry points for getting, setting, and checking, using the familiar functions.
Base.haskey(vnt::VarNamedTuple, vn::VarName) = _haskey(vnt, vn)
Base.getindex(vnt::VarNamedTuple, vn::VarName) = _getindex(vnt, vn)
BangBang.setindex!!(vnt::VarNamedTuple, value, vn::VarName) = _setindex!!(vnt, value, vn)

Base.haskey(vnt::PartialArray, key) = _haskey(vnt, key)
Base.getindex(vnt::PartialArray, inds...) = _getindex(vnt, inds...)
BangBang.setindex!!(vnt::PartialArray, value, inds...) = _setindex!!(vnt, value, inds...)

"""
    make_leaf(value, optic)

Make a new leaf node for a VarNamedTuple.

This is the function that sets any `optic` that is a `PropertyLens` to be stored as a
`VarNamedTuple`, any `IndexLens` to be stored as a `PartialArray`, and other `identity`
optics to be stored as raw values. It is the link that joins `VarNamedTuple` and
`PartialArray` together.
"""
make_leaf(value, ::typeof(identity)) = value
make_leaf(value, ::PropertyLens{S}) where {S} = VarNamedTuple(NamedTuple{(S,)}((value,)))

function make_leaf(value, optic::ComposedFunction)
    sub = make_leaf(value, optic.outer)
    return make_leaf(sub, optic.inner)
end

function make_leaf(value, optic::IndexLens)
    inds = optic.indices
    num_inds = length(inds)
    # Check if any of the indices are ranges or colons. If yes, value needs to be an
    # AbstractArray. Otherwise it needs to be an individual value.
    et = _is_multiindex(inds) ? eltype(value) : typeof(value)
    pa = PartialArray{et,num_inds}()
    return _setindex!!(pa, value, optic)
end

end
