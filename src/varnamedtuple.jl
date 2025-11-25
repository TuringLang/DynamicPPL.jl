# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using BangBang
using Accessors
using ..DynamicPPL: _compose_no_identity

export VarNamedTuple

# We define our own getindex, setindex!!, and haskey functions to be able to override their
# behaviour for some types exported from elsewhere without type piracy. This is needed
# because
# 1. We want to index into things with lenses (from Accessors.jl) using getindex and
# setindex!!.
# 2. We want to use getindex, setindex!!, and haskey as the universal functions for getting,
# setting, checking. This includes e.g. checking whether an index is valid for an Array,
# which would normally be done with checkbounds.
_haskey(x, key) = Base.haskey(x, key)
_getindex(x, inds...) = Base.getindex(x, inds...)
_setindex!!(x, value, inds...) = BangBang.setindex!!(x, value, inds...)
_getindex(arr::AbstractArray, optic::IndexLens) = _getindex(arr, optic.indices...)
_haskey(arr::AbstractArray, optic::IndexLens) = _haskey(arr, optic.indices)
function _setindex!!(arr::AbstractArray, value, optic::IndexLens)
    return _setindex!!(arr, value, optic.indices...)
end
_haskey(arr::AbstractArray, inds) = checkbounds(Bool, arr, inds...)

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
is `y[a]`, whereas `_merge_recursive(x, y)[a]` be `_merge_recursive(x[a], y[a])`, unless
no specific method is defined for the type of `x` and `y`, in which case
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

All indexing into `PartialArray`s are done with `getindex` and `setindex!!`. `setindex!`,
`push!`, etc. are not defined. The element type of a `PartialArray` will change as needed
under `setindex!!` to accomoddate the new values.

Like `Base.Array`s, `PartialArray`s have a well-defined, compile-time-known element type
`ElType` and number of dimensions `numdims`.

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
    PartialArray{ElType,num_dims}(min_size=nothing)

Create a new empty `PartialArray` with set element type and number of dimensions.

The optional argument `min_size` can be used to specify the minimum initial size. This is
purely a performance optimisation, to avoid resizing if the eventual size is known ahead of
time.
"""
function PartialArray{ElType,num_dims}(
    min_size::Union{Tuple,Nothing}=nothing
) where {ElType,num_dims}
    if min_size === nothing
        dims = ntuple(_ -> PARTIAL_ARRAY_DIM_GROWTH_FACTOR, num_dims)
    else
        dims = map(_partial_array_dim_size, min_size)
    end
    dims = ntuple(_ -> PARTIAL_ARRAY_DIM_GROWTH_FACTOR, num_dims)
    data = Array{ElType,num_dims}(undef, dims)
    mask = fill(false, dims)
    return PartialArray(data, mask)
end

Base.ndims(::PartialArray{ElType,num_dims}) where {ElType,num_dims} = num_dims
Base.eltype(::PartialArray{ElType}) where {ElType} = ElType

# We deliberately don't define Base.size for PartialArray, because it is ill-defined.
# The size of the .data field is an implementation detail.
_internal_size(pa::PartialArray, args...) = size(pa.data, args...)

function Base.copy(pa::PartialArray)
    return PartialArray(copy(pa.data), copy(pa.mask))
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
        new_mask[i] = mask_val
        if mask_val
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
            ArgumentError("Cannot merge PartialArrays with different number of dimensions")
        )
    end
    num_dims = ndims(pa1)
    merge_size = ntuple(i -> max(_internal_size(pa1, i), _internal_size(pa2, i)), num_dims)
    result = if merge_size == _internal_size(pa2)
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
    return result
end

struct VarNamedTuple{Names,Values}
    data::NamedTuple{Names,Values}
end

# TODO(mhauru) Since I define this, should I also define `isequal` and `hash`? Same for
# PartialArrays.
Base.:(==)(vnt1::VarNamedTuple, vnt2::VarNamedTuple) = vnt1.data == vnt2.data

Base.merge(x1::VarNamedTuple, x2::VarNamedTuple) = _merge_recursive(x1, x2)

function make_leaf(value, ::PropertyLens{S}) where {S}
    return VarNamedTuple(NamedTuple{(S,)}((value,)))
end
make_leaf(value, ::typeof(identity)) = value
function make_leaf(value, optic::ComposedFunction)
    sub = make_leaf(value, optic.outer)
    return make_leaf(sub, optic.inner)
end

function make_leaf(value, optic::IndexLens{T}) where {T}
    inds = optic.indices
    num_inds = length(inds)
    # Check if any of the indices are ranges or colons. If yes, value needs to be an
    # AbstractArray. Otherwise it needs to be an individual value.
    et = _is_multiindex(optic.indices) ? eltype(value) : typeof(value)
    iarr = PartialArray{et,num_inds}()
    return setindex!!(iarr, value, optic)
end

VarNamedTuple() = VarNamedTuple((;))

function Base.show(io::IO, vnt::VarNamedTuple)
    print(io, "(")
    for (i, (name, value)) in enumerate(pairs(vnt.data))
        if i > 1
            print(io, ", ")
        end
        print(io, name, " -> ")
        print(io, value)
    end
    return print(io, ")")
end

_getindex(vnt::VarNamedTuple, name::Symbol) = vnt.data[name]

function varname_to_lens(name::VarName{S}) where {S}
    return _compose_no_identity(getoptic(name), PropertyLens{S}())
end

function _getindex(vnt::VarNamedTuple, name::VarName)
    return _getindex(vnt, varname_to_lens(name))
end
function _getindex(vnt::VarNamedTuple, ::PropertyLens{S}) where {S}
    return _getindex(vnt.data, S)
end

function _haskey(vnt::VarNamedTuple, name::VarName)
    return _haskey(vnt, varname_to_lens(name))
end

_haskey(vnt::VarNamedTuple, ::typeof(identity)) = true

_haskey(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = _haskey(vnt.data, S)
_haskey(::VarNamedTuple, ::IndexLens) = false

function _setindex!!(vnt::VarNamedTuple, value, name::VarName)
    return _setindex!!(vnt, value, varname_to_lens(name))
end

function _setindex!!(vnt::VarNamedTuple, value, ::PropertyLens{S}) where {S}
    # I would like this to just read
    # return VarNamedTuple(_setindex!!(vnt.data, value, S))
    # but that seems to be type unstable. Why? Shouldn't it obviously be the same as the
    # below?
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((value,))))
end

function apply(func, vnt::VarNamedTuple, name::VarName)
    if !haskey(vnt.data, name.name)
        throw(KeyError(repr(name)))
    end
    subdata = _getindex(vnt, name)
    new_subdata = func(subdata)
    return _setindex!!(vnt, new_subdata, name)
end

function Base.map(func, vnt::VarNamedTuple)
    new_data = NamedTuple{keys(vnt.data)}(map(func, values(vnt.data)))
    return VarNamedTuple(new_data)
end

function Base.keys(vnt::VarNamedTuple)
    result = ()
    for sym in keys(vnt.data)
        subdata = vnt.data[sym]
        if subdata isa VarNamedTuple
            subkeys = keys(subdata)
            result = (
                (AbstractPPL.prefix(sk, VarName{sym}()) for sk in subkeys)..., result...
            )
        else
            result = (VarName{sym}(), result...)
        end
        subkeys = keys(vnt.data[sym])
    end
    return result
end

function _haskey(vnt::VarNamedTuple, name::VarName{S,Optic}) where {S,Optic}
    if !haskey(vnt.data, S)
        return false
    end
    subdata = vnt.data[S]
    return if Optic === typeof(identity)
        true
    elseif Optic <: IndexLens
        try
            AbstractPPL.getoptic(name)(subdata)
            true
        catch e
            if e isa BoundsError || e isa KeyError
                false
            else
                rethrow(e)
            end
        end
    else
        haskey(subdata, AbstractPPL.unprefix(name, VarName{S}()))
    end
end

# TODO(mhauru) Check the performance of this, and make it into a generated function if
# necessary.
function _merge_recursive(vnt1::VarNamedTuple, vnt2::VarNamedTuple)
    result_data = vnt1.data
    for k in keys(vnt2.data)
        val = if haskey(result_data, k)
            _merge_recursive(result_data[k], vnt2.data[k])
        else
            vnt2.data[k]
        end
        Accessors.@reset result_data[k] = val
    end
    return VarNamedTuple(result_data)
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
Base.haskey(vnt::VarNamedTuple, key) = _haskey(vnt, key)
Base.getindex(vnt::VarNamedTuple, inds...) = _getindex(vnt, inds...)
BangBang.setindex!!(vnt::VarNamedTuple, value, inds...) = _setindex!!(vnt, value, inds...)
Base.haskey(vnt::PartialArray, key) = _haskey(vnt, key)
Base.getindex(vnt::PartialArray, inds...) = _getindex(vnt, inds...)
BangBang.setindex!!(vnt::PartialArray, value, inds...) = _setindex!!(vnt, value, inds...)

end
