# Some utilities for checking what sort of indices we are dealing with.
# The non-generated function implementations of these would be
# _has_colon(::T) where {T<:Tuple} = any(x <: Colon for x in T.parameters)
# function _is_multiindex(::T) where {T<:Tuple}
#     return any(x <: AbstractUnitRange || x <: Colon for x in T.parameters)
# end
# However, constant propagation sometimes fails if the index tuple is too big (e.g. length
# 4), so we play it safe and use generated functions. Constant propagating these is
# important, because many functions choose different paths based on their values, which
# would lead to type instability if they were only evaluated at runtime.
@generated function _has_colon_or_dynamicindex(::T) where {T<:Tuple}
    for x in T.parameters
        if x <: Colon || x <: AbstractPPL.DynamicIndex
            return :(return true)
        end
    end
    return :(return false)
end
@generated function _is_multiindex(::T) where {T<:Tuple}
    for x in T.parameters
        if x <: AbstractUnitRange || x <: Colon
            return :(return true)
        end
    end
    return :(return false)
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
`_merge_recursive(x, y) === y`.
"""
_merge_recursive(_, x2) = x2

"""The factor by which we increase the dimensions of PartialArrays when resizing them."""
const PARTIAL_ARRAY_DIM_GROWTH_FACTOR = 4

"""A convenience for defining method argument type bounds."""
const INDEX_TYPES = Union{Integer,AbstractUnitRange,Colon,AbstractPPL.DynamicIndex}

"""
    SkipSizeCheck()

A special return value for `vnt_size` indicating that size checks should be skipped.
"""
struct SkipSizeCheck end

"""
    vnt_size(x)

Get the size of an object `x` for use in `VarNamedTuple` and `PartialArray`.

By default, this falls back onto `Base.size`, but can be overloaded for custom types.
This notion of type is used to determine whether a value can be set into a `PartialArray`
as a block, see the docstring of `PartialArray` and `ArrayLikeBlock` for details.

A special return value of `SkipSizeCheck()` indicates that the size check should be skipped.
"""
vnt_size(x) = size(x)

"""
    ArrayLikeBlock{T,I}

A wrapper for non-array blocks stored in `PartialArray`s.

When setting a value in a `PartialArray` over a range of indices, if the value being set
is not itself an `AbstractArray`, but has a well-defined size, we wrap it in an
`ArrayLikeBlock`, which records both the value and the indices it was set with.

When getting values from a `PartialArray`, if any of the requested indices correspond to
an `ArrayLikeBlock`, we check that the requested indices match the ones used to set the
value. If they do, we return the underlying block, otherwise we throw an error.
"""
struct ArrayLikeBlock{T,I}
    block::T
    inds::I

    function ArrayLikeBlock(block::T, inds::I) where {T,I}
        if !_is_multiindex(inds)
            throw(ArgumentError("ArrayLikeBlock must be constructed with a multi-index"))
        end
        return new{T,I}(block, inds)
    end
end

function Base.show(io::IO, alb::ArrayLikeBlock)
    # Note the distinction: The raw strings that form part of the structure of the print
    # out are `print`ed, whereas the keys and values are `show`n. The latter ensures
    # that strings are quoted, Symbols are prefixed with :, etc.
    print(io, "ArrayLikeBlock(")
    show(io, alb.block)
    print(io, ", ")
    show(io, alb.inds)
    print(io, ")")
    return nothing
end

_blocktype(::Type{ArrayLikeBlock{T}}) where {T} = T

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

One can set values in a `PartialArray` either element-by-element, or with ranges like
`arr[1:3,2] = [5,10,15]`. When setting values over a range of indices, the value being set
must either be an `AbstractArray` or otherwise something for which `vnt_size(value)` or
`Base.size(value)` (which `vnt_size` falls back onto) is defined, and the size matches the
range. If the value is an `AbstractArray`, the elements are copied individually, but if it
is not, the value is stored as a block, that takes up the whole range, e.g. `[1:3,2]`, but
is only a single object. Getting such a block-value must be done with the exact same range
of indices, otherwise an error is thrown.

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
    # TODO(mhauru) Consider trying FixedSizeArrays instead, see how it would change
    # performance. We reallocate new Arrays every time when resizing anyway, except for
    # Vectors, which can be extended in place.
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
        pa = setindex!!(pa, convert(ElType, value), inds...)
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

# Even though a PartialArray has no well-defined size, we still allow it to be used as an
# ArrayLikeBlock. This enables setting values for keys like @varname(x[1:3][1]), which will
# be stored as a PartialArray wrapped in an ArrayLikeBlock, stored in another PartialArray.
# Note that this bypasses _any_ size checks, so that e.g. @varname(x[1:3][1,15]) is also a
# valid key.
vnt_size(::PartialArray) = SkipSizeCheck()

function Base.copy(pa::PartialArray)
    # Make a shallow copy of pa, except for any VarNamedTuple elements, which we recursively
    # copy.
    pa_copy = PartialArray(copy(pa.data), copy(pa.mask))
    et = eltype(pa)
    if (
        VarNamedTuple <: et ||
        et <: VarNamedTuple ||
        PartialArray <: et ||
        et <: PartialArray
    )
        @inbounds for i in eachindex(pa.mask)
            if pa.mask[i]
                val = @inbounds pa_copy.data[i]
                if val isa VarNamedTuple || val isa PartialArray
                    pa_copy.data[i] = copy(val)
                end
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
    result = true
    for i in CartesianIndices(merge_size)
        m1 = checkbounds(Bool, pa1.mask, Tuple(i)...) ? pa1.mask[i] : false
        m2 = checkbounds(Bool, pa2.mask, Tuple(i)...) ? pa2.mask[i] : false
        if m1 != m2
            return false
        end
        if m1
            elements_equal = pa1.data[i] == pa2.data[i]
            if elements_equal === false
                return false
            elseif elements_equal === missing
                # This branch can't short-circuit and just return missing, because some
                # later values may be straight-up not equal.
                result = missing
            end
        end
    end
    return result
end

# Exactly as == above, except the comparison of the data elements uses isequal.
function Base.isequal(pa1::PartialArray, pa2::PartialArray)
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
        if m1 && !isequal(pa1.data[i], pa2.data[i])
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

Base.isempty(pa::PartialArray) = !any(pa.mask)
Base.empty(pa::PartialArray) = PartialArray(similar(pa.data), fill(false, size(pa.mask)))
function BangBang.empty!!(pa::PartialArray)
    fill!(pa.mask, false)
    return pa
end

# This is a tad hacky: We use _mapreduce_recursive which requires a prefix VarName. We give
# it the non-sense @varname(_), and then strip it away with the mapping function, returning
# only the optic.
function Base.keys(pa::PartialArray)
    return _mapreduce_recursive(pair -> first(pair).optic, push!, pa, @varname(_), Any[])
end

# Length could be defined as a special case of mapreduce, but it's harder to keep it type
# stable that way: If the element type is abstract, we end up calling _mapreduce_recursive
# on an abstract type, which makes the type of the cumulant Any.
function Base.length(pa::PartialArray)
    len = 0
    @inbounds for i in eachindex(pa.mask)
        if !pa.mask[i]
            continue
        end
        val = pa.data[i]
        len += val isa VarNamedTuple || val isa PartialArray ? length(val) : 1
    end
    return len
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
    # We could use promote_type here, instead of typejoin. However, that would e.g.
    # cause Ints to be converted to Float64s, since
    # promote_type(Int, Float64) == Float64, which can cause problems. See
    # https://github.com/TuringLang/DynamicPPL.jl/pull/1098#discussion_r2472636188.
    # Base.promote_typejoin would be like typejoin, but creates Unions out of Nothing
    # and Missing, rather than falling back on Any. However, it's not exported.
    new_et = typejoin((typeof(pa.data[i]) for i in eachindex(pa.mask) if pa.mask[i])...)
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
_length_needed(r::AbstractUnitRange) = last(r)

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

"""Throw an appropriate error if the given indices are invalid for `pa`."""
function _check_index_validity(pa::PartialArray, inds::NTuple{N,INDEX_TYPES}) where {N}
    if length(inds) != ndims(pa)
        throw(BoundsError(pa, inds))
    end
    if _has_colon_or_dynamicindex(inds)
        msg = """
            Indexing PartialArrays with Colon or AbstractPPL.DynamicIndex is not supported.
            You may need to concretise the `VarName` first."""
        throw(ArgumentError(msg))
    end
    return nothing
end

"""
    Base.getindex(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)

Obtain the value at the given indices from the `PartialArray`. This needs to be smarter than
just calling Base.getindex on the internal data array, because we need to check if the
requested indices correspond to an ArrayLikeBlock.
"""
function Base.getindex(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    isempty(kw) || error_kw_indices()
    # The unmodified inds is needed later for ArrayLikeBlock checks.
    orig_inds = inds
    _check_index_validity(pa, inds)
    if !(checkbounds(Bool, pa.mask, inds...) && all(@inbounds(getindex(pa.mask, inds...))))
        throw(BoundsError(pa, inds))
    end
    val = getindex(pa.data, inds...)

    # If not for ArrayLikeBlocks, at this point we could just return val directly. However,
    # we need to check if val contains any ArrayLikeBlocks, and if so, make sure that that
    # we are retrieving exactly that block and nothing else.

    # The error we'll throw if the retrieval is invalid.
    err = ArgumentError("""
        A non-Array value set with a range of indices must be retrieved with the same
        range of indices.
        """)
    if val isa ArrayLikeBlock
        # Tried to get a single value, but it's an ArrayLikeBlock.
        throw(err)
    elseif val isa Array && (eltype(val) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(val))
        # Tried to get a range of values, and at least some of them may be ArrayLikeBlocks.
        # The below isempty check is deliberately kept separate from the outer elseif,
        # because the outer one can be resolved at compile time.
        if isempty(val)
            # We need to return an empty array, but for type stability, we want to unwrap
            # any ArrayLikeBlock types in the element type.
            return if eltype(val) <: ArrayLikeBlock
                Array{_blocktype(eltype(val)),ndims(val)}()
            else
                val
            end
        end
        first_elem = first(val)
        if !(first_elem isa ArrayLikeBlock)
            throw(err)
        end
        if orig_inds != first_elem.inds
            # The requested indices do not match the ones used to set the value.
            throw(err)
        end
        # If _setindex!! works correctly, we should only be able to reach this point if all
        # the elements in `val` are identical to first_elem. Thus we just return that one.
        return first(val).block
    else
        return val
    end
end

function Base.haskey(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    isempty(kw) || error_kw_indices()
    _check_index_validity(pa, inds)
    hasall =
        checkbounds(Bool, pa.mask, inds...) && all(@inbounds(getindex(pa.mask, inds...)))

    # If not for ArrayLikeBlocks, we could just return hasall directly. However, we need to
    # check that if any ArrayLikeBlocks are included, they are fully included.
    et = eltype(pa)
    if !(et <: ArrayLikeBlock || ArrayLikeBlock <: et)
        # pa can't possibly hold any ArrayLikeBlocks, so nothing to do.
        return hasall
    end

    if !hasall
        return false
    end
    # From this point on we can assume that all the requested elements are set, and the only
    # thing to check is that we are not partially indexing into any ArrayLikeBlocks.
    # We've already checked checkbounds at the top of the function, and returned if it
    # wasn't true, so @inbounds is safe.
    subdata = @inbounds getindex(pa.data, inds...)
    if !_is_multiindex(inds)
        return !(subdata isa ArrayLikeBlock)
    end
    return !any(elem -> elem isa ArrayLikeBlock && elem.inds != inds, subdata)
end

function BangBang.delete!!(pa::PartialArray, inds::Vararg{INDEX_TYPES})
    _check_index_validity(pa, inds)
    if _is_multiindex(inds)
        pa.mask[inds...] .= false
    else
        pa.mask[inds...] = false
    end
    return pa
end

_ensure_range(r::AbstractUnitRange) = r
_ensure_range(i::Integer) = i:i

"""
    _remove_partial_blocks!!(pa::PartialArray, inds::Vararg{INDEX_TYPES})

Remove any ArrayLikeBlocks that overlap with the given indices from the PartialArray.

Note that this removes the whole block, even the parts that are within `inds`, to avoid
partially indexing into ArrayLikeBlocks.
"""
function _remove_partial_blocks!!(pa::PartialArray, inds::Vararg{INDEX_TYPES})
    et = eltype(pa)
    if !(et <: ArrayLikeBlock || ArrayLikeBlock <: et)
        # pa can't possibly hold any ArrayLikeBlocks, so nothing to do.
        return pa
    end

    for i in CartesianIndices(map(_ensure_range, inds))
        if pa.mask[i]
            val = @inbounds pa.data[i]
            if val isa ArrayLikeBlock
                pa = delete!!(pa, val.inds...)
            end
        end
    end
    return pa
end

"""
    _needs_arraylikeblock(value, inds::Vararg{INDEX_TYPES})

Check if the given value needs to be wrapped in an `ArrayLikeBlock` when being set at inds.

The value only depends on the types of the arguments, and should be constant propagated.
"""
function _needs_arraylikeblock(value, inds::Vararg{INDEX_TYPES})
    return _is_multiindex(inds) &&
           !isa(value, AbstractArray) &&
           hasmethod(vnt_size, Tuple{typeof(value)})
end

function BangBang.setindex!!(pa::PartialArray, value, inds::Vararg{INDEX_TYPES}; kw...)
    isempty(kw) || error_kw_indices()
    orig_inds = inds
    _check_index_validity(pa, inds)
    pa = if checkbounds(Bool, pa.mask, inds...)
        pa
    else
        _resize_partialarray!!(pa, inds)
    end
    pa = _remove_partial_blocks!!(pa, inds...)

    new_data = pa.data
    if _needs_arraylikeblock(value, inds...)
        inds_size = reduce((x, y) -> tuple(x..., y...), map(size, inds))
        if !(vnt_size(value) isa SkipSizeCheck) && vnt_size(value) != inds_size
            throw(
                DimensionMismatch(
                    "Assigned value has size $(vnt_size(value)), which does not match " *
                    "the size implied by the indices $(map(x -> _length_needed(x), inds)).",
                ),
            )
        end
        # At this point we know we have a value that is not an AbstractArray, but it has
        # some notion of size, and that size matches the indices that are being set. In this
        # case we wrap the value in an ArrayLikeBlock, and set all the individual indices
        # to point to that.
        alb = ArrayLikeBlock(value, orig_inds)
        new_data = setindex!!(new_data, fill(alb, inds_size...), inds...)
    else
        new_data = setindex!!(new_data, value, inds...)
    end

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
            # We could use promote_type here, instead of typejoin. However, that would e.g.
            # cause Ints to be converted to Float64s, since
            # promote_type(Int, Float64) == Float64, which can cause problems. See
            # https://github.com/TuringLang/DynamicPPL.jl/pull/1098#discussion_r2472636188.
            # Base.promote_typejoin would be like typejoin, but creates Unions out of
            # Nothing and Missing, rather than falling back on Any. However, it's not
            # exported.
            et = typejoin(eltype(pa1), eltype(pa2))
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

"""
    _dense_array(pa::PartialArray)

Return a `Base.Array` of the elements of the `PartialArray`.

If the `PartialArray` has any missing elements that are within the block of set elements,
this will error. For instance, if `pa` is two-dimensional and (2,2) is set, but one of
(1,1), (1,2), or (2,1) is not.

Likewise, if `pa` includes any blocks set as `ArrayLikeBlocks`, this will error.
"""
function _dense_array(pa::PartialArray)
    # Find the size of the dense array, by checking what are the largest indices set in pa.
    num_dims = ndims(pa)
    size_needed = fill(0, num_dims)
    # TODO(mhauru) This could be optimised by not looping over the whole array: If e.g.
    # (3,3) is set, we have no need to check any indices within the block (3,3).
    for ind in CartesianIndices(pa.mask)
        @inbounds if !pa.mask[ind]
            continue
        end
        for d in 1:num_dims
            size_needed[d] = max(size_needed[d], ind[d])
        end
    end

    # Check that all indices within size_needed are set.
    slice = ntuple(d -> 1:size_needed[d], num_dims)
    if !all(pa.mask[slice...])
        throw(
            ArgumentError(
                "Cannot convert PartialArray to dense Array when some elements within " *
                "the dense block are not set.",
            ),
        )
    end

    retval = pa.data[slice...]
    if eltype(pa) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(pa)
        for ind in CartesianIndices(retval)
            @inbounds if retval[ind] isa ArrayLikeBlock
                throw(
                    ArgumentError(
                        "Cannot convert PartialArray to dense Array when some elements " *
                        "are set as ArrayLikeBlocks.",
                    ),
                )
            end
        end
    end
    return retval
end
