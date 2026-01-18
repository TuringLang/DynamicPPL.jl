# Some utilities for checking what sort of indices we are dealing with.
# The non-generated function implementations of these would be
# _has_colon(::T) where {T<:Tuple} = any(x <: Colon for x in T.parameters)
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
    ArrayLikeBlock{T,I,S}

A wrapper for non-array blocks stored in `PartialArray`s.

When setting a value in a `PartialArray` over a range of indices, if the value being set
is not itself an `AbstractArray`, but has a well-defined size, we wrap it in an
`ArrayLikeBlock`, which records both the value and the indices it was set with.

When getting values from a `PartialArray`, if any of the requested indices correspond to
an `ArrayLikeBlock`, we check that the requested indices match the ones used to set the
value. If they do, we return the underlying block, otherwise we throw an error.
"""
struct ArrayLikeBlock{T,I,N,S}
    block::T
    idxs::I
    kw::N
    index_size::S
end

function Base.show(io::IO, alb::ArrayLikeBlock)
    print(io, "ArrayLikeBlock(")
    show(io, alb.block)
    print(io, ", ")
    show(io, alb.idxs)
    print(io, ", ")
    show(io, alb.kw)
    print(io, ", ")
    show(io, alb.index_size)
    print(io, ")")
    return nothing
end

_blocktype(::Type{ArrayLikeBlock{T}}) where {T} = T

"""
    PartialArray{D,M}

An array-like like structure that may only have some of its elements defined.

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
struct PartialArray{
    ElType,num_dims,D<:AbstractArray{ElType,num_dims},M<:AbstractArray{Bool,num_dims}
}
    data::D
    mask::M

    """
        PartialArray(data::AbstractArray, mask::AbstractArray{Bool})

    Create a new `PartialArray` with the given `data` and `mask` arrays. Note that this
    constructor does not copy its arguments, so you should make sure that you do not
    alias them elsewhere!
    """
    function PartialArray(
        data::AbstractArray{ElType,N}, mask::AbstractArray{Bool,N}
    ) where {ElType,N}
        if size(data) != size(mask)
            throw(ArgumentError("Data and mask arrays must have the same size"))
        end
        return new{ElType,N,typeof(data),typeof(mask)}(data, mask)
    end
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

Base.size(pa::PartialArray) = size(pa.data)
Base.isassigned(pa::PartialArray, ix...; kw...) = isassigned(pa.data, ix...; kw...)

# Even though a PartialArray may have its own size, we still allow it to be used as an
# ArrayLikeBlock. This enables setting values for keys like @varname(x[1:3][1]), which will
# be stored as a PartialArray wrapped in an ArrayLikeBlock, stored in another PartialArray.
# Note that this bypasses _any_ size checks, so that e.g. @varname(x[1:3][1,15]) is also a
# valid key.
# TODO(penelopeysm) check if this is still needed.
vnt_size(pa::PartialArray) = size(pa)

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
    return size(pa1) == size(pa2) &&
           pa1.mask == pa2.mask &&
           all(pa1.data[pa1.mask] .== pa2.data[pa2.mask])
end

function Base.isequal(pa1::PartialArray, pa2::PartialArray)
    return size(pa1) == size(pa2) &&
           pa1.mask == pa2.mask &&
           all(isequal.(pa1.data[pa1.mask], pa2.data[pa2.mask]))
end

function Base.hash(pa::PartialArray, h::UInt)
    h = hash(typeof(pa.data), h)
    for i in eachindex(pa.mask)
        @inbounds if pa.mask[i]
            h = hash(i, h)
            h = hash(pa.data[i], h)
        end
    end
    return h
end

Base.isempty(pa::PartialArray) = !any(pa.mask)
function Base.empty(pa::PartialArray)
    return PartialArray(similar(pa.data), fill!(false, similar(pa.mask)))
end
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
    new_data = similar(pa.data, new_et)
    @inbounds for i in eachindex(pa.mask)
        if pa.mask[i]
            new_data[i] = pa.data[i]
        end
    end
    return PartialArray(new_data, pa.mask)
end

"""
    Base.getindex(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)

Obtain the value at the given indices from the `PartialArray`. This needs to be smarter than
just calling Base.getindex on the internal data array, because we need to check if the
requested indices correspond to an ArrayLikeBlock.
"""
function Base.getindex(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    # The unmodified inds is needed later for ArrayLikeBlock checks.
    if !(checkbounds(Bool, pa.mask, inds...; kw...) && all(getindex(pa.mask, inds...)))
        throw(BoundsError(pa, (inds..., kw)))
    end
    val = getindex(pa.data, inds...; kw...)

    # If not for ArrayLikeBlocks, at this point we could just return val directly. However,
    # we need to check if val contains any ArrayLikeBlocks, and if so, make sure that that
    # we are retrieving exactly that block and nothing else.

    # The error we'll throw if the retrieval is invalid.
    err = ArgumentError("""
        A non-Array value set with a range of indices must be retrieved with the same
        range of indices.
        """)
    if val isa Base.RefValue{<:ArrayLikeBlock}
        # Tried to get a single value, but it's an ArrayLikeBlock.
        throw(err)
    elseif val isa Array && (
        eltype(val) <: Base.RefValue{<:ArrayLikeBlock} ||
        Base.RefValue{<:ArrayLikeBlock} <: eltype(val)
    )
        # Tried to get a range of values, and at least some of them may be ArrayLikeBlocks.
        # The below isempty check is deliberately kept separate from the outer elseif,
        # because the outer one can be resolved at compile time.
        if isempty(val)
            # We need to return an empty array, but for type stability, we want to unwrap
            # any ArrayLikeBlock types in the element type.
            return if eltype(val) <: Base.RefValue{<:ArrayLikeBlock}
                Array{_blocktype(eltype(val)),ndims(val)}()
            else
                val
            end
        end
        # check that all elements that we're trying to access are a Ref to the same
        # ArrayLikeBlock.
        first_elem = first(val)
        if !(first_elem isa Base.RefValue{<:ArrayLikeBlock}) ||
            any(v -> v !== first_elem, val)
            throw(err)
        end
        # check that there are no other elements that also refer to the same ArrayLikeBlock.
        if size(val) != first_elem[].index_size
            throw(err)
        end
        # If _setindex!! works correctly, we should only be able to reach this point if all
        # the elements in `val` are identical to first_elem. Thus we just return that one.
        return first(val)[].block
    else
        return val
    end
end

function Base.haskey(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    hasall =
        checkbounds(Bool, pa.mask, inds...; kw...) && all(getindex(pa.mask, inds...; kw...))

    # If not for ArrayLikeBlocks, we could just return hasall directly. However, we need to
    # check that if any ArrayLikeBlocks are included, they are fully included.
    et = eltype(pa)
    if !(et <: Base.RefValue{<:ArrayLikeBlock} || Base.RefValue{<:ArrayLikeBlock} <: et)
        # pa can't possibly hold any ArrayLikeBlocks, so nothing to do.
        return hasall
    end

    if !hasall
        return false
    end
    # From this point on we can assume that all the requested elements are set, and the only
    # thing to check is that we are not partially indexing into any ArrayLikeBlocks.
    subview = view(pa.data, inds...; kw...)
    return if prod(size(subview)) == 1
        # Only one index being accessed
        !(subview[] isa Base.RefValue{<:ArrayLikeBlock})
    else
        # Multiple indices being accessed
        all_elems_are_equal = all(v -> v === first(subview), subview)
        idx_size_matches = size(subview) == first(subview)[].index_size
        all_elems_are_equal && idx_size_matches
    end
end

function BangBang.delete!!(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    fill!(view(pa.mask, inds...; kw...), false)
    return pa
end

"""
    _remove_partial_blocks!!(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)

Remove any ArrayLikeBlocks that overlap with the given indices from the PartialArray.

Note that this removes the whole block, even the parts that are within `inds`, to avoid
partially indexing into ArrayLikeBlocks.
"""
function _remove_partial_blocks!!(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    et = eltype(pa)
    if !(et <: ArrayLikeBlock || ArrayLikeBlock <: et)
        # pa can't possibly hold any ArrayLikeBlocks, so nothing to do.
        return pa
    end

    # Generate two views, which ensures that the indices will line up
    dataview = view(pa.data, inds...; kw...)
    maskview = view(pa.mask, inds...; kw...)
    for i in eachindex(dataview)
        if maskview[i] && (dataview[i] isa Base.RefValue{<:ArrayLikeBlock})
            val = dataview[i][]
            pa = BangBang.delete!!(pa, val.idxs...; val.kw...)
        end
    end
    return pa
end

function _is_multiindex(f::AbstractArray, ix...; kw...)
    return ndims(view(f, ix...; kw...)) > 0
end
function _is_multiindex(f::PartialArray, ix...; kw...)
    return ndims(view(f.data, ix...; kw...)) > 0
end

"""
    _needs_arraylikeblock(pa::PartialArray, value, inds::Vararg{INDEX_TYPES}; kw...)

Check if the given value needs to be wrapped in an `ArrayLikeBlock` when being set at the
indices `inds` in the `PartialArray` `pa`.

The value only depends on the types of the arguments, and should be constant propagated.
"""
function _needs_arraylikeblock(pa::PartialArray, value, inds::Vararg{INDEX_TYPES}; kw...)
    return _is_multiindex(pa.data, inds...; kw...) &&
           !isa(value, AbstractArray) &&
           !isa(value, PartialArray) &&
           hasmethod(vnt_size, Tuple{typeof(value)})
end

function BangBang.setindex!!(pa::PartialArray, value, inds::Vararg{INDEX_TYPES}; kw...)
    # Delete any overlapping ArrayLikeBlocks first
    pa = _remove_partial_blocks!!(pa, inds...; kw...)

    new_data = pa.data
    new_mask = pa.mask
    if _needs_arraylikeblock(pa, value, inds...; kw...)
        # Check that we're trying to set a block that has the right size.
        idx_sz = size(@view pa.data[inds..., kw...])
        vnt_sz = vnt_size(value)
        if !(vnt_sz isa SkipSizeCheck) && vnt_sz != idx_sz
            throw(
                DimensionMismatch(
                    "Assigned value has size $(vnt_sz), which does not match " *
                    "the size implied by the indices $(idx_sz).",
                ),
            )
        end

        alb_ref = Ref(ArrayLikeBlock(value, inds, NamedTuple(kw), idx_sz))
        new_data = setindex!!(new_data, fill(alb_ref, idx_sz...), inds...; kw...)
        fill!(view(new_mask, inds...; kw...), true)
    else
        if value isa PartialArray
            if _is_multiindex(pa.data, inds...; kw...)
                # Overwriting multiple parts of a PA with data from another PA.
                new_data = setindex!!(new_data, value.data, inds...; kw...)
                new_mask = setindex!(
                    new_mask, getindex(value.mask, inds...; kw...), inds...; kw...
                )
            else
                # Overwriting one element of a PA with another PA. The PA is the value
                # itself!
                new_data = setindex!!(new_data, value, inds...; kw...)
                setindex!(new_mask, true, inds...; kw...)
            end
        else
            new_data = setindex!!(new_data, value, inds...; kw...)
            fill!(view(new_mask, inds...; kw...), true)
        end
    end

    return _concretise_eltype!!(PartialArray(new_data, new_mask))
end

function _subset_partialarray(pa::PartialArray, inds::Vararg{INDEX_TYPES}; kw...)
    new_data = view(pa.data, inds...; kw...)
    new_mask = view(pa.mask, inds...; kw...)
    return PartialArray(new_data, new_mask)
end

Base.merge(x1::PartialArray, x2::PartialArray) = _merge_recursive(x1, x2)

function _merge_element_recursive(x1::PartialArray, x2::PartialArray, ind)
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

function _merge_recursive(pa1::PartialArray, pa2::PartialArray)
    if size(pa1.data) != size(pa2.data)
        throw(ArgumentError("Cannot merge PartialArrays with different sizes"))
    end
    result = copy(pa2)
    for i in eachindex(pa1.mask)
        if pa1.mask[i]
            new_elem = _merge_element_recursive(pa1, result, i)
            # This is not only a performance optimisation: it also avoids a potential bug
            # where calling setindex!! on `result` would lead to the deletion of overlapping
            # ArrayLikeBlocks in `result`, thereby changing `result.mask` and making
            # subsequent iterations behave wrongly.
            if !isassigned(result.data, i) || new_elem !== result.data[i]
                result = setindex!!(result, new_elem, i)
            end
        end
    end
    return result
end

"""
    as_array(pa::PartialArray)

Return the underlying data of the `PartialArray`.

If the `PartialArray` has any elements that are masked, or if any of the elements are `ArrayLikeBlock`s, this will error.
"""
function as_array(pa::PartialArray)
    if !(all(pa.mask))
        throw(
            ArgumentError(
                "Cannot extract data from PartialArray because some elements are not set."
            ),
        )
    end

    retval = pa.data
    if eltype(retval) <: Base.RefValue{<:ArrayLikeBlock} ||
        Base.RefValue{<:ArrayLikeBlock} <: eltype(retval)
        for ind in eachindex(retval)
            @inbounds if retval[ind] isa Base.RefValue{<:ArrayLikeBlock}
                throw(
                    ArgumentError(
                        "Cannot extract data from PartialArray when some elements " *
                        "are set as ArrayLikeBlocks.",
                    ),
                )
            end
        end
    end
    return retval
end
