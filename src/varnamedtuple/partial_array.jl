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
    _merge(x1, x2, recursive=Val(true))

Merge two values `x1` and `x2`. If `recursive isa Val{true}`, additionally recurses into
elements of `x1` and `x2` to merge those.

Unlike `Base.merge`, this function is defined for all types, and by default returns the
second argument. It is overridden for `PartialArray` and `VarNamedTuple`, since they are
nested containers, and the `recursive` argument allows us to control whether we want to
merge its nested values as well.
"""
_merge(_, x2, _) = x2

"""
    vnt_size(x)

Get the size of an object `x` for use in `VarNamedTuple` and `PartialArray`.

By default, this falls back onto `Base.size`, but can be overloaded for custom types.
This notion of type is used to determine whether a value can be set into a `PartialArray`
as a block, see the docstring of `PartialArray` and `ArrayLikeBlock` for details.
"""
vnt_size(x) = size(x)

"""
    ArrayLikeBlock{T,I,N,S}

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
    ix::I
    kw::N
    index_size::S
end
# When broadcasting (e.g. my_array[1:5] .= array_like_block), we want to treat
# ArrayLikeBlocks as scalars.
Base.broadcastable(o::ArrayLikeBlock) = Ref(o)

_blocktype(::Type{ArrayLikeBlock{T}}) where {T} = T

"""
    GrowableArray{T,N}

A wrapper around an `Array{T,N}`. This represents an array whose shape is not actually
known, but is presumed by the VNT to be at least as large as the largest index used to set
values into it. It is possible to call setindex!! on a GrowableArray with indices that are
out of bounds; in such a case, the GrowableArray will automatically resize itself to
accommodate the new index.
"""
struct GrowableArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
# I don't think this is the full AbstractArray interface, but it's enough for us (probably a
# bit too much even).
Base.size(ga::GrowableArray) = size(ga.data)
Base.axes(ga::GrowableArray) = axes(ga.data)
Base.view(ga::GrowableArray, ix...) = view(ga.data, ix...)
Base.copy(ga::GrowableArray) = GrowableArray(copy(ga.data))
Base.:(==)(ga1::GrowableArray, ga2::GrowableArray) = ga1.data == ga2.data
Base.isequal(ga1::GrowableArray, ga2::GrowableArray) = isequal(ga1.data, ga2.data)
Base.hash(ga::GrowableArray, h::UInt) = hash(GrowableArray, hash(ga.data, h))
Base.collect(ga::GrowableArray) = collect(ga.data)
Base.similar(ga::GrowableArray, ::Type{T}) where {T} = GrowableArray(similar(ga.data, T))
Base.similar(ga::GrowableArray, sz::Tuple) = GrowableArray(similar(ga.data, sz))
# single-element indexing
Base.getindex(ga::GrowableArray, ix::Vararg{Int}) = getindex(ga.data, ix...)
function Base.copyto!(dest::GrowableArray, src::GrowableArray, args...)
    return copyto!(dest.data, src.data, args...)
end
Base.copy!(dest::GrowableArray, src::GrowableArray) = copy!(dest.data, src.data)
Base.getindex(ga::GrowableArray, ix::CartesianIndex) = getindex(ga.data, ix)
# multi-element indexing
Base.getindex(ga::GrowableArray, ix...) = GrowableArray(getindex(ga.data, ix...))
Base.setindex!(ga::GrowableArray, value, ix...) = setindex!(ga.data, value, ix...)
# This function is exported so we can override it!
function AbstractPPL.concretize_top_level(idx::Index, val::GrowableArray)
    if any(ix -> ix isa AbstractPPL.DynamicIndex, idx.ix)
        _warn_growable_array_extraction()
    end
    return concretize_top_level(idx, val.data)
end

function throw_kw_error()
    throw(
        ArgumentError(
            "Attempted to set a value with an keyword index, but no" *
            " template (or an unsuitable template) was provided for the" *
            " array. A proper template is needed to determine the shape" *
            " and type of the array so that the indexed data can be stored" *
            " correctly.",
        ),
    )
end

# Helper functions to determine the largest index from various index types.
largest_index(ix::Integer) = ix
largest_index(r::AbstractUnitRange) = last(r)
largest_index(r::AbstractVector{<:Integer}) = maximum(r)
function largest_index(x)
    throw(
        ArgumentError(
            "Attempted to set a value with an index of $x, but no" *
            " template was provided for the array. For such an index," *
            " a template is needed to determine the shape and type of the" *
            " array so that the indexed data can be stored correctly.",
        ),
    )
end
function get_maximum_size_from_indices(ix...; kw...)
    isempty(kw) || throw_kw_error()
    return tuple(map(largest_index, ix)...)
end

# This determines the required size for setting into the indices ix. Note that ix is not
# splatted! and this function takes no keywords! For example, if ix is (3:5,1), this
# function will return (3,); but get_maximum_size_from_indices will return (5, 1).
@generated function get_required_size_from_indices(ix::Tuple)
    x = Expr(:tuple)
    for (i, ti) in enumerate(ix.parameters)
        if ti <: AbstractVector{<:Integer}
            push!(x.args, :(length(ix[$i])))
        elseif i isa Colon
            error("nope")
        end
    end
    return x
end

"""
    PartialArray{
        ElType,
        num_dims,
        D<:AbstractArray{ElType,num_dims},
        M<:AbstractArray{Bool,num_dims}
    }

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
    h = hash(size(pa.data), h)
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
    return PartialArray(similar(pa.data), fill!(similar(pa.mask), false))
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
    _can_get_arraylikeblock(pa_data::AbstractArray)

Returns a Boolean indicating whether `pa_data` refers to a single `ArrayLikeBlock` that can
be retrieved.
"""
function _can_get_arraylikeblock(pa_data::AbstractArray)
    # Check that all elements that we're trying to access are the same ArrayLikeBlock.
    first_elem = first(pa_data)
    if !(first_elem isa ArrayLikeBlock) || any(v -> v !== first_elem, pa_data)
        return false
    end
    # check that there are no other elements that also refer to the same ArrayLikeBlock.
    if size(pa_data) != first_elem.index_size
        return false
    end
    # If _setindex!! works correctly, we should only be able to reach this point if all
    # the elements in `val` are identical to first_elem. In this case we can return true.
    return true
end

"""
    Base.getindex(pa::PartialArray, inds::Vararg{Any}; kw...)

Obtain the value at the given indices from the `PartialArray`. This needs to be smarter than
just calling Base.getindex on the internal data array, because we need to check if the
requested indices correspond to an ArrayLikeBlock.
"""
function Base.getindex(pa::PartialArray, inds::Vararg{Any}; kw...)
    # Check the mask first. We defer bounds checking to the sub-arrays.
    if !(all(getindex(pa.mask, inds...; kw...)))
        throw(BoundsError(pa, (inds..., kw)))
    end
    val = getindex(pa.data, inds...; kw...)

    # If not for ArrayLikeBlocks, at this point we could just return val directly. However,
    # we need to check if val contains any ArrayLikeBlocks, and if so, make sure that that
    # we are retrieving exactly that block and nothing else.
    is_multiindex = _is_multiindex(pa.data, inds...; kw...)

    # The error we'll throw if the retrieval is invalid.
    alb_err = ArgumentError("""
        A non-Array value set with a range of indices must be retrieved with the same
        range of indices.
        """)
    if val isa ArrayLikeBlock
        # Error if we try to get a single value, but it's an ArrayLikeBlock.
        throw(alb_err)
    elseif is_multiindex &&
        val isa AbstractArray &&
        (eltype(val) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(val))
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
        if _can_get_arraylikeblock(val)
            return first(val).block
        else
            throw(alb_err)
        end
    elseif val isa GrowableArray
        # This code path is hit for things like `vnt[@varname(x[:])]` where `x` is a PA that
        # stores a GrowableArray. We warn the user that the result may be wrong.
        # TODO(penelopeysm): The DynamicIndex check actually doesn't work here because if
        # you index with `x[end]` for example, `inds` is already concretised outside of this
        # function.
        if any(ind -> ind isa AbstractPPL.DynamicIndex || ind isa Colon, inds)
            _warn_growable_array_extraction()
        end
        return unwrap_internal_array(val)
    else
        return val
    end
end

function Base.haskey(pa::PartialArray, inds::Vararg{Any}; kw...)
    hasall =
        checkbounds(Bool, pa.mask, inds...; kw...) && all(view(pa.mask, inds...; kw...))

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
    subview = view(pa.data, inds...; kw...)
    return if ndims(subview) == 0
        # Only one index being accessed -- just check that it isn't an ArrayLikeBlock.
        isassigned(subview) && !(subview[] isa ArrayLikeBlock)
    else
        # Multiple indices being accessed. We need to check that we are accessing an
        # ArrayLikeBlock in its entirety.
        _can_get_arraylikeblock(subview)
    end
end

function BangBang.delete!!(pa::PartialArray, inds::Vararg{Any}; kw...)
    fill!(view(pa.mask, inds...; kw...), false)
    return pa
end

"""
    _remove_partial_blocks!!(pa::PartialArray, inds::Vararg{Any}; kw...)

Remove any ArrayLikeBlocks that overlap with the given indices from the PartialArray.

Note that this removes the whole block, even the parts that are within `inds`, to avoid
partially indexing into ArrayLikeBlocks.
"""
function _remove_partial_blocks!!(
    pa_data::AbstractArray, pa_mask::AbstractArray{Bool}, inds::Vararg{Any}; kw...
)
    et = eltype(pa_data)
    if !(et <: ArrayLikeBlock || ArrayLikeBlock <: et)
        # pa can't possibly hold any ArrayLikeBlocks, so nothing to do.
        return pa_data, pa_mask
    end

    # Generate two views, which ensures that the indices will line up
    dataview = view(pa_data, inds...; kw...)
    maskview = view(pa_mask, inds...; kw...)
    for i in eachindex(dataview)
        if maskview[i] && (dataview[i] isa ArrayLikeBlock)
            val = dataview[i]
            fill!(view(pa_mask, val.ix...; val.kw...), false)
        end
    end
    return pa_data, pa_mask
end

function _is_multiindex(f::AbstractArray, ix...; kw...)
    return ndims(view(f, ix...; kw...)) > 0
end
function _is_multiindex(f::PartialArray, ix...; kw...)
    return ndims(view(f.data, ix...; kw...)) > 0
end

"""
    _needs_arraylikeblock(pa_data::AbstractArray, value, inds::Vararg{Any}; kw...)

Check if the given value needs to be wrapped in an `ArrayLikeBlock` when being set at the
indices `inds` in the `PartialArray` with data array `pa_data`.

The value only depends on the types of the arguments, and should be constant propagated.
"""
function _needs_arraylikeblock(pa_data::AbstractArray, value, inds::Vararg{Any}; kw...)
    return !isa(value, AbstractArray) &&
           !isa(value, PartialArray) &&
           hasmethod(vnt_size, Tuple{typeof(value)}) &&
           _is_multiindex(pa_data, inds...; kw...)
end

function grow_to_indices!!(
    pa::PartialArray{T,ndims,A}, inds::Vararg{Any,ndims}; kw...
) where {T,ndims,A<:GrowableArray}
    # pa.data = GrowableArray{T}
    # pa.mask = GrowableArray{Bool}
    required_size = get_maximum_size_from_indices(inds...; kw...)
    current_size = size(pa.data)
    return if any(required_size .> current_size)
        old_data = pa.data.data
        old_mask = pa.mask.data
        # Need to make a new Array that is big enough to hold the new index. Note that these
        # are just Arrays, not GrowableArrays.
        new_size = map(max, required_size, current_size)
        new_data = similar(old_data, new_size)
        new_mask = fill(false, new_size)
        # Copy the old mask into the new array. copyto! works if we are extending only the
        # last dimension of the array, but if we are extending any other dimension, it will
        # copy elements wrongly
        for c in CartesianIndices(old_mask)
            # If the old value was masked, it might be undefined. Don't copy it over.
            if old_mask[c]
                new_mask[c] = old_mask[c]
                new_data[c] = old_data[c]
            end
        end
        PartialArray(GrowableArray(new_data), GrowableArray(new_mask))
    else
        # Can reuse the old one.
        pa
    end
end
function grow_to_indices!!(
    pa::PartialArray{T,ndims,A}, inds::Vararg{Any,ndims2}; kw...
) where {T,ndims,ndims2,A<:GrowableArray}
    throw(
        ArgumentError(
            "Cannot expand a GrowableArray with $ndims dimensions" *
            " using an index with $ndims2 dimensions. GrowableArrays" *
            " are created when no template is provided when setting" *
            " VarNames with indices, e.g. `@varname(x[1])`, which causes" *
            " DynamicPPL to have no knowledge of the shape of `x`. To" *
            " fix this, you should provide a template for `x` when creating" *
            " the VarNamedTuple. Alternatively, if `x` is an array with" *
            " `N` dimensions, you should always index into it with `N`" *
            " indices. This means avoiding using, for example, a mixture of" *
            " linear and Cartesian indexing for arrays with more than one" *
            " dimension.",
        ),
    )
end
grow_to_indices!!(pa::PartialArray, inds::Vararg{Any}; kw...) = pa

function BangBang.setindex!!(pa::PartialArray, value, inds::Vararg{Any}; kw...)
    # If pa.data and pa.mask are GrowableArrays, we may need to resize them before doing
    # anything else. For other AbstractArrays, grow_to_indices is a no-op.
    new_pa = grow_to_indices!!(pa, inds...; kw...)
    new_data, new_mask = new_pa.data, new_pa.mask

    # Then delete any overlapping ArrayLikeBlocks
    new_data, new_mask = _remove_partial_blocks!!(new_data, new_mask, inds...; kw...)

    if _needs_arraylikeblock(new_data, value, inds...; kw...)
        # Check that we're trying to set a block that has the right size.
        idx_sz = size(@view new_data[inds..., kw...])
        vnt_sz = vnt_size(value)
        if vnt_sz != idx_sz
            throw(
                DimensionMismatch(
                    "Assigned value has size $(vnt_sz), which does not match " *
                    "the size implied by the indices $(idx_sz).",
                ),
            )
        end
        alb = ArrayLikeBlock(value, inds, NamedTuple(kw), idx_sz)
        new_data = setindex!!(new_data, fill(alb, idx_sz...), inds...; kw...)
        fill!(view(new_mask, inds...; kw...), true)
    else
        if value isa PartialArray
            if _is_multiindex(new_data, inds...; kw...)
                # This branch occurs if we are overwriting a slice of a PA with data from
                # another PA.
                #
                # A naive implementation of this would be:
                #
                #    new_data = setindex!!(new_data, value.data, inds...; kw...)
                #    setindex!(new_mask, value.mask, inds...; kw...)
                #
                # However, there are two problems with this.
                #
                # 1. We only really want to overwrite the parts of `value` that are
                #    unmasked. Right now we are just copying over everything.
                # 2. If it was just the above, that would be fine because it would just
                #    be a performance loss. The problem is that `value.data` may be
                #    uninitialised in the places where it is masked. Attempting to copy
                #    over that data would lead to an error.
                #
                # So we need to do it manually. :(
                new_eltype = promote_type(eltype(value.data), eltype(new_data))
                new_data = if new_eltype <: eltype(new_data)
                    new_data
                else
                    broadened = similar(new_data, new_eltype)
                    copy!(broadened, new_data)
                    broadened
                end
                new_data_view = view(new_data, inds...; kw...)
                new_mask_view = view(new_mask, inds...; kw...)
                for i in eachindex(new_mask_view)
                    if value.mask[i]
                        # If value.mask[i] is true, we can guarantee that value.data[i] is
                        # initialised to a concrete value
                        new_data_view[i] = value.data[i]
                        new_mask_view[i] = true
                    end
                end
            else
                # Overwriting one element of a PA with another PA. The PA is the value
                # itself! -- i.e. nested PAs! This can happen with things like x[1][1]
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

function _subset_partialarray(pa::PartialArray, inds::Vararg{Any}; kw...)
    if pa.data isa GrowableArray &&
        any(ind -> ind isa AbstractPPL.DynamicIndex || ind isa Colon, inds)
        _warn_growable_array_extraction()
    end
    new_data = view(pa.data, inds...; kw...)
    new_mask = view(pa.mask, inds...; kw...)
    return PartialArray(new_data, new_mask)
end

Base.merge(x1::PartialArray, x2::PartialArray) = _merge(x1, x2, Val(true))
function _merge(pa1::PartialArray, pa2::PartialArray, recurse::Val)
    # If both `pa1` and `pa2` are GrowableArrays, we can grow them before merging
    if pa1.data isa GrowableArray && pa2.data isa GrowableArray && ndims(pa1) == ndims(pa2)
        size1 = size(pa1.data)
        size2 = size(pa2.data)
        new_size = map(max, size1, size2)
        pa1 = grow_to_indices!!(pa1, new_size...)
        pa2 = grow_to_indices!!(pa2, new_size...)
    end

    # TODO(penelopeysm) In general, we should like to catch more cases (e.g. where the
    # underlying Array type is different). However I don't know how to reliably check for
    # that.
    if axes(pa1.data) != axes(pa2.data)
        throw(ArgumentError("Cannot merge PartialArrays with different axes"))
    end
    result = copy(pa2)
    new_data, new_mask = result.data, result.mask
    for i in eachindex(pa1.mask)
        if pa1.mask[i]
            new_elem, new_mask_val = _merge_element(pa1, pa2, i, recurse)
            new_mask[i] = new_mask_val
            if new_mask_val
                new_data = setindex!!(new_data, new_elem, i)
            end
        end
    end
    return _concretise_eltype!!(PartialArray(new_data, new_mask))
end

"""
    _merge_element(x1::PartialArray, x2::PartialArray, ind, ::Val{true})

Performs an elementwise merge of two `PartialArray`s. If both `x1` and `x2` have the element
at `ind` set, this function will attempt to merge the values recursively.

Returns the value to be set at `ind`, plus the value of the mask at `ind`.
"""
function _merge_element(x1::PartialArray, x2::PartialArray, ind, ::Val{true})
    m1 = x1.mask[ind]
    m2 = x2.mask[ind]
    return if m1 && m2
        _merge(x1.data[ind], x2.data[ind], Val(true)), true
    else
        _merge_element(x1, x2, ind, Val(false))
    end
end

"""
    _merge_element(x1::PartialArray, x2::PartialArray, ind, ::Val{false})

Performs an elementwise merge of two `PartialArray`s, but does not attempt to recurse into
nested values to merge those as well. In particular, if both `x1` and `x2` have the element
at `ind` set, the value from `x2` is taken directly.

Returns the value to be set at `ind`, plus the value of the mask at `ind`.
"""
function _merge_element(x1::PartialArray, x2::PartialArray, ind, ::Val{false})
    m1 = x1.mask[ind]
    m2 = x2.mask[ind]
    return if m1 && !m2
        # This is the only potential case where we need to return something in x1. However,
        # there is one additional check: we only want to copy ALBs over if all the target
        # indices are completely 'free' in pa2. If pa2 has any overlapping indices set, we
        # shouldn't copy the ALB over.
        #
        # It's safe to check x1.data[ind] here because the mask was set, so it can't be
        # an uninitialized value.
        d1 = x1.data[ind]
        if d1 isa ArrayLikeBlock
            if any(view(x2.mask, d1.ix...; d1.kw...))
                # There is some overlap. Return d1 as the data (because x2.data[ind] might
                # be uninitialized data!), but indicate that the mask should be false (which
                # also causes setindex!! to be skipped).
                d1, false
            else
                # No overlap -- safe to copy. Since all the target indices are free, this
                # means that we will naturally copy over all the bits and pieces of the ALB
                # eventually.
                d1, true
            end
        else
            # Just some other scalar value that we can copy over.
            d1, true
        end
    else
        x2.data[ind], true
    end
end

"""
    unwrap_internal_array(x)

`x` here is some data obtained by indexing with a slice into an AbstractArray or
PartialArray stored in a VNT. (It might be an AbstractArray if for example the VarName `x`
is just set to an array, rather than its individual elements being set one by one.)

This function unwraps any PartialArray or GrowableArray wrappers, returning the underlying
Base.Array. It errors if any elements are not set, or if any elements are still
ArrayLikeBlocks.

Sometimes, indexing with a slice into a PartialArray returns an unwrapped ArrayLikeBlock.
In that case, this function does nothing.
"""
unwrap_internal_array(x) = x
function unwrap_internal_array(pa::PartialArray)
    if !(all(pa.mask))
        throw(
            ArgumentError(
                "Cannot extract data from PartialArray because some elements are not set."
            ),
        )
    end

    retval = pa.data
    if retval isa GrowableArray
        _warn_growable_array_extraction()
    end
    if eltype(retval) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(retval)
        for ind in eachindex(retval)
            @inbounds if retval[ind] isa ArrayLikeBlock
                throw(
                    ArgumentError(
                        "Cannot extract data from PartialArray when some elements " *
                        "are set as ArrayLikeBlocks.",
                    ),
                )
            end
        end
    end
    return unwrap_internal_array(retval)
end
unwrap_internal_array(ga::GrowableArray) = ga.data

function _warn_growable_array_extraction()
    @warn (
        "Returning a `Base.Array` with a presumed size based on the indices" *
        " used to set values; but this may not be the actual type or size" *
        " of the actual `AbstractArray` that was inside the DynamicPPL model." *
        " You should inspect the returned result to make sure that it has the" *
        " correct value.\n\n" *
        "To find out how to avoid this warning, please see: " *
        "https://turinglang.org/docs/uri/growablearray"
    )
end
function _warn_growable_array_creation(size)
    @warn (
        "Creating a growable `Base.Array` of dimension $(length(size)) to store" *
        " values. This may not match the actual type or size of the actual" *
        " `AbstractArray` that will be used inside the DynamicPPL model.\n\n" *
        " If this is not the type or size that you expect, please see:" *
        " https://turinglang.org/docs/uri/growablearray"
    )
end
