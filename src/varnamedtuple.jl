# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using BangBang
using Accessors
using ..DynamicPPL: _compose_no_identity

export VarNamedTuple

"""The factor by which we increase the dimensions of PartialArrays when resizing them."""
const PARTIAL_ARRAY_DIM_GROWTH_FACTOR = 4

const INDEX_TYPES = Union{Integer,UnitRange,Colon}

_has_colon(::T) where {T<:Tuple} = any(x <: Colon for x in T.parameters)

function _is_multiindex(::T) where {T<:Tuple}
    return any(x <: UnitRange || x <: Colon for x in T.parameters)
end

struct VarNamedTuple{Names,Values}
    data::NamedTuple{Names,Values}
end

# TODO(mhauru) Since I define this, should I also define `isequal` and `hash`? Same for
# PartialArrays.
Base.:(==)(vnt1::VarNamedTuple, vnt2::VarNamedTuple) = vnt1.data == vnt2.data

struct PartialArray{ElType,numdims}
    data::Array{ElType,numdims}
    mask::Array{Bool,numdims}
end

function PartialArray(eltype, num_dims)
    dims = ntuple(_ -> PARTIAL_ARRAY_DIM_GROWTH_FACTOR, num_dims)
    data = Array{eltype,num_dims}(undef, dims)
    mask = fill(false, dims)
    return PartialArray(data, mask)
end

Base.ndims(iarr::PartialArray) = ndims(iarr.data)

# We deliberately don't define Base.size for PartialArray, because it is ill-defined.
# The size of the .data field is an implementation detail.
_internal_size(iarr::PartialArray, args...) = size(iarr.data, args...)

function Base.copy(pa::PartialArray)
    return PartialArray(copy(pa.data), copy(pa.mask))
end

function Base.:(==)(pa1::PartialArray, pa2::PartialArray)
    if ndims(pa1) != ndims(pa2)
        return false
    end
    size1 = _internal_size(pa1)
    size2 = _internal_size(pa2)
    # TODO(mhauru) This could be optimised, but not sure it's worth it.
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

_length_needed(i::Integer) = i
_length_needed(r::UnitRange) = last(r)
_length_needed(::Colon) = 0

"""Take the minimum size that a dimension of a PartialArray needs to be, and return the size
we choose it to be. This size will be the smallest possible power of
PARTIAL_ARRAY_DIM_GROWTH_FACTOR. Growing PartialArrays in big jumps like this helps reduce
data copying, as resizes aren't needed as often.
"""
function _partial_array_dim_size(min_dim)
    factor = PARTIAL_ARRAY_DIM_GROWTH_FACTOR
    return factor^(Int(ceil(log(factor, min_dim))))
end

function _min_size(iarr::PartialArray, inds)
    return ntuple(i -> max(_internal_size(iarr, i), _length_needed(inds[i])), length(inds))
end

function _resize_partialarray(iarr::PartialArray, inds)
    min_size = _min_size(iarr, inds)
    new_size = map(_partial_array_dim_size, min_size)
    # Generic multidimensional Arrays can not be resized, so we need to make a new one.
    # See https://github.com/JuliaLang/julia/issues/37900
    new_data = Array{eltype(iarr.data),ndims(iarr)}(undef, new_size)
    new_mask = fill(false, new_size)
    # Note that we have to use CartesianIndices instead of eachindex, because the latter
    # may use a linear index that does not match between the old and the new arrays.
    for i in CartesianIndices(iarr.data)
        mask_val = iarr.mask[i]
        @inbounds new_mask[i] = mask_val
        if mask_val
            @inbounds new_data[i] = iarr.data[i]
        end
    end
    return PartialArray(new_data, new_mask)
end

# The below implements the same functionality as above, but more performantly for 1D arrays.
function _resize_partialarray(iarr::PartialArray{Eltype,1}, (ind,)) where {Eltype}
    # Resize arrays to accommodate new indices.
    old_size = _internal_size(iarr, 1)
    min_size = max(old_size, _length_needed(ind))
    new_size = _partial_array_dim_size(min_size)
    resize!(iarr.data, new_size)
    resize!(iarr.mask, new_size)
    @inbounds iarr.mask[(old_size + 1):new_size] .= false
    return iarr
end

function BangBang.setindex!!(pa::PartialArray, value, optic::IndexLens)
    return BangBang.setindex!!(pa, value, optic.indices...)
end
Base.getindex(pa::PartialArray, optic::IndexLens) = Base.getindex(pa, optic.indices...)
Base.haskey(pa::PartialArray, optic::IndexLens) = Base.haskey(pa, optic.indices)

function BangBang.setindex!!(iarr::PartialArray, value, inds::Vararg{INDEX_TYPES})
    if length(inds) != ndims(iarr)
        throw(BoundsError(iarr, inds))
    end
    if _has_colon(inds)
        throw(ArgumentError("Indexing with colons is not supported"))
    end
    iarr = if checkbounds(Bool, iarr.mask, inds...)
        iarr
    else
        _resize_partialarray(iarr, inds)
    end
    new_data = setindex!!(iarr.data, value, inds...)
    if _is_multiindex(inds)
        iarr.mask[inds...] .= true
    else
        iarr.mask[inds...] = true
    end
    return PartialArray(new_data, iarr.mask)
end

function Base.getindex(iarr::PartialArray, inds::Vararg{INDEX_TYPES})
    if length(inds) != ndims(iarr)
        throw(ArgumentError("Invalid index $(inds)"))
    end
    if _has_colon(inds)
        throw(ArgumentError("Indexing with colons is not supported"))
    end
    if !haskey(iarr, inds)
        throw(BoundsError(iarr, inds))
    end
    return getindex(iarr.data, inds...)
end

function Base.haskey(iarr::PartialArray, inds)
    if _has_colon(inds)
        throw(ArgumentError("Indexing with colons is not supported"))
    end
    return checkbounds(Bool, iarr.mask, inds...) &&
           all(@inbounds(getindex(iarr.mask, inds...)))
end

Base.merge(x1::PartialArray, x2::PartialArray) = _merge_recursive(x1, x2)
Base.merge(x1::VarNamedTuple, x2::VarNamedTuple) = _merge_recursive(x1, x2)
_merge_recursive(_, x2) = x2

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
        # Either pa2 is strictly bigger than pa1, or they are equal in size.
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
    iarr = PartialArray(et, num_inds)
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

Base.getindex(vnt::VarNamedTuple, name::Symbol) = vnt.data[name]

function varname_to_lens(name::VarName{S}) where {S}
    return _compose_no_identity(getoptic(name), PropertyLens{S}())
end

function Base.getindex(vnt::VarNamedTuple, name::VarName)
    return getindex(vnt, varname_to_lens(name))
end
function Base.getindex(x::Union{VarNamedTuple,PartialArray}, optic::ComposedFunction)
    subdata = getindex(x, optic.inner)
    return getindex(subdata, optic.outer)
end
function Base.getindex(vnt::VarNamedTuple, ::PropertyLens{S}) where {S}
    return getindex(vnt.data, S)
end

function Base.haskey(vnt::VarNamedTuple, name::VarName)
    return haskey(vnt, varname_to_lens(name))
end

Base.haskey(vnt::VarNamedTuple, ::typeof(identity)) = true

function Base.haskey(vnt::VarNamedTuple, optic::ComposedFunction)
    return haskey(vnt, optic.inner) && haskey(getindex(vnt, optic.inner), optic.outer)
end

Base.haskey(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = haskey(vnt.data, S)
Base.haskey(::VarNamedTuple, ::IndexLens) = false

# TODO(mhauru) This is type piracy.
Base.getindex(arr::AbstractArray, optic::IndexLens) = getindex(arr, optic.indices...)

# TODO(mhauru) This is type piracy.
function BangBang.setindex!!(arr::AbstractArray, value, optic::IndexLens)
    return BangBang.setindex!!(arr, value, optic.indices...)
end

function BangBang.setindex!!(vnt::VarNamedTuple, value, name::VarName)
    return BangBang.setindex!!(vnt, value, varname_to_lens(name))
end

function BangBang.setindex!!(
    vnt::Union{VarNamedTuple,PartialArray}, value, optic::ComposedFunction
)
    sub = if haskey(vnt, optic.inner)
        BangBang.setindex!!(getindex(vnt, optic.inner), value, optic.outer)
    else
        make_leaf(value, optic.outer)
    end
    return BangBang.setindex!!(vnt, sub, optic.inner)
end

function BangBang.setindex!!(vnt::VarNamedTuple, value, ::PropertyLens{S}) where {S}
    # I would like this to just read
    # return VarNamedTuple(BangBang.setindex!!(vnt.data, value, S))
    # but that seems to be type unstable. Why? Shouldn't it obviously be the same as the
    # below?
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((value,))))
end

function apply(func, vnt::VarNamedTuple, name::VarName)
    if !haskey(vnt.data, name.name)
        throw(KeyError(repr(name)))
    end
    subdata = getindex(vnt, name)
    new_subdata = func(subdata)
    return BangBang.setindex!!(vnt, new_subdata, name)
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

function Base.haskey(vnt::VarNamedTuple, name::VarName{S,Optic}) where {S,Optic}
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

end
