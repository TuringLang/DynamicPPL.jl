# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using BangBang
using Accessors
using DynamicPPL: _compose_no_identity

export VarNamedTuple

struct VarNamedTuple{T<:Function,Names,Values}
    data::NamedTuple{Names,Values}
    make_leaf::T
end

struct IndexDict{T<:Function,Keys,Values}
    data::Dict{Keys,Values}
    make_leaf::T
end

struct IndexArray{T<:Function,ElType,numdims}
    data::Array{ElType,numdims}
    mask::Array{Bool,numdims}
    make_leaf::T
end

function IndexArray(eltype, num_dims, make_leaf)
    dims = ntuple(_ -> 0, num_dims)
    data = Array{eltype,num_dims}(undef, dims)
    mask = fill(false, dims)
    return IndexArray(data, mask, make_leaf)
end

_length_needed(i::Integer) = i
_length_needed(r::UnitRange) = last(r)
_length_needed(::Colon) = 0

function _resize_indexarray(iarr::IndexArray, inds)
    # Resize arrays to accommodate new indices.
    new_sizes = ntuple(i -> max(size(iarr.data, i), _length_needed(inds[i])), length(inds))
    # Generic multidimensional Arrays can not be resized, so we need to make a new one.
    # See https://github.com/JuliaLang/julia/issues/37900
    new_data = Array{eltype(iarr.data),ndims(iarr.data)}(undef, new_sizes)
    new_mask = fill(false, new_sizes)
    for i in eachindex(iarr.data)
        @inbounds new_data[i] = iarr.data[i]
        @inbounds new_mask[i] = iarr.mask[i]
    end
    return IndexArray(new_data, new_mask, iarr.make_leaf)
end

function BangBang.setindex!!(iarr::IndexArray, value, lens::IndexLens)
    inds = lens.indices
    iarr = if checkbounds(Bool, iarr.mask, inds...)
        iarr
    else
        _resize_indexarray(iarr, inds)
    end
    new_data = setindex!!(iarr.data, value, inds...)
    new_mask = setindex!!(iarr.mask, true, inds...)
    return IndexArray(new_data, new_mask, iarr.make_leaf)
end

function Base.getindex(iarr::IndexArray, lens::IndexLens)
    if !haskey(iarr, lens)
        throw(BoundsError("No value set at indices $(lens)"))
    end
    inds = lens.indices
    return getindex(iarr.data, inds...)
end

function Base.haskey(iarr::IndexArray, lens::IndexLens)
    inds = lens.indices
    return checkbounds(Bool, iarr.mask, inds...) &&
           all(@inbounds(getindex(iarr.mask, inds...)))
end

function make_leaf_array(value, ::PropertyLens{S}) where {S}
    return VarNamedTuple(NamedTuple{(S,)}((value,)), make_leaf_array)
end
make_leaf_array(value, ::typeof(identity)) = value
function make_leaf_array(value, optic::ComposedFunction)
    sub = make_leaf_array(value, optic.outer)
    return make_leaf_array(sub, optic.inner)
end

function make_leaf_array(value, optic::IndexLens)
    num_inds = length(optic.indices)
    iarr = IndexArray(typeof(value), num_inds, make_leaf_array)
    return setindex!!(iarr, value, optic)
end

function make_leaf_dict(value, ::PropertyLens{S}) where {S}
    return VarNamedTuple(NamedTuple{(S,)}((value,)), make_leaf_dict)
end
make_leaf_dict(value, ::typeof(identity)) = value
function make_leaf_dict(value, optic::ComposedFunction)
    sub = make_leaf_dict(value, optic.outer)
    return make_leaf_dict(sub, optic.inner)
end
function make_leaf_dict(value, optic::IndexLens)
    return IndexDict(Dict(optic.indices => value), make_leaf_dict)
end

VarNamedTuple() = VarNamedTuple((;), make_leaf_array)

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

function Base.show(io::IO, id::IndexDict)
    return print(io, id.data)
end

Base.getindex(vnt::VarNamedTuple, name::Symbol) = vnt.data[name]

function varname_to_lens(name::VarName{S}) where {S}
    return _compose_no_identity(getoptic(name), PropertyLens{S}())
end

function Base.getindex(vnt::VarNamedTuple, name::VarName)
    return getindex(vnt, varname_to_lens(name))
end
function Base.getindex(x::Union{VarNamedTuple,IndexDict,IndexArray}, lens::ComposedFunction)
    subdata = getindex(x, lens.inner)
    return getindex(subdata, lens.outer)
end
function Base.getindex(vnt::VarNamedTuple, ::PropertyLens{S}) where {S}
    return getindex(vnt.data, S)
end
function Base.getindex(id::IndexDict, lens::IndexLens)
    return getindex(id.data, lens.indices)
end

function Base.haskey(vnt::VarNamedTuple, name::VarName)
    return haskey(vnt, varname_to_lens(name))
end

Base.haskey(vnt::VarNamedTuple, ::typeof(identity)) = true

function Base.haskey(vnt::VarNamedTuple, lens::ComposedFunction)
    return haskey(vnt, lens.inner) && haskey(getindex(vnt, lens.inner), lens.outer)
end

Base.haskey(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = haskey(vnt.data, S)
Base.haskey(id::IndexDict, lens::IndexLens) = haskey(id.data, lens.indices)
Base.haskey(::VarNamedTuple, ::IndexLens) = false
Base.haskey(::IndexDict, ::PropertyLens) = false

# TODO(mhauru) This is type piracy.
Base.getindex(arr::AbstractArray, lens::IndexLens) = getindex(arr, lens.indices...)

# TODO(mhauru) This is type piracy.
function BangBang.setindex!!(arr::AbstractArray, value, lens::IndexLens)
    return BangBang.setindex!!(arr, value, lens.indices...)
end

function BangBang.setindex!!(vnt::VarNamedTuple, value, name::VarName)
    return BangBang.setindex!!(vnt, value, varname_to_lens(name))
end

function BangBang.setindex!!(
    vnt::Union{VarNamedTuple,IndexDict,IndexArray}, value, lens::ComposedFunction
)
    sub = if haskey(vnt, lens.inner)
        BangBang.setindex!!(getindex(vnt, lens.inner), value, lens.outer)
    else
        vnt.make_leaf(value, lens.outer)
    end
    return BangBang.setindex!!(vnt, sub, lens.inner)
end

function BangBang.setindex!!(vnt::VarNamedTuple, value, ::PropertyLens{S}) where {S}
    # I would like this to just read
    # return VarNamedTuple(BangBang.setindex!!(vnt.data, value, S), vnt.make_leaf)
    # but that seems to be type unstable. Why? Shouldn't it obviously be the same as the
    # below?
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((value,))), vnt.make_leaf)
end

function BangBang.setindex!!(id::IndexDict, value, lens::IndexLens)
    return IndexDict(setindex!!(id.data, value, lens.indices), id.make_leaf)
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
    return VarNamedTuple(new_data, vnt.make_leaf)
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

end
