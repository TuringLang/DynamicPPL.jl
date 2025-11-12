module VarNamedTuples

using AbstractPPL
using BangBang
using Accessors
using DynamicPPL: _compose_no_identity

export VarNamedTuple

# @varname(a.b[3].c[:].d)
#
# VarNamedTuple(
#   (; a=(; b=[
#       (; c=[
#           (; d=...),
#           (; d=...),
#           (; d=...),
#       ]),
#       (; c=[
#           (; d=...),
#           (; d=...),
#           (; d=...),
#       ]),
#       (; c=[
#           (; d=...),
#           (; d=...),
#           (; d=...),
#       ]),
#   ))
#)

struct VarNamedTuple{T<:Function,Names,Values}
    data::NamedTuple{Names,Values}
    make_leaf::T
end

struct IndexDict{T<:Function,Keys,Values}
    data::Dict{Keys,Values}
    make_leaf::T
end

function make_leaf_raw(value, ::PropertyLens{S}) where {S}
    return VarNamedTuple(NamedTuple{(S,)}((value,)), make_leaf_raw)
end
make_leaf_raw(value, ::typeof(identity)) = value
function make_leaf_raw(value, optic::IndexLens)
    return IndexDict(Dict(optic.indices => value), make_leaf_raw)
end
function make_leaf_raw(value, optic::ComposedFunction)
    sub = make_leaf_raw(value, optic.outer)
    return make_leaf_raw(sub, optic.inner)
end

VarNamedTuple() = VarNamedTuple((;), make_leaf_raw)

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

function Base.show(io::IO, vnt::IndexDict)
    return print(io, vnt.data)
end

Base.getindex(vnt::VarNamedTuple, name::Symbol) = vnt.data[name]

function varname_to_lens(name::VarName{S}) where {S}
    return _compose_no_identity(getoptic(name), PropertyLens{S}())
end
function Base.getindex(vnt::VarNamedTuple, name::VarName)
    return getindex(vnt, varname_to_lens(name))
end
function Base.getindex(vnt::VarNamedTuple, lens::ComposedFunction)
    subdata = getindex(vnt, lens.inner)
    return getindex(subdata, lens.outer)
end
function Base.getindex(vnt::VarNamedTuple, ::PropertyLens{S}) where {S}
    return getindex(vnt.data, S)
end
function Base.getindex(vnt::IndexDict, lens::IndexLens)
    return getindex(vnt.data, lens.indices)
end

function Base.haskey(vnt::VarNamedTuple, name::VarName)
    return haskey(vnt, varname_to_lens(name))
end

Base.haskey(vnt::VarNamedTuple, ::typeof(identity)) = true

function Base.haskey(vnt::VarNamedTuple, lens::ComposedFunction)
    return haskey(vnt, lens.inner) && haskey(getindex(vnt, lens.inner), lens.outer)
end

Base.haskey(vnt::VarNamedTuple, ::PropertyLens{S}) where {S} = haskey(vnt.data, S)
Base.haskey(vnt::IndexDict, lens::IndexLens) = haskey(vnt.data, lens.indices)
Base.haskey(vnt::VarNamedTuple, lens::IndexLens) = false
Base.haskey(vnt::IndexDict, lens::PropertyLens) = false

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
    vnt::VarNamedTuple, value, lens::ComposedFunction{Outer,Inner}
) where {Outer,Inner}
    sub = if haskey(vnt, lens.inner)
        BangBang.setindex!!(lens.inner(vnt.data), value, lens.outer)
    else
        vnt.make_leaf(value, lens.outer)
    end
    return BangBang.setindex!!(vnt, sub, lens.inner)
end

function BangBang.setindex!!(vnt::VarNamedTuple, value, ::PropertyLens{S}) where {S}
    return VarNamedTuple(BangBang.setindex!!(vnt.data, value, S), vnt.make_leaf)
end

function BangBang.setindex!!(vnt::IndexDict, value, lens::IndexLens)
    return setindex!(vnt.data, value, lens.indices)
end

# function BangBang.setindex!!(
#     vnt::VarNamedTuple, value, name::{S,Optic}
# ) where {S,Optic}
#     new_data = if haskey(vnt.data, S)
#         if Optic === typeof(identity)
#             BangBang.setindex!!(vnt.data, vnt.make_leaf(value, getoptic(name)), S)
#         elseif Optic <: IndexLens
#             new_subdata = BangBang.setindex!!(vnt.data[S], value, getoptic(name))
#             BangBang.setindex!!(vnt.data, new_subdata, S)
#         else
#             new_subdata = BangBang.setindex!!(
#                 vnt.data[S], value, AbstractPPL.unprefix(name, VarName{S}())
#             )
#             BangBang.setindex!!(vnt.data, new_subdata, S)
#         end
#     else
#         new_subdata = if Optic === typeof(identity) || Optic <: IndexLens
#             vnt.make_leaf(value, getoptic(name))
#             # if Optic === typeof(identity)
#             #     BangBang.setindex!!(vnt.data, value, S)
#             # elseif Optic <: IndexLens
#             #     new_subdata = BangBang.setindex!!(Dict{Union{},Union{}}(), value, getoptic(name).indices...)
#             #     BangBang.setindex!!(vnt.data, new_subdata, S)
#         else
#             BangBang.setindex!!(
#                 VarNamedTuple((;), vnt.make_leaf),
#                 value,
#                 AbstractPPL.unprefix(name, VarName{S}()),
#             )
#         end
#         BangBang.setindex!!(vnt.data, new_subdata, S)
#     end
#     return VarNamedTuple(new_data, vnt.make_leaf)
# end

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

# module AdHocTests
#
# using DynamicPPL: VarNamedTuples, @varname
# using BangBang
#
# vnt = VarNamedTuples.VarNamedTuple()
# display(vnt)
#
# vnt = setindex!!(vnt, 32.0, @varname(a))
# println("a = $(vnt[@varname(a)])")
# display(vnt)
#
# vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
# println("b[2] = $(vnt[@varname(b[2])])")
# display(vnt)
#
# vnt = setindex!!(vnt, 64.0, @varname(a))
# display(vnt)
#
# vnt = setindex!!(vnt, 15, @varname(b[2]))
# display(vnt)
#
# vnt = setindex!!(vnt, [10], @varname(c.x.y))
# println("c.x = $(vnt[@varname(c.x)])")
# display(vnt)
#
# vnt = setindex!!(vnt, 11, @varname(c.x.y[1]))
# display(vnt)
#
# vnt = setindex!!(vnt, -1.0, @varname(d[4]))
# display(vnt)
#
# vnt = setindex!!(vnt, -2.0, @varname(d[4]))
# display(vnt)
#
# vnt = setindex!!(vnt, -3.0, @varname(d[5]))
# display(vnt)
#
# println("d = $(vnt[@varname(d)])")
#
# vnt = setindex!!(vnt, 1.0, @varname(e.f[3].g.h[2].i))
# display(vnt)
# end
