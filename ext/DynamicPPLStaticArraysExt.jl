module DynamicPPLStaticArraysExt
using DynamicPPL: DynamicPPL
using AbstractPPL: AbstractPPL
using StaticArrays: SVector, MVector, similar_type

DynamicPPL._binding_optic(::SVector) = AbstractPPL.Index
function DynamicPPL._binding_optic(x::MVector)
    return isbitstype(eltype(x)) ? AbstractPPL.Index : nothing
end
DynamicPPL._rebuild_buffer(x::SVector) = x
# Mooncake cannot differentiate Base.isassigned's try/catch.
DynamicPPL._isassigned(x::SVector, indices...) = checkbounds(Bool, x, indices...)
function DynamicPPL._isassigned(x::MVector, indices...)
    return checkbounds(Bool, x, indices...) &&
           (isbitstype(eltype(x)) || isdefined(x, :data))
end
function DynamicPPL._setindex!!(x::Union{SVector,MVector}, value, index::Int)
    # Promote first: StaticArrays' setindex converts to the existing eltype.
    if !(typeof(value) <: eltype(x))
        x = similar_type(x, promote_type(eltype(x), typeof(value)))(x)
    end
    return Base.setindex(x, value, index)
end
end
