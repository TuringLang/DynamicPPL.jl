module DynamicPPLReverseDiffExt

using DynamicPPL
using ReverseDiff

@inline DynamicPPL.maybe_view_ad(vect::ReverseDiff.TrackedArray, range) =
    getindex(vect, range)

function DynamicPPL._setindex!!(
    array::ReverseDiff.TrackedArray, value, indices...; kwargs...
)
    return DynamicPPL._setindex!!(map(identity, array), value, indices...; kwargs...)
end

DynamicPPL._argument_ad_storage(::Type{<:ReverseDiff.TrackedArray}) = true
# Every tracked value reaches the shared tape, which is not argument storage.
DynamicPPL._argument_may_alias(::Type{ReverseDiff.InstructionTape}) = false
DynamicPPL._reached_latent(latent, ::ReverseDiff.InstructionTape, seen, memory) = nothing
# Scalar origins and derivative buffers belong to the tape, not argument storage.
function DynamicPPL._argument_may_alias(::Type{<:ReverseDiff.TrackedReal{V}}) where {V}
    return DynamicPPL._argument_may_alias(V)
end
function DynamicPPL._foreach_argument_child(f::F, value::ReverseDiff.TrackedReal) where {F}
    return f(ReverseDiff.value(value))
end
function DynamicPPL._argument_opaque_value(
    ::Union{ReverseDiff.TrackedArray,ReverseDiff.TrackedReal}
)
    return true
end
function DynamicPPL._retain_argument_value!(memo, value::ReverseDiff.TrackedArray, seen)
    _retain_buffer!(memo, ReverseDiff.value(value))
    _retain_buffer!(memo, ReverseDiff.deriv(value))
    memo[ReverseDiff.tape(value)] = ReverseDiff.tape(value)
    return nothing
end

function _retain_buffer!(memo, buffer)
    memo[buffer] = buffer
    storage = parent(buffer)
    storage === buffer || _retain_buffer!(memo, storage)
    return nothing
end

end # module
