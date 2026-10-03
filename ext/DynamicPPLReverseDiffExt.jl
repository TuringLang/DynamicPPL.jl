module DynamicPPLReverseDiffExt

using DynamicPPL
using ReverseDiff

@inline DynamicPPL.maybe_view_ad(vect::ReverseDiff.TrackedArray, range) =
    getindex(vect, range)

# `copy` returns the same non-writable TrackedArray. Keep its scalar tape connections
# while allocating storage that argument reconstruction and latent tildes can update.
function DynamicPPL._writable_model_argument(value::ReverseDiff.TrackedArray)
    return map(identity, value)
end

DynamicPPL._argument_ad_storage(::Type{<:ReverseDiff.TrackedArray}) = true
DynamicPPL._copy_model_argument(value::ReverseDiff.TrackedArray) = map(identity, value)
function DynamicPPL._retain_argument_children!(memo, value::ReverseDiff.TrackedArray, seen)
    # A tracked array is an immutable facade. Keep its tape and origin buffers so
    # reads of a copied facade still connect to the enclosing differentiation.
    memo[DynamicPPL.ModelArgumentCopy] = true
    _retain_buffer!(memo, ReverseDiff.value(value))
    _retain_buffer!(memo, ReverseDiff.deriv(value))
    memo[ReverseDiff.tape(value)] = ReverseDiff.tape(value)
    return nothing
end

# Views and reshapes also need their parent buffers: deepcopy may rebuild a
# wrapper through its parent even when the wrapper itself is in the memo.
function _retain_buffer!(memo, buffer)
    memo[buffer] = buffer
    storage = parent(buffer)
    storage === buffer || _retain_buffer!(memo, storage)
    return nothing
end

# Deepcopy must preserve types. Afterwards, adapt immutable TrackedArray facades
# to writable scalar storage, rebuilding only parents whose child type changes.
# This walks Julia's already-copied graph; it never allocates user structs itself.
function DynamicPPL._adapt_copied_argument(value)
    memo = IdDict()
    DynamicPPL._retain_bound_argument_storage!(memo, value)
    return _writable_graph(value, memo)
end
function _writable_graph(
    value::Union{
        Number,
        Type,
        Symbol,
        AbstractString,
        Nothing,
        DynamicPPL.NoModelBinding,
        DynamicPPL.ModelArgumentLeaf,
    },
    memo,
)
    return value
end
function _writable_graph(value::ReverseDiff.TrackedArray, memo)
    return get!(memo, value) do
        map(identity, value)
    end
end
function _writable_graph(value::Union{Tuple,NamedTuple}, memo)
    return map(child -> _writable_graph(child, memo), value)
end
function _writable_graph(value::DynamicPPL.ModelArgumentStorage, memo)
    return DynamicPPL.ModelArgumentStorage(
        _writable_graph(value.value, memo), _writable_graph(value.children, memo)
    )
end
function _writable_graph(value::SubArray, memo)
    return get!(memo, value) do
        storage = _writable_graph(parent(value), memo)
        storage === parent(value) ? value : view(storage, parentindices(value)...)
    end
end
function _writable_graph(value::AbstractArray, memo)
    eltype(value) <: Number && return value
    haskey(memo, value) && return memo[value]
    memo[value] = value
    result = value
    for i in eachindex(value)
        isassigned(value, i) || continue
        child = _writable_graph(value[i], memo)
        child === value[i] && continue
        result = DynamicPPL._set_argument_index(result, child, i)
    end
    return memo[value] = result
end
function _writable_graph(value, memo)
    isbits(value) && return value
    haskey(memo, value) && return memo[value]
    memo[value] = value
    result = value
    for name in fieldnames(typeof(value))
        isdefined(value, name) || continue
        original = getfield(value, name)
        child = _writable_graph(original, memo)
        child === original && continue
        result = DynamicPPL._set_argument_property(result, Val(name), child)
    end
    return memo[value] = result
end

end # module
