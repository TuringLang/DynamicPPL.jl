module DynamicPPLReverseDiffExt

using DynamicPPL
using ReverseDiff

@inline DynamicPPL.maybe_view_ad(vect::ReverseDiff.TrackedArray, range) =
    getindex(vect, range)

DynamicPPL._argument_ad_storage(::Type{<:ReverseDiff.TrackedArray}) = true
DynamicPPL._copy_model_argument(value::ReverseDiff.TrackedArray) = map(identity, value)
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
function _writable_graph(value::Ref, memo)
    haskey(memo, value) && return memo[value]
    child = _writable_graph(value[], memo)
    result = child === value[] ? value : Ref(child)
    memo[value] = result
    return result
end
function _writable_graph(value::Union{Dict,IdDict}, memo)
    haskey(memo, value) && return memo[value]
    pairs = Pair[]
    changed = false
    for (key, child) in value
        adapted = _writable_graph(child, memo)
        changed |= adapted !== child
        push!(pairs, key => adapted)
    end
    changed || return value
    value_type = if isempty(pairs)
        valtype(typeof(value))
    else
        typejoin(valtype(typeof(value)), typeof.(last.(pairs))...)
    end
    dict_type = typeof(value).name.wrapper{keytype(typeof(value)),value_type}
    result = dict_type(pairs)
    memo[value] = result
    return result
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
        declared_type = Base.fieldtype(typeof(result), name)
        result = if child isa declared_type && !isconst(typeof(result), name)
            DynamicPPL._set_argument_property(result, Val(name), child)
        else
            DynamicPPL._rebuild_argument_property(result, Val(name), child)
        end
    end
    return memo[value] = result
end

end # module
