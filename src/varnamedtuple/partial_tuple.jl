# An element of a `PartialTuple`, whose `value` is undefined while unset. `T` fixes the
# element's type either way, so writing a value of that type keeps the tuple's type, even at
# an index known only at run time. Not `Union{Nothing,Some{T}}`: Enzyme's type analysis
# throws on some inline-union fields with multi-field payloads, such as
# `Some{UnitRange{Int}}`.
struct TupleSlot{T}
    isset::Bool
    value::T
    TupleSlot{T}() where {T} = new{T}(false)
    TupleSlot{T}(value) where {T} = new{T}(true, value)
end
TupleSlot(value) = TupleSlot{typeof(value)}(value)

_isset(slot::TupleSlot) = slot.isset

"""
    PartialTuple{T<:Tuple}

A tuple whose elements may be unset: the storage `VarNamedTuple` creates for values set by
index into a tuple template, such as `x[1]` with `x = (0, 0.0)`.

Each element keeps its own type, so a tuple whose elements have different types round-trips
without promotion. Elements are addressed by a single integer, such as `x[1]` or `x[end]`;
other indices throw an `ArgumentError`.
"""
struct PartialTuple{T<:Tuple{Vararg{TupleSlot}}}
    data::T
end

# Must infer `Bool` for an unknown tuple type, where `all(_isset, pt.data)` gives
# `Union{Missing,Bool}`.
_all_set(pt::PartialTuple) = count(_isset, pt.data) == length(pt.data)

# Resolve `begin` and `end` against `slots`, a tuple or a `PartialTuple`'s data. `vn` names
# the variable in errors, where the caller knows it.
function _tuple_index(optic::AbstractPPL.Index, slots::Tuple, vn=nothing)
    coptic = AbstractPPL.concretize_top_level(optic, slots)
    coptic.ix isa Tuple{Integer} && isempty(coptic.kw) && return coptic
    name = something(vn, AbstractPPL.append_optic(@varname(_), optic))
    throw(
        ArgumentError(
            "Cannot index a tuple with `$name`: tuple elements are addressed by a single " *
            "integer, such as `x[1]`.",
        ),
    )
end
# Report an index outside the tuple against `pt`, whose summary hides the slot types.
function _slot(pt::PartialTuple, i)
    return checkbounds(Bool, keys(pt.data), i) ? pt.data[i] : throw(BoundsError(pt, i))
end
Base.summary(io::IO, pt::PartialTuple) = print(io, length(pt.data), "-element PartialTuple")

# Infer the result type of `f` from the element type rather than the value, so the result's
# type does not depend on which elements are set.
function _map_slot(f, slot::TupleSlot{T}) where {T}
    R = Core.Compiler.return_type(f, Tuple{T})
    S = TupleSlot{R === Union{} ? Any : R}
    return _isset(slot) ? S(f(slot.value)) : S()
end
_map_slots(f, pt::PartialTuple) = PartialTuple(map(slot -> _map_slot(f, slot), pt.data))

# A set element as `(value,)`, an unset one as `()`, whose `value` is undefined.
_slot_values(pt::PartialTuple) = map(s -> _isset(s) ? (s.value,) : (), pt.data)
Base.:(==)(pt1::PartialTuple, pt2::PartialTuple) = _slot_values(pt1) == _slot_values(pt2)
function Base.isequal(pt1::PartialTuple, pt2::PartialTuple)
    return isequal(_slot_values(pt1), _slot_values(pt2))
end
Base.hash(pt::PartialTuple, h::UInt) = hash(_slot_values(pt), h)

# Keeps each slot's type, which `_map_slots` would re-derive by inference.
function Base.copy(pt::PartialTuple)
    return PartialTuple(
        map(pt.data) do slot
            _isset(slot) || return slot
            return typeof(slot)(_copy_storage(slot.value))
        end,
    )
end

function Base.length(pt::PartialTuple)
    len = 0
    for slot in pt.data
        _isset(slot) && (len += _leaf_count(slot.value))
    end
    return len
end

function _merge(pt1::PartialTuple, pt2::PartialTuple, recurse::Val{true})
    length(pt1.data) == length(pt2.data) ||
        throw(ArgumentError("Cannot merge PartialTuples of different lengths"))
    return PartialTuple(
        map(pt1.data, pt2.data) do slot1, slot2
            _isset(slot1) || return slot2
            _isset(slot2) || return slot1
            return TupleSlot(_merge(slot1.value, slot2.value, recurse))
        end,
    )
end

function unwrap_internal_array(pt::PartialTuple)
    _all_set(pt) || throw(
        ArgumentError(
            "Cannot extract data from PartialTuple because some elements are not set."
        ),
    )
    return map(slot -> unwrap_internal_array(slot.value), pt.data)
end

# Matching nodes have their own methods; a tuple node and an array node cannot be merged.
function _merge(
    ::Union{PartialTuple,PartialArray}, ::Union{PartialTuple,PartialArray}, ::Val
)
    throw(ArgumentError("Cannot merge tuple and array nodes; give the tuple's template."))
end
