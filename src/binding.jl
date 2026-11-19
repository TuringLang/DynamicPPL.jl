abstract type BindingRole end
struct ConditionRole <: BindingRole end
struct ArgumentRole <: BindingRole end
struct FixRole <: BindingRole end
struct BoundValue{R<:BindingRole,T}
    value::T
    function BoundValue{R}(value::T) where {R,T}
        R in (ConditionRole, ArgumentRole, FixRole) ||
            throw(ArgumentError("Invalid binding role"))
        return new{R,T}(value)
    end
end
struct NoBinding end
struct BindingConstraint{V,R,S,M<:Union{Val{:type},Val{:shape}}}
    owner::V
    role::Type{R}
    storage::S
    mode::M
end
struct BindingRoot{V,S,C}
    value::V
    storage::S
    constraints::C
end
function _effective_constraints(root, base)
    return if root.value isa BoundValue{ArgumentRole}
        (BindingConstraint(base, ConditionRole, root.value.value, Val(:shape)),)
    else
        root.constraints
    end
end
# `reserved` names roots that an enclosing model binds for its own tildes while its
# unprefixed child takes them as covariates; a descendant tilde there would duplicate them.
struct BindingTable{V<:VarNamedTuple,R<:Tuple}
    roots::V
    reserved::R
end
BindingTable(roots::VarNamedTuple) = BindingTable(roots, ())
maybe_prefix(vn, ::Nothing) = vn
maybe_prefix(vn, prefix::VarName) = AbstractPPL.prefix(vn, prefix)
maybe_prefix(::Nothing, prefix::VarName) = prefix
_throw_binding_error(vn, rule) = throw(ArgumentError("`$vn`: $rule"))
function _throw_missing_storage(vn)
    return _throw_binding_error(
        vn,
        "partial argument binding lacks storage; bind the whole value or pass a concrete argument (for example one with missing entries)",
    )
end

function _find_placeholder(value, seen=Base.IdSet{Any}())
    value === missing && return :missing
    value === nothing && return :nothing
    value isa Union{Number,AbstractString,Symbol,Type} && return :none
    value isa AbstractArray{<:Number} && return :none
    if ismutabletype(typeof(value))
        value in seen && return :none
        push!(seen, value)
    end
    children = if value isa AbstractArray
        (value[i] for i in eachindex(value) if isassigned(value, i))
    elseif value isa Union{Tuple,NamedTuple,Base.Pairs}
        values(value)
    else
        (getfield(value, i) for i in 1:fieldcount(typeof(value)) if isdefined(value, i))
    end
    result = :none
    for child in children
        kind = _find_placeholder(child, seen)
        kind === :missing && return kind
        kind === :nothing && (result = kind)
    end
    return result
end
_rewrap_role(::BoundValue{R}, value) where {R} = BoundValue{R}(value)
_binding_role(::BoundValue{R}) where {R} = R === FixRole ? FixRole() : ConditionRole()
_matches_role(R, x) = x isa BoundValue{R}
_matches_role(::Type{ConditionRole}, x::BoundValue{ArgumentRole}) = true
"""
    _element_optic(x)

Choose how binding-managed storage holds and rebuilds children without mutating
the caller's storage. Return `AbstractPPL.Index` for position-addressed containers
rebuilt with preserved axes, `AbstractPPL.Property` for field-addressed
containers rebuilt with `ConstructionBase.setproperties`, or `nothing` for
unclaimed values. Unclaimed values are shared as leaves, except arrays and
dictionaries, which are refused where storage is required. This method never
decides whether a value can be bound whole; any non-placeholder value can.
"""
_element_optic(x) = nothing
_element_optic(::Array) = AbstractPPL.Index
function _element_optic(::LinearAlgebra.Transpose{<:Union{Missing,Number},<:Matrix})
    return AbstractPPL.Index
end
_element_optic(::Tuple) = AbstractPPL.Index
_element_optic(::NamedTuple) = AbstractPPL.Property
_rebuild_buffer(x::AbstractArray) = similar(x)
_rebuild_buffer(x::Tuple) = x
_rebuild_buffer(x, ::Type{T}) where {T} = similar(x, T)
_argument_buffer(storage, binding) = _rebuild_buffer(storage)
# Widen up front: widening on write in the `_bound_data` and `_fill_latent_storage` loops
# gives a non-concrete result type. Mixed value types take their union; promotion may round.
function _argument_buffer(
    storage::Array{Missing}, binding::VarNamedTuples.PartialArray{<:BoundValue}
)
    T = fieldtype(eltype(binding.data), :value)
    isconcretetype(T) ||
        (T = Union{unique(typeof(b.value) for b in view(binding.data, binding.mask))...})
    return similar(storage, Union{Missing,T})
end
_positional_indices(x::AbstractArray) = CartesianIndices(x)
_positional_indices(x::Tuple) = keys(x)
_isassigned(x, indices...) = isassigned(x, indices...)
_isassigned(x::Tuple, i) = i in keys(x)
_selection_positions(x) = _positional_indices(x)
_index_tuple(i::CartesianIndex) = Tuple(i)
_index_tuple(i::Integer) = (i,)

const _PositionalNode = Union{VarNamedTuples.PartialArray,VarNamedTuples.PartialTuple}
const _InnerNode = Union{VarNamedTuple,_PositionalNode}
_is_complete(x::VarNamedTuples.PartialArray) = all(x.mask)
_is_complete(x::VarNamedTuples.PartialTuple) = VarNamedTuples._all_set(x)

_unwrap_storage(x) = x
_unwrap_storage(x::VarNamedTuples.PartialArray) = x.data
# An unset slot holds no storage, so it reads as `NoTemplate()`.
function _unwrap_storage(x::VarNamedTuples.PartialTuple)
    return map(s -> VarNamedTuples._isset(s) ? s.value : NoTemplate(), x.data)
end
_unwrap_storage(x::BoundValue) = x.value
_unwrap_storage(::Union{Missing,Nothing}) = NoTemplate()
# Root storage from a `VarNamedTuple` input, such as a replayed listing: partial nodes become
# plain containers, so later bindings find `Array`, tuple or field storage.
_plain_storage(x) = _unwrap_storage(x)
_plain_storage(x::VarNamedTuple) = VarNamedTuple(map(_plain_storage, x.data))
function _plain_storage(x::VarNamedTuples.PartialTuple)
    return map(
        s -> VarNamedTuples._isset(s) ? _plain_storage(s.value) : NoTemplate(), x.data
    )
end
function _plain_storage(x::VarNamedTuples.PartialArray)
    any(i -> x.mask[i] && x.data[i] isa _InnerNode, eachindex(x.mask)) || return x.data
    return map(
        i -> x.mask[i] ? _plain_storage(x.data[i]) : NoTemplate(), CartesianIndices(x.data)
    )
end
function VarNamedTuples.maybe_index_template(::BoundValue, template, optic)
    return VarNamedTuples.maybe_index_template(template, optic)
end

function VarNamedTuples._getindex_optic(x::BoundValue, optic::AbstractPPL.AbstractOptic, vn)
    return _rewrap_role(x, VarNamedTuples._getindex_optic(x.value, optic, vn))
end
VarNamedTuples._getindex_optic(x::BoundValue, ::AbstractPPL.Iden, vn) = x
function VarNamedTuples._haskey_optic(
    x::BoundValue, optic::Union{AbstractPPL.Property,AbstractPPL.Index}
)
    return VarNamedTuples._haskey_optic(x.value, optic)
end

function _split_bound_value(x::BoundValue{R}, vn) where {R}
    value = x.value
    if _element_optic(value) === AbstractPPL.Property
        names = propertynames(value)
        return VarNamedTuple(
            NamedTuple{names}(map(n -> BoundValue{R}(getproperty(value, n)), names))
        )
    end
    _element_optic(value) === AbstractPPL.Index || _throw_unsupported_storage(value, vn)
    value isa Tuple && return VarNamedTuples.PartialTuple(
        map(v -> VarNamedTuples.TupleSlot(BoundValue{R}(v)), value)
    )
    types = Base.uniontypes(eltype(value))
    T = if !isempty(types) && all(isconcretetype, types)
        Union{map(T -> BoundValue{R,T}, types)...}
    else
        BoundValue{R}
    end
    data = _rebuild_buffer(value, T)
    mask = fill!(similar(data, Bool), false)
    for i in _positional_indices(value)
        if _isassigned(value, i)
            data[i] = BoundValue{R}(value[i])
            mask[i] = true
        end
    end
    return VarNamedTuples.PartialArray(data, mask)
end
function VarNamedTuples._setindex_optic!!(
    x::BoundValue, value, optic::AbstractPPL.AbstractOptic, template, perms
)
    return VarNamedTuples._setindex_optic!!(
        _split_bound_value(x, @varname(_)), value, optic, template, perms
    )
end
function VarNamedTuples._setindex_optic!!(
    ::BoundValue, value, ::AbstractPPL.Iden, template, perms::VarNamedTuples.SetPermissions
)
    return value
end

function _binding_node_at(x, optic::AbstractPPL.AbstractOptic)
    x isa Union{Nothing,NoBinding} && return nothing
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(x, head) || return nothing
    return _binding_node_at(
        VarNamedTuples._getindex_optic(x, head, @varname(_)), optic.child
    )
end
function _binding_node_at(x::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return haskey(x.data, S) ? _binding_node_at(x.data[S], optic.child) : nothing
end
_binding_node_at(x, ::AbstractPPL.Iden) = x
const _PositionalIndices = Tuple{
    Vararg{
        Union{Integer,Colon,CartesianIndex,AbstractArray{<:Union{Integer,CartesianIndex}}}
    },
}
function _positional_index(optic::AbstractPPL.Index)
    return isempty(optic.kw) && optic.ix isa _PositionalIndices
end
function _has_binding_position(x, optic)
    if !_positional_index(optic)
        isempty(view(x, optic.ix...; optic.kw...)) &&
            throw(ArgumentError("wrapper index selects no elements"))
        return true
    end
    checkbounds(Bool, x, optic.ix...) || return false
    return VarNamedTuples._is_multiindex(x, optic.ix...) || _isassigned(x, optic.ix...)
end
function _binding_node_at(x::AbstractArray, optic::AbstractPPL.Index)
    head = AbstractPPL.concretize_top_level(AbstractPPL.ohead(optic), x)
    _has_binding_position(x, head) || return nothing
    return _binding_node_at(getindex(x, head.ix...; head.kw...), optic.child)
end
function _binding_node_at(x::VarNamedTuples.PartialArray, optic::AbstractPPL.Index)
    optic = AbstractPPL.concretize_top_level(optic, x.data)
    _has_binding_position(x.mask, optic) || return nothing
    child = if VarNamedTuples._is_multiindex(x.data, optic.ix...; optic.kw...)
        VarNamedTuples._subset_partialarray(x, optic.ix...; optic.kw...)
    elseif getindex(x.mask, optic.ix...; optic.kw...)
        getindex(x.data, optic.ix...; optic.kw...)
    else
        nothing
    end
    return _binding_node_at(child, optic.child)
end
# A set slot is a node even when partly bound; `_haskey_optic` requires all its entries.
function _binding_node_at(x::VarNamedTuples.PartialTuple, optic::AbstractPPL.Index)
    i = only(VarNamedTuples._tuple_index(optic, x.data, @varname(_)).ix)
    checkbounds(Bool, keys(x.data), i) && VarNamedTuples._isset(x.data[i]) || return nothing
    return _binding_node_at(x.data[i].value, optic.child)
end
function _binding_node_at(
    x::BoundValue{R,<:AbstractArray}, optic::AbstractPPL.Index
) where {R}
    array = x.value
    optic = AbstractPPL.concretize_top_level(optic, array)
    _has_binding_position(array, optic) || return nothing
    if VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...) &&
        _element_optic(array) === AbstractPPL.Index &&
        !all(i -> _isassigned(array, i), _positional_indices(array))
        return _binding_node_at(_split_bound_value(x, @varname(_)), optic)
    end
    return _binding_node_at(
        BoundValue{R}(getindex(array, optic.ix...; optic.kw...)), optic.child
    )
end
_binding_role(::Union{Nothing,NoBinding}, vn, storage=NoTemplate()) = nothing
_binding_role(x::BoundValue, vn, storage=NoTemplate()) = _binding_role(x)
function _binding_role(x::_InnerNode, vn, storage=NoTemplate())
    if x isa VarNamedTuples.PartialArray{<:BoundValue}
        observed, fixed, complete = false, false, true
        for i in eachindex(x.mask)
            if x.mask[i]
                observed |= _matches_role(ConditionRole, x.data[i])
                fixed |= _matches_role(FixRole, x.data[i])
            else
                complete = false
            end
        end
    else
        observed = _has_role(ConditionRole, x)
        fixed = _has_role(FixRole, x)
        complete = true
    end
    observed || fixed || return nothing
    observed && fixed && _throw_binding_error(vn, "a tilde cannot mix binding roles")
    if !(x isa VarNamedTuples.PartialArray{<:BoundValue})
        complete = _is_fully_bound(x, storage)
    end
    if !complete
        parts = x isa VarNamedTuple ? "fields" : "positions"
        message = "a tilde cannot mix bound and latent " * parts
        if _has_field_bindings(x)
            message *=
                "; record fields need stored shape: bind the whole value " *
                "(the whole array for array elements) or supply argument storage; " *
                "flat records also accept an @of template"
        end
        _throw_binding_error(vn, message)
    end
    return fixed ? FixRole() : ConditionRole()
end
_bound_data(x::BoundValue) = x.value
_bound_data(x::VarNamedTuple) = map(_bound_data, x.data)
function _bound_data(x::_PositionalNode)
    _is_complete(x) || throw(ArgumentError("Incomplete binding"))
    return map(_bound_data, _unwrap_storage(x))
end
_bound_data(x, storage) = _bound_data(x)
function _bound_data(x::VarNamedTuple, storage)
    _element_optic(storage) === AbstractPPL.Property || return _bound_data(x)
    names = propertynames(storage)
    fields = map(names) do name
        _bound_data(x.data[name], getproperty(storage, name))
    end
    return ConstructionBase.setproperties(storage, NamedTuple{names}(fields))
end
function _bound_data(x::_PositionalNode, storage::Union{AbstractArray,Tuple})
    _is_complete(x) || throw(ArgumentError("Incomplete binding"))
    data = _unwrap_storage(x)
    @assert axes(storage) == axes(data)
    result = _argument_buffer(storage, x)
    for i in _positional_indices(storage)
        result = _set_optic!!(
            result,
            _bound_data(data[i], _isassigned(storage, i) ? storage[i] : NoTemplate()),
            AbstractPPL.Index(_index_tuple(i), (;)),
            @varname(_),
        )
    end
    return result
end
_has_field_bindings(x) = false
_has_field_bindings(::VarNamedTuple) = true
_has_field_bindings(::VarNamedTuples.PartialArray{<:BoundValue}) = false
function _has_field_bindings(x::VarNamedTuples.PartialArray)
    return any(i -> x.mask[i] && _has_field_bindings(x.data[i]), eachindex(x.mask))
end
function _has_field_bindings(x::VarNamedTuples.PartialTuple)
    return any(_has_field_bindings, _unwrap_storage(x))
end
function _has_role(R, x::_InnerNode)
    return VarNamedTuples._mapreduce_recursive(
        p -> _has_role(R, last(p)), |, x, @varname(_), false
    )
end
_has_role(R, x) = _matches_role(R, x)
function _has_bound_value(x::_InnerNode)
    return _has_role(ConditionRole, x) || _has_role(FixRole, x)
end
_has_bound_value(x) = x isa BoundValue

function _check_nested_selection(optic, vn)
    optic isa AbstractPPL.Iden && return nothing
    if optic isa AbstractPPL.Index &&
        optic.child isa AbstractPPL.Index &&
        any(i -> i isa AbstractArray, optic.ix) &&
        all(i -> i isa Union{Integer,AbstractArray}, optic.ix)
        indices = map(i -> i isa AbstractVector{Bool} ? Base.OneTo(count(i)) : i, optic.ix)
        selected = CartesianIndices(Base.index_shape(indices...))
        child = optic.child
        try
            while selected isa AbstractArray && child isa AbstractPPL.Index
                head = AbstractPPL.concretize_top_level(child, selected)
                selected = getindex(selected, head.ix...; head.kw...)
                child = child.child
            end
        catch err
            err isa BoundsError || rethrow()
            _throw_binding_error(vn, "index is outside the preceding selection")
        end
    end
    return _check_nested_selection(optic.child, vn)
end

function _expand_selection(storage, optic::AbstractPPL.Iden, vn, value, node=nothing)
    return ((vn, value),)
end
function _expand_selection(
    storage, optic::AbstractPPL.Property{S}, vn, value, node=nothing
) where {S}
    node isa BoundValue && (storage = node.value)
    storage = _unwrap_storage(storage)
    node isa BoundValue || storage isa Number && (storage = NoTemplate())
    (
        storage isa Union{NoTemplate,VarNamedTuple} ||
        _element_optic(storage) === AbstractPPL.Property
    ) || _throw_unsupported_storage(storage, vn)
    head = AbstractPPL.Property{S}()
    child = VarNamedTuples.SharedGetProperty{S}()(storage)
    _element_optic(storage) === AbstractPPL.Property &&
        !hasproperty(storage, S) &&
        _throw_binding_error(vn, "field is outside retained storage")
    return _expand_selection(
        child,
        optic.child,
        AbstractPPL.append_optic(vn, head),
        value,
        _binding_node_at(node, head),
    )
end
function _selection_indices(storage, optic, vn)
    try
        optic = AbstractPPL.concretize_top_level(optic, storage)
        selected = getindex(_selection_positions(storage), optic.ix...; optic.kw...)
        child = optic.child
        while selected isa AbstractArray && child isa AbstractPPL.Index
            head = AbstractPPL.concretize_top_level(child, selected)
            selected = getindex(selected, head.ix...; head.kw...)
            child = child.child
        end
        return selected, child
    catch err
        err isa BoundsError || rethrow()
        _throw_binding_error(vn, "index is outside retained storage")
    end
end
function _check_selection_shape(value, selected, vn)
    try
        Base.setindex_shape_check(value, size(selected)...)
        checkbounds(value, LinearIndices(size(selected)))
    catch err
        err isa BoundsError && _throw_binding_error(vn, "index is outside retained storage")
        err isa DimensionMismatch || rethrow()
        _throw_binding_error(vn, "selection and value shapes must match")
    end
    return nothing
end
function _expand_selection(storage, optic::AbstractPPL.Index, vn, value, node=nothing)
    node isa BoundValue && (storage = node.value)
    # A node's unset positions hold no storage; their data may be left from a removed binding.
    from_node = node isa _PositionalNode && storage isa NoTemplate
    from_node && (storage = _unwrap_storage(node))
    storage = _unwrap_storage(storage)
    # A removal finds nothing bound below a position that holds neither a binding nor storage.
    value isa NoBinding && node === nothing && storage isa NoTemplate && return ()
    _element_optic(storage) === AbstractPPL.Index ||
        _throw_unsupported_storage(storage, AbstractPPL.append_optic(vn, optic))
    selected, child = _selection_indices(storage, optic, vn)
    if selected isa Union{Integer,CartesianIndex}
        location = AbstractPPL.append_optic(
            vn, AbstractPPL.Index(_index_tuple(selected), (;))
        )
        child_node = _binding_node_at(node, AbstractPPL.Index(_index_tuple(selected), (;)))
        held = _isassigned(storage, selected) && !(from_node && child_node === nothing)
        return _expand_selection(
            held ? storage[selected] : NoTemplate(), child, location, value, child_node
        )
    end
    child isa AbstractPPL.Iden ||
        _throw_binding_error(vn, "field access on an array selection is unsupported")
    if !(value isa NoBinding)
        if storage isa Tuple
            value isa Union{Tuple,AbstractArray} ||
                _throw_binding_error(vn, "selection requires a tuple or array value")
            length(value) == length(selected) ||
                _throw_binding_error(vn, "selection and value lengths must match")
        else
            value isa AbstractArray ||
                _throw_binding_error(vn, "selection requires an array value")
            _check_selection_shape(value, selected, vn)
        end
    end
    # Pair positions in linear order, as `setindex!` does: `[1.0, 2.0]` fills `x[1:1, 1:2]`.
    return map(eachindex(selected), LinearIndices(size(selected))) do i, k
        location = AbstractPPL.append_optic(
            vn, AbstractPPL.Index(_index_tuple(selected[i]), (;))
        )
        (location, value isa NoBinding ? value : value[k])
    end
end

# A range has a `Vector`'s shape; with `claimed` it differs, so `_latent_source` keeps
# storage that can hold latent values.
function _same_shape(x, y, claimed=false, seen=Set{Tuple{UInt,UInt}}())
    a, b = _unwrap_storage(x), _unwrap_storage(y)
    pair = (objectid(a), objectid(b))
    pair in seen && return true
    push!(seen, pair)
    (a isa NoTemplate || b isa NoTemplate) && return true
    function positional(v)
        return _element_optic(v) === AbstractPPL.Index || !claimed && v isa AbstractArray
    end
    if positional(a) || positional(b)
        positional(a) && positional(b) || return false
        axes(a) == axes(b) || return false
        return eltype(a) <: Union{Missing,Number} && eltype(b) <: Union{Missing,Number} ||
               all(
            ((i, j),) ->
                !_isassigned(a, i) ||
                    !_isassigned(b, j) ||
                    _same_shape(a[i], b[j], claimed, seen),
            zip(_positional_indices(a), _positional_indices(b)),
        )
    elseif a isa VarNamedTuple ||
        b isa VarNamedTuple ||
        _element_optic(a) === AbstractPPL.Property ||
        _element_optic(b) === AbstractPPL.Property
        loose = a isa VarNamedTuple || b isa VarNamedTuple
        p = a isa VarNamedTuple ? a.data : a
        q = b isa VarNamedTuple ? b.data : b
        _element_optic(p) === _element_optic(q) === AbstractPPL.Property || return false
        loose || propertynames(p) == propertynames(q) || return false
        return all(
            n -> _same_shape(getproperty(p, n), getproperty(q, n), claimed, seen),
            intersect(propertynames(p), propertynames(q)),
        )
    end
    return true
end
function _matches_local_shape(a, b)
    a, b = _unwrap_storage(a), _unwrap_storage(b)
    if _element_optic(a) === AbstractPPL.Property || a isa Base.Pairs
        (_element_optic(b) === AbstractPPL.Property || b isa Base.Pairs) || return false
        names = a isa Base.Pairs ? keys(a) : propertynames(a)
        return all(n -> b isa Base.Pairs ? haskey(b, n) : hasproperty(b, n), names)
    elseif a isa Tuple || b isa Tuple
        # An `of(Array, n)` template gives a tuple variable only its length.
        return b isa Tuple && a isa Union{Tuple,AbstractVector} && axes(a) == axes(b)
    elseif a isa AbstractArray || b isa AbstractArray
        return a isa AbstractArray && b isa AbstractArray && axes(a) == axes(b)
    end
    return true
end
_convert_exact(::Type{T}, value::T, vn) where {T} = value
function _convert_exact(::Type{T}, value, vn) where {T}
    element_type = T <: Union{AbstractArray,Tuple} ? eltype(T) : T
    # Must throw before `convert`, which raises a bare `ErrorException` for these types.
    element_type <: Union{Missing,Nothing} && _throw_binding_error(
        vn,
        "storage element type $element_type cannot hold $(repr(value)); use an `Array` with an element type such as `Union{Missing,Float64}`, or bind the whole value",
    )
    try
        converted = convert(T, value)
        isequal(converted, value) && return converted
    catch err
        err isa MethodError &&
            _throw_binding_error(vn, "storage type $T cannot hold $(repr(value))")
        err isa InexactError || rethrow()
    end
    return _throw_binding_error(
        vn,
        "conversion of $(repr(value)) into storage element type $element_type is inexact",
    )
end
function _convert_to_storage(value, storage, vn)
    storage = _unwrap_storage(storage)
    storage isa NoTemplate && return value
    _same_shape(storage, value) ||
        _throw_binding_error(vn, "value disagrees with retained shape")
    if _element_optic(storage) === AbstractPPL.Property &&
        _element_optic(value) === AbstractPPL.Property
        names = propertynames(storage)
        fields = map(
            n -> _convert_to_storage(getproperty(value, n), getproperty(storage, n), vn),
            names,
        )
        return ConstructionBase.setproperties(value, NamedTuple{names}(fields))
    end
    return _convert_exact(typeof(storage), value, vn)
end
function _check_partial_storage(value, optic, vn)
    optic isa AbstractPPL.Iden && return nothing
    value isa Union{Missing,Nothing} && _throw_missing_storage(vn)
    _element_optic(value) !== nothing || _throw_unsupported_storage(value, vn)
    head = AbstractPPL.ohead(optic)
    head isa AbstractPPL.Index && (head = AbstractPPL.concretize_top_level(head, value))
    head isa AbstractPPL.Index && !_positional_index(head) ||
        VarNamedTuples._haskey_optic(value, head) ||
        return nothing
    _check_partial_storage(_binding_node_at(value, head), optic.child, vn)
    return nothing
end
_position_type(storage, optic) = eltype(storage)
# Tuple positions keep their own types.
function _position_type(storage::Tuple, optic)
    return fieldtype(typeof(storage), only(VarNamedTuples._tuple_index(optic, storage).ix))
end
function _convert_partial(value, storage, optic, vn)
    optic isa AbstractPPL.Iden && return _convert_to_storage(value, storage, vn)
    head = AbstractPPL.ohead(optic)
    storage = _unwrap_storage(storage)
    storage isa NoTemplate && return value
    if _element_optic(storage) === AbstractPPL.Index && optic.child isa AbstractPPL.Iden
        # Nothing converts into `Array{Missing}`, whose copy `_argument_buffer` widens, or
        # into a tuple slot of unknown type (`NoTemplate`).
        T = _position_type(storage, head)
        converted = if storage isa Array{Missing} || T === NoTemplate
            value
        else
            _convert_exact(T, value, vn)
        end
        previous = _binding_node_at(storage, head)
        previous === nothing ||
            _same_shape(previous, converted) ||
            _throw_binding_error(vn, "value disagrees with retained shape")
        return converted
    end
    child = _binding_node_at(storage, head)
    return child === nothing ? value : _convert_partial(value, child, optic.child, vn)
end

function _check_role_overlap(previous, incoming::BoundValue, vn; disjoint=false)
    previous === nothing && return nothing
    disjoint && previous isa BoundValue{ArgumentRole} && return nothing
    # A placeholder argument entry observes nothing; storage conversion reports it.
    previous isa BoundValue{ArgumentRole,<:Union{Missing,Nothing}} && return nothing
    if previous isa BoundValue
        disjoint ||
            typeof(_binding_role(previous)) === typeof(_binding_role(incoming)) ||
            _throw_binding_error(vn, "observed and fixed positions overlap")
        disjoint && _throw_binding_error(vn, "parent and child bindings overlap")
    elseif previous isa _InnerNode
        VarNamedTuples._mapreduce_recursive(
            identity,
            (a, p) -> (_check_role_overlap(last(p), incoming, first(p); disjoint); nothing),
            previous,
            vn,
            nothing,
        )
    end
    return nothing
end

function _check_role_overlap(previous, incoming::VarNamedTuple, vn; disjoint=false)
    for name in keys(incoming.data)
        optic = AbstractPPL.Property{name}()
        _check_role_overlap(
            _binding_node_at(previous, optic),
            incoming.data[name],
            AbstractPPL.append_optic(vn, optic);
            disjoint,
        )
    end
    return nothing
end
function _check_role_overlap(previous, incoming::_PositionalNode, vn; disjoint=false)
    for (i, child) in VarNamedTuples._set_entries(incoming)
        optic = AbstractPPL.Index(i, (;))
        _check_role_overlap(
            _binding_node_at(previous, optic),
            child,
            AbstractPPL.append_optic(vn, optic);
            disjoint,
        )
    end
    return nothing
end
function _delete_role_at(x, ::AbstractPPL.Iden, R, vn)
    return _matches_role(R, x) ? NoBinding() : x
end
function _delete_role_at(x::_PositionalNode, ::AbstractPPL.Iden, R, vn)
    result = copy(x)
    for (i, child) in VarNamedTuples._set_entries(x)
        child = _delete_role_at(child, AbstractPPL.Iden(), R, vn)
        optic = AbstractPPL.Index(i, (;))
        result = if child isa NoBinding
            _delete_position(result, optic)
        else
            VarNamedTuples._setindex_optic!!(
                result, child, optic, NoTemplate(), VarNamedTuples.AllowAll()
            )
        end
    end
    return _has_bound_value(result) ? result : NoBinding()
end
function _delete_position(x::VarNamedTuples.PartialArray, optic)
    return BangBang.delete!!(x, optic.ix...; optic.kw...)
end
function _delete_position(x::VarNamedTuples.PartialTuple, optic)
    i = only(VarNamedTuples._tuple_index(optic, x.data).ix)
    return VarNamedTuples.PartialTuple(Base.setindex(x.data, typeof(x.data[i])(), i))
end
function _delete_role_at(x::VarNamedTuple, ::AbstractPPL.Iden, R, vn)
    result = VarNamedTuple(map(v -> _delete_role_at(v, AbstractPPL.Iden(), R, vn), x.data))
    for name in keys(result.data)
        result.data[name] isa NoBinding || continue
        result = VarNamedTuple(Base.structdiff(result.data, NamedTuple{(name,)}))
    end
    return isempty(result) ? NoBinding() : result
end
function _delete_role_at(x, optic::AbstractPPL.AbstractOptic, R, vn; prune=true)
    x isa Union{Nothing,NoBinding} && return NoBinding()
    x isa BoundValue && (x = _split_bound_value(x, vn))
    head = AbstractPPL.ohead(optic)
    child = _binding_node_at(x, head)
    child === nothing && return x
    replacement = _delete_role_at(child, optic.child, R, vn)
    result = if replacement isa NoBinding && x isa VarNamedTuple
        name = AbstractPPL.getsym(head)
        VarNamedTuple(Base.structdiff(x.data, NamedTuple{(name,)}))
    elseif replacement isa NoBinding && x isa _PositionalNode
        _delete_position(x, head)
    else
        VarNamedTuples._setindex_optic!!(
            x, replacement, head, _unwrap_storage(x), VarNamedTuples.AllowAll()
        )
    end
    return !prune || _has_bound_value(result) ? result : NoBinding()
end

function _collapse_fields(node, constraints)
    for constraint in reverse(constraints)
        (; owner, storage) = constraint
        R = constraint.role
        constraint.mode === Val(:shape) || continue
        _element_optic(storage) === AbstractPPL.Property || continue
        selected = _binding_node_at(node, AbstractPPL.getoptic(owner))
        selected isa VarNamedTuple && _is_fully_bound(selected, storage) || continue
        _has_role(ArgumentRole, selected) && continue
        _has_role(R === FixRole ? ConditionRole : FixRole, selected) && continue
        value = _bound_data(selected, storage)
        node = VarNamedTuples._setindex_optic!!(
            node,
            BoundValue{R}(value),
            AbstractPPL.getoptic(owner),
            NoTemplate(),
            VarNamedTuples.AllowAll(),
        )
    end
    return node
end

function VarNamedTuples.make_leaf(value::BoundValue, optic::AbstractPPL.Property, ::Number)
    return VarNamedTuples.make_leaf(value, optic, NoTemplate())
end

_role_values(x, R) = NoBinding()
_role_values(x::BoundValue, R) = _matches_role(R, x) ? x.value : NoBinding()
function _role_values(x::VarNamedTuple, R)
    fields = map(v -> _role_values(v, R), x.data)
    for name in keys(fields)
        fields[name] isa NoBinding || continue
        fields = Base.structdiff(fields, NamedTuple{(name,)})
    end
    return isempty(fields) ? NoBinding() : VarNamedTuple(fields)
end
function _role_values(x::VarNamedTuples.PartialArray, R)
    data = similar(x.data, Union{})
    mask = falses(size(x))
    result = VarNamedTuples.PartialArray(data, mask)
    for i in CartesianIndices(x.mask)
        x.mask[i] || continue
        value = _role_values(x.data[i], R)
        value isa NoBinding || (result = setindex!!(result, value, i))
    end
    return any(result.mask) ? result : NoBinding()
end
function _role_values(x::VarNamedTuples.PartialTuple, R)
    slots = map(_unwrap_storage(x)) do child
        value = _role_values(child, R)
        return if value isa NoBinding
            VarNamedTuples.TupleSlot{Any}()
        else
            VarNamedTuples.TupleSlot(value)
        end
    end
    return if any(VarNamedTuples._isset, slots)
        VarNamedTuples.PartialTuple(slots)
    else
        NoBinding()
    end
end
function _equality_key(x::_InnerNode)
    entries = Dict(
        first(p) => _equality_key(last(p)) for
        p in mapreduce(identity, push!, VarNamedTuple((; root=x)); init=Pair[]) if
        !(last(p) isa NoBinding)
    )
    return isempty(entries) ? nothing : entries
end
_equality_key(x::BoundValue) = (typeof(_binding_role(x)), x.value)
_equality_key(::NoBinding) = nothing
function _equality_key(table::BindingTable)
    result = Dict()
    for (name, root) in pairs(table.roots.data)
        value = _equality_key(root.value)
        storage = _shape_signature(root.storage)
        constraints = Set(
            (c.owner, c.role, _shape_signature(c.storage, c.mode === Val(:type)), c.mode)
            for c in root.constraints
        )
        value === nothing && storage === nothing && isempty(constraints) && continue
        result[name] = (value, storage, constraints)
    end
    return result
end
function Base.isequal(a::BindingTable, b::BindingTable)
    return isequal(_equality_key(a), _equality_key(b))
end
Base.:(==)(a::BindingTable, b::BindingTable) = isequal(a, b)
Base.hash(x::BindingTable, h::UInt) = hash(_equality_key(x), h)

function _shape_signature(x, typed=false)
    if _element_optic(x) === AbstractPPL.Index
        children = if eltype(x) <: Number
            nothing
        else
            map(
                i -> _isassigned(x, i) ? _shape_signature(x[i], typed) : nothing,
                _positional_indices(x),
            )
        end
        return (axes(x), typed ? eltype(x) : nothing, children)
    elseif _element_optic(x) === AbstractPPL.Property
        names = propertynames(x)
        return NamedTuple{names}(
            map(n -> _shape_signature(getproperty(x, n), typed), names)
        )
    end
    return typed ? typeof(x) : nothing
end
_shape_signature(::NoTemplate, typed=false) = nothing
function _shape_signature(x::_PositionalNode, typed=false)
    return _shape_signature(_unwrap_storage(x), typed)
end
function _shape_signature(x::VarNamedTuple, typed=false)
    values = Dict(
        n => _shape_signature(v, typed) for
        (n, v) in pairs(x.data) if _shape_signature(v, typed) !== nothing
    )
    return isempty(values) ? nothing : values
end

_is_fully_bound(x, storage) = x isa BoundValue
function _is_fully_bound(x::_PositionalNode, storage)
    _is_complete(x) || return false
    data = _unwrap_storage(x)
    return all(
        i -> _is_fully_bound(
            data[i],
            if _element_optic(storage) === AbstractPPL.Index && _isassigned(storage, i)
                storage[i]
            else
                NoTemplate()
            end,
        ),
        eachindex(data),
    )
end
function _is_fully_bound(x::VarNamedTuple, storage)
    _element_optic(storage) === AbstractPPL.Property &&
        issetequal(keys(x.data), propertynames(storage)) || return false
    return all(
        n -> _is_fully_bound(x.data[n], getproperty(storage, n)), propertynames(storage)
    )
end
