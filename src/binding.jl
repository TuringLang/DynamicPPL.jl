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
struct BindingConstraint{V,R,S,T}
    owner::V
    role::Type{R}
    storage::S
    typed::Val{T}
end
Base.@constprop :aggressive function BindingConstraint(owner, role, storage, typed::Bool)
    return BindingConstraint(owner, role, storage, Val(typed))
end
struct BindingRoot{V,S,C}
    value::V
    storage::S
    constraints::C
end
function _effective_constraints(root, base)
    return if root.value isa BoundValue{ArgumentRole}
        (BindingConstraint(base, ConditionRole, root.value.value, false),)
    else
        root.constraints
    end
end
struct BindingTable{V<:VarNamedTuple}
    roots::V
end
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
    _binding_optic(x)

Choose how binding-managed storage holds and rebuilds children without mutating
the caller's storage. Return `AbstractPPL.Index` for position-addressed containers
rebuilt with preserved axes, `AbstractPPL.Property` for field-addressed
containers rebuilt with `ConstructionBase.setproperties`, or `nothing` for
unclaimed values. Unclaimed values are shared as leaves, except arrays, tuples and
dictionaries, which are refused where storage is required. This method never
decides whether a value can be bound whole; any non-placeholder value can.
"""
_binding_optic(x) = nothing
_binding_optic(::Array) = AbstractPPL.Index
_binding_optic(::Tuple{T,Vararg{T}}) where {T} = AbstractPPL.Index
_binding_optic(::Tuple{}) = AbstractPPL.Index
_binding_optic(::NamedTuple) = AbstractPPL.Property
_rebuild_buffer(x::AbstractArray) = similar(x)
_rebuild_buffer(x::Tuple) = x
_rebuild_buffer(x::Tuple, ::Type{T}) where {T} = Vector{T}(undef, length(x))
_rebuild_buffer(x, ::Type{T}) where {T} = similar(x, T)
_positional_indices(x::AbstractArray) = CartesianIndices(x)
_positional_indices(x::Tuple) = keys(x)
_index_tuple(i::CartesianIndex) = Tuple(i)
_index_tuple(i::Integer) = (i,)
_isassigned(x, indices...) = isassigned(x, indices...)
_isassigned(x::Tuple, i) = i in keys(x)

_unwrap_storage(x) = x
_unwrap_storage(x::VarNamedTuples.PartialArray) = x.data
_unwrap_storage(x::BoundValue) = x.value
_unwrap_storage(::Union{Missing,Nothing}) = NoTemplate()
function VarNamedTuples.maybe_index_template(::BoundValue, template, optic)
    return VarNamedTuples.maybe_index_template(template, optic)
end

function VarNamedTuples._getindex_optic(x::BoundValue, optic::AbstractPPL.AbstractOptic, vn)
    return _rewrap_role(x, VarNamedTuples._getindex_optic(x.value, optic, vn))
end
VarNamedTuples._getindex_optic(x::BoundValue, ::AbstractPPL.Iden, vn) = x
function VarNamedTuples._haskey_optic(x::BoundValue, optic::AbstractPPL.AbstractOptic)
    return VarNamedTuples._haskey_optic(x.value, optic)
end
VarNamedTuples._haskey_optic(::BoundValue, ::AbstractPPL.Iden) = true

function _split_bound_value(x::BoundValue{R}, vn) where {R}
    value = x.value
    if _binding_optic(value) === AbstractPPL.Property
        names = propertynames(value)
        return VarNamedTuple(
            NamedTuple{names}(map(n -> BoundValue{R}(getproperty(value, n)), names))
        )
    end
    _binding_optic(value) === AbstractPPL.Index || _throw_unsupported_storage(value, vn)
    data = _rebuild_buffer(
        value, isconcretetype(eltype(value)) ? BoundValue{R,eltype(value)} : BoundValue{R}
    )
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
function _binding_node_at(x::AbstractArray, optic::AbstractPPL.Index)
    head = AbstractPPL.concretize_top_level(AbstractPPL.ohead(optic), x)
    checkbounds(Bool, x, head.ix...) || return nothing
    VarNamedTuples._is_multiindex(x, head.ix...) ||
        _isassigned(x, head.ix...) ||
        return nothing
    return _binding_node_at(getindex(x, head.ix...), optic.child)
end
function _binding_node_at(x::VarNamedTuples.PartialArray, optic::AbstractPPL.Index)
    optic = AbstractPPL.concretize_top_level(optic, x.data)
    checkbounds(Bool, x, optic.ix...) || return nothing
    child = if VarNamedTuples._is_multiindex(x.data, optic.ix...; optic.kw...)
        VarNamedTuples._subset_partialarray(x, optic.ix...; optic.kw...)
    elseif x.mask[optic.ix...]
        x.data[optic.ix...]
    else
        nothing
    end
    return _binding_node_at(child, optic.child)
end
function _binding_node_at(
    x::BoundValue{R,<:AbstractArray}, optic::AbstractPPL.Index
) where {R}
    array = x.value
    optic = AbstractPPL.concretize_top_level(optic, array)
    checkbounds(Bool, array, optic.ix...) || return nothing
    if VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...)
        if _binding_optic(array) === AbstractPPL.Index &&
            !all(i -> _isassigned(array, i), _positional_indices(array))
            return _binding_node_at(_split_bound_value(x, @varname(_)), optic)
        end
    elseif !_isassigned(array, optic.ix...)
        return nothing
    end
    return _binding_node_at(
        BoundValue{R}(getindex(array, optic.ix...; optic.kw...)), optic.child
    )
end
_binding_role(::Union{Nothing,NoBinding}, vn) = nothing
_binding_role(x::BoundValue, vn) = _binding_role(x)
function _binding_role(x::VarNamedTuple, vn)
    isempty(x) && return nothing
    return _throw_binding_error(vn, "a tilde cannot mix bound and latent fields")
end
function _binding_role(x::VarNamedTuples.PartialArray, vn)
    found = nothing
    for i in eachindex(x.mask)
        x.mask[i] || continue
        r = _binding_role(x.data[i], vn)
        r === nothing && continue
        found === nothing ||
            typeof(found) === typeof(r) ||
            _throw_binding_error(vn, "a tilde cannot mix binding roles")
        found = r
    end
    found === nothing && return nothing
    all(x.mask) || _throw_binding_error(vn, "a tilde cannot mix bound and latent positions")
    return found
end
_bound_data(x::BoundValue) = x.value
_bound_data(x::VarNamedTuple) = map(_bound_data, x.data)
function _bound_data(x::VarNamedTuples.PartialArray)
    all(x.mask) || throw(ArgumentError("Incomplete binding"))
    return map(_bound_data, x.data)
end
_bound_data(x, storage) = _bound_data(x)
function _bound_data(x::VarNamedTuple, storage)
    _binding_optic(storage) === AbstractPPL.Property || return _bound_data(x)
    names = propertynames(storage)
    fields = map(names) do name
        _bound_data(x.data[name], getproperty(storage, name))
    end
    return ConstructionBase.setproperties(storage, NamedTuple{names}(fields))
end
function _bound_data(x::VarNamedTuples.PartialArray, storage::Union{AbstractArray,Tuple})
    all(x.mask) || throw(ArgumentError("Incomplete binding"))
    @assert axes(storage) == axes(x.data)
    result = _rebuild_buffer(storage)
    for i in _positional_indices(storage)
        result = _setindex!!(
            result,
            _bound_data(x.data[i], _isassigned(storage, i) ? storage[i] : NoTemplate()),
            _index_tuple(i)...,
        )
    end
    return result
end
function _has_role(R, x::Union{VarNamedTuple,VarNamedTuples.PartialArray})
    return VarNamedTuples._mapreduce_recursive(
        p -> _has_role(R, last(p)), |, x, @varname(_), false
    )
end
_has_role(R, x) = _matches_role(R, x)
function _has_bound_value(x::Union{VarNamedTuple,VarNamedTuples.PartialArray})
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
        _binding_optic(storage) === AbstractPPL.Property
    ) || _throw_unsupported_storage(storage, vn)
    head = AbstractPPL.Property{S}()
    child = VarNamedTuples.SharedGetProperty{S}()(storage)
    _binding_optic(storage) === AbstractPPL.Property &&
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
function _expand_selection(storage, optic::AbstractPPL.Index, vn, value, node=nothing)
    try
        node isa BoundValue && (storage = node.value)
        node isa VarNamedTuples.PartialArray &&
            storage isa NoTemplate &&
            (storage = node.data)
        storage = _unwrap_storage(storage)
        _binding_optic(storage) === AbstractPPL.Index ||
            _throw_unsupported_storage(storage, AbstractPPL.append_optic(vn, optic))
        optic = AbstractPPL.concretize_top_level(optic, storage)
        selected = getindex(_positional_indices(storage), optic.ix...; optic.kw...)
        child = optic.child
        while selected isa AbstractArray && child isa AbstractPPL.Index
            head = AbstractPPL.concretize_top_level(child, selected)
            selected = getindex(selected, head.ix...; head.kw...)
            child = child.child
        end
        if selected isa Union{Integer,CartesianIndex}
            location = AbstractPPL.append_optic(
                vn, AbstractPPL.Index(_index_tuple(selected), (;))
            )
            return _expand_selection(
                _isassigned(storage, selected) ? storage[selected] : NoTemplate(),
                child,
                location,
                value,
                _binding_node_at(node, AbstractPPL.Index(_index_tuple(selected), (;))),
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
                try
                    Base.setindex_shape_check(value, size(selected)...)
                catch err
                    err isa DimensionMismatch || rethrow()
                    _throw_binding_error(vn, "selection and value shapes must match")
                end
            end
        end
        return map(eachindex(selected)) do i
            location = AbstractPPL.append_optic(
                vn, AbstractPPL.Index(_index_tuple(selected[i]), (;))
            )
            (location, value isa NoBinding ? value : value[i])
        end
    catch err
        err isa BoundsError || rethrow()
        _throw_binding_error(vn, "index is outside retained storage")
    end
end

function _same_shape(a, b, seen=Set{Tuple{UInt,UInt}}())
    a, b = _unwrap_storage(a), _unwrap_storage(b)
    pair = (objectid(a), objectid(b))
    pair in seen && return true
    push!(seen, pair)
    (a isa NoTemplate || b isa NoTemplate) && return true
    if _binding_optic(a) === AbstractPPL.Index || _binding_optic(b) === AbstractPPL.Index
        _binding_optic(a) === AbstractPPL.Index &&
            _binding_optic(b) === AbstractPPL.Index || return false
        axes(a) == axes(b) || return false
        return eltype(a) <: Number && eltype(b) <: Number || all(
            ((i, j),) ->
                !_isassigned(a, i) || !_isassigned(b, j) || _same_shape(a[i], b[j], seen),
            zip(_positional_indices(a), _positional_indices(b)),
        )
    elseif a isa VarNamedTuple ||
        b isa VarNamedTuple ||
        _binding_optic(a) === AbstractPPL.Property ||
        _binding_optic(b) === AbstractPPL.Property
        loose = a isa VarNamedTuple || b isa VarNamedTuple
        a = a isa VarNamedTuple ? a.data : a
        b = b isa VarNamedTuple ? b.data : b
        _binding_optic(a) === _binding_optic(b) === AbstractPPL.Property || return false
        loose || propertynames(a) == propertynames(b) || return false
        return all(
            n -> _same_shape(getproperty(a, n), getproperty(b, n), seen),
            intersect(propertynames(a), propertynames(b)),
        )
    end
    return true
end
function _matches_local_shape(a, b)
    a, b = _unwrap_storage(a), _unwrap_storage(b)
    if _binding_optic(a) === AbstractPPL.Property || a isa Base.Pairs
        (_binding_optic(b) === AbstractPPL.Property || b isa Base.Pairs) || return false
        names = a isa Base.Pairs ? keys(a) : propertynames(a)
        return all(n -> b isa Base.Pairs ? haskey(b, n) : hasproperty(b, n), names)
    elseif a isa Tuple
        return b isa Tuple && length(a) == length(b)
    elseif a isa AbstractArray || b isa AbstractArray
        return a isa AbstractArray && b isa AbstractArray && axes(a) == axes(b)
    end
    return true
end
function _convert_exact(::Type{T}, value, vn) where {T}
    try
        converted = convert(T, value)
        isequal(converted, value) && return converted
    catch err
        err isa InexactError || rethrow()
    end
    element_type = T <: Union{AbstractArray,Tuple} ? eltype(T) : T
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
    if _binding_optic(storage) === AbstractPPL.Property &&
        _binding_optic(value) === AbstractPPL.Property
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
    _binding_optic(value) !== nothing || _throw_unsupported_storage(value, vn)
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(value, head) || return nothing
    _check_partial_storage(_binding_node_at(value, head), optic.child, vn)
    return nothing
end
function _convert_partial(value, storage, optic, vn)
    optic isa AbstractPPL.Iden && return _convert_to_storage(value, storage, vn)
    head = AbstractPPL.ohead(optic)
    storage = _unwrap_storage(storage)
    storage isa NoTemplate && return value
    if _binding_optic(storage) === AbstractPPL.Index && optic.child isa AbstractPPL.Iden
        converted = _convert_exact(eltype(storage), value, vn)
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
    if previous isa BoundValue
        disjoint ||
            typeof(_binding_role(previous)) === typeof(_binding_role(incoming)) ||
            _throw_binding_error(vn, "observed and fixed positions overlap")
        disjoint && _throw_binding_error(vn, "parent and child bindings overlap")
    elseif previous isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
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
function _check_role_overlap(
    previous, incoming::VarNamedTuples.PartialArray, vn; disjoint=false
)
    for i in CartesianIndices(incoming.mask)
        incoming.mask[i] || continue
        optic = AbstractPPL.Index(Tuple(i), (;))
        _check_role_overlap(
            _binding_node_at(previous, optic),
            incoming.data[i],
            AbstractPPL.append_optic(vn, optic);
            disjoint,
        )
    end
    return nothing
end
function _delete_role_at(x, ::AbstractPPL.Iden, R, vn)
    return _matches_role(R, x) ? NoBinding() : x
end
function _delete_role_at(x::VarNamedTuples.PartialArray, ::AbstractPPL.Iden, R, vn)
    result = copy(x)
    for i in CartesianIndices(x.mask)
        x.mask[i] || continue
        child = _delete_role_at(x.data[i], AbstractPPL.Iden(), R, vn)
        result = if child isa NoBinding
            BangBang.delete!!(result, i)
        else
            setindex!!(result, child, i)
        end
    end
    return _has_bound_value(result) ? result : NoBinding()
end
function _delete_role_at(x::VarNamedTuple, ::AbstractPPL.Iden, R, vn)
    result = VarNamedTuple(map(v -> _delete_role_at(v, AbstractPPL.Iden(), R, vn), x.data))
    for name in keys(result.data)
        result.data[name] isa NoBinding || continue
        result = VarNamedTuple(Base.structdiff(result.data, NamedTuple{(name,)}))
    end
    return isempty(result) ? NoBinding() : result
end
function _delete_role_at(x, optic::AbstractPPL.AbstractOptic, R, vn)
    x isa Union{Nothing,NoBinding} && return NoBinding()
    x isa BoundValue && (x = _split_bound_value(x, vn))
    head = AbstractPPL.ohead(optic)
    child = _binding_node_at(x, head)
    child === nothing && return x
    replacement = _delete_role_at(child, optic.child, R, vn)
    result = if replacement isa NoBinding && x isa VarNamedTuple
        name = AbstractPPL.getsym(head)
        VarNamedTuple(Base.structdiff(x.data, NamedTuple{(name,)}))
    elseif replacement isa NoBinding && x isa VarNamedTuples.PartialArray
        BangBang.delete!!(x, head.ix...; head.kw...)
    else
        VarNamedTuples._setindex_optic!!(
            x, replacement, head, _unwrap_storage(x), VarNamedTuples.AllowAll()
        )
    end
    return _has_bound_value(result) ? result : NoBinding()
end

function _collapse_fields(node, constraints)
    for constraint in reverse(constraints)
        (; owner, storage) = constraint
        R = constraint.role
        constraint.typed === Val(true) && continue
        _binding_optic(storage) === AbstractPPL.Property || continue
        selected = _binding_node_at(node, AbstractPPL.getoptic(owner))
        selected isa VarNamedTuple && _is_fully_bound(selected, storage) || continue
        _has_role(R === FixRole ? ConditionRole : FixRole, selected) && continue
        R in (ConditionRole, FixRole) || continue
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
    return result
end
function _equality_key(x::Union{VarNamedTuple,VarNamedTuples.PartialArray})
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
            (c.owner, c.role, _shape_signature(c.storage, c.typed === Val(true)), c.typed)
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
    if _binding_optic(x) === AbstractPPL.Index
        children = if eltype(x) <: Number
            nothing
        else
            map(
                i -> _isassigned(x, i) ? _shape_signature(x[i], typed) : nothing,
                _positional_indices(x),
            )
        end
        return (axes(x), typed ? eltype(x) : nothing, children)
    elseif _binding_optic(x) === AbstractPPL.Property
        names = propertynames(x)
        return NamedTuple{names}(
            map(n -> _shape_signature(getproperty(x, n), typed), names)
        )
    end
    return typed ? typeof(x) : nothing
end
_shape_signature(::NoTemplate, typed=false) = nothing
function _shape_signature(x::VarNamedTuple, typed=false)
    values = Dict(
        n => _shape_signature(v, typed) for
        (n, v) in pairs(x.data) if _shape_signature(v, typed) !== nothing
    )
    return isempty(values) ? nothing : values
end

_is_fully_bound(x, storage) = x isa BoundValue
function _is_fully_bound(x::VarNamedTuples.PartialArray, storage)
    all(x.mask) || return false
    return all(
        i -> _is_fully_bound(
            x.data[i],
            if _binding_optic(storage) === AbstractPPL.Index && _isassigned(storage, i)
                storage[i]
            else
                NoTemplate()
            end,
        ),
        eachindex(x.mask),
    )
end
function _is_fully_bound(x::VarNamedTuple, storage)
    _binding_optic(storage) === AbstractPPL.Property &&
        issetequal(keys(x.data), propertynames(storage)) || return false
    return all(
        n -> _is_fully_bound(x.data[n], getproperty(storage, n)), propertynames(storage)
    )
end
