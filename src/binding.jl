struct Condition end
struct ArgumentCondition end
struct Fix end
struct ModelValue{R,T}
    value::T
    function ModelValue{R}(value::T) where {R,T}
        R in (Condition, ArgumentCondition, Fix) ||
            throw(ArgumentError("Invalid binding role"))
        return new{R,T}(value)
    end
end
struct NoModelBinding end
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
function _root_constraints(root, base)
    return if root.value isa ModelValue{ArgumentCondition}
        (BindingConstraint(base, Condition, root.value.value, false),)
    else
        root.constraints
    end
end
struct ResolvedBindings{V<:VarNamedTuple}
    roots::V
end
maybe_prefix(vn, ::Nothing) = vn
maybe_prefix(vn, prefix::VarName) = AbstractPPL.prefix(vn, prefix)
maybe_prefix(::Nothing, prefix::VarName) = prefix
_binding_error(vn, rule) = throw(ArgumentError("`$vn`: $rule"))
function _missing_argument_error(vn)
    return _binding_error(
        vn,
        "partial argument binding lacks storage; bind the whole value or pass a concrete argument (for example one with missing entries)",
    )
end
const BindingArray = VarNamedTuples.PartialArray
const BindingNodes = Union{VarNamedTuple,BindingArray}
@enum _PlaceholderKind _NoPlaceholder _NothingPlaceholder _MissingPlaceholder

function _classify_placeholder(value, seen=Base.IdSet{Any}())
    value === missing && return _MissingPlaceholder
    value === nothing && return _NothingPlaceholder
    value isa Union{Number,AbstractString,Symbol,Type} && return _NoPlaceholder
    value isa AbstractArray{<:Number} && return _NoPlaceholder
    if ismutabletype(typeof(value))
        value in seen && return _NoPlaceholder
        push!(seen, value)
    end
    children = if value isa AbstractArray
        (value[i] for i in eachindex(value) if isassigned(value, i))
    elseif value isa Union{Tuple,NamedTuple,Base.Pairs}
        values(value)
    else
        (getfield(value, i) for i in 1:fieldcount(typeof(value)) if isdefined(value, i))
    end
    result = _NoPlaceholder
    for child in children
        kind = _classify_placeholder(child, seen)
        kind === _MissingPlaceholder && return kind
        kind === _NothingPlaceholder && (result = kind)
    end
    return result
end
_model_value_like(::ModelValue{R}, value) where {R} = ModelValue{R}(value)
_model_role(::ModelValue{R}) where {R} = R === Fix ? Fix() : Condition()
_matches_model_role(R, x) = x isa ModelValue{R}
_matches_model_role(::Type{Condition}, x::ModelValue{ArgumentCondition}) = true
"""
    _binding_optic(x)

Choose how binding-managed storage holds and rebuilds children without aliasing
the caller's storage. Return `AbstractPPL.Index` for position-addressed containers
with fresh buffers preserving axes, `AbstractPPL.Property` for field-addressed
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

_binding_storage(x) = x
_binding_storage(x::BindingArray) = x.data
_binding_storage(x::ModelValue) = x.value
_binding_storage(::Union{Missing,Nothing}) = NoTemplate()
function VarNamedTuples.maybe_index_template(::ModelValue, template, optic)
    return VarNamedTuples.maybe_index_template(template, optic)
end

function VarNamedTuples._getindex_optic(x::ModelValue, optic::AbstractPPL.AbstractOptic, vn)
    return _model_value_like(x, VarNamedTuples._getindex_optic(x.value, optic, vn))
end
VarNamedTuples._getindex_optic(x::ModelValue, ::AbstractPPL.Iden, vn) = x
function VarNamedTuples._haskey_optic(x::ModelValue, optic::AbstractPPL.AbstractOptic)
    return VarNamedTuples._haskey_optic(x.value, optic)
end
VarNamedTuples._haskey_optic(::ModelValue, ::AbstractPPL.Iden) = true

function _expand_binding(x::ModelValue{R}, vn) where {R}
    value = x.value
    if _binding_optic(value) === AbstractPPL.Property
        names = propertynames(value)
        return VarNamedTuple(
            NamedTuple{names}(map(n -> ModelValue{R}(getproperty(value, n)), names))
        )
    end
    _binding_optic(value) === AbstractPPL.Index || _unsupported_binding_storage(value, vn)
    data = _rebuild_buffer(
        value, isconcretetype(eltype(value)) ? ModelValue{R,eltype(value)} : ModelValue{R}
    )
    mask = fill!(similar(data, Bool), false)
    for i in _positional_indices(value)
        if _isassigned(value, i)
            data[i] = ModelValue{R}(value[i])
            mask[i] = true
        end
    end
    return BindingArray(data, mask)
end
function VarNamedTuples._setindex_optic!!(
    x::ModelValue, value, optic::AbstractPPL.AbstractOptic, template, perms
)
    return VarNamedTuples._setindex_optic!!(
        _expand_binding(x, @varname(_)), value, optic, template, perms
    )
end
function VarNamedTuples._setindex_optic!!(
    ::ModelValue, value, ::AbstractPPL.Iden, template, perms::VarNamedTuples.SetPermissions
)
    return value
end

function _node_at(x, optic::AbstractPPL.AbstractOptic)
    x isa Union{Nothing,NoModelBinding} && return nothing
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(x, head) || return nothing
    return _node_at(VarNamedTuples._getindex_optic(x, head, @varname(_)), optic.child)
end
function _node_at(x::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return haskey(x.data, S) ? _node_at(x.data[S], optic.child) : nothing
end
_node_at(x, ::AbstractPPL.Iden) = x
function _node_at(x::AbstractArray, optic::AbstractPPL.Index)
    head = AbstractPPL.concretize_top_level(AbstractPPL.ohead(optic), x)
    checkbounds(Bool, x, head.ix...) || return nothing
    VarNamedTuples._is_multiindex(x, head.ix...) ||
        _isassigned(x, head.ix...) ||
        return nothing
    return _node_at(getindex(x, head.ix...), optic.child)
end
function _node_at(x::BindingArray, optic::AbstractPPL.Index)
    optic = AbstractPPL.concretize_top_level(optic, x.data)
    checkbounds(Bool, x, optic.ix...) || return nothing
    child = if VarNamedTuples._is_multiindex(x.data, optic.ix...; optic.kw...)
        VarNamedTuples._subset_partialarray(x, optic.ix...; optic.kw...)
    elseif x.mask[optic.ix...]
        x.data[optic.ix...]
    else
        nothing
    end
    return _node_at(child, optic.child)
end
function _node_at(x::ModelValue{R,<:AbstractArray}, optic::AbstractPPL.Index) where {R}
    array = x.value
    optic = AbstractPPL.concretize_top_level(optic, array)
    checkbounds(Bool, array, optic.ix...) || return nothing
    if VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...)
        if _binding_optic(array) === AbstractPPL.Index &&
            !all(i -> _isassigned(array, i), _positional_indices(array))
            return _node_at(_expand_binding(x, @varname(_)), optic)
        end
    elseif !_isassigned(array, optic.ix...)
        return nothing
    end
    return _node_at(ModelValue{R}(getindex(array, optic.ix...; optic.kw...)), optic.child)
end
_model_role(::Union{Nothing,NoModelBinding}, vn) = nothing
_model_role(x::ModelValue, vn) = _model_role(x)
function _model_role(x::VarNamedTuple, vn)
    isempty(x) && return nothing
    return _binding_error(vn, "a tilde cannot mix bound and latent fields")
end
function _model_role(x::BindingArray, vn)
    found = nothing
    for i in eachindex(x.mask)
        x.mask[i] || continue
        r = _model_role(x.data[i], vn)
        r === nothing && continue
        found === nothing ||
            typeof(found) === typeof(r) ||
            _binding_error(vn, "a tilde cannot mix binding roles")
        found = r
    end
    found === nothing && return nothing
    all(x.mask) || _binding_error(vn, "a tilde cannot mix bound and latent positions")
    return found
end
function _model_role_at(x, optic, vn)
    _check_namedtuple_index(x, optic, AbstractPPL.Property{AbstractPPL.getsym(vn)}())
    node = _node_at(x, optic)
    node === nothing &&
        x isa ModelValue{<:Any,<:AbstractArray} &&
        optic isa AbstractPPL.Index &&
        _node_at(x, AbstractPPL.ohead(optic)) === nothing &&
        _binding_error(vn, "executed index is outside bound storage")
    return _model_role(node, vn)
end
_model_data(x::ModelValue) = x.value
_model_data(x::VarNamedTuple) = map(_model_data, x.data)
function _model_data(x::BindingArray)
    all(x.mask) || throw(ArgumentError("Incomplete binding"))
    return map(_model_data, x.data)
end
_model_data(x, storage) = _model_data(x)
function _model_data(x::VarNamedTuple, storage)
    _binding_optic(storage) === AbstractPPL.Property || return _model_data(x)
    names = propertynames(storage)
    fields = map(names) do name
        _model_data(x.data[name], getproperty(storage, name))
    end
    return ConstructionBase.setproperties(storage, NamedTuple{names}(fields))
end
function _model_data(x::BindingArray, storage::Union{AbstractArray,Tuple})
    all(x.mask) || throw(ArgumentError("Incomplete binding"))
    @assert axes(storage) == axes(x.data)
    result = _rebuild_buffer(storage)
    for i in _positional_indices(storage)
        result = _setindex!!(
            result,
            _model_data(x.data[i], _isassigned(storage, i) ? storage[i] : NoTemplate()),
            _index_tuple(i)...,
        )
    end
    return result
end
function _has_binding_role(R, x::BindingNodes)
    return VarNamedTuples._mapreduce_recursive(
        p -> _has_binding_role(R, last(p)), |, x, @varname(_), false
    )
end
_has_binding_role(R, x) = _matches_model_role(R, x)
function _contains_binding(x::BindingNodes)
    return _has_binding_role(Condition, x) || _has_binding_role(Fix, x)
end
_contains_binding(x) = x isa ModelValue

function _check_selection_indices(optic, vn)
    optic isa AbstractPPL.Iden && return nothing
    if optic isa AbstractPPL.Index &&
        optic.child isa AbstractPPL.Index &&
        any(i -> i isa AbstractArray, optic.ix) &&
        all(i -> i isa Union{Integer,AbstractArray}, optic.ix)
        selected = CartesianIndices(VarNamedTuples._selected_index_shape(optic.ix...))
        child = optic.child
        try
            while selected isa AbstractArray && child isa AbstractPPL.Index
                head = AbstractPPL.concretize_top_level(child, selected)
                selected = getindex(selected, head.ix...; head.kw...)
                child = child.child
            end
        catch err
            err isa BoundsError || rethrow()
            _binding_error(vn, "index is outside the preceding selection")
        end
    end
    return _check_selection_indices(optic.child, vn)
end

function _selection(storage, optic::AbstractPPL.Iden, vn, value, node=nothing)
    return ((vn, value),)
end
function _selection(
    storage, optic::AbstractPPL.Property{S}, vn, value, node=nothing
) where {S}
    node isa ModelValue && (storage = node.value)
    storage = _binding_storage(storage)
    node isa ModelValue || storage isa Number && (storage = NoTemplate())
    (
        storage isa Union{NoTemplate,VarNamedTuple} ||
        _binding_optic(storage) === AbstractPPL.Property
    ) || _unsupported_binding_storage(storage, vn)
    head = AbstractPPL.Property{S}()
    child = VarNamedTuples.SharedGetProperty{S}()(storage)
    _binding_optic(storage) === AbstractPPL.Property &&
        !hasproperty(storage, S) &&
        _binding_error(vn, "field is outside retained storage")
    return _selection(
        child, optic.child, AbstractPPL.append_optic(vn, head), value, _node_at(node, head)
    )
end
function _selection(storage, optic::AbstractPPL.Index, vn, value, node=nothing)
    try
        node isa ModelValue && (storage = node.value)
        node isa BindingArray && storage isa NoTemplate && (storage = node.data)
        storage = _binding_storage(storage)
        _binding_optic(storage) === AbstractPPL.Index ||
            _unsupported_binding_storage(storage, AbstractPPL.append_optic(vn, optic))
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
            return _selection(
                _isassigned(storage, selected) ? storage[selected] : NoTemplate(),
                child,
                location,
                value,
                _node_at(node, AbstractPPL.Index(_index_tuple(selected), (;))),
            )
        end
        child isa AbstractPPL.Iden ||
            _binding_error(vn, "field access on an array selection is unsupported")
        if !(value isa NoModelBinding)
            if storage isa Tuple
                value isa Union{Tuple,AbstractArray} ||
                    _binding_error(vn, "selection requires a tuple or array value")
                length(value) == length(selected) ||
                    _binding_error(vn, "selection and value lengths must match")
            else
                value isa AbstractArray ||
                    _binding_error(vn, "selection requires an array value")
                try
                    Base.setindex_shape_check(value, size(selected)...)
                catch err
                    err isa DimensionMismatch || rethrow()
                    _binding_error(vn, "selection and value shapes must match")
                end
            end
        end
        return map(eachindex(selected)) do i
            location = AbstractPPL.append_optic(
                vn, AbstractPPL.Index(_index_tuple(selected[i]), (;))
            )
            (location, value isa NoModelBinding ? value : value[i])
        end
    catch err
        err isa BoundsError || rethrow()
        _binding_error(vn, "index is outside retained storage")
    end
end

function _same_storage_shape(a, b, seen=Set{Tuple{UInt,UInt}}())
    a, b = _binding_storage(a), _binding_storage(b)
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
                !_isassigned(a, i) ||
                    !_isassigned(b, j) ||
                    _same_storage_shape(a[i], b[j], seen),
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
            n -> _same_storage_shape(getproperty(a, n), getproperty(b, n), seen),
            intersect(propertynames(a), propertynames(b)),
        )
    end
    return true
end
function _runtime_shape(a, b)
    a, b = _binding_storage(a), _binding_storage(b)
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
function _convert_binding(value, storage, vn)
    storage = _binding_storage(storage)
    storage isa NoTemplate && return value
    _same_storage_shape(storage, value) ||
        _binding_error(vn, "value disagrees with retained shape")
    if _binding_optic(storage) === AbstractPPL.Property &&
        _binding_optic(value) === AbstractPPL.Property
        names = propertynames(storage)
        fields = map(
            n -> _convert_binding(getproperty(value, n), getproperty(storage, n), vn), names
        )
        return ConstructionBase.setproperties(value, NamedTuple{names}(fields))
    end
    T = typeof(storage)
    converted = convert(T, value)
    isequal(converted, value) ||
        _binding_error(vn, "conversion into retained storage is inexact")
    return converted
end
function _check_partial_storage(value, optic, vn)
    optic isa AbstractPPL.Iden && return nothing
    value isa Union{Missing,Nothing} && _missing_argument_error(vn)
    _binding_optic(value) !== nothing || _unsupported_binding_storage(value, vn)
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(value, head) || return nothing
    _check_partial_storage(_node_at(value, head), optic.child, vn)
    return nothing
end
function _partial_conversion(value, storage, optic, vn)
    optic isa AbstractPPL.Iden && return _convert_binding(value, storage, vn)
    head = AbstractPPL.ohead(optic)
    storage = _binding_storage(storage)
    storage isa NoTemplate && return value
    if _binding_optic(storage) === AbstractPPL.Index && optic.child isa AbstractPPL.Iden
        converted = convert(eltype(storage), value)
        previous = _node_at(storage, head)
        previous === nothing ||
            _same_storage_shape(previous, converted) ||
            _binding_error(vn, "value disagrees with retained shape")
        isequal(converted, value) ||
            _binding_error(vn, "conversion into storage is inexact")
        return converted
    end
    child = _node_at(storage, head)
    return child === nothing ? value : _partial_conversion(value, child, optic.child, vn)
end

function _check_cross_roles(previous, incoming::ModelValue, vn; disjoint=false)
    previous === nothing && return nothing
    if previous isa ModelValue
        disjoint ||
            typeof(_model_role(previous)) === typeof(_model_role(incoming)) ||
            _binding_error(vn, "observed and fixed positions overlap")
        disjoint && _binding_error(vn, "parent and child bindings overlap")
    elseif previous isa BindingNodes
        VarNamedTuples._mapreduce_recursive(
            identity,
            (a, p) -> (_check_cross_roles(last(p), incoming, first(p); disjoint); nothing),
            previous,
            vn,
            nothing,
        )
    end
    return nothing
end

function _check_cross_roles(previous, incoming::VarNamedTuple, vn; disjoint=false)
    for name in keys(incoming.data)
        optic = AbstractPPL.Property{name}()
        _check_cross_roles(
            _node_at(previous, optic),
            incoming.data[name],
            AbstractPPL.append_optic(vn, optic);
            disjoint,
        )
    end
    return nothing
end
function _check_cross_roles(previous, incoming::BindingArray, vn; disjoint=false)
    for i in CartesianIndices(incoming.mask)
        incoming.mask[i] || continue
        optic = AbstractPPL.Index(Tuple(i), (;))
        _check_cross_roles(
            _node_at(previous, optic),
            incoming.data[i],
            AbstractPPL.append_optic(vn, optic);
            disjoint,
        )
    end
    return nothing
end
function _delete_binding(x, ::AbstractPPL.Iden, R, vn)
    return _matches_model_role(R, x) ? NoModelBinding() : x
end
function _delete_binding(x::BindingArray, ::AbstractPPL.Iden, R, vn)
    result = copy(x)
    for i in CartesianIndices(x.mask)
        x.mask[i] || continue
        child = _delete_binding(x.data[i], AbstractPPL.Iden(), R, vn)
        result = if child isa NoModelBinding
            BangBang.delete!!(result, i)
        else
            setindex!!(result, child, i)
        end
    end
    return _contains_binding(result) ? result : NoModelBinding()
end
function _delete_binding(x::VarNamedTuple, ::AbstractPPL.Iden, R, vn)
    result = VarNamedTuple(map(v -> _delete_binding(v, AbstractPPL.Iden(), R, vn), x.data))
    for name in keys(result.data)
        result.data[name] isa NoModelBinding || continue
        result = VarNamedTuple(Base.structdiff(result.data, NamedTuple{(name,)}))
    end
    return isempty(result) ? NoModelBinding() : result
end
function _delete_binding(x, optic::AbstractPPL.AbstractOptic, R, vn)
    x isa Union{Nothing,NoModelBinding} && return NoModelBinding()
    x isa ModelValue && (x = _expand_binding(x, vn))
    head = AbstractPPL.ohead(optic)
    child = _node_at(x, head)
    child === nothing && return x
    replacement = _delete_binding(child, optic.child, R, vn)
    result = if replacement isa NoModelBinding && x isa VarNamedTuple
        name = _property_name(head)
        VarNamedTuple(Base.structdiff(x.data, NamedTuple{(name,)}))
    elseif replacement isa NoModelBinding && x isa BindingArray
        BangBang.delete!!(x, head.ix...; head.kw...)
    else
        VarNamedTuples._setindex_optic!!(
            x, replacement, head, _binding_storage(x), VarNamedTuples.AllowAll()
        )
    end
    return _contains_binding(result) ? result : NoModelBinding()
end
_property_name(::AbstractPPL.Property{S}) where {S} = S

function _collapse_bindings(node, constraints)
    for constraint in reverse(constraints)
        (; owner, storage) = constraint
        R = constraint.role
        constraint.typed === Val(true) && continue
        _binding_optic(storage) === AbstractPPL.Property || continue
        selected = _node_at(node, AbstractPPL.getoptic(owner))
        selected isa VarNamedTuple && _complete_binding(selected, storage) || continue
        _has_binding_role(R === Fix ? Condition : Fix, selected) && continue
        R in (Condition, Fix) || continue
        value = _model_data(selected, storage)
        node = VarNamedTuples._setindex_optic!!(
            node,
            ModelValue{R}(value),
            AbstractPPL.getoptic(owner),
            NoTemplate(),
            VarNamedTuples.AllowAll(),
        )
    end
    return node
end

function VarNamedTuples.make_leaf(value::ModelValue, optic::AbstractPPL.Property, ::Number)
    return VarNamedTuples.make_leaf(value, optic, NoTemplate())
end

_binding_listing(x, R) = NoModelBinding()
_binding_listing(x::ModelValue, R) = _matches_model_role(R, x) ? x.value : NoModelBinding()
function _binding_listing(x::VarNamedTuple, R)
    fields = map(v -> _binding_listing(v, R), x.data)
    for name in keys(fields)
        fields[name] isa NoModelBinding || continue
        fields = Base.structdiff(fields, NamedTuple{(name,)})
    end
    return isempty(fields) ? NoModelBinding() : VarNamedTuple(fields)
end
function _binding_listing(x::BindingArray, R)
    data = similar(x.data, Union{})
    mask = falses(size(x))
    result = BindingArray(data, mask)
    for i in CartesianIndices(x.mask)
        x.mask[i] || continue
        value = _binding_listing(x.data[i], R)
        value isa NoModelBinding || (result = setindex!!(result, value, i))
    end
    return result
end
function _table_listing(table, R)
    result = VarNamedTuple()
    for name in keys(table.roots.data)
        root = table.roots.data[name]
        value = _binding_listing(root.value, R)
        value isa NoModelBinding || (result = setindex!!(result, value, VarName{name}()))
    end
    return result
end
function _binding_content(x::BindingNodes)
    entries = Dict(
        first(p) => _binding_content(last(p)) for
        p in mapreduce(identity, push!, VarNamedTuple((; root=x)); init=Pair[]) if
        !(last(p) isa NoModelBinding)
    )
    return isempty(entries) ? nothing : entries
end
_binding_content(x::ModelValue) = (typeof(_model_role(x)), x.value)
_binding_content(::NoModelBinding) = nothing
function _binding_content(table::ResolvedBindings)
    result = Dict()
    for (name, root) in pairs(table.roots.data)
        value = _binding_content(root.value)
        storage = _storage_signature(root.storage)
        constraints = Set(
            (
                c.owner,
                c.role,
                _storage_signature(c.storage, c.typed === Val(true)),
                c.typed,
            ) for c in root.constraints
        )
        value === nothing && storage === nothing && isempty(constraints) && continue
        result[name] = (value, storage, constraints)
    end
    return result
end
function Base.isequal(a::ResolvedBindings, b::ResolvedBindings)
    return isequal(_binding_content(a), _binding_content(b))
end
Base.:(==)(a::ResolvedBindings, b::ResolvedBindings) = isequal(a, b)
Base.hash(x::ResolvedBindings, h::UInt) = hash(_binding_content(x), h)

function _storage_signature(x, typed=false)
    if _binding_optic(x) === AbstractPPL.Index
        children = if eltype(x) <: Number
            nothing
        else
            map(
                i -> _isassigned(x, i) ? _storage_signature(x[i], typed) : nothing,
                _positional_indices(x),
            )
        end
        return (axes(x), typed ? eltype(x) : nothing, children)
    elseif _binding_optic(x) === AbstractPPL.Property
        names = propertynames(x)
        return NamedTuple{names}(
            map(n -> _storage_signature(getproperty(x, n), typed), names)
        )
    end
    return typed ? typeof(x) : nothing
end
_storage_signature(::NoTemplate, typed=false) = nothing
function _storage_signature(x::VarNamedTuple, typed=false)
    values = Dict(
        n => _storage_signature(v, typed) for
        (n, v) in pairs(x.data) if _storage_signature(v, typed) !== nothing
    )
    return isempty(values) ? nothing : values
end

_complete_binding(x, storage) = x isa ModelValue
function _complete_binding(x::BindingArray, storage)
    all(x.mask) || return false
    return all(
        i -> _complete_binding(
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
function _complete_binding(x::VarNamedTuple, storage)
    _binding_optic(storage) === AbstractPPL.Property &&
        issetequal(keys(x.data), propertynames(storage)) || return false
    return all(
        n -> _complete_binding(x.data[n], getproperty(storage, n)), propertynames(storage)
    )
end
