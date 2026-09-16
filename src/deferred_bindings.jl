struct DeferredWrite{V,G}
    value::V
    group::G
end

struct RemovedBinding end

struct DeferredRemoval{V,O}
    value::V
    optic::O
end

# A NamedTuple below an unresolved child can be a whole variable or a namespace.
# Keep updates ordered until the executing child distinguishes them.
struct DeferredBinding{T<:Tuple{Vararg{Union{DeferredWrite,DeferredRemoval}}}}
    updates::T
end

# Keep field releases relative: a field becomes an independent root only if
# the executing child identifies its parent as a namespace.
struct DeferredShape{T<:Tuple{Vararg{Tuple{Any,Tuple}}}}
    updates::T
end

_deferred_updates(x) = (DeferredWrite(x, nothing),)
_deferred_updates(::NoModelBinding) = ()
_deferred_updates(x::DeferredBinding) = x.updates

_deferred_merge(a, b) = DeferredBinding((_deferred_updates(a)..., _deferred_updates(b)...))

function _merge_model_node(a::DeferredBinding, b)
    return _deferred_merge(a, b)
end
function _merge_model_node(a, b::DeferredBinding)
    return _deferred_merge(a, b)
end
_merge_model_node(a::DeferredBinding, b::DeferredBinding) = _deferred_merge(a, b)
_merge_model_node(a::DeferredBinding, b::NoModelBinding) = a
_merge_model_node(a::DeferredBinding, b::VarNamedTuple) = _deferred_merge(a, b)
function _merge_model_node(a::DeferredBinding, b::VarNamedTuples.PartialArray)
    return _deferred_merge(a, b)
end
_merge_model_node(a::DeferredBinding, b::PendingBindings) = _deferred_merge(a, b)
_merge_model_node(a::PendingBindings, b::DeferredBinding) = _deferred_merge(a, b)
_merge_model_node(a::DeferredBinding, b::ModelValueTree) = _deferred_merge(a, b)
function _merge_model_node(
    a::DeferredBinding, b::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}}
) where {T,N,D<:AbstractArray{T,N}}
    return _deferred_merge(a, b)
end

function _check_model_binding(
    a::DeferredBinding, b::Union{VarNamedTuple,VarNamedTuples.PartialArray}, vn; kwargs...
)
    return nothing
end
function _check_model_binding(
    a::DeferredBinding,
    b::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}},
    vn;
    kwargs...,
) where {T,N,D<:AbstractArray{T,N}}
    return nothing
end

Base.copy(x::DeferredBinding) = DeferredBinding(map(_copy_model_node, x.updates))

_copy_model_node(x::DeferredBinding) = copy(x)
_copy_model_node(x::DeferredWrite) = DeferredWrite(_copy_model_node(x.value), x.group)

_valid_model_binding(x::DeferredBinding) = any(_valid_model_binding, x.updates)
_valid_model_binding(x::DeferredWrite) = _valid_model_binding(x.value)

function _classify_placeholder_children(x::DeferredBinding, seen)
    return any(v -> _classify_placeholder(v, seen), x.updates)
end
_classify_placeholder_children(::DeferredRemoval, seen) = false
function _classify_placeholder_children(x::DeferredWrite, seen)
    return _classify_placeholder(x.value, seen)
end

function _binding_storage(x::DeferredBinding)
    for update in reverse(x.updates)
        update isa DeferredWrite || continue
        value = update.value
        if value isa ModelValue && !(value.value isa NamedTuple)
            return _binding_storage(value)
        end
    end
    return NoTemplate()
end

function _binding_address_storage(
    x::DeferredBinding, optic::AbstractPPL.Property{S}
) where {S}
    child = _model_argument_binding(x, AbstractPPL.Property{S}())
    storage = _binding_address_storage(child, optic.child)
    return storage isa NoTemplate ? storage : VarNamedTuple(NamedTuple{(S,)}((storage,)))
end

_shape_template(::DeferredBinding, optic, previous=nothing) = NoTemplate()
function _shape_template(x::DeferredShape, optic, previous=nothing)
    return _shape_template(_submodel_namespace(x), optic, _deferred_projection(previous))
end

_prune_model_node(x::DeferredBinding) = _prune_deferred_updates(x.updates)
function _prune_model_node(x::DeferredWrite)
    value = _prune_model_node(x.value)
    return value isa NoModelBinding ? value : DeferredWrite(value, x.group)
end
_prune_model_node(x::DeferredRemoval) = x.value isa NoModelBinding ? x.value : x

function _prune_deferred_updates(updates)
    updates = filter(v -> !(v isa NoModelBinding), map(_prune_model_node, updates))
    return if updates isa Tuple{Vararg{DeferredRemoval}}
        NoModelBinding()
    else
        DeferredBinding(updates)
    end
end

function VarNamedTuples._map_values_recursive!!(f, x::DeferredBinding)
    return _prune_deferred_updates(
        map(v -> VarNamedTuples._map_values_recursive!!(f, v), x.updates)
    )
end
function VarNamedTuples._map_values_recursive!!(f, x::DeferredRemoval)
    return _prune_model_node(DeferredRemoval(f(x.value), x.optic))
end
function VarNamedTuples._map_values_recursive!!(f, x::DeferredWrite)
    return _prune_model_node(
        DeferredWrite(VarNamedTuples._map_values_recursive!!(f, x.value), x.group)
    )
end

function VarNamedTuples._map_pairs_recursive!!(f, x::DeferredBinding, vn)
    return DeferredBinding(
        map(v -> VarNamedTuples._map_pairs_recursive!!(f, v, vn), x.updates)
    )
end
function VarNamedTuples._map_pairs_recursive!!(f, x::DeferredRemoval, vn)
    return DeferredRemoval(f(vn => x.value), x.optic)
end
function VarNamedTuples._map_pairs_recursive!!(f, x::DeferredWrite, vn)
    return DeferredWrite(VarNamedTuples._map_pairs_recursive!!(f, x.value, vn), x.group)
end

function VarNamedTuples._mapreduce_recursive(f, op, x::DeferredBinding, vn, init)
    return op(init, f(vn => x))
end
VarNamedTuples._mapreduce_recursive(f, op, x::DeferredRemoval, vn, init) = init
function VarNamedTuples._mapreduce_recursive(f, op, x::DeferredWrite, vn, init)
    return VarNamedTuples._mapreduce_recursive(f, op, x.value, vn, init)
end

_model_data(x::DeferredBinding) = VarNamedTuples._map_values_recursive!!(_model_data, x)

function _select_model_node(::Type{R}, x::DeferredBinding) where {R}
    projected = Base.afoldl(_deferred_listing_update, NoModelBinding(), x.updates...)
    return _deferred_listing(R, projected)
end
function _deferred_listing(::Type{R}, projected) where {R}
    selected = _select_model_node(R, projected)
    return if _complete_deferred_listing(selected)
        _deferred_listing_value(selected)
    else
        selected
    end
end
function _deferred_listing(::Type{R}, projected::VarNamedTuple) where {R}
    return _map_model_nodes(
        v -> _select_model_node(R, v), projected, _deferred_listing_data
    )
end
function _deferred_listing_data(data::NamedTuple)
    isempty(data) && return NoModelBinding()
    complete = all(_complete_deferred_listing, values(data))
    return complete ? map(_deferred_listing_value, data) : VarNamedTuple(data)
end
function _select_model_node(::Type{R}, x::DeferredWrite) where {R}
    return _prune_model_node(DeferredWrite(_select_model_node(R, x.value), x.group))
end
function _select_model_node(::Type{R}, x::DeferredRemoval) where {R}
    return _prune_model_node(DeferredRemoval(_select_model_node(R, x.value), x.optic))
end

_deferred_listing_update(result, update) = _deferred_value_update(result, update, nothing)
function _deferred_listing_update(result, update::DeferredWrite)
    return _merge_model_node(result, update.value)
end
function _deferred_listing_update(
    result, update::DeferredWrite{<:ModelValue{<:Any,<:NamedTuple}}
)
    return _merge_model_node(result, _deferred_namespace(update.value))
end

_complete_deferred_listing(x) = _has_complete_model_data(x)
function _complete_deferred_listing(x::VarNamedTuple)
    return all(_complete_deferred_listing, values(x.data))
end
function _complete_deferred_listing(x::VarNamedTuples.PartialArray)
    return !(x.data isa VarNamedTuples.GrowableArray) &&
           all(x.mask) &&
           all(_complete_deferred_listing, x.data)
end

_deferred_listing_value(x) = x
_deferred_listing_value(x::VarNamedTuple) = map(_deferred_listing_value, x.data)
_deferred_listing_value(x::VarNamedTuples.PartialArray) = _model_data(x)

_has_binding_role(R, x::DeferredBinding) = any(v -> _has_binding_role(R, v), x.updates)
_has_binding_role(R, x::DeferredWrite) = _has_binding_role(R, x.value)

_model_argument_binding(x::DeferredBinding, ::AbstractPPL.Iden) = x
function _model_argument_binding(x::DeferredBinding, optic::AbstractPPL.AbstractOptic)
    value = _model_argument_binding(_deferred_projection(x), optic)
    return value isa NoModelBinding ? nothing : value
end
_model_argument_binding(x::DeferredShape, ::AbstractPPL.Iden) = x
_model_argument_binding(x::DeferredShape, optic::AbstractPPL.AbstractOptic) = nothing
_model_argument_binding(::DeferredRemoval, optic::AbstractPPL.AbstractOptic) = nothing
function _model_argument_binding(x::DeferredWrite, optic::AbstractPPL.AbstractOptic)
    value = _model_argument_binding(x.value, optic)
    return value === nothing ? nothing : DeferredWrite(value, x.group)
end
_model_argument_binding(x::DeferredWrite, ::AbstractPPL.Iden) = x
_model_argument_binding(::DeferredRemoval, ::AbstractPPL.Iden) = nothing

function VarNamedTuples._haskey_optic(x::DeferredBinding, optic::AbstractPPL.AbstractOptic)
    return any(v -> VarNamedTuples._haskey_optic(v, optic), x.updates)
end
function VarNamedTuples._haskey_optic(x::DeferredWrite, optic::AbstractPPL.AbstractOptic)
    return VarNamedTuples._haskey_optic(x.value, optic)
end
VarNamedTuples._haskey_optic(::DeferredBinding, ::AbstractPPL.Iden) = true
function VarNamedTuples._haskey_optic(x::DeferredWrite, optic::AbstractPPL.Iden)
    return VarNamedTuples._haskey_optic(x.value, optic)
end

_defer_named_root(x) = x
function _defer_named_root(x::ModelValue{R,<:NamedTuple}) where {R}
    return DeferredBinding((DeferredWrite(x, nothing),))
end
_defer_named_root(x::BindingShape{<:FieldBindingGeometry}) = DeferredShape(((x, ()),))

function _deferred_namespace(x::Union{ModelValue,BindingShape})
    namespace = _submodel_namespace(x)
    return VarNamedTuple(map(_defer_named_root, namespace.data))
end
_deferred_namespace(x) = x
function _deferred_namespace(x::PendingBindings)
    return _fold_model_indices(VarNamedTuple(), x) do result, value, optic, template
        node = DeferredBinding((DeferredWrite(value, _pending_group(x, optic)),))
        update = VarNamedTuples.make_leaf(node, optic, NoTemplate())
        _merge_model_values(result, update)
    end
end
_deferred_namespace(::NoModelBinding) = VarNamedTuple()
function _deferred_namespace(x::DeferredWrite)
    x.group === nothing && return _deferred_namespace(x.value)
    children = _submodel_namespace(x.value)
    return VarNamedTuple(
        map(v -> DeferredBinding((DeferredWrite(v, x.group),)), children.data)
    )
end

function _submodel_namespace(x::DeferredBinding)
    return Base.afoldl(_deferred_namespace_update, VarNamedTuple(), x.updates...)
end
_submodel_namespace(x::DeferredShape) = _reduce_deferred_shapes(x, nothing, true)

function _resolves_deferred_binding(model, vn)
    return _is_namespace_storage(model, vn) ||
           _binding_schema_owned(model, vn) ||
           AbstractPPL.getsym(vn) in _args_on_lhs(model)
end

function _resolve_deferred_binding(x::DeferredBinding, model, vn)
    _resolves_deferred_binding(model, vn) || return x
    if _is_namespace_storage(model, vn)
        return _resolve_deferred_binding(_submodel_namespace(x), model, vn)
    end
    x = DeferredBinding(
        map(update -> _resolve_deferred_write(update, model, vn), x.updates)
    )
    _check_deferred_groups(x, model, vn)
    return Base.afoldl(
        (result, update) -> _deferred_value_update(result, update, vn),
        NoModelBinding(),
        x.updates...,
    )
end
_resolve_deferred_binding(x, model, vn) = x
@generated function _resolve_deferred_binding(
    x::VarNamedTuple{names}, model, vn
) where {names}
    children = map(names) do name
        quote
            let child = x.data.$name, address = maybe_prefix(VarName{$(QuoteNode(name))}(), vn)

                # Decide scope before recursion can widen the address type in inference.
                if _resolves_deferred_binding(model, address)
                    _resolve_deferred_binding(child, model, address)
                else
                    child
                end
            end
        end
    end
    return quote
        _has_deferred_binding(x) || return x
        VarNamedTuple(NamedTuple{$names}(($(children...),)))
    end
end
function _resolve_deferred_binding(x::DeferredShape, model, vn)
    _resolves_deferred_binding(model, vn) || return x
    if _is_namespace_storage(model, vn)
        return _resolve_deferred_binding(_submodel_namespace(x), model, vn)
    end
    return _reduce_deferred_shapes(x, vn, false)
end
function _resolve_deferred_binding(x::PendingBindings, model, vn)
    return _pending_map(
        e ->
            first(e) => _resolve_deferred_binding(
                last(e), model, AbstractPPL.append_optic(vn, first(e))
            ),
        x,
    )
end

function _check_namedtuple_index(
    ::DeferredBinding, optic::AbstractPPL.AbstractOptic, prefix; kwargs...
)
    return nothing
end
_check_namedtuple_index(::DeferredBinding, ::AbstractPPL.Iden, prefix; kwargs...) = nothing

function _snapshot_shape(x::DeferredBinding)
    return DeferredShape(map(v -> (_snapshot_shape(v), ()), x.updates))
end
_snapshot_shape(::DeferredRemoval) = nothing
function _snapshot_shape(x::VarNamedTuples.PartialArray{<:DeferredBinding})
    T = Core.Compiler.return_type(_snapshot_shape, Tuple{eltype(x)})
    data = similar(x.data, T)
    for i in eachindex(x.mask)
        x.mask[i] && (data[i] = _snapshot_shape(x.data[i]))
    end
    return VarNamedTuples.PartialArray(data, copy(x.mask))
end
_snapshot_shape(x::DeferredWrite) = _snapshot_shape(x.value)

_deferred_shape_updates(x) = ((x, ()),)
_deferred_shape_updates(x::DeferredShape) = x.updates

function _deferred_shape_merge(op, a, b, vn)
    previous = _deferred_shape_projection(a, vn)
    incoming = _deferred_shape_projection(b, vn)
    rules = ShapeOp{false}(op.model, op.whole)(previous, incoming, vn)
    released = if !_shape_occupied(op.model, vn)
        (AbstractPPL.Iden(),)
    else
        map(rules) do rule
            AbstractPPL._unprefix_optic(
                AbstractPPL.getoptic(first(rule)), AbstractPPL.getoptic(vn)
            )
        end
    end
    updates = map(_deferred_shape_updates(b)) do (shape, prior)
        (shape, (released..., prior...))
    end
    return DeferredShape((_deferred_shape_updates(a)..., updates...))
end

_deferred_shape_projection(x, vn) = x
function _deferred_shape_projection(x::BindingShape{<:FieldBindingGeometry}, vn)
    return _deferred_shape_projection(_submodel_namespace(x), vn)
end
function _deferred_shape_projection(x::VarNamedTuple{names}, vn) where {names}
    return VarNamedTuple(
        NamedTuple{names}(
            map(names) do name
                _deferred_shape_projection(x.data[name], maybe_prefix(VarName{name}(), vn))
            end,
        ),
    )
end
function _deferred_shape_projection(x::DeferredShape, vn)
    return Base.afoldl(nothing, x.updates...) do result, (update, released)
        shape = _deferred_shape_projection(update, vn)
        rules = _deferred_shape_rules(released, result, vn, true)
        ShapeOp{true}(rules, rules)(result, shape, vn)
    end
end

_may_have_deferred_shape(::Type{T}) where {T} = !isconcretetype(T)
_may_have_deferred_shape(::Type{<:DeferredShape}) = true
_may_have_deferred_shape(::Type{<:PendingBindings}) = true
function _may_have_deferred_shape(::Type{<:PendingBindings{N,E}}) where {N,E<:AbstractDict}
    return _may_have_deferred_shape(valtype(E))
end
function _may_have_deferred_shape(::Type{<:VarNamedTuple{names,T}}) where {names,T}
    return any(_may_have_deferred_shape, fieldtypes(T))
end
function _may_have_deferred_shape(::Type{<:VarNamedTuples.PartialArray{T}}) where {T}
    return _may_have_deferred_shape(T)
end
@generated function _deferred_shape_check_needed(::Type{T}) where {T}
    return _may_have_deferred_shape(T)
end

_check_deferred_shape_writes(x, vn) = nothing
function _check_deferred_shape_writes(x::DeferredShape, vn)
    _deferred_shape_projection(x, vn)
    return nothing
end
function _check_deferred_shape_writes(
    x::Union{VarNamedTuple,VarNamedTuples.PartialArray,PendingBindings}, vn
)
    _deferred_shape_check_needed(typeof(x)) || return nothing
    return _fold_model_indices(nothing, x) do _, child, optic, _
        _check_deferred_shape_writes(child, _shape_child_name(vn, optic))
    end
end

function (op::ShapeOp{Build})(a::DeferredShape, b, vn) where {Build}
    return Build ? _deferred_shape_merge(op, a, b, vn) : ()
end
function (op::ShapeOp{Build})(a, b::DeferredShape, vn) where {Build}
    return Build ? _deferred_shape_merge(op, a, b, vn) : ()
end
function (op::ShapeOp{Build})(a::DeferredShape, b::DeferredShape, vn) where {Build}
    return Build ? _deferred_shape_merge(op, a, b, vn) : ()
end
(op::ShapeOp{Build})(::Nothing, b::DeferredShape, vn) where {Build} = Build ? b : ()
(op::ShapeOp{Build})(a::DeferredShape, ::Nothing, vn) where {Build} = Build ? a : ()
function (op::ShapeOp{Build})(a::InheritedShapes, b::DeferredShape, vn) where {Build}
    return Build ? _deferred_shape_merge(op, a, b, vn) : ()
end
function (op::ShapeOp{Build})(a::DeferredShape, b::InheritedShapes, vn) where {Build}
    return Build ? _deferred_shape_merge(op, a, b, vn) : ()
end

function _deferred_shape_rules(released, previous, vn, namespace)
    rules = mapreduce((a, b) -> (a..., b...), released; init=()) do optic
        if optic isa AbstractPPL.Iden
            _released_shape_rules(previous, vn)
        elseif namespace
            (_shape_child_name(vn, optic) => true,)
        else
            ()
        end
    end
    return ShapeRules(rules)
end

_released_shape_rules(x, vn) = (vn => true,)
_released_shape_rules(::Nothing, vn) = ()
function _released_shape_rules(x::VarNamedTuple{names}, vn) where {names}
    return mapreduce((a, b) -> (a..., b...), names; init=()) do name
        _released_shape_rules(x.data[name], maybe_prefix(VarName{name}(), vn))
    end
end

function _reduce_deferred_shapes(x, vn, namespace)
    reduce = function (result, (update, released))
        shape = if update isa DeferredShape
            _reduce_deferred_shapes(update, vn, namespace)
        else
            namespace ? _deferred_namespace(update) : update
        end
        rules = _deferred_shape_rules(released, result, vn, namespace)
        return ShapeOp{true}(rules, rules)(result, shape, vn)
    end
    return Base.afoldl(reduce, nothing, x.updates...)
end

function _remove_model_binding(
    ::Type{R}, x::DeferredBinding, optic::AbstractPPL.AbstractOptic
) where {R}
    if optic isa AbstractPPL.Iden
        return VarNamedTuples._map_values_recursive!!(
            v -> _matches_model_role(R, v) ? NoModelBinding() : v, x
        )
    end
    return DeferredBinding((
        x.updates..., DeferredRemoval(ModelValue{R}(RemovedBinding()), optic)
    ))
end

_deferred_value_update(result, update, vn) = _merge_model_node(result, update, vn)
function _deferred_value_update(
    result, update::DeferredRemoval{<:ModelValue{R}}, vn
) where {R}
    return _remove_model_binding(R, result, update.optic)
end
function _deferred_value_update(result, x::DeferredWrite, vn)
    return if result isa NoModelBinding
        _copy_model_node(x.value)
    else
        _merge_model_node(result, x.value, vn)
    end
end

function _deferred_namespace_update(result, update)
    return _merge_model_values(result, _deferred_namespace(update))
end
function _deferred_namespace_update(
    result, update::DeferredRemoval{<:ModelValue{R}}
) where {R}
    return _remove_model_binding(R, result, update.optic)
end

function _deferred_group_entries(f, x::DeferredWrite, optic, group)
    return _deferred_group_entries(f, x.value, optic, x.group)
end
function _deferred_group_entries(f, x::PendingBindings, optic, group)
    for (selection, value) in x.entries
        _deferred_group_entries(
            f,
            value,
            selection ∘ optic,
            something(_pending_group(x, selection), Some(group)),
        )
    end
end
function _deferred_group_entries(f, x::VarNamedTuple, optic, group)
    for (name, value) in pairs(x.data)
        _deferred_group_entries(f, value, AbstractPPL.Property{name}() ∘ optic, group)
    end
end
_deferred_group_entries(f, x, optic, group) = f(optic, group)
_deferred_group_entries(f, x::DeferredRemoval, optic, group) = nothing
function _deferred_group_entries(f, x::ModelValueTree, optic, group)
    return _deferred_group_entries(f, x.values, optic, group)
end

function _check_deferred_groups(x, model, vn)
    groups = IdDict{Any,Any}()
    for update in x.updates
        _deferred_group_entries(update, AbstractPPL.Iden(), nothing) do optic, group
            group === nothing && return nothing
            seen = get!(groups, group) do
                _binding_overlap_state(nothing)
            end
            name = AbstractPPL.append_optic(vn, optic)
            storage = _binding_address_template(model, _model_values(model.values), name)
            _check_binding_overlap!(seen, model, name, storage)
        end
    end
    return nothing
end
function _check_deferred_groups(
    ::DeferredBinding{<:Tuple{DeferredWrite{V,G}}}, model, vn
) where {V<:ModelValue,G}
    return nothing
end

function _defer_binding_overlap(model, vn, value)
    return value isa Union{NamedTuple,DeferredBinding} &&
           _pending_binding(model, vn) &&
           !_is_namespace_storage(model, vn)
end

function _check_deferred_overlap!(seen, model, vn, value::NamedTuple)
    for (name, child) in pairs(value)
        _check_deferred_overlap!(
            seen, model, AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()), child
        )
    end
    return nothing
end
function _check_deferred_overlap!(seen, model, vn, value)
    return _check_binding_overlap!(seen, model, vn, NoTemplate())
end
_check_deferred_overlap!(seen, model, vn, value::DeferredBinding) = nothing

Base.:(==)(a::DeferredWrite, b::DeferredWrite) = a.value == b.value
Base.:(==)(a::DeferredBinding, b::DeferredBinding) = a.updates == b.updates
function Base.:(==)(a::DeferredRemoval, b::DeferredRemoval)
    return a.value == b.value && a.optic == b.optic
end

Base.isequal(a::DeferredWrite, b::DeferredWrite) = isequal(a.value, b.value)
Base.isequal(a::DeferredBinding, b::DeferredBinding) = isequal(a.updates, b.updates)
function Base.isequal(a::DeferredRemoval, b::DeferredRemoval)
    return isequal(a.value, b.value) && isequal(a.optic, b.optic)
end

Base.hash(x::DeferredWrite, h::UInt) = hash(x.value, h)
Base.hash(x::DeferredBinding, h::UInt) = hash(x.updates, h)
Base.hash(x::DeferredRemoval, h::UInt) = hash(x.optic, hash(x.value, h))

function Base.show(io::IO, x::DeferredBinding)
    print(io, "pending(")
    for (i, update) in enumerate(x.updates)
        i == 1 || print(io, ", ")
        show(io, update isa DeferredWrite ? update.value : update)
    end
    return print(io, ")")
end
function Base.show(io::IO, x::DeferredRemoval)
    print(io, "remove(")
    AbstractPPL._pretty_print_optic(io, x.optic)
    return print(io, ")")
end

function VarNamedTuples._storage_template(x::DeferredBinding)
    throw(
        ArgumentError(
            "Cannot make a storage template from unresolved NamedTuple bindings; supply concrete storage with `@of` instead.",
        ),
    )
end

_resolve_deferred_write(update, model, vn) = update
function _resolve_deferred_write(update::DeferredWrite{<:DeferredBinding}, model, vn)
    return DeferredWrite(_resolve_deferred_binding(update.value, model, vn), update.group)
end

_deferred_input_storage(x) = x
_deferred_input_storage(::DeferredBinding) = NoTemplate()
function _deferred_input_storage(x::VarNamedTuple)
    return VarNamedTuple(map(_deferred_input_storage, x.data))
end

function _deferred_projection(x::DeferredBinding)
    # Layer concatenation preserves each role's order, not their relative chronology.
    observations = _remove_model_binding(Fix, x, AbstractPPL.Iden())
    fixed = _remove_model_binding(Condition, x, AbstractPPL.Iden())
    return _merge_model_node(
        _prune_model_node(_deferred_layer_projection(observations)),
        _prune_model_node(_deferred_layer_projection(fixed)),
    )
end
_deferred_layer_projection(x) = x
function _deferred_layer_projection(x::DeferredBinding)
    return foldl(x.updates; init=NoModelBinding()) do result, update
        if update isa DeferredRemoval
            _deferred_value_update(result, update, nothing)
        else
            _merge_model_node(result, _deferred_projection(update.value))
        end
    end
end
_deferred_projection(x) = x
_deferred_projection(x::VarNamedTuple) = VarNamedTuple(map(_deferred_projection, x.data))
function _deferred_projection(x::ModelValue{R,<:NamedTuple}) where {R}
    return _deferred_projection(_submodel_namespace(x))
end
_deferred_projection(x::ModelValueTree) = _deferred_projection(x.values)
_deferred_projection(x::PendingBindings) = _deferred_projection(_submodel_namespace(x))

_has_deferred_binding(x) = false
_has_deferred_binding(::Union{DeferredBinding,DeferredShape}) = true
_has_deferred_binding(x::VarNamedTuple) = any(_has_deferred_binding, values(x.data))
function _has_deferred_binding(x::PendingBindings)
    return any(e -> _has_deferred_binding(last(e)), x.entries)
end

_listing_known_namespaces(x, model, vn) = x
function _listing_known_namespaces(x::DeferredBinding, model, vn)
    return if _is_namespace_storage(model, vn)
        _listing_known_namespaces(_submodel_namespace(x), model, vn)
    else
        x
    end
end
function _listing_known_namespaces(x::VarNamedTuple{names}, model, vn) where {names}
    return VarNamedTuple(
        NamedTuple{names}(
            map(names) do name
                _listing_known_namespaces(
                    x.data[name], model, maybe_prefix(VarName{name}(), vn)
                )
            end,
        ),
    )
end

# Namespace projection replaces a subtree; pending binding writes would merge it.
function _replace_listing_subtree(values, child, head)
    return VarNamedTuples._setindex_optic!!(
        copy(values), child, head, NoTemplate(), VarNamedTuples.AllowAll()
    )
end
function _replace_listing_subtree(values::PendingBindings, child, head)
    values.base isa VarNamedTuples.PartialArray &&
        return _replace_listing_subtree(_materialize_pending(values), child, head)
    @assert values.base === nothing
    entries = empty(_pending_dict(values.entries))
    groups = Dict{AbstractPPL.AbstractOptic,Union{Nothing,Base.RefValue{Nothing}}}()
    for (optic, value) in values.entries
        remainder = _pending_remainder(optic, value, head, true)
        remainder === nothing && continue
        entries = _pending_set!!(entries, last(remainder), first(remainder))
        groups[first(remainder)] = _pending_group(values, optic)
    end
    entries = _pending_set!!(entries, child, head)
    groups[head] = _pending_group(values, head)
    return PendingBindings(values.name, entries, nothing, groups)
end

function _listing_at_prefix(values, model, ::AbstractPPL.Iden)
    namespace = _listing_known_namespaces(_submodel_namespace(values), model, nothing)
    return _listing_argument_children(model, namespace; local_names=true)
end
function _listing_at_prefix(values, model, optic::AbstractPPL.AbstractOptic)
    optic isa AbstractPPL.Property && (values = _submodel_namespace(values))
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(values, head)
    child === nothing && return values
    child = _listing_at_prefix(child, model, optic.child)
    return _replace_listing_subtree(values, child, head)
end

@inline function _listing_argument_children(model, values; local_names=false)
    sites = _submodel_arguments(_binding_metadata(model))
    arguments = merge(model.args, model.defaults)
    for (lhs, name) in sites
        name isa Symbol || continue
        lhs === nothing || count(site -> first(site) == lhs, sites) == 1 || continue
        child = get(arguments, name, nothing)
        child isa Model || continue
        address = local_names ? lhs : maybe_prefix(lhs, model.prefix)
        optic =
            address === nothing ? AbstractPPL.Iden() : AbstractPPL.varname_to_optic(address)
        node = _model_argument_binding(values, optic)
        node === nothing && continue
        prefix = child.prefix
        prefix_optic =
            prefix === nothing ? AbstractPPL.Iden() : AbstractPPL.varname_to_optic(prefix)
        namespace = if lhs === nothing && count(site -> first(site) === nothing, sites) > 1
            @inline _listing_unprefixed_child(
                node, _binding_metadata(model), arguments, child, prefix_optic
            )
        else
            _listing_at_prefix(node, child, prefix_optic)
        end
        values = _replace_listing_subtree(values, namespace, optic)
    end
    return values
end

_listing_namespace_agrees(child, vn) = true
@inline function _listing_namespace_agrees(child::Model, vn)
    local_name = _local_removal_name(child, vn; operation="bind")
    local_name === nothing && return true
    AbstractPPL.getsym(local_name) in _args_on_lhs(child) && return false
    for (lhs, submodel) in _lhs_addresses(_binding_metadata(child))
        submodel && continue
        (subsumes(lhs, local_name) || subsumes(local_name, lhs)) || continue
        _is_namespace_storage(child, vn) || return false
    end
    arguments = merge(child.args, child.defaults)
    for (lhs, name) in _submodel_arguments(_binding_metadata(child))
        name isa Symbol || continue
        nested = get(arguments, name, nothing)
        nested isa Model || continue
        relative = if lhs === nothing
            local_name
        else
            (lhs != local_name && subsumes(lhs, local_name)) || continue
            AbstractPPL.unprefix(local_name, lhs)
        end
        _listing_namespace_agrees(nested, relative) || return false
    end
    return true
end

@inline function _listing_namespace_agrees(other, root, child::Model, prefix=nothing)
    prefix = maybe_prefix(child.prefix, prefix)
    for (lhs, submodel) in _lhs_addresses(_binding_metadata(child))
        submodel || continue
        address = maybe_prefix(lhs, prefix)
        subsumes(root, address) || continue
        _listing_namespace_agrees(other, address) || return false
    end
    arguments = merge(child.args, child.defaults)
    for (lhs, name) in _submodel_arguments(_binding_metadata(child))
        name isa Symbol || continue
        nested = get(arguments, name, nothing)
        nested isa Model || continue
        _listing_namespace_agrees(other, root, nested, maybe_prefix(lhs, prefix)) ||
            return false
    end
    return true
end

# Literal names let namespace agreement fold away for unambiguous children.
@generated function _listing_unprefixed_child(
    values::VarNamedTuple{names},
    ::ModelBindingMetadata{A,L,U,T,N,B,C},
    arguments,
    child,
    optic,
) where {names,A,L,U,T,N,B,C}
    any(site -> first(site) === nothing && !(last(site) isa Symbol), C) && return :(values)
    checks = map(names) do name
        children = map(
            filter(site -> first(site) === nothing && last(site) isa Symbol, C)
        ) do (_, argument)
            quote
                _listing_namespace_agrees(
                    get(arguments, $(QuoteNode(argument)), nothing),
                    VarName{$(QuoteNode(name))}(),
                ) && _listing_namespace_agrees(
                    get(arguments, $(QuoteNode(argument)), nothing),
                    VarName{$(QuoteNode(name))}(),
                    child,
                )
            end
        end
        foldl((a, b) -> :($a && $b), children; init=true)
    end
    agreed = foldl((a, b) -> :($a && $b), checks; init=true)
    children = map(names, checks) do name, check
        quote
            if $check
                namespace = VarNamedTuple(NamedTuple{$((name,))}((values.data.$name,)))
                _listing_at_prefix(namespace, child, optic).data.$name
            else
                values.data.$name
            end
        end
    end
    return quote
        $agreed && return _listing_at_prefix(values, child, optic)
        VarNamedTuple(NamedTuple{$names}(($(children...),)))
    end
end
