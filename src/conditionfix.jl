Base.@constprop :aggressive function _argument_bindings(arguments, metadata)
    return _argument_bindings(arguments, Val(_lhs_arguments(metadata)))
end
@generated function _argument_bindings(arguments::NamedTuple{N}, ::Val{L}) where {N,L}
    names = Tuple(n for n in N if unsplat_symbol(n) in L)
    fields = map(names) do n
        # An empty keyword splat is unobserved, like a `missing` argument.
        empty = is_splat_symbol(n) && fieldtype(arguments, n) <: NamedTuple{()}
        :(BindingRoot(
            if $empty || arguments.$n isa Union{Missing,Nothing}
                NoBinding()
            else
                BoundValue{ArgumentRole}(arguments.$n)
            end,
            arguments.$n,
            (),
        ))
    end
    roots = Tuple(unsplat_symbol(n) for n in names)
    return :(BindingTable(VarNamedTuple(NamedTuple{$roots}(($(fields...),)))))
end
function _binding_root(model, vn)
    return get(model.values.roots.data, AbstractPPL.getsym(vn), nothing)
end
function _binding_at(model, vn)
    root = _binding_root(model, vn)
    return if root === nothing
        nothing
    else
        _binding_node_at(root.value, AbstractPPL.getoptic(vn))
    end
end
@inline function _bound_data_at(model, vn)
    node = _binding_at(model, vn)
    return if _has_field_bindings(node)
        storage = _binding_node_at(
            _binding_root(model, vn).storage, AbstractPPL.getoptic(vn)
        )
        _bound_data(node, storage)
    else
        _bound_data(node)
    end
end
function _bound_data_at(model, vn, local_value)
    _check_local_shape(model, vn, local_value, FixRole)
    return _local_bound_data(model, vn, local_value)
end
# A tuple variable takes its tuple type from the model value; an `of(Array, n)` template
# gives it only its length.
function _local_bound_data(model, vn, local_value)
    node = _binding_at(model, vn)
    if node isa _PositionalNode
        value = _binding_node_at(local_value, AbstractPPL.getoptic(vn))
        value isa Tuple && return _bound_data(node, value)
    end
    return _bound_data_at(model, vn)
end
function _argument_role(model, vn, argument, local_value=NoTemplate())
    binding = _binding_at(model, argument)
    (
        binding isa Union{BoundValue{ConditionRole},BoundValue{ArgumentRole}} ||
        local_value isa Base.Pairs && binding isa BoundValue
    ) && return _binding_role(binding, vn)
    original = get(merge(model.args, model.defaults), AbstractPPL.getsym(argument), nothing)
    optic = AbstractPPL.getoptic(vn)
    if local_value isa Union{AbstractArray{>:Missing},Tuple} &&
        _element_optic(local_value) === AbstractPPL.Index &&
        (optic isa AbstractPPL.Iden || optic.child isa AbstractPPL.Iden) &&
        !(original isa Union{Missing,Nothing})
        value = _binding_node_at(local_value, optic)
        value isa Union{AbstractArray,Tuple} &&
            any(ismissing, value) &&
            _throw_binding_error(
                maybe_prefix(vn, _model_prefix(model)),
                "a multivariate tilde cannot impute `missing` argument entries; write one tilde per imputed entry, such as `x[i] ~ Normal()`",
            )
    end
    return _stored_role(binding, vn, model)
end
# The role of a local LHS variable, after checking its storage against the bound shape.
function _local_role(binding, vn, template, model)
    found = _stored_role(binding, vn, model)
    if binding isa Union{BoundValue,_InnerNode}
        _check_local_shape(model, vn, template, ConditionRole)
    end
    return found
end
function _stored_role(binding, vn, model)
    optic = AbstractPPL.getoptic(vn)
    address = maybe_prefix(vn, _model_prefix(model))
    root = _binding_root(model, vn)
    _check_address_kind(binding, optic, AbstractPPL.Property{AbstractPPL.getsym(address)}())
    node = _binding_node_at(binding, optic)
    node === nothing &&
        binding isa BoundValue &&
        binding.value isa AbstractArray &&
        optic isa AbstractPPL.Index &&
        _binding_node_at(binding, AbstractPPL.ohead(optic)) === nothing &&
        _throw_binding_error(address, "executed index is outside bound storage")
    storage = if node isa _InnerNode
        _binding_node_at(root.storage, optic)
    else
        NoTemplate()
    end
    return _binding_role(node, address, storage)
end
function _check_local_shape(model, vn, local_value, role; submodel=false)
    local_value isa NoTemplate && return nothing
    root = _binding_root(model, vn)
    root === nothing && return nothing
    address = maybe_prefix(vn, _model_prefix(model))
    _check_node_shape(root.value, local_value, AbstractPPL.getoptic(vn), address; submodel)
    for constraint in root.constraints
        (; owner, storage) = constraint
        constraint.role === role || continue
        _skip_bound_shape_check(storage) && continue
        subsumes(owner, vn) || continue
        # A submodel site's own binding is the child's namespace, not a local shape.
        submodel && owner == vn && continue
        value = _binding_node_at(local_value, AbstractPPL.getoptic(owner))
        value === nothing &&
            _throw_binding_error(address, "local storage omits bound fields")
        _matches_local_shape(storage, value) ||
            _throw_local_shape_error(address, storage, value)
    end
    return nothing
end
function _throw_local_shape_error(vn, bound, local_value)
    bound = _unwrap_storage(bound)
    positional =
        bound isa Union{AbstractArray,Tuple} && local_value isa Union{AbstractArray,Tuple}
    positional &&
        axes(bound) == axes(local_value) &&
        _throw_binding_error(
            vn,
            "local storage type changed; bound a $(bound isa Tuple ? "tuple" : "array"), local $(typeof(local_value))",
        )
    detail =
        positional ? "; bound axes $(axes(bound)), local axes $(axes(local_value))" : ""
    return _throw_binding_error(vn, "local storage shape changed" * detail)
end
function _check_node_shape(binding, value, optic, vn; submodel=false)
    binding isa Union{Nothing,NoBinding} && return nothing
    submodel && optic isa AbstractPPL.Iden && return nothing
    # Every bound node on the path must fit, whatever its role: a fixed `x = (1.0, 2.0)`
    # must not leave `x[3]` latent.
    bound = _unwrap_storage(binding)
    if binding isa Union{BoundValue,_PositionalNode} && !_skip_bound_shape_check(bound)
        _matches_local_shape(bound, value) || _throw_local_shape_error(vn, bound, value)
    end
    (optic isa AbstractPPL.Iden || submodel && optic.child isa AbstractPPL.Iden) &&
        return nothing
    head = AbstractPPL.ohead(optic)
    head isa AbstractPPL.Index && (head = AbstractPPL.concretize_top_level(head, value))
    child = _binding_node_at(binding, head)
    child === nothing && return nothing
    if head isa AbstractPPL.Index &&
        value isa AbstractArray &&
        _positional_index(head) &&
        !VarNamedTuples._is_multiindex(value, head.ix...) &&
        checkbounds(Bool, value, head.ix...) &&
        !_isassigned(value, head.ix...)
        return nothing
    end
    local_child = _binding_node_at(value, head)
    local_child === nothing &&
        _throw_binding_error(vn, "local storage omits a bound address")
    return _check_node_shape(child, local_child, optic.child, vn; submodel)
end
function _check_address_kind(value, optic, prefix=AbstractPPL.Iden())
    optic isa AbstractPPL.Iden && return nothing
    storage = _unwrap_storage(value)
    kind = _element_optic(storage isa VarNamedTuple ? storage.data : storage)
    (kind !== nothing && !(optic isa kind)) && _throw_binding_error(
        AbstractPPL.optic_to_varname(optic ∘ prefix),
        "address kind disagrees with storage",
    )
    value isa VarNamedTuples.PartialTuple && VarNamedTuples._tuple_index(
        optic, value.data, AbstractPPL.optic_to_varname(optic ∘ prefix)
    )
    optic.child isa AbstractPPL.Iden && return nothing
    child = _binding_node_at(value, AbstractPPL.ohead(optic))
    child === nothing ||
        _check_address_kind(child, optic.child, AbstractPPL.ohead(optic) ∘ prefix)
    return nothing
end

_binding_inputs(x) = (x,)
_binding_inputs(x::Tuple) = mapreduce(_binding_inputs, (a, b) -> (a..., b...), x; init=())
_binding_inputs(p::Pair{Symbol}) = (VarName{first(p)}() => last(p),)
function _flatten_bindings(input::VarNamedTuple)
    entries = Tuple[]
    for (name, value) in pairs(input.data)
        _flatten_bindings!(entries, value, VarName{name}(), value)
    end
    return entries
end
function _flatten_bindings!(entries, value::VarNamedTuple, vn, storage)
    for (name, child) in pairs(value.data)
        _flatten_bindings!(
            entries,
            child,
            AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()),
            storage,
        )
    end
    return entries
end
function _flatten_bindings!(entries, value::_PositionalNode, vn, storage)
    children = VarNamedTuples._set_entries(value)
    isempty(children) && return push!(entries, (vn, value, storage))
    for (i, child) in children
        _flatten_bindings!(
            entries, child, AbstractPPL.append_optic(vn, AbstractPPL.Index(i, (;))), storage
        )
    end
    return entries
end
_flatten_bindings!(entries, value, vn, storage) = push!(entries, (vn, value, storage))
function _flatten_bindings(input::NamedTuple{N}) where {N}
    any(x -> x isa _InnerNode, values(input)) &&
        return _flatten_bindings(VarNamedTuple(input))
    return map(n -> (VarName{n}(), input[n], NoTemplate()), N)
end
_flatten_bindings(p::Pair{<:VarName}) = ((first(p), last(p), NoTemplate()),)
function _flatten_bindings(input)
    throw(
        ArgumentError("Bindings require NamedTuple, VarNamedTuple, or Symbol/VarName pairs")
    )
end
function _relative_name(model, vn)
    prefix = _model_prefix(model)
    prefix === nothing && return vn
    template = _prefixed_template(_model_prefix_template(model), NoTemplate())
    depth = optic_skip_length(AbstractPPL.getoptic(prefix))
    prefix = _expand_cartesian(_concretize_prefix(prefix, template; depth, canonical=true))
    vn = _expand_cartesian(_concretize_prefix(vn, template; depth, canonical=true))
    subsumes(prefix, vn) && return prefix == vn ? nothing : AbstractPPL.unprefix(vn, prefix)
    subsumes(vn, prefix) && return nothing
    return _throw_binding_error(vn, "address is outside model prefix $prefix")
end
function _check_binding_address(model, vn, address=vn)
    _check_nested_selection(AbstractPPL.getoptic(vn), address)
    names = _lhs_names(_model_metadata(model))
    names === nothing ||
        AbstractPPL.getsym(vn) in names ||
        _throw_binding_error(address, "not a tilde LHS root")
    return nothing
end
function _is_namespace(model, vn)
    root = _binding_root(model, vn)
    root !== nothing &&
        _binding_node_at(root.storage, AbstractPPL.getoptic(vn)) isa VarNamedTuple &&
        return true
    any(p -> last(p) && first(p) == vn, _lhs_addresses(_model_metadata(model))) &&
        return true
    arguments = merge(model.args, model.defaults)
    for (site, name) in _submodel_arguments(_model_metadata(model))
        site === nothing && continue
        name isa Symbol || continue
        child = get(arguments, name, nothing)
        child isa Model || continue
        prefix = maybe_prefix(_model_prefix(child), site)
        subsumes(vn, prefix) && return true
        subsumes(prefix, vn) &&
            return _is_namespace(child, AbstractPPL.unprefix(vn, prefix))
    end
    return false
end
_has_index(::AbstractPPL.Iden) = false
_has_index(::AbstractPPL.Index) = true
_has_index(optic::AbstractPPL.Property) = _has_index(optic.child)
# A child model passed as an argument supplies its retained storage to its site.
function _site_storage(model, site)
    arguments = merge(model.args, model.defaults)
    for (address, name) in _submodel_arguments(_model_metadata(model))
        address == site && name isa Symbol || continue
        child = get(arguments, name, nothing)
        child isa Model || continue
        return _site_storage(child)
    end
    return NoTemplate()
end
function _site_storage(model)
    roots = model.values.roots.data
    names = Tuple(n for n in keys(roots) if !(roots[n].storage isa NoTemplate))
    storage = VarNamedTuple(NamedTuple{names}(map(n -> roots[n].storage, names)))
    for (site, name) in _submodel_arguments(_model_metadata(model))
        site === nothing && continue
        _has_index(AbstractPPL.getoptic(site)) && continue
        child = _site_storage(model, site)
        child isa NoTemplate && continue
        storage = merge(
            templated_setindex!!(VarNamedTuple(), child, site, storage), storage
        )
    end
    prefix = _model_prefix(model)
    return if prefix === nothing
        storage
    else
        templated_setindex!!(
            VarNamedTuple(),
            storage,
            prefix,
            _prefixed_template(_model_prefix_template(model), NoTemplate()),
        )
    end
end

function _split_namespace_entry(model, vn, value, storage)
    if value isa NamedTuple && _is_namespace(model, vn)
        return mapreduce((a, b) -> (a..., b...), keys(value); init=()) do name
            _split_namespace_entry(
                model,
                AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()),
                value[name],
                storage,
            )
        end
    else
        return ((vn, value, storage),)
    end
end
function _check_field_under_whole(model, vn, storage)
    any(
        ((a, child),) -> child && subsumes(a, vn), _lhs_addresses(_model_metadata(model))
    ) || return nothing
    optic = AbstractPPL.getoptic(vn)
    base = VarName{AbstractPPL.getsym(vn)}()
    while !(optic isa AbstractPPL.Iden)
        node = _binding_at(model, base)
        retained = _binding_node_at(storage, AbstractPPL.getoptic(base))
        root = _binding_root(model, base)
        typed =
            root !== nothing &&
            any(c -> c.mode === Val(:type) && subsumes(c.owner, base), root.constraints)
        if node isa BoundValue &&
            _element_optic(node.value) === AbstractPPL.Property &&
            (
                !(
                    retained isa VarNamedTuple ||
                    _element_optic(retained) === AbstractPPL.Property
                ) || typed
            )
            _throw_binding_error(vn, "field edit below an unresolved child needs storage")
        end
        base = AbstractPPL.append_optic(base, AbstractPPL.ohead(optic))
        optic = optic.child
    end
    return nothing
end
_template_owner(storage, optic, vn) = (vn, storage)
function _template_owner(storage, optic::AbstractPPL.Property{S}, vn) where {S}
    head = AbstractPPL.ohead(optic)
    return _template_owner(
        _binding_node_at(storage, head), optic.child, AbstractPPL.append_optic(vn, head)
    )
end
function _apply_constraints(value, optic, address, R, constraints; error_address=address)
    for constraint in constraints
        (; owner, role) = constraint
        template = constraint.storage
        typed = constraint.mode === Val(:type)
        if owner != address && subsumes(address, owner)
            selected = _binding_node_at(
                value, AbstractPPL._unprefix_optic(AbstractPPL.getoptic(owner), optic)
            )
            selected === nothing &&
                _throw_binding_error(error_address, "value omits retained fields")
            _same_shape(template, selected) ||
                _throw_binding_error(error_address, "value disagrees with retained shape")
            continue
        end
        subsumes(owner, address) || continue
        selected = _binding_node_at(
            template, AbstractPPL._unprefix_optic(optic, AbstractPPL.getoptic(owner))
        )
        selected === nothing ||
            _same_shape(selected, value) ||
            _throw_binding_error(error_address, "value disagrees with retained shape")
        (role === R && (typed || owner != address)) || continue
        relative = AbstractPPL._unprefix_optic(optic, AbstractPPL.getoptic(owner))
        typed &&
            relative isa AbstractPPL.Iden &&
            !(value isa typeof(template)) &&
            _throw_binding_error(error_address, "whole value must match template type")
        value = _convert_partial(value, template, relative, error_address)
    end
    return value
end
function _set_binding_node(node, name, value, vn, storage)
    wrapper = if node isa NoBinding
        VarNamedTuple()
    else
        VarNamedTuple(NamedTuple{(name,)}((node,)))
    end
    return templated_setindex!!(wrapper, value, vn, storage).data[name]
end
# Resolve root storage and constraints for entry `vn`, possibly indexed, bound as `R`.
# `original` is the root's argument value, or `NoTemplate()` if it is not an LHS argument.
Base.@nospecializeinfer function _resolve_root(
    model, old, vn, ::Type{R}, schema, supplied, original
) where {R}
    @nospecialize model original
    name = AbstractPPL.getsym(vn)
    base = VarName{name}()
    optic = AbstractPPL.getoptic(vn)
    typed = haskey(schema, name)
    storage = _unwrap_storage(old.storage)
    offered = get(schema, name, _plain_storage(supplied))
    if offered isa NamedTuple && _is_namespace(model, base)
        offered = VarNamedTuple(offered)
    end
    # A concrete argument is the storage; templates cannot replace it.
    original isa Union{Missing,Nothing,NoTemplate} || (offered = NoTemplate())
    fresh = !(offered isa NoTemplate) && old.value isa NoBinding
    if storage isa NoTemplate || fresh
        storage = offered
    elseif !(offered isa NoTemplate)
        _same_shape(storage, offered) ||
            _throw_binding_error(vn, "template disagrees with retained storage")
    end
    if original isa Union{Missing,Nothing} &&
        !(optic isa AbstractPPL.Iden) &&
        storage isa NoTemplate
        _throw_missing_storage(vn)
    end
    _check_field_under_whole(model, vn, storage)
    constraints = fresh ? () : _effective_constraints(old, base)
    if !(offered isa NoTemplate) &&
        (typed || old.storage isa NoTemplate && !(optic isa AbstractPPL.Iden))
        owner, template = _template_owner(offered, optic, base)
        template = _unwrap_storage(template)
        any(
            c ->
                c.owner == owner &&
                    c.role === R &&
                    c.mode === Val(:type) &&
                    typeof(c.storage) !== typeof(template),
            constraints,
        ) && _throw_binding_error(vn, "template disagrees with retained type")
        held = _binding_node_at(old.value, AbstractPPL.getoptic(owner))
        if held isa BoundValue
            _same_shape(held.value, template) ||
                _throw_binding_error(vn, "template disagrees with held shape")
            typed &&
                _matches_role(R, held) &&
                !(held.value isa typeof(template)) &&
                _throw_binding_error(vn, "template disagrees with held type")
        end
        if !(template isa Union{Nothing,NoTemplate}) && !(
            typed && any(
                c -> c.owner == owner && c.role === R && c.mode === Val(:type), constraints
            )
        )
            constraints = (
                constraints...,
                BindingConstraint(owner, R, template, typed ? Val(:type) : Val(:shape)),
            )
        end
    end
    return storage, constraints
end
# Write `element` at `address`, a selection of entry `vn`, into its root's `node`.
# Return the updated node, storage and constraints.
function _write_selection(
    node, storage, constraints, ::Type{R}, vn, address, element, located, argument
) where {R}
    optic = AbstractPPL.getoptic(address)
    previous = _binding_node_at(node, optic)
    previous isa VarNamedTuple &&
        !(_element_optic(_binding_node_at(storage, optic)) === AbstractPPL.Property) &&
        _throw_binding_error(address, "namespace cannot become a whole value")
    _check_role_overlap(previous, BoundValue{R}(element), address)
    if argument && !(optic isa AbstractPPL.Iden)
        element = _convert_partial(element, storage, optic, vn)
    end
    previous isa BoundValue &&
        !_same_shape(previous.value, element) &&
        _throw_binding_error(address, "remove the binding before changing shape")
    element = _apply_constraints(element, optic, address, R, constraints)
    incoming = BoundValue{R}(element)
    if optic isa AbstractPPL.Iden
        node = incoming
        storage = element
        constraints = filter(c -> c.mode === Val(:type), constraints)
    else
        node = _set_binding_node(
            node, AbstractPPL.getsym(address), incoming, address, located
        )
    end
    constraints = filter(
        c -> !(c.owner == address && c.role === R && c.mode === Val(:shape)), constraints
    )
    # Numeric value nodes already enforce their role and shape.
    element isa Number || (
        constraints = (constraints..., BindingConstraint(address, R, element, Val(:shape)))
    )
    return node, storage, constraints
end
# Avoid inference per model history, but retain the role and stored types:
# `condition` can run inside an evaluator.
Base.@nospecializeinfer function _add_bindings(
    model, table, ::Type{R}, entries, schema
) where {R}
    @nospecialize model
    roots = table.roots
    arguments = merge(model.args, model.defaults)
    seen = VarName[]
    copied = Set{Symbol}()
    unused = filter(n -> n ∉ AbstractPPL.getsym.(first.(entries)), keys(schema))
    templates = map(n -> (VarName{n}(), NoBinding(), NoTemplate()), unused)
    for (vn, value, supplied) in (templates..., entries...)
        value isa NoBinding || _check_binding_address(model, vn)
        _find_placeholder(value) !== :none &&
            _throw_binding_error(vn, "explicit bindings cannot contain missing or nothing")
        name = AbstractPPL.getsym(vn)
        base = VarName{name}()
        optic = AbstractPPL.getoptic(vn)
        old = get(roots.data, name, BindingRoot(NoBinding(), NoTemplate(), ()))
        argument = name in _lhs_arguments(model)
        original = argument ? get(arguments, name, NoTemplate()) : NoTemplate()
        storage, constraints = _resolve_root(model, old, vn, R, schema, supplied, original)
        if value isa NoBinding
            roots = setindex!!(roots, BindingRoot(old.value, storage, constraints), base)
            continue
        end
        if optic isa AbstractPPL.Iden && (
            !(old.value isa NoBinding) || !isempty(constraints) || storage isa VarNamedTuple
        )
            _same_shape(storage, value) ||
                _throw_binding_error(vn, "remove the whole binding before changing shape")
            (
                storage isa VarNamedTuple ||
                old.value isa VarNamedTuple && storage isa NoTemplate
            ) && _throw_binding_error(vn, "namespace cannot become a whole value")
        end
        if value isa VarNamedTuples.PartialArray && !any(value.mask)
            node = old.value
            if _binding_node_at(node, optic) isa Union{Nothing,NoBinding}
                empty = VarNamedTuples.PartialArray(
                    similar(value.data, BoundValue{R,eltype(value)}), copy(value.mask)
                )
                node = _set_binding_node(
                    node isa NoBinding ? node : copy(node), name, empty, vn, storage
                )
            end
            roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
            continue
        end
        argument && !(storage isa NoTemplate) && _check_partial_storage(storage, optic, vn)
        # Storage from a child model argument only locates indexed positions; it is not
        # retained, so whole and field bindings keep their unconstrained behaviour.
        located = if storage isa NoTemplate && _has_index(optic)
            _site_storage(model, base)
        else
            storage
        end
        selections = _expand_selection(located, optic, base, value, old.value)
        node = old.value
        if !(name in copied)
            node isa _InnerNode && (node = copy(node))
            push!(copied, name)
        end
        for (address, element) in selections
            any(other -> subsumes(other, address) || subsumes(address, other), seen) &&
                _throw_binding_error(address, "inputs overlap in this call")
            push!(seen, address)
            node, storage, constraints = _write_selection(
                node, storage, constraints, R, vn, address, element, located, argument
            )
        end
        if isempty(selections)
            node = _set_binding_node(node, name, map(BoundValue{R}, value), vn, storage)
        end
        node = _collapse_fields(node, constraints)
        roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
    end
    return BindingTable(roots)
end

"""
    condition(model::Model, values...; kwargs...)

Bind observations. Pass an optional storage template last.
"""
AbstractPPL.condition(model::Model, inputs...; kwargs...) =
    _bind(model, ConditionRole, inputs; kwargs...)
"""
    fix(model::Model, values...; kwargs...)

Bind values whose distributions contribute no density. Pass an optional storage template last.
"""
fix(model::Model, inputs...; kwargs...) = _bind(model, FixRole, inputs; kwargs...)
function _bind(
    model, ::Type{R}, inputs::Tuple{NamedTuple{N,Tuple{T}}}; kwargs...
) where {R,N,T<:Number}
    vn = VarName{only(N)}()
    old = _binding_root(model, vn)
    if isempty(kwargs) &&
        _model_prefix(model) === nothing &&
        !any(is_splat_symbol, keys(model.defaults)) &&
        (
            old === nothing ||
            isempty(old.constraints) &&
            old.storage isa Union{Number,Missing,Nothing,NoTemplate} &&
            (
                old.value isa NoBinding ||
                old.value isa BoundValue && old.value.value isa Number
            )
        )
        _check_binding_address(model, vn)
        value = only(values(only(inputs)))
        incoming = BoundValue{R}(value)
        old === nothing || _check_role_overlap(old.value, incoming, vn)
        table = BindingTable(
            setindex!!(model.values.roots, BindingRoot(incoming, value, ()), vn)
        )
        result = _reconstruct_model(model; values=table)
        _check_argument_bindings(result)
        _check_argument_aliasing(result)
        return result
    end
    return invoke(_bind, Tuple{Any,Type,Any}, model, R, inputs; kwargs...)
end
function _bind(model, ::Type{R}, inputs; kwargs...) where {R}
    @nospecialize model
    flat = _binding_inputs(inputs)
    count(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, flat) <= 1 ||
        throw(ArgumentError("Only one binding template is allowed"))
    schema = (;)
    entries = ()
    for (i, input) in enumerate((flat..., (; kwargs...)))
        if input isa Type{<:AbstractPPL.OfNamedTuple}
            i == length(flat) && !isempty(inputs) && last(inputs) === input || throw(
                ArgumentError("Binding template must be the final positional argument")
            )
            schema = zero(input)
            prefix = _model_prefix(model)
            if prefix !== nothing
                schema = _binding_node_at(schema, AbstractPPL.varname_to_optic(prefix))
                schema isa NamedTuple ||
                    _throw_binding_error(prefix, "template needs absolute names")
            end
            continue
        end
        for (vn, value, storage) in _flatten_bindings(input)
            name = AbstractPPL.getsym(vn)
            if any(
                n -> is_splat_symbol(n) && unsplat_symbol(n) == name, keys(model.defaults)
            )
                AbstractPPL.getoptic(vn) isa AbstractPPL.Iden ||
                    _throw_binding_error(vn, "splat arguments require whole bindings")
            end
            local_vn = _relative_name(model, vn)
            local_vn === nothing &&
                _throw_binding_error(vn, "address is not below model prefix")
            prefix = _model_prefix(model)
            if prefix !== nothing && !(storage isa NoTemplate)
                root = AbstractPPL.prefix(VarName{AbstractPPL.getsym(local_vn)}(), prefix)
                storage = _binding_node_at(storage, AbstractPPL.getoptic(root))
            end
            entries = (
                entries..., _split_namespace_entry(model, local_vn, value, storage)...
            )
        end
    end
    table = _add_bindings(model, model.values, R, entries, schema)
    result = @noinline _reconstruct_model(model; values=table)
    _check_argument_bindings(result)
    _check_argument_aliasing(result)
    return result
end
function Base.:|(
    model::Model, values::Union{NamedTuple,VarNamedTuple,Pair,Tuple,AbstractDict}
)
    any(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, _binding_inputs(values)) &&
        throw(ArgumentError("Templates require condition or fix"))
    return condition(model, values)
end
"""
    conditioned(model::Model)

Return observations, including argument observations, as a `VarNamedTuple`.
"""
conditioned(model::Model) = _role_bindings(model, ConditionRole)
"""
    fixed(model::Model)

Return fixed values as a `VarNamedTuple`.
"""
fixed(model::Model) = _role_bindings(model, FixRole)
function _role_bindings(model, R)
    result = VarNamedTuple()
    for name in keys(model.values.roots.data)
        root = model.values.roots.data[name]
        value = _role_values(root.value, R)
        value isa NoBinding || (result = setindex!!(result, value, VarName{name}()))
    end
    prefix = _model_prefix(model)
    (prefix === nothing || isempty(result)) && return result
    return templated_setindex!!(
        VarNamedTuple(),
        result,
        prefix,
        _prefixed_template(_model_prefix_template(model), NoTemplate()),
    )
end
_removal_markers(::Type{ConditionRole}, model) = model.observation_removals
_removal_markers(::Type{FixRole}, model) = model.fixed_removals
function _remove_role(table, R, names; arguments=(;), validate=true)
    roots = table.roots
    for name in keys(roots.data)
        root = roots.data[name]
        base = VarName{name}()
        selectors =
            isempty(names) ? (base,) : filter(vn -> AbstractPPL.getsym(vn) == name, names)
        occupied = _has_bound_value(root.value)
        node = if root.value isa _InnerNode
            copy(root.value)
        else
            root.value
        end
        for vn in selectors
            root.value isa NoBinding &&
                (!validate || _unwrap_storage(root.storage) isa NoTemplate) &&
                continue
            for (address, _) in _expand_selection(
                root.storage, AbstractPPL.getoptic(vn), base, NoBinding(), node
            )
                optic = AbstractPPL.getoptic(address)
                # Prune once per root below: a scan per address is quadratic.
                node = if optic isa AbstractPPL.Iden
                    _delete_role_at(node, optic, R, address)
                else
                    _delete_role_at(node, optic, R, address; prune=false)
                end
            end
        end
        !occupied || isempty(selectors) || _has_bound_value(node) || (node = NoBinding())
        constraints = filter(_effective_constraints(root, base)) do c
            (c.role === R && any(vn -> subsumes(vn, c.owner), selectors)) || return true
            return c.mode === Val(:type) && !any(==(base), selectors)
        end
        storage = if base in selectors && node isa NoBinding && haskey(arguments, name)
            arguments[name]
        elseif node isa NoBinding &&
            isempty(constraints) &&
            !(
                root.storage isa VarNamedTuple ||
                _element_optic(root.storage) === AbstractPPL.Property
            )
            get(arguments, name, NoTemplate())
        else
            root.storage
        end
        node = _collapse_fields(node, constraints)
        roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
    end
    return BindingTable(roots)
end
"""
    decondition(model::Model, names...)

Remove observations recursively; omit names to remove all observations.
"""
AbstractPPL.decondition(model::Model, names::Union{Symbol,VarName}...) =
    _unbind_recursive(model, names, ConditionRole)
"""
    unfix(model::Model, names...)

Remove fixed values recursively; omit names to remove all fixed values.
"""
unfix(model::Model, names::Union{Symbol,VarName}...) =
    _unbind_recursive(model, names, FixRole)
function _relative_names(model, names)
    names = map(n -> _relative_name(model, n isa Symbol ? VarName{n}() : n), names)
    return any(isnothing, names) ? () : names
end
# Remove `R` bindings at `names` from `model`'s own table; submodels keep theirs.
function _unbind_local(model, names, R)
    names = _relative_names(model, names)
    for vn in names
        _check_binding_address(model, vn)
        root = _binding_root(model, vn)
        root === nothing || _check_field_under_whole(model, vn, root.storage)
    end
    table = _remove_role(
        model.values, R, names; arguments=merge(model.args, model.defaults)
    )
    result = _reconstruct_model(model; values=table)
    _check_argument_aliasing(result)
    return result
end
# Also record removal markers, which each submodel applies to its own bindings when reached.
function _unbind_recursive(model, names, R)
    result = _unbind_local(model, names, R)
    names = _relative_names(model, names)
    markers = if isempty(names)
        ((name=nothing, active=true),)
    else
        # A repeated name would grow the model type on every call.
        foldl(names; init=_removal_markers(R, model)) do markers, name
            any(m -> isequal(m.name, name), markers) && return markers
            return (markers..., (; name, active=true))
        end
    end
    return if R === ConditionRole
        _reconstruct_model(result; observation_removals=markers)
    else
        _reconstruct_model(result; fixed_removals=markers)
    end
end

function _prepare_argument(binding, value, vn, owned, storage=value)
    binding isa BoundValue{ArgumentRole} && return value
    binding isa BoundValue && return binding.value
    if !owned && _is_fully_bound(binding, storage)
        return _bound_data(binding, storage)
    end
    source = _latent_source(value, storage)
    return _fill_latent_storage(binding, source, vn, storage, owned && source === value)
end
function _latent_source(value, storage)
    value === storage && return value
    (value isa Union{Missing,Nothing} || _element_optic(storage) === nothing) &&
        return value
    return if _element_optic(value) !== nothing && _same_shape(value, storage, true)
        value
    else
        storage
    end
end
function _fill_latent_storage(
    binding, value::Union{Missing,Nothing,NoTemplate}, vn, storage, owned
)
    binding isa _InnerNode || return value
    return _throw_missing_storage(vn)
end
@generated function _fill_latent_storage(
    binding, value, vn, storage, owned, ::Val{N}
) where {N}
    fields = map(N) do name
        quote
            optic = AbstractPPL.Property{$(QuoteNode(name))}()
            retained =
                if _element_optic(storage) === AbstractPPL.Property &&
                    hasproperty(storage, $(QuoteNode(name)))
                    storage.$name
                else
                    value.$name
                end
            _prepare_argument(
                _binding_node_at(binding, optic),
                value.$name,
                AbstractPPL.append_optic(vn, optic),
                owned,
                retained,
            )
        end
    end
    return :(ConstructionBase.setproperties(value, NamedTuple{$N}(($(fields...),))))
end
function _throw_unsupported_storage(::NoTemplate, vn)
    return _throw_binding_error(
        vn,
        "indexed binding lacks storage; pass an `@of`/`of` template as the last positional argument, or bind the whole value",
    )
end
function _throw_unsupported_storage(value, vn)
    remedy = if value isa AbstractArray
        "use collect or pass a plain Array before constructing the model (axes or metadata may be lost)"
    else
        "use supported Array/NamedTuple storage and matching index/field addresses"
    end
    return _throw_binding_error(vn, "unsupported binding storage $(typeof(value)); $remedy")
end
function _check_partial_storage(value, vn)
    if _element_optic(value) === nothing &&
        value isa Union{AbstractArray,AbstractDict,Tuple}
        _throw_unsupported_storage(value, vn)
    end
    return value
end
function _fill_latent_storage(
    binding::VarNamedTuples.PartialArray{<:BoundValue},
    value::Array{<:Union{Missing,Number}},
    vn,
    storage,
    owned,
)
    result = if value isa Array{Missing}
        copyto!(_argument_buffer(value, binding), value)
    else
        owned ? value : copy(value)
    end
    for i in _positional_indices(value)
        indices = _index_tuple(i)
        checkbounds(Bool, binding.mask, indices...) && binding.mask[indices...] || continue
        binding.data[indices...] isa BoundValue{ArgumentRole} && continue
        result = _set_optic!!(
            result, binding.data[indices...].value, AbstractPPL.Index(indices, (;)), vn
        )
    end
    return result
end
function _fill_latent_storage(binding, value, vn, storage, owned)
    _check_partial_storage(value, vn)
    if _element_optic(value) === AbstractPPL.Index
        result = owned ? value : _argument_buffer(value, binding)
        for i in _positional_indices(value)
            optic = AbstractPPL.Index(_index_tuple(i), (;))
            child = _binding_node_at(binding, optic)
            prepared = if child isa BoundValue
                child isa BoundValue{ArgumentRole} ? value[i] : child.value
            else
                _isassigned(value, i) || continue
                _prepare_argument(
                    child, value[i], AbstractPPL.append_optic(vn, optic), owned
                )
            end
            result = _set_optic!!(result, prepared, optic, vn)
        end
        return result
    elseif _element_optic(value) === AbstractPPL.Property
        return _fill_latent_storage(
            binding, value, vn, storage, owned, Val(propertynames(value))
        )
    else
        return value
    end
end
function prepare_model_argument(model::Model, vn::VarName, value)
    name = AbstractPPL.getsym(vn)
    # Distinct positional storage is owned: convert_model_argument deep-copies it.
    owned = haskey(model.args, name) && value !== model.args[name]
    root = _binding_root(model, vn)
    root === nothing && return _prepare_argument(nothing, value, vn, owned)
    return _prepare_argument(
        root.value, value, maybe_prefix(vn, _model_prefix(model)), owned, root.storage
    )
end
@generated function _check_argument_bindings(model::Model{F,A,D}) where {F,A,D}
    calls = map(union(A, D)) do n
        :(_check_argument_binding(
            model, Val($(QuoteNode(n))), Val($(QuoteNode(unsplat_symbol(n))))
        ))
    end
    return Expr(:block, calls..., :(nothing))
end
@inline function _check_argument_binding(
    model, ::Val{stored}, ::Val{name}
) where {stored,name}
    arguments = merge(model.args, model.defaults)
    vn = maybe_prefix(VarName{name}(), _model_prefix(model))
    root = _binding_root(model, VarName{name}())
    root === nothing && return nothing
    name in _lhs_arguments(model) || _throw_binding_error(vn, "covariates cannot be bound")
    original = arguments[stored]
    if original isa Union{Missing,Nothing} &&
        root.value isa _InnerNode &&
        !_is_fully_bound(root.value, root.storage)
        _throw_missing_storage(vn)
    end
    if is_splat_symbol(stored) &&
        haskey(model.defaults, stored) &&
        !(root.value isa Union{BoundValue,NoBinding})
        _throw_binding_error(vn, "splat arguments require whole bindings")
    elseif root.value isa BoundValue || _is_fully_bound(root.value, root.storage)
        T = _declared_argument_type(_model_metadata(model), stored)
        _bound_data(root.value, root.storage) isa T ||
            _throw_binding_error(vn, "binding requires declared argument type $T")
    end
    return nothing
end
function _collect_argument_arrays!(arrays, value, binding, vn, parents, storage=value)
    binding isa Union{BoundValue{ConditionRole},BoundValue{FixRole}} && return arrays
    if binding isa _InnerNode && _is_fully_bound(binding, storage)
        binding = BoundValue{ArgumentRole}(_bound_data(binding, storage))
    end
    value isa Union{Missing,Nothing} && binding isa _InnerNode && _throw_missing_storage(vn)
    value = if binding isa BoundValue
        binding.value
    else
        _check_partial_storage(_latent_source(value, storage), vn)
    end
    value isa AbstractArray && push!(arrays, (value, !(binding isa BoundValue), vn))
    kind = _element_optic(value)
    kind === nothing && return arrays
    if value in parents
        binding isa BoundValue && return arrays
        _throw_binding_error(vn, "cyclic latent storage is unsupported")
    end
    children = if kind === AbstractPPL.Index
        T = Base.nonmissingtype(eltype(value))
        (eltype(value) === Missing || isprimitivetype(T) && T <: Number) &&
            return arrays
        (
            (AbstractPPL.Index(_index_tuple(i), (;)), value[i]) for
            i in _positional_indices(value) if _isassigned(value, i)
        )
    else
        binding isa BoundValue ||
            ConstructionBase.setproperties(value, ConstructionBase.getproperties(value))
        ((AbstractPPL.Property{n}(), getproperty(value, n)) for n in propertynames(value))
    end
    push!(parents, value)
    for (optic, child) in children
        _collect_argument_arrays!(
            arrays,
            child,
            if binding isa BoundValue
                BoundValue{ArgumentRole}(child)
            else
                _binding_node_at(binding, optic)
            end,
            AbstractPPL.append_optic(vn, optic),
            parents,
            kind === AbstractPPL.Property ? _binding_node_at(storage, optic) : child,
        )
    end
    delete!(parents, value)
    return arrays
end
@generated function _collect_argument_arrays!(arrays, model::Model{F,A,D}) where {F,A,D}
    calls = map(union(A, D)) do stored
        name = unsplat_symbol(stored)
        quote
            value = arguments.$stored
            vn = $(VarName{name}())
            lhs = $(QuoteNode(name)) in _lhs_arguments(model)
            binding = lhs ? _binding_at(model, vn) : BoundValue{ArgumentRole}(value)
            _collect_argument_arrays!(
                arrays,
                value,
                binding,
                maybe_prefix(vn, _model_prefix(model)),
                Base.IdSet{Any}(),
                lhs ? _binding_root(model, vn).storage : value,
            )
        end
    end
    return quote
        arguments = merge(model.args, model.defaults)
        $(calls...)
        arrays
    end
end

function _check_argument_aliasing(model)
    any(
        root -> !_is_fully_bound(root.value, root.storage), values(model.values.roots.data)
    ) || return nothing
    arrays = Tuple[]
    _collect_argument_arrays!(arrays, model)
    for i in eachindex(arrays)
        a, latent, _ = arrays[i]
        latent || continue
        for j in eachindex(arrays)
            b, other_latent, _ = arrays[j]
            (i == j || (other_latent && j > i)) && continue
            (ismutable(a) && a === b || Base.mightalias(a, b)) && _throw_binding_error(
                arrays[max(i, j)][3], "latent storage aliases another argument path"
            )
        end
    end
    return nothing
end

_skip_bound_shape_check(value) = false
