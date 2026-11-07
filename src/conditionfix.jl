_binding_check_child(ctx::AbstractContext, parent, child, prefix, template) = ctx
function _binding_check_child(ctx::AbstractParentContext, parent, child, prefix, template)
    inner = childcontext(ctx)
    checked = _binding_check_child(inner, parent, child, prefix, template)
    return checked === inner ? ctx : setchildcontext(ctx, checked)
end
_binding_check_model(::AbstractContext, model) = nothing
function _binding_check_model(ctx::AbstractParentContext, model)
    return _binding_check_model(childcontext(ctx), model)
end
_binding_check_read(::AbstractContext, model, vn, template) = nothing
function _binding_check_read(ctx::AbstractParentContext, model, vn, template)
    return _binding_check_read(childcontext(ctx), model, vn, template)
end

Base.@constprop :aggressive function _initial_bindings(arguments, metadata)
    return _initial_bindings(arguments, Val(_args_on_lhs(metadata)))
end
@generated function _initial_bindings(arguments::NamedTuple{N}, ::Val{L}) where {N,L}
    names = Tuple(n for n in N if unsplat_symbol(n) in L)
    fields = map(names) do n
        :(BindingRoot(_argument_observation(arguments.$n), arguments.$n, ()))
    end
    roots = Tuple(unsplat_symbol(n) for n in names)
    return :(BindingTable(VarNamedTuple(NamedTuple{$roots}(($(fields...),)))))
end
_argument_observation(value) = BoundValue{ArgumentRole}(value)
_argument_observation(::Union{Missing,Nothing}) = NoBinding()
function _root(model, vn)
    return get(model.values.roots.data, AbstractPPL.getsym(vn), nothing)
end
function _get_model_binding(model, vn)
    root = _root(model, vn)
    return root === nothing ? nothing : _node_at(root.value, AbstractPPL.getoptic(vn))
end
_get_model_data(model, vn) = _model_data(_get_model_binding(model, vn))
function _get_model_data(model, vn, root, local_value)
    _check_runtime_shape(model, vn, local_value, FixRole)
    return _get_model_data(model, vn)
end
function _get_argument_role(model, vn, argument, local_value=NoTemplate())
    binding = _get_model_binding(model, argument)
    (
        binding isa Union{BoundValue{ConditionRole},BoundValue{ArgumentRole}} ||
        local_value isa Base.Pairs && binding isa BoundValue
    ) && return _model_role(binding, vn)
    original = get(merge(model.args, model.defaults), AbstractPPL.getsym(argument), nothing)
    if local_value isa AbstractArray{>:Missing} &&
        _binding_optic(local_value) === AbstractPPL.Index &&
        !(original isa Union{Missing,Nothing})
        value = _node_at(local_value, AbstractPPL.getoptic(vn))
        value isa AbstractArray && any(ismissing, value) && return ConditionRole()
    end
    return _local_binding_role(binding, vn, local_value, model, ArgumentRole)
end
function _local_binding_role(binding, vn, template, model, role=ConditionRole)
    found = _model_role_at(
        binding, AbstractPPL.getoptic(vn), maybe_prefix(vn, _model_prefix(model))
    )
    if role !== ArgumentRole && (
        found !== nothing ||
        binding isa
        Union{BoundValue{ConditionRole},VarNamedTuple,VarNamedTuples.PartialArray}
    )
        _check_runtime_shape(model, vn, template, role)
    end
    return found
end
function _check_runtime_shape(model, vn, local_value, role; submodel=false)
    local_value isa NoTemplate && return nothing
    root = _root(model, vn)
    root === nothing && return nothing
    _check_runtime_path(
        root.value, local_value, AbstractPPL.getoptic(vn), vn, role; submodel
    )
    if root.value isa VarNamedTuples.PartialArray
        local_value isa Union{AbstractArray,Tuple} &&
            axes(root.value.data) == axes(local_value) ||
            _binding_error(vn, "local storage shape changed")
    end
    for constraint in root.constraints
        (; owner, storage) = constraint
        constraint.role === role || continue
        _skip_bound_shape_check(storage) && continue
        subsumes(owner, vn) || continue
        value = _node_at(local_value, AbstractPPL.getoptic(owner))
        value === nothing && _binding_error(vn, "local storage omits bound fields")
        _runtime_shape(storage, value) || _binding_error(vn, "local storage shape changed")
    end
    return nothing
end
function _check_runtime_path(binding, value, optic, vn, role; submodel=false)
    binding isa Union{Nothing,NoBinding} && return nothing
    submodel && optic isa AbstractPPL.Iden && return nothing
    if binding isa BoundValue{role} && !_skip_bound_shape_check(binding.value)
        _runtime_shape(binding.value, value) ||
            _binding_error(vn, "local storage shape changed")
    end
    (optic isa AbstractPPL.Iden || submodel && optic.child isa AbstractPPL.Iden) &&
        return nothing
    head = AbstractPPL.ohead(optic)
    head isa AbstractPPL.Index && (head = AbstractPPL.concretize_top_level(head, value))
    child = _node_at(binding, head)
    child === nothing && role === ConditionRole && return nothing
    child === nothing && _binding_error(vn, "fixed binding omits an executed address")
    if head isa AbstractPPL.Index &&
        value isa AbstractArray &&
        !VarNamedTuples._is_multiindex(value, head.ix...) &&
        checkbounds(Bool, value, head.ix...) &&
        !_isassigned(value, head.ix...)
        return nothing
    end
    local_child = _node_at(value, head)
    local_child === nothing && _binding_error(vn, "local storage omits a bound address")
    return _check_runtime_path(child, local_child, optic.child, vn, role; submodel)
end
function _check_namedtuple_index(value, optic, prefix=AbstractPPL.Iden())
    optic isa AbstractPPL.Iden && return nothing
    value = _binding_storage(value)
    VarNamedTuples._check_tuple_storage(value)
    kind = _binding_optic(value isa VarNamedTuple ? value.data : value)
    (kind !== nothing && !(optic isa kind)) && _binding_error(
        AbstractPPL.optic_to_varname(optic ∘ prefix),
        "address kind disagrees with storage",
    )
    optic.child isa AbstractPPL.Iden && return nothing
    child = _node_at(value, AbstractPPL.ohead(optic))
    child === nothing || _check_namedtuple_index(child, optic.child, prefix)
    return nothing
end

_binding_inputs(x) = (x,)
_binding_inputs(x::Tuple) = mapreduce(_binding_inputs, (a, b) -> (a..., b...), x; init=())
_binding_inputs(p::Pair{Symbol}) = (VarName{first(p)}() => last(p),)
function _binding_entries(input::VarNamedTuple)
    entries = Tuple[]
    for (name, value) in pairs(input.data)
        _binding_entries!(entries, value, VarName{name}(), value)
    end
    return entries
end
function _binding_entries!(entries, value::VarNamedTuple, vn, storage)
    for (name, child) in pairs(value.data)
        _binding_entries!(
            entries,
            child,
            AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()),
            storage,
        )
    end
    return entries
end
function _binding_entries!(entries, value::VarNamedTuples.PartialArray, vn, storage)
    if !any(value.mask)
        push!(entries, (vn, value, storage))
    else
        for i in CartesianIndices(value.mask)
            value.mask[i] || continue
            _binding_entries!(
                entries,
                value.data[i],
                AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(i), (;))),
                storage,
            )
        end
    end
    return entries
end
_binding_entries!(entries, value, vn, storage) = push!(entries, (vn, value, storage))
function _binding_entries(input::NamedTuple{N}) where {N}
    any(x -> x isa Union{VarNamedTuple,VarNamedTuples.PartialArray}, values(input)) &&
        return _binding_entries(VarNamedTuple(input))
    return map(n -> (VarName{n}(), input[n], NoTemplate()), N)
end
_binding_entries(p::Pair{<:VarName}) = ((first(p), last(p), NoTemplate()),)
function _binding_entries(input)
    throw(
        ArgumentError("Bindings require NamedTuple, VarNamedTuple, or Symbol/VarName pairs")
    )
end
function _public_binding_name(model, vn)
    prefix = _model_prefix(model)
    prefix === nothing && return vn
    template = _apply_prefix_template(_model_prefix_template(model), NoTemplate())
    depth = optic_skip_length(AbstractPPL.getoptic(prefix))
    prefix = _expand_cartesian(_concretize_prefix(prefix, template; depth, canonical=true))
    vn = _expand_cartesian(_concretize_prefix(vn, template; depth, canonical=true))
    subsumes(prefix, vn) && return prefix == vn ? nothing : AbstractPPL.unprefix(vn, prefix)
    subsumes(vn, prefix) && return nothing
    return _binding_error(vn, "address is outside model prefix $prefix")
end
function _check_binding_address(model, vn, address=vn)
    _check_selection_indices(AbstractPPL.getoptic(vn), address)
    names = _lhs_names(_binding_metadata(model))
    names === nothing ||
        AbstractPPL.getsym(vn) in names ||
        _binding_error(address, "not a tilde LHS root")
    return nothing
end
function _known_namespace(model, vn)
    root = _root(model, vn)
    root !== nothing &&
        _node_at(root.storage, AbstractPPL.getoptic(vn)) isa VarNamedTuple &&
        return true
    any(p -> last(p) && first(p) == vn, _lhs_addresses(_binding_metadata(model))) &&
        return true
    arguments = merge(model.args, model.defaults)
    for (site, name) in _submodel_arguments(_binding_metadata(model))
        site === nothing && continue
        name isa Symbol || continue
        child = get(arguments, name, nothing)
        child isa Model || continue
        prefix = maybe_prefix(_model_prefix(child), site)
        subsumes(vn, prefix) && return true
        subsumes(prefix, vn) &&
            return _known_namespace(child, AbstractPPL.unprefix(vn, prefix))
    end
    return false
end
function _expand_binding_entries(model, vn, value, storage)
    if value isa NamedTuple && _known_namespace(model, vn)
        return mapreduce((a, b) -> (a..., b...), keys(value); init=()) do name
            _expand_binding_entries(
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
function _check_ambiguous_field(model, vn, storage)
    any(
        ((a, child),) -> child && subsumes(a, vn), _lhs_addresses(_binding_metadata(model))
    ) || return nothing
    optic = AbstractPPL.getoptic(vn)
    base = VarName{AbstractPPL.getsym(vn)}()
    while !(optic isa AbstractPPL.Iden)
        node = _get_model_binding(model, base)
        retained = _node_at(storage, AbstractPPL.getoptic(base))
        root = _root(model, base)
        typed =
            root !== nothing &&
            any(c -> c.typed === Val(true) && subsumes(c.owner, base), root.constraints)
        if node isa BoundValue &&
            _binding_optic(node.value) === AbstractPPL.Property &&
            (
                !(
                    retained isa VarNamedTuple ||
                    _binding_optic(retained) === AbstractPPL.Property
                ) || typed
            )
            _binding_error(vn, "field edit below an unresolved child needs storage")
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
        _node_at(storage, head), optic.child, AbstractPPL.append_optic(vn, head)
    )
end
function _constrained_value(value, optic, address, R, constraints; error_address=address)
    for constraint in constraints
        (; owner, role) = constraint
        template = constraint.storage
        typed = constraint.typed === Val(true)
        if owner != address && subsumes(address, owner)
            selected = _node_at(
                value, AbstractPPL._unprefix_optic(AbstractPPL.getoptic(owner), optic)
            )
            selected === nothing &&
                _binding_error(error_address, "value omits retained fields")
            _same_storage_shape(template, selected) ||
                _binding_error(error_address, "value disagrees with retained shape")
            continue
        end
        subsumes(owner, address) || continue
        selected = _node_at(
            template, AbstractPPL._unprefix_optic(optic, AbstractPPL.getoptic(owner))
        )
        selected === nothing ||
            _same_storage_shape(selected, value) ||
            _binding_error(error_address, "value disagrees with retained shape")
        (role === R && (typed || owner != address)) || continue
        relative = AbstractPPL._unprefix_optic(optic, AbstractPPL.getoptic(owner))
        typed &&
            relative isa AbstractPPL.Iden &&
            !(value isa typeof(template)) &&
            _binding_error(error_address, "whole value must match template type")
        value = _partial_conversion(value, template, relative, error_address)
    end
    return value
end
function _set_root_node(node, name, value, vn, storage)
    wrapper = if node isa NoBinding
        VarNamedTuple()
    else
        VarNamedTuple(NamedTuple{(name,)}((node,)))
    end
    return templated_setindex!!(wrapper, value, vn, storage).data[name]
end
# Avoid inference per model history, but retain the role and stored types:
# `condition` can run inside an evaluator.
Base.@nospecializeinfer function _bind_table(
    model, table, ::Type{R}, entries, schema; disjoint=false
) where {R}
    @nospecialize model
    roots = table.roots
    seen = VarName[]
    copied = Set{Symbol}()
    unused = filter(n -> n ∉ AbstractPPL.getsym.(first.(entries)), keys(schema))
    templates = map(n -> (VarName{n}(), NoBinding(), NoTemplate()), unused)
    for (vn, value, supplied) in (templates..., entries...)
        value isa NoBinding || _check_binding_address(model, vn)
        _classify_placeholder(value) !== :none &&
            _binding_error(vn, "explicit bindings cannot contain missing or nothing")
        name = AbstractPPL.getsym(vn)
        base = VarName{name}()
        old = get(roots.data, name, BindingRoot(NoBinding(), NoTemplate(), ()))
        storage = _binding_storage(old.storage)
        offered = get(schema, name, _binding_storage(supplied))
        if offered isa NamedTuple && _known_namespace(model, base)
            offered = VarNamedTuple(offered)
        end
        arguments = merge(model.args, model.defaults)
        original = get(arguments, name, NoTemplate())
        if name in _args_on_lhs(model) && !(original isa Union{Missing,Nothing,NoTemplate})
            offered = NoTemplate()
        end
        if storage isa NoTemplate || (!(offered isa NoTemplate) && old.value isa NoBinding)
            storage = offered
        elseif !(offered isa NoTemplate)
            _same_storage_shape(storage, offered) ||
                _binding_error(vn, "template disagrees with retained storage")
        end
        if name in _args_on_lhs(model) &&
            original isa Union{Missing,Nothing} &&
            !(AbstractPPL.getoptic(vn) isa AbstractPPL.Iden) &&
            storage isa NoTemplate
            _missing_argument_error(vn)
        end
        _check_ambiguous_field(model, vn, storage)
        constraints = if !(offered isa NoTemplate) && old.value isa NoBinding
            ()
        else
            _root_constraints(old, base)
        end
        if !(offered isa NoTemplate) && (
            haskey(schema, name) ||
            old.storage isa NoTemplate && !(AbstractPPL.getoptic(vn) isa AbstractPPL.Iden)
        )
            owner, template = _template_owner(offered, AbstractPPL.getoptic(vn), base)
            template = _binding_storage(template)
            any(
                c ->
                    c.owner == owner &&
                        c.role === R &&
                        c.typed === Val(true) &&
                        typeof(c.storage) !== typeof(template),
                constraints,
            ) && _binding_error(vn, "template disagrees with retained type")
            held = _node_at(old.value, AbstractPPL.getoptic(owner))
            if held isa BoundValue
                _same_storage_shape(held.value, template) ||
                    _binding_error(vn, "template disagrees with held shape")
                _matches_model_role(R, held) &&
                    !(held.value isa typeof(template)) &&
                    _binding_error(vn, "template disagrees with held type")
            end
            if !(template isa Union{Nothing,NoTemplate}) && !(
                haskey(schema, name) && any(
                    c -> c.owner == owner && c.role === R && c.typed === Val(true),
                    constraints,
                )
            )
                constraints = (
                    constraints...,
                    BindingConstraint(owner, R, template, haskey(schema, name)),
                )
            end
        end
        if value isa NoBinding
            roots = setindex!!(roots, BindingRoot(old.value, storage, constraints), base)
            continue
        end
        if AbstractPPL.getoptic(vn) isa AbstractPPL.Iden && (
            !(old.value isa NoBinding) || !isempty(constraints) || storage isa VarNamedTuple
        )
            _same_storage_shape(storage, value) ||
                _binding_error(vn, "remove the whole binding before changing shape")
            (
                storage isa VarNamedTuple ||
                old.value isa VarNamedTuple && storage isa NoTemplate
            ) && _binding_error(vn, "namespace cannot become a whole value")
        end
        if value isa VarNamedTuples.PartialArray && !any(value.mask)
            node = old.value
            if _node_at(node, AbstractPPL.getoptic(vn)) isa Union{Nothing,NoBinding}
                empty = VarNamedTuples.PartialArray(
                    similar(value.data, BoundValue{R,eltype(value)}), copy(value.mask)
                )
                node = _set_root_node(
                    node isa NoBinding ? node : copy(node), name, empty, vn, storage
                )
            end
            roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
            continue
        end
        if name in _args_on_lhs(model) && !(storage isa NoTemplate)
            _check_partial_storage(storage, AbstractPPL.getoptic(vn), vn)
        end
        selections = _selection(storage, AbstractPPL.getoptic(vn), base, value, old.value)
        node = old.value
        name in copied ||
            (node = if node isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
                copy(node)
            else
                node
            end;
            push!(copied, name))
        for (address, element) in selections
            any(other -> subsumes(other, address) || subsumes(address, other), seen) &&
                _binding_error(address, "inputs overlap in this call")
            push!(seen, address)
            optic = AbstractPPL.getoptic(address)
            previous = _node_at(node, optic)
            previous isa VarNamedTuple &&
                !(_binding_optic(_node_at(storage, optic)) === AbstractPPL.Property) &&
                _binding_error(address, "namespace cannot become a whole value")
            incoming = BoundValue{R}(element)
            _check_cross_roles(previous, incoming, address; disjoint)
            if !(optic isa AbstractPPL.Iden) && name in _args_on_lhs(model)
                element = _partial_conversion(element, storage, optic, address)
            end
            previous isa BoundValue && _same_storage_shape(previous.value, element) ||
                previous isa BoundValue &&
                    _binding_error(address, "remove the binding before changing shape")
            element = _constrained_value(element, optic, address, R, constraints)
            incoming = BoundValue{R}(element)
            if optic isa AbstractPPL.Iden
                node = incoming
                storage = element
                constraints = filter(c -> c.typed === Val(true), constraints)
            else
                node = _set_root_node(node, name, incoming, address, storage)
            end
            constraints = filter(
                c -> !(c.owner == address && c.role === R && c.typed === Val(false)),
                constraints,
            )
            # Numeric value nodes already enforce their role and shape.
            element isa Number || (
                constraints = (
                    constraints..., BindingConstraint(address, R, element, false)
                )
            )
        end
        if isempty(selections)
            node = _set_root_node(node, name, map(BoundValue{R}, value), vn, storage)
        end
        node = _collapse_bindings(node, constraints)
        roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
    end
    return BindingTable(roots)
end

"""
    condition(model::Model, values...; kwargs...)

Bind observations. Pass an optional storage template last.
"""
AbstractPPL.condition(model::Model, inputs...; kwargs...) =
    _conditionfix(model, ConditionRole, inputs; kwargs...)
"""
    fix(model::Model, values...; kwargs...)

Bind values whose distributions contribute no density. Pass an optional storage template last.
"""
fix(model::Model, inputs...; kwargs...) = _conditionfix(model, FixRole, inputs; kwargs...)
function _conditionfix(
    model, ::Type{R}, inputs::Tuple{NamedTuple{N,Tuple{T}}}; kwargs...
) where {R,N,T<:Number}
    vn = VarName{only(N)}()
    old = _root(model, vn)
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
        old === nothing || _check_cross_roles(old.value, incoming, vn)
        table = BindingTable(
            setindex!!(model.values.roots, BindingRoot(incoming, value, ()), vn)
        )
        result = _reconstruct_model(model; values=table)
        _check_argument_bindings(result)
        return _check_latent_arguments(result)
    end
    return invoke(_conditionfix, Tuple{Any,Any,Any}, model, R, inputs; kwargs...)
end
function _conditionfix(model, ::Type{R}, inputs; kwargs...) where {R}
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
                schema = _node_at(schema, AbstractPPL.varname_to_optic(prefix))
                schema isa NamedTuple ||
                    _binding_error(prefix, "template needs absolute names")
            end
            continue
        end
        for (vn, value, storage) in _binding_entries(input)
            name = AbstractPPL.getsym(vn)
            if any(
                n -> is_splat_symbol(n) && unsplat_symbol(n) == name, keys(model.defaults)
            )
                AbstractPPL.getoptic(vn) isa AbstractPPL.Iden ||
                    _binding_error(vn, "keyword splats require whole bindings")
                value = (; value...)
            end
            local_vn = _public_binding_name(model, vn)
            if local_vn === nothing
                prefix = _model_prefix(model)
                optic = AbstractPPL._unprefix_optic(
                    AbstractPPL.getoptic(prefix), AbstractPPL.getoptic(vn)
                )
                selected = _node_at(value, optic)
                selected isa NamedTuple ||
                    _binding_error(vn, "prefix binding requires named child values")
                for (n, v) in pairs(selected)
                    entries = (
                        entries...,
                        _expand_binding_entries(model, VarName{n}(), v, NoTemplate())...,
                    )
                end
            else
                entries = (
                    entries..., _expand_binding_entries(model, local_vn, value, storage)...
                )
            end
        end
    end
    table = _bind_table(model, model.values, R, entries, schema)
    metadata = _with_binding_template_names(
        _binding_metadata(model), _binding_template_names(schema)
    )
    result = _reconstruct_model(model; values=table, metadata)
    _check_argument_bindings(result)
    return _check_latent_arguments(result)
end
function _binding_template_names(schema::NamedTuple, prefix=nothing)
    return mapreduce((a, b) -> (a..., b...), keys(schema); init=()) do name
        vn = maybe_prefix(VarName{name}(), prefix)
        value = schema[name]
        value isa NamedTuple ? _binding_template_names(value, vn) : (vn,)
    end
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
conditioned(model::Model) = _listed_bindings(model, ConditionRole)
"""
    fixed(model::Model)

Return fixed values as a `VarNamedTuple`.
"""
fixed(model::Model) = _listed_bindings(model, FixRole)
function _listed_bindings(model, R)
    result = _table_listing(model.values, R)
    prefix = _model_prefix(model)
    (prefix === nothing || isempty(result)) && return result
    return templated_setindex!!(
        VarNamedTuple(),
        result,
        prefix,
        _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
    )
end
function _removal_matches(table, R, name)
    if name === nothing
        return any(root -> _has_binding_role(R, root.value), values(table.roots.data))
    end
    root = get(table.roots.data, AbstractPPL.getsym(name), nothing)
    root === nothing && return false
    return _has_binding_role(R, _node_at(root.value, AbstractPPL.getoptic(name)))
end
_removals(::Type{ConditionRole}, model) = model.observation_removals
_removals(::Type{FixRole}, model) = model.fixed_removals
function _remove_table(table, R, names; arguments=(;), validate=true)
    roots = table.roots
    for name in keys(roots.data)
        root = roots.data[name]
        base = VarName{name}()
        selectors =
            isempty(names) ? (base,) : filter(vn -> AbstractPPL.getsym(vn) == name, names)
        node = if root.value isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            copy(root.value)
        else
            root.value
        end
        for vn in selectors
            root.value isa NoBinding &&
                (!validate || _binding_storage(root.storage) isa NoTemplate) &&
                continue
            for (address, _) in
                _selection(root.storage, AbstractPPL.getoptic(vn), base, NoBinding(), node)
                node = _delete_binding(node, AbstractPPL.getoptic(address), R, address)
            end
        end
        previous = _root_constraints(root, base)
        constraints = filter(
            !isnothing,
            map(previous) do c
                (c.role === R && any(vn -> subsumes(vn, c.owner), selectors)) || return c
                c.typed === Val(true) && !any(==(base), selectors) && return c
                enclosing = any(previous) do parent
                    parent.owner != c.owner &&
                        subsumes(parent.owner, c.owner) &&
                        !any(vn -> subsumes(vn, parent.owner), selectors)
                end
                enclosing |=
                    haskey(arguments, name) &&
                    !any(==(base), selectors) &&
                    !(arguments[name] isa Union{Missing,Nothing}) &&
                    _storage_signature(c.storage) !== nothing
                enclosing || return nothing
                return BindingConstraint(c.owner, Nothing, c.storage, false)
            end,
        )
        storage = if base in selectors && node isa NoBinding && haskey(arguments, name)
            arguments[name]
        elseif node isa NoBinding &&
            isempty(constraints) &&
            !(
                root.storage isa VarNamedTuple ||
                _binding_optic(root.storage) === AbstractPPL.Property
            )
            get(arguments, name, NoTemplate())
        else
            root.storage
        end
        node = _collapse_bindings(node, constraints)
        roots = setindex!!(roots, BindingRoot(node, storage, constraints), base)
    end
    return BindingTable(roots)
end
"""
    decondition(model::Model, names...)

Remove observations recursively; omit names to remove all observations.
"""
AbstractPPL.decondition(model::Model, names::Union{Symbol,VarName}...) =
    _decondition_unfix(model, names, ConditionRole, Val(true))
"""
    unfix(model::Model, names...)

Remove fixed values recursively; omit names to remove all fixed values.
"""
unfix(model::Model, names::Union{Symbol,VarName}...) =
    _decondition_unfix(model, names, FixRole, Val(true))
function _decondition_unfix(model, names, R, ::Val{recursive}=Val(false)) where {recursive}
    names = map(n -> _public_binding_name(model, n isa Symbol ? VarName{n}() : n), names)
    any(isnothing, names) && (names = ())
    for vn in names
        _check_binding_address(model, vn)
        root = _root(model, vn)
        root === nothing || _check_ambiguous_field(model, vn, root.storage)
    end
    table = _remove_table(
        model.values, R, names; arguments=merge(model.args, model.defaults)
    )
    markers = _removals(R, model)
    if recursive
        markers = if isempty(names)
            ((name=nothing, active=true, matched=false),)
        else
            (
                markers...,
                map(names) do name
                    (; name, active=true, matched=_removal_matches(model.values, R, name))
                end...,
            )
        end
    end
    result = _reconstruct_model(
        model;
        values=table,
        observation_removals=R === ConditionRole ? markers : model.observation_removals,
        fixed_removals=R === FixRole ? markers : model.fixed_removals,
    )
    return _check_latent_arguments(result)
end

function _prepare_argument(binding, value, vn, storage=value)
    binding isa BoundValue && return binding.value
    if _complete_binding(binding, storage)
        return _model_data(binding, storage)
    end
    return _prepare_latent(binding, _latent_argument_source(value, storage), vn, storage)
end
function _latent_argument_source(value, storage)
    (value isa Union{Missing,Nothing} || _binding_optic(storage) === nothing) &&
        return value
    return if _binding_optic(value) !== nothing && _same_storage_shape(value, storage)
        value
    else
        storage
    end
end
function _prepare_latent(binding, value::Union{Missing,Nothing,NoTemplate}, vn, storage)
    binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} || return value
    return _missing_argument_error(vn)
end
@generated function _prepare_latent_fields(binding, value, vn, storage, ::Val{N}) where {N}
    fields = map(N) do name
        quote
            optic = AbstractPPL.Property{$(QuoteNode(name))}()
            retained =
                if _binding_optic(storage) === AbstractPPL.Property &&
                    hasproperty(storage, $(QuoteNode(name)))
                    storage.$name
                else
                    value.$name
                end
            _prepare_argument(
                _node_at(binding, optic),
                value.$name,
                AbstractPPL.append_optic(vn, optic),
                retained,
            )
        end
    end
    return :(ConstructionBase.setproperties(value, NamedTuple{$N}(($(fields...),))))
end
function _unsupported_binding_storage(value, vn)
    VarNamedTuples._check_tuple_storage(value)
    return _binding_error(
        vn,
        "unsupported binding storage $(typeof(value)); bind the whole value or use supported storage",
    )
end
function _check_latent_storage(value, vn)
    if _binding_optic(value) === nothing &&
        value isa Union{AbstractArray,AbstractDict,Tuple}
        _unsupported_binding_storage(value, vn)
    end
    return value
end
function _prepare_latent(binding, value, vn, storage)
    _check_latent_storage(value, vn)
    if _binding_optic(value) === AbstractPPL.Index
        result = _rebuild_buffer(value)
        for i in _positional_indices(value)
            optic = AbstractPPL.Index(_index_tuple(i), (;))
            child = _node_at(binding, optic)
            prepared = if child isa BoundValue
                child.value
            else
                _isassigned(value, i) || continue
                _prepare_argument(child, value[i], AbstractPPL.append_optic(vn, optic))
            end
            result = _setindex!!(result, prepared, _index_tuple(i)...)
        end
        return result
    elseif _binding_optic(value) === AbstractPPL.Property
        return _prepare_latent_fields(
            binding, value, vn, storage, Val(propertynames(value))
        )
    else
        return value
    end
end
function prepare_model_argument(model::Model, vn::VarName, value)
    root = _root(model, vn)
    root === nothing && return _prepare_argument(nothing, value, vn)
    return _prepare_argument(
        root.value, value, maybe_prefix(vn, _model_prefix(model)), root.storage
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
    root = _root(model, VarName{name}())
    root === nothing && return nothing
    name in _args_on_lhs(model) || _binding_error(vn, "covariates cannot be bound")
    original = arguments[stored]
    if original isa Union{Missing,Nothing} &&
        root.value isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        !_complete_binding(root.value, root.storage)
        _missing_argument_error(vn)
    end
    if root.value isa BoundValue || _complete_binding(root.value, root.storage)
        T = _declared_argument_type(_binding_metadata(model), stored)
        _model_data(root.value, root.storage) isa T ||
            _binding_error(vn, "binding requires declared argument type $T")
    elseif is_splat_symbol(stored)
        _binding_error(vn, "splat arguments require whole bindings")
    end
    return nothing
end
function _argument_arrays!(arrays, value, binding, vn, parents, storage=value)
    binding isa Union{BoundValue{ConditionRole},BoundValue{FixRole}} && return arrays
    if binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        _complete_binding(binding, storage)
        binding = BoundValue{ArgumentRole}(_model_data(binding, storage))
    end
    value isa Union{Missing,Nothing} &&
        binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        _missing_argument_error(vn)
    value = if binding isa BoundValue
        binding.value
    else
        _check_latent_storage(_latent_argument_source(value, storage), vn)
    end
    kind = _binding_optic(value)
    kind === nothing && return arrays
    if value in parents
        binding isa BoundValue && return arrays
        _binding_error(vn, "cyclic latent storage is unsupported")
    end
    children = if kind === AbstractPPL.Index
        value isa AbstractArray && push!(arrays, (value, !(binding isa BoundValue), vn))
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
        _argument_arrays!(
            arrays,
            child,
            if binding isa BoundValue
                BoundValue{ArgumentRole}(child)
            else
                _node_at(binding, optic)
            end,
            AbstractPPL.append_optic(vn, optic),
            parents,
            kind === AbstractPPL.Property ? _node_at(storage, optic) : child,
        )
    end
    delete!(parents, value)
    return arrays
end
_check_latent_arguments(model) = (_validate_latent_arguments(model); model)
function _validate_latent_arguments(model)
    any(
        root -> !_complete_binding(root.value, root.storage),
        values(model.values.roots.data),
    ) || return nothing
    arrays = Tuple[]
    for (stored, value) in pairs(merge(model.args, model.defaults))
        vn = VarName{unsplat_symbol(stored)}()
        lhs = AbstractPPL.getsym(vn) in _args_on_lhs(model)
        binding = lhs ? _get_model_binding(model, vn) : BoundValue{ArgumentRole}(value)
        _argument_arrays!(
            arrays,
            value,
            binding,
            maybe_prefix(vn, _model_prefix(model)),
            Base.IdSet{Any}(),
            lhs ? _root(model, vn).storage : value,
        )
    end
    for i in eachindex(arrays), j in 1:(i - 1)
        a, la, vn = arrays[i]
        b, lb, _ = arrays[j]
        (la || lb) &&
            (a === b || Base.mightalias(a, b)) &&
            _binding_error(vn, "latent storage aliases another argument path")
    end
    return nothing
end

_skip_bound_shape_check(value) = false
