function _binding_check_child(
    strategy::AbstractInitStrategy, parent, child, prefix, template
)
    return strategy
end
function _binding_check_child(ctx::Context, parent, child, prefix, template)
    checked = _binding_check_child(ctx.strategy, parent, child, prefix, template)
    return Context(ctx.rng, checked, ctx.transform_strategy)
end
_binding_check_model(::AbstractInitStrategy, model) = nothing
_binding_check_model(ctx::Context, model) = _binding_check_model(ctx.strategy, model)
_binding_check_read(::AbstractInitStrategy, model, vn, template) = nothing
function _binding_check_read(ctx::Context, model, vn, template)
    return _binding_check_read(ctx.strategy, model, vn, template)
end

Base.@constprop :aggressive function _argument_bindings(arguments, metadata)
    return _argument_bindings(arguments, Val(_lhs_arguments(metadata)))
end
@generated function _argument_bindings(arguments::NamedTuple{N}, ::Val{L}) where {N,L}
    names = Tuple(n for n in N if unsplat_symbol(n) in L)
    fields = map(names) do n
        :(BindingRoot(
            if arguments.$n isa Union{Missing,Nothing}
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
_bound_data_at(model, vn) = _bound_data(_binding_at(model, vn))
function _bound_data_at(model, vn, root, local_value)
    _check_local_shape(model, vn, local_value, FixRole)
    return _bound_data_at(model, vn)
end
function _argument_role(model, vn, argument, local_value=NoTemplate())
    binding = _binding_at(model, argument)
    (
        binding isa Union{BoundValue{ConditionRole},BoundValue{ArgumentRole}} ||
        local_value isa Base.Pairs && binding isa BoundValue
    ) && return _binding_role(binding, vn)
    original = get(merge(model.args, model.defaults), AbstractPPL.getsym(argument), nothing)
    if local_value isa AbstractArray{>:Missing} &&
        _binding_optic(local_value) === AbstractPPL.Index &&
        !(original isa Union{Missing,Nothing})
        value = _binding_node_at(local_value, AbstractPPL.getoptic(vn))
        value isa AbstractArray && any(ismissing, value) && return ConditionRole()
    end
    return _local_role(binding, vn, local_value, model, ArgumentRole)
end
function _local_role(binding, vn, template, model, role=ConditionRole)
    optic = AbstractPPL.getoptic(vn)
    address = maybe_prefix(vn, getprefix(model))
    _check_address_kind(binding, optic, AbstractPPL.Property{AbstractPPL.getsym(address)}())
    node = _binding_node_at(binding, optic)
    node === nothing &&
        binding isa BoundValue &&
        binding.value isa AbstractArray &&
        optic isa AbstractPPL.Index &&
        _binding_node_at(binding, AbstractPPL.ohead(optic)) === nothing &&
        _throw_binding_error(address, "executed index is outside bound storage")
    found = _binding_role(node, address)
    if role !== ArgumentRole && (
        found !== nothing ||
        binding isa
        Union{BoundValue{ConditionRole},VarNamedTuple,VarNamedTuples.PartialArray}
    )
        _check_local_shape(model, vn, template, role)
    end
    return found
end
function _check_local_shape(model, vn, local_value, role; submodel=false)
    local_value isa NoTemplate && return nothing
    root = _binding_root(model, vn)
    root === nothing && return nothing
    _check_local_shape(
        root.value, local_value, AbstractPPL.getoptic(vn), vn, role; submodel
    )
    if root.value isa VarNamedTuples.PartialArray
        local_value isa Union{AbstractArray,Tuple} &&
            axes(root.value.data) == axes(local_value) || _throw_local_shape_error(
            VarName{AbstractPPL.getsym(vn)}(), root.value.data, local_value
        )
    end
    for constraint in root.constraints
        (; owner, storage) = constraint
        constraint.role === role || continue
        _skip_bound_shape_check(storage) && continue
        subsumes(owner, vn) || continue
        value = _binding_node_at(local_value, AbstractPPL.getoptic(owner))
        value === nothing && _throw_binding_error(vn, "local storage omits bound fields")
        _matches_local_shape(storage, value) ||
            _throw_local_shape_error(owner, storage, value)
    end
    return nothing
end
function _throw_local_shape_error(vn, bound, local_value)
    detail =
        if bound isa Union{AbstractArray,Tuple} &&
            local_value isa Union{AbstractArray,Tuple}
            "; bound axes $(axes(bound)), local axes $(axes(local_value))"
        else
            ""
        end
    return _throw_binding_error(vn, "local storage shape changed" * detail)
end
function _check_local_shape(
    binding, value, optic, vn, role; submodel=false, owner=VarName{AbstractPPL.getsym(vn)}()
)
    binding isa Union{Nothing,NoBinding} && return nothing
    submodel && optic isa AbstractPPL.Iden && return nothing
    if binding isa BoundValue{role} && !_skip_bound_shape_check(binding.value)
        _matches_local_shape(binding.value, value) ||
            _throw_local_shape_error(owner, binding.value, value)
    end
    (optic isa AbstractPPL.Iden || submodel && optic.child isa AbstractPPL.Iden) &&
        return nothing
    head = AbstractPPL.ohead(optic)
    head isa AbstractPPL.Index && (head = AbstractPPL.concretize_top_level(head, value))
    child = _binding_node_at(binding, head)
    child === nothing && role === ConditionRole && return nothing
    child === nothing && _throw_binding_error(vn, "fixed binding omits an executed address")
    if head isa AbstractPPL.Index &&
        value isa AbstractArray &&
        !VarNamedTuples._is_multiindex(value, head.ix...) &&
        checkbounds(Bool, value, head.ix...) &&
        !_isassigned(value, head.ix...)
        return nothing
    end
    local_child = _binding_node_at(value, head)
    local_child === nothing &&
        _throw_binding_error(vn, "local storage omits a bound address")
    return _check_local_shape(
        child,
        local_child,
        optic.child,
        vn,
        role;
        submodel,
        owner=AbstractPPL.append_optic(owner, head),
    )
end
function _check_address_kind(value, optic, prefix=AbstractPPL.Iden())
    optic isa AbstractPPL.Iden && return nothing
    value = _unwrap_storage(value)
    VarNamedTuples._check_tuple_storage(value)
    kind = _binding_optic(value isa VarNamedTuple ? value.data : value)
    (kind !== nothing && !(optic isa kind)) && _throw_binding_error(
        AbstractPPL.optic_to_varname(optic ∘ prefix),
        "address kind disagrees with storage",
    )
    optic.child isa AbstractPPL.Iden && return nothing
    child = _binding_node_at(value, AbstractPPL.ohead(optic))
    child === nothing || _check_address_kind(child, optic.child, prefix)
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
function _flatten_bindings!(entries, value::VarNamedTuples.PartialArray, vn, storage)
    if !any(value.mask)
        push!(entries, (vn, value, storage))
    else
        for i in CartesianIndices(value.mask)
            value.mask[i] || continue
            _flatten_bindings!(
                entries,
                value.data[i],
                AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(i), (;))),
                storage,
            )
        end
    end
    return entries
end
_flatten_bindings!(entries, value, vn, storage) = push!(entries, (vn, value, storage))
function _flatten_bindings(input::NamedTuple{N}) where {N}
    any(x -> x isa Union{VarNamedTuple,VarNamedTuples.PartialArray}, values(input)) &&
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
    prefix = getprefix(model)
    prefix === nothing && return vn
    template = _prefixed_template(model.prefix, NoTemplate())
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
        prefix = maybe_prefix(getprefix(child), site)
        subsumes(vn, prefix) && return true
        subsumes(prefix, vn) &&
            return _is_namespace(child, AbstractPPL.unprefix(vn, prefix))
    end
    return false
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
            any(c -> c.typed === Val(true) && subsumes(c.owner, base), root.constraints)
        if node isa BoundValue &&
            _binding_optic(node.value) === AbstractPPL.Property &&
            (
                !(
                    retained isa VarNamedTuple ||
                    _binding_optic(retained) === AbstractPPL.Property
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
        typed = constraint.typed === Val(true)
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
# Avoid inference per model history, but retain the role and stored types:
# `condition` can run inside an evaluator.
Base.@nospecializeinfer function _add_bindings(
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
        _find_placeholder(value) !== :none &&
            _throw_binding_error(vn, "explicit bindings cannot contain missing or nothing")
        name = AbstractPPL.getsym(vn)
        base = VarName{name}()
        old = get(roots.data, name, BindingRoot(NoBinding(), NoTemplate(), ()))
        storage = _unwrap_storage(old.storage)
        offered = get(schema, name, _unwrap_storage(supplied))
        if offered isa NamedTuple && _is_namespace(model, base)
            offered = VarNamedTuple(offered)
        end
        arguments = merge(model.args, model.defaults)
        original = get(arguments, name, NoTemplate())
        if name in _lhs_arguments(model) &&
            !(original isa Union{Missing,Nothing,NoTemplate})
            offered = NoTemplate()
        end
        if storage isa NoTemplate || (!(offered isa NoTemplate) && old.value isa NoBinding)
            storage = offered
        elseif !(offered isa NoTemplate)
            _same_shape(storage, offered) ||
                _throw_binding_error(vn, "template disagrees with retained storage")
        end
        if name in _lhs_arguments(model) &&
            original isa Union{Missing,Nothing} &&
            !(AbstractPPL.getoptic(vn) isa AbstractPPL.Iden) &&
            storage isa NoTemplate
            _throw_missing_storage(vn)
        end
        _check_field_under_whole(model, vn, storage)
        constraints = if !(offered isa NoTemplate) && old.value isa NoBinding
            ()
        else
            _effective_constraints(old, base)
        end
        if !(offered isa NoTemplate) && (
            haskey(schema, name) ||
            old.storage isa NoTemplate && !(AbstractPPL.getoptic(vn) isa AbstractPPL.Iden)
        )
            owner, template = _template_owner(offered, AbstractPPL.getoptic(vn), base)
            template = _unwrap_storage(template)
            any(
                c ->
                    c.owner == owner &&
                        c.role === R &&
                        c.typed === Val(true) &&
                        typeof(c.storage) !== typeof(template),
                constraints,
            ) && _throw_binding_error(vn, "template disagrees with retained type")
            held = _binding_node_at(old.value, AbstractPPL.getoptic(owner))
            if held isa BoundValue
                _same_shape(held.value, template) ||
                    _throw_binding_error(vn, "template disagrees with held shape")
                _matches_role(R, held) &&
                    !(held.value isa typeof(template)) &&
                    _throw_binding_error(vn, "template disagrees with held type")
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
            _same_shape(storage, value) ||
                _throw_binding_error(vn, "remove the whole binding before changing shape")
            (
                storage isa VarNamedTuple ||
                old.value isa VarNamedTuple && storage isa NoTemplate
            ) && _throw_binding_error(vn, "namespace cannot become a whole value")
        end
        if value isa VarNamedTuples.PartialArray && !any(value.mask)
            node = old.value
            if _binding_node_at(node, AbstractPPL.getoptic(vn)) isa Union{Nothing,NoBinding}
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
        if name in _lhs_arguments(model) && !(storage isa NoTemplate)
            _check_partial_storage(storage, AbstractPPL.getoptic(vn), vn)
        end
        selections = _expand_selection(
            storage, AbstractPPL.getoptic(vn), base, value, old.value
        )
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
                _throw_binding_error(address, "inputs overlap in this call")
            push!(seen, address)
            optic = AbstractPPL.getoptic(address)
            previous = _binding_node_at(node, optic)
            previous isa VarNamedTuple &&
                !(
                    _binding_optic(_binding_node_at(storage, optic)) ===
                    AbstractPPL.Property
                ) &&
                _throw_binding_error(address, "namespace cannot become a whole value")
            incoming = BoundValue{R}(element)
            _check_role_overlap(previous, incoming, address; disjoint)
            if !(optic isa AbstractPPL.Iden) && name in _lhs_arguments(model)
                element = _convert_partial(element, storage, optic, address)
            end
            previous isa BoundValue && _same_shape(previous.value, element) ||
                previous isa BoundValue && _throw_binding_error(
                    address, "remove the binding before changing shape"
                )
            element = _apply_constraints(element, optic, address, R, constraints)
            incoming = BoundValue{R}(element)
            if optic isa AbstractPPL.Iden
                node = incoming
                storage = element
                constraints = filter(c -> c.typed === Val(true), constraints)
            else
                node = _set_binding_node(node, name, incoming, address, storage)
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
        getprefix(model) === nothing &&
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
    return invoke(_bind, Tuple{Any,Type{R},Any}, model, R, inputs; kwargs...)
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
            prefix = getprefix(model)
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
                    _throw_binding_error(vn, "keyword splats require whole bindings")
                value = (; value...)
            end
            local_vn = _relative_name(model, vn)
            if local_vn === nothing
                prefix = getprefix(model)
                optic = AbstractPPL._unprefix_optic(
                    AbstractPPL.getoptic(prefix), AbstractPPL.getoptic(vn)
                )
                selected = _binding_node_at(value, optic)
                selected isa NamedTuple ||
                    _throw_binding_error(vn, "prefix binding requires named child values")
                for (n, v) in pairs(selected)
                    entries = (
                        entries...,
                        _split_namespace_entry(model, VarName{n}(), v, NoTemplate())...,
                    )
                end
            else
                prefix = getprefix(model)
                if prefix !== nothing && !(storage isa NoTemplate)
                    root = AbstractPPL.prefix(
                        VarName{AbstractPPL.getsym(local_vn)}(), prefix
                    )
                    storage = _binding_node_at(storage, AbstractPPL.getoptic(root))
                end
                entries = (
                    entries..., _split_namespace_entry(model, local_vn, value, storage)...
                )
            end
        end
    end
    table = _add_bindings(model, model.values, R, entries, schema)
    metadata = _with_binding_template_names(_model_metadata(model), _template_names(schema))
    result = _reconstruct_model(model; values=table, metadata)
    _check_argument_bindings(result)
    _check_argument_aliasing(result)
    return result
end
function _template_names(schema::NamedTuple, prefix=nothing)
    return mapreduce((a, b) -> (a..., b...), keys(schema); init=()) do name
        vn = maybe_prefix(VarName{name}(), prefix)
        value = schema[name]
        value isa NamedTuple ? _template_names(value, vn) : (vn,)
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
    prefix = getprefix(model)
    (prefix === nothing || isempty(result)) && return result
    return templated_setindex!!(
        VarNamedTuple(), result, prefix, _prefixed_template(model.prefix, NoTemplate())
    )
end
function _removal_hits(table, R, name)
    if name === nothing
        return any(root -> _has_role(R, root.value), values(table.roots.data))
    end
    root = get(table.roots.data, AbstractPPL.getsym(name), nothing)
    root === nothing && return false
    return _has_role(R, _binding_node_at(root.value, AbstractPPL.getoptic(name)))
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
        node = if root.value isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
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
                node = _delete_role_at(node, AbstractPPL.getoptic(address), R, address)
            end
        end
        previous = _effective_constraints(root, base)
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
                    _shape_signature(c.storage) !== nothing
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
    _unbind(model, names, ConditionRole, Val(true))
"""
    unfix(model::Model, names...)

Remove fixed values recursively; omit names to remove all fixed values.
"""
unfix(model::Model, names::Union{Symbol,VarName}...) =
    _unbind(model, names, FixRole, Val(true))
function _unbind(model, names, R, ::Val{recursive}=Val(false)) where {recursive}
    names = map(n -> _relative_name(model, n isa Symbol ? VarName{n}() : n), names)
    any(isnothing, names) && (names = ())
    for vn in names
        _check_binding_address(model, vn)
        root = _binding_root(model, vn)
        root === nothing || _check_field_under_whole(model, vn, root.storage)
    end
    table = _remove_role(
        model.values, R, names; arguments=merge(model.args, model.defaults)
    )
    markers = _removal_markers(R, model)
    if recursive
        markers = if isempty(names)
            ((name=nothing, active=true, matched=false),)
        else
            (
                markers...,
                map(names) do name
                    (; name, active=true, matched=_removal_hits(model.values, R, name))
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
    _check_argument_aliasing(result)
    return result
end

function _prepare_argument(binding, value, vn, storage=value)
    binding isa BoundValue && return binding.value
    if _is_fully_bound(binding, storage)
        return _bound_data(binding, storage)
    end
    return _fill_latent_storage(binding, _latent_source(value, storage), vn, storage)
end
function _latent_source(value, storage)
    (value isa Union{Missing,Nothing} || _binding_optic(storage) === nothing) &&
        return value
    return if _binding_optic(value) !== nothing && _same_shape(value, storage)
        value
    else
        storage
    end
end
function _fill_latent_storage(
    binding, value::Union{Missing,Nothing,NoTemplate}, vn, storage
)
    binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} || return value
    return _throw_missing_storage(vn)
end
@generated function _fill_latent_storage(binding, value, vn, storage, ::Val{N}) where {N}
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
                _binding_node_at(binding, optic),
                value.$name,
                AbstractPPL.append_optic(vn, optic),
                retained,
            )
        end
    end
    return :(ConstructionBase.setproperties(value, NamedTuple{$N}(($(fields...),))))
end
function _throw_unsupported_storage(value, vn)
    VarNamedTuples._check_tuple_storage(value)
    remedy = if value isa AbstractArray
        "use collect or pass a plain Array before constructing the model (axes or metadata may be lost)"
    else
        "use supported Array/NamedTuple storage and matching index/field addresses"
    end
    return _throw_binding_error(vn, "unsupported binding storage $(typeof(value)); $remedy")
end
function _check_partial_storage(value, vn)
    if _binding_optic(value) === nothing &&
        value isa Union{AbstractArray,AbstractDict,Tuple}
        _throw_unsupported_storage(value, vn)
    end
    return value
end
function _fill_latent_storage(binding, value, vn, storage)
    _check_partial_storage(value, vn)
    if _binding_optic(value) === AbstractPPL.Index
        result = _rebuild_buffer(value)
        for i in _positional_indices(value)
            optic = AbstractPPL.Index(_index_tuple(i), (;))
            child = _binding_node_at(binding, optic)
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
        return _fill_latent_storage(binding, value, vn, storage, Val(propertynames(value)))
    else
        return value
    end
end
function prepare_model_argument(model::Model, vn::VarName, value)
    root = _binding_root(model, vn)
    root === nothing && return _prepare_argument(nothing, value, vn)
    return _prepare_argument(
        root.value, value, maybe_prefix(vn, getprefix(model)), root.storage
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
    vn = maybe_prefix(VarName{name}(), getprefix(model))
    root = _binding_root(model, VarName{name}())
    root === nothing && return nothing
    name in _lhs_arguments(model) || _throw_binding_error(vn, "covariates cannot be bound")
    original = arguments[stored]
    if original isa Union{Missing,Nothing} &&
        root.value isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        !_is_fully_bound(root.value, root.storage)
        _throw_missing_storage(vn)
    end
    if root.value isa BoundValue || _is_fully_bound(root.value, root.storage)
        T = _declared_argument_type(_model_metadata(model), stored)
        _bound_data(root.value, root.storage) isa T ||
            _throw_binding_error(vn, "binding requires declared argument type $T")
    elseif is_splat_symbol(stored)
        _throw_binding_error(vn, "splat arguments require whole bindings")
    end
    return nothing
end
function _collect_argument_arrays!(arrays, value, binding, vn, parents, storage=value)
    binding isa Union{BoundValue{ConditionRole},BoundValue{FixRole}} && return arrays
    if binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        _is_fully_bound(binding, storage)
        binding = BoundValue{ArgumentRole}(_bound_data(binding, storage))
    end
    value isa Union{Missing,Nothing} &&
        binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
        _throw_missing_storage(vn)
    value = if binding isa BoundValue
        binding.value
    else
        _check_partial_storage(_latent_source(value, storage), vn)
    end
    kind = _binding_optic(value)
    kind === nothing && return arrays
    if value in parents
        binding isa BoundValue && return arrays
        _throw_binding_error(vn, "cyclic latent storage is unsupported")
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
function _check_argument_aliasing(model)
    any(
        root -> !_is_fully_bound(root.value, root.storage), values(model.values.roots.data)
    ) || return nothing
    arrays = Tuple[]
    for (stored, value) in pairs(merge(model.args, model.defaults))
        vn = VarName{unsplat_symbol(stored)}()
        lhs = AbstractPPL.getsym(vn) in _lhs_arguments(model)
        binding = lhs ? _binding_at(model, vn) : BoundValue{ArgumentRole}(value)
        _collect_argument_arrays!(
            arrays,
            value,
            binding,
            maybe_prefix(vn, getprefix(model)),
            Base.IdSet{Any}(),
            lhs ? _binding_root(model, vn).storage : value,
        )
    end
    for i in eachindex(arrays), j in 1:(i - 1)
        a, la, vn = arrays[i]
        b, lb, _ = arrays[j]
        (la || lb) &&
            (a === b || Base.mightalias(a, b)) &&
            _throw_binding_error(vn, "latent storage aliases another argument path")
    end
    return nothing
end

_skip_bound_shape_check(value) = false
