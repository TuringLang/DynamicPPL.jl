function _get_model_binding(model, vn)
    return _model_argument_binding(model.values.values, AbstractPPL.varname_to_optic(vn))
end
function _get_argument_role(model, vn, argument, local_value=NoTemplate())
    binding = _get_model_binding(model, argument)
    # Skip the local-storage check for whole argument observations: the body may resize them.
    binding isa Union{ModelValue{Condition},ModelValue{ArgumentCondition}} &&
        return _model_role(binding, vn)
    return _local_binding_role(binding, vn, local_value, model, ArgumentCondition)
end
function _get_model_data(model, vn)
    binding = _get_model_binding(model, vn)
    binding === nothing && throw(KeyError(vn))
    return _model_data(binding)
end
function _get_model_data(model, vn, root, local_value)
    binding = _get_model_binding(model, root)
    _check_bound_shape(
        binding,
        local_value,
        AbstractPPL.getoptic(vn),
        maybe_prefix(vn, _model_prefix(model)),
    )
    return _get_model_data(model, vn)
end
function _check_bound_shape(binding, local_value, optic, vn, role=Fix; submodel=false)
    if binding isa VarNamedTuples.PartialArray && local_value isa AbstractArray
        axes(binding.data) == axes(local_value) || _bound_shape_error(
            vn,
            role,
            "a static size and shape; storage axes must match the model's local array.",
        )
    end
    if (
        role === Fix && (
            optic isa AbstractPPL.Iden ||
            _inherits_shape(binding) ||
            binding isa VarNamedTuples.PartialArray && binding.mask isa ModelBindingArray
        )
    ) || (role === Condition && _whole_observation(binding))
        template = if binding isa VarNamedTuples.PartialArray
            binding.data
        elseif binding isa ModelValueTree
            binding.template
        else
            nothing
        end
        template === nothing || _check_bound_shape(
            ModelValue{role}(template),
            local_value,
            AbstractPPL.Iden(),
            vn,
            role;
            submodel,
        )
    end
    return _check_bound_shape_child(binding, local_value, optic, vn, role; submodel)
end
function _check_bound_shape(
    binding::ModelValue{R}, local_value, optic, vn, role=Fix; submodel=false
) where {R}
    role === ArgumentCondition && return nothing
    submodel && optic isa AbstractPPL.Iden && return nothing
    R === role || submodel || return nothing
    value = binding.value
    _skip_bound_shape_check(value) &&
        return _check_bound_shape_child(binding, local_value, optic, vn, role; submodel)
    if value isa Tuple
        local_value isa Tuple && length(value) == length(local_value) || _bound_shape_error(
            vn,
            role,
            "a static size and shape; the model body supplied a different length.",
        )
    elseif value isa Union{NamedTuple,Base.Pairs}
        local_value isa Union{NamedTuple,Base.Pairs} &&
            all(name -> haskey(local_value, name), keys(value)) || _bound_shape_error(
            vn,
            role,
            "a static size and shape; the model body supplied different fields.",
        )
    elseif value isa AbstractArray || local_value isa AbstractArray
        value isa AbstractArray &&
            local_value isa AbstractArray &&
            axes(value) == axes(local_value) || _bound_shape_error(
            vn,
            role,
            "a static size and shape; the model body supplied a different shape.",
        )
    end
    return _check_bound_shape_child(binding, local_value, optic, vn, role; submodel)
end
_skip_bound_shape_check(::Any) = false

function _check_bound_shape_child(
    binding, local_value, ::AbstractPPL.Iden, vn, role; submodel=false
)
    return nothing
end
function _check_bound_shape_child(
    binding, local_value, optic::AbstractPPL.Property{S}, vn, role; submodel=false
) where {S}
    submodel && optic.child isa AbstractPPL.Iden && return nothing
    child = _model_argument_binding(binding, AbstractPPL.Property{S}())
    child === nothing && role === Condition && return nothing
    child !== nothing && hasproperty(local_value, S) ||
        _bound_shape_error(vn, role, "coverage with a static size and shape.")
    return _check_bound_shape(
        child, getproperty(local_value, S), optic.child, vn, role; submodel
    )
end
function _check_bound_shape_child(
    binding, local_value, optic::AbstractPPL.Index, vn, role; submodel=false
)
    submodel && optic.child isa AbstractPPL.Iden && return nothing
    optic = AbstractPPL.concretize_top_level(optic, local_value)
    child = _model_argument_binding(binding, AbstractPPL.Index(optic.ix, optic.kw))
    child === nothing && role === Condition && return nothing
    child === nothing &&
        _bound_shape_error(vn, role, "coverage with a static size and shape.")
    selected = if VarNamedTuples._is_multiindex(local_value, optic.ix...; optic.kw...)
        Base.maybeview(local_value, optic.ix...; optic.kw...)
    else
        if local_value isa AbstractArray &&
            optic.child isa AbstractPPL.Iden &&
            isempty(optic.kw) &&
            checkbounds(Bool, local_value, optic.ix...) &&
            !isassigned(local_value, optic.ix...)
            return nothing
        end
        getindex(local_value, optic.ix...; optic.kw...)
    end
    return _check_bound_shape(child, selected, optic.child, vn, role; submodel)
end

@inline function _check_partial_binding(
    value, optic, prefix=AbstractPPL.Iden(); operation="bind"
)
    return _check_namedtuple_index(value, optic, prefix; check_container=true, operation)
end
@inline function _check_namedtuple_index(
    value,
    optic,
    prefix=AbstractPPL.Iden();
    check_container=false,
    operation="bind",
    check_owner=true,
)
    optic isa AbstractPPL.Iden && return nothing
    template = value isa ModelValue ? value.value : value
    template = template isa ModelValueTree ? template.template : template
    if check_container &&
        check_owner &&
        template isa AbstractArray &&
        !_partial_binding_array(template)
        _unsupported_partial_binding(
            template,
            AbstractPPL.optic_to_varname(optic ∘ prefix),
            AbstractPPL.optic_to_varname(prefix);
            operation,
        )
    end
    if check_container && !(
        template isa Union{
            AbstractArray,
            NamedTuple,
            NoTemplate,
            Missing,
            Nothing,
            VarNamedTuple,
            VarNamedTuples.PartialArray,
            VarNamedTuples.NestedTemplate,
            VarNamedTuples.SkipTemplate,
        }
    )
        if operation == "bind" && template isa Number && isempty(propertynames(template))
            _incompatible_partial_binding(template, AbstractPPL.optic_to_varname(prefix))
        end
        address = AbstractPPL.optic_to_varname(optic ∘ prefix)
        _unsupported_partial_binding(
            template, address, AbstractPPL.optic_to_varname(prefix); operation
        )
    end
    if check_container &&
        template isa AbstractArray &&
        optic isa AbstractPPL.Property &&
        !hasproperty(template, _binding_property_name(optic))
        _unsupported_partial_binding(
            template,
            AbstractPPL.optic_to_varname(optic ∘ prefix),
            AbstractPPL.optic_to_varname(prefix);
            operation,
        )
    end
    if template isa NamedTuple && optic isa AbstractPPL.Index
        key = if length(optic.ix) == 1
            only(AbstractPPL.concretize_top_level(optic, template).ix)
        else
            nothing
        end
        field = if key isa Symbol
            key
        elseif key isa Integer
            get(keys(template), key, nothing)
        else
            nothing
        end
        kind = if key isa Symbol
            "Symbol indexing"
        elseif key isa Integer
            "integer indexing"
        else
            "indexing"
        end
        suggestion = if field === nothing
            "a field name"
        else
            "`$(AbstractPPL.optic_to_varname(AbstractPPL.Property{field}(optic.child) ∘ prefix))`"
        end
        vn = AbstractPPL.optic_to_varname(optic ∘ prefix)
        message = if operation == "remove"
            "Cannot remove `$vn`: $kind into a NamedTuple is unsupported; use $suggestion instead."
        else
            "$(uppercasefirst(kind)) into a NamedTuple at `$vn` is unsupported; use $suggestion instead."
        end
        throw(ArgumentError(message))
    end

    optic.child isa AbstractPPL.Iden && return nothing
    check_child_owner = !_partial_binding_selector(template, optic)
    if template isa AbstractArray && optic isa AbstractPPL.Index
        coptic = AbstractPPL.concretize_top_level(optic, template)
        if checkbounds(Bool, template, coptic.ix...; coptic.kw...)
            check_child_owner &=
                !VarNamedTuples._is_multiindex(template, coptic.ix...; coptic.kw...)
            _check_assigned_binding_storage(
                template, coptic, AbstractPPL.optic_to_varname(optic ∘ prefix)
            )
        end
    end
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(value, head)
    return _check_namedtuple_index(
        child,
        optic.child,
        head ∘ prefix;
        check_container,
        operation,
        check_owner=check_child_owner,
    )
end

function _local_binding_role(node, vn, template, model, shape_role=Condition)
    node === nothing && return nothing
    if template isa NamedTuple && node isa VarNamedTuples.PartialArray
        _fold_model_indices(nothing, node) do _, _, optic, _
            _check_namedtuple_index(
                template, optic, AbstractPPL.Property{AbstractPPL.getsym(vn)}()
            )
        end
    end
    role = _model_role_at(
        node, AbstractPPL.getoptic(vn), maybe_prefix(vn, _model_prefix(model))
    )
    if !(template isa NoTemplate) &&
        (role !== nothing || node isa ModelValue{Condition} || _whole_observation(node))
        _check_bound_shape(
            node,
            template,
            AbstractPPL.getoptic(vn),
            maybe_prefix(vn, _model_prefix(model)),
            shape_role,
        )
    end
    return role
end
@generated function _argument_defaults(
    arguments::NamedTuple{names}, ::Val{args_on_lhs}
) where {names,args_on_lhs}
    fields = map(names) do stored_name
        name = unsplat_symbol(stored_name)
        name in args_on_lhs || return :((;))
        :(NamedTuple{($(QuoteNode(name)),)}((
            ModelValue{ArgumentCondition}(arguments.$stored_name),
        )))
    end
    return :(VarNamedTuple(merge((;), $(fields...))))
end
function _binding_local_name(vn, prefix, operation)
    if vn === nothing || prefix === nothing
        return vn
    elseif subsumes(prefix, vn)
        return prefix == vn ? nothing : AbstractPPL.unprefix(vn, prefix)
    elseif subsumes(vn, prefix)
        return nothing
    end
    throw(
        ArgumentError(
            "Cannot $operation `$vn`: it is outside this model's prefix `$prefix`."
        ),
    )
end

function _public_binding_name(model, vn; operation="bind")
    prefix = _model_prefix(model)
    template = _apply_prefix_template(_model_prefix_template(model), NoTemplate())
    depth = optic_skip_length(AbstractPPL.getoptic(prefix))
    prefix = _expand_cartesian(_concretize_prefix(prefix, template; depth, canonical=true))
    vn = _expand_cartesian(_concretize_prefix(vn, template; depth, canonical=true))
    return _binding_local_name(vn, prefix, operation)
end
function _local_binding_input(model, input::Pair{<:VarName})
    name = _public_binding_name(model, first(input))
    name !== nothing && return name => last(input)
    prefix = _model_prefix(model)
    value = ModelValue{Condition}(last(input))
    optic = AbstractPPL._unprefix_optic(
        AbstractPPL.getoptic(prefix), AbstractPPL.getoptic(first(input))
    )
    selected = _model_argument_binding(value, optic)
    return _plain_model_values(
        _select_model_values(Condition, _submodel_namespace(selected))
    )
end
function _local_binding_input(model, input::Union{NamedTuple,VarNamedTuple})
    values = input isa NamedTuple ? VarNamedTuple(input) : input
    foreach(vn -> _public_binding_name(model, vn), keys(values))
    tagged = _tag_model_values(Condition, values)
    return _select_model_values(Condition, _submodel_values(tagged, _model_prefix(model)))
end
_local_binding_input(model, input::Tuple) = map(x -> _local_binding_input(model, x), input)
_local_binding_input(model, input) = input
_record_binding_read(context, model, vn, template) = context
function _record_binding_read(context::AbstractParentContext, model, vn, template)
    return setchildcontext(
        context, _record_binding_read(childcontext(context), model, vn, template)
    )
end

function _inherited_model_values(values::VarNamedTuple)
    return _prune_model_bindings(
        VarNamedTuples._map_values_recursive!!(
            v -> v isa ModelValue && !_inherits_binding(v) ? NoModelBinding() : v,
            copy(values),
        ),
    )
end

struct ModelRemoval{N}
    name::N
    matched::Bool
    token::Base.RefValue{Nothing}
end
ModelRemoval(name, matched) = ModelRemoval(name, matched, Ref(nothing))
maybe_prefix(::Nothing, prefix::VarName) = prefix

Base.@constprop :aggressive function _initial_bindings(arguments, metadata)
    names = _args_on_lhs(metadata)
    tagged = _argument_defaults(arguments, Val(names))
    stored = VarNamedTuple(map(v -> v.value, tagged.data))
    roots = _initial_binding_roots(metadata, stored)
    return ResolvedBindings(tagged, stored, VarNamedTuple(), VarNamedTuple(), stored, roots)
end
@generated function _initial_binding_roots(
    ::ModelBindingMetadata{A,L,T,N}, ::VarNamedTuple{names}
) where {A,L,T,N,names}
    roots = Tuple(VarName{AbstractPPL.getsym(first(pair))}() for pair in N if !last(pair))
    return QuoteNode((roots..., map(name -> VarName{name}(), names)...))
end
@generated function _initial_binding_roots(::Tuple, ::VarNamedTuple{names}) where {names}
    return QuoteNode(map(name -> VarName{name}(), names))
end

function _check_binding_addresses(model, values, error_prefix=_model_prefix(model))
    names = _lhs_names(_binding_metadata(model))
    names === nothing && return nothing
    for name in keys(values.data)
        name in names || throw(
            ArgumentError(
                "Cannot bind `$(maybe_prefix(VarName{name}(), error_prefix))`: it is not an LHS top symbol of this model.",
            ),
        )
    end
    return nothing
end

@generated function _check_argument_bindings(
    model::Model,
    values,
    inherited=false,
    error_prefix=_model_prefix(model),
    templates=merge(model.values.observed_templates, model.values.fixed_templates),
)
    names = (
        fieldnames(fieldtype(model, :args))..., fieldnames(fieldtype(model, :defaults))...
    )
    checks = map(names) do stored_name
        name = unsplat_symbol(stored_name)
        keyword =
            is_splat_symbol(stored_name) &&
            stored_name in fieldnames(fieldtype(model, :defaults))
        quote
            vn = $(VarName{name}())
            binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
            if binding !== nothing
                $(QuoteNode(name)) in _args_on_lhs(model) || throw(
                    ArgumentError(
                        "Argument `$(maybe_prefix(vn, error_prefix))` does not occur on the left-hand side of `~` and cannot be conditioned or fixed; construct the model with a new argument value instead.",
                    ),
                )
                if !(binding isa ModelValue)
                    $keyword && _keyword_binding_error($(QuoteNode(name)))
                    argument = get(
                        merge(model.args, model.defaults),
                        $(QuoteNode(stored_name)),
                        nothing,
                    )
                    if argument === missing
                        template = get(templates.data, $(QuoteNode(name)), NoTemplate())
                        if binding isa VarNamedTuples.PartialArray &&
                            template isa AbstractArray
                            mask = if binding.mask isa ModelBindingArray
                                binding.mask.data
                            else
                                binding.mask
                            end
                            binding = VarNamedTuples.PartialArray(
                                binding.data, ModelBindingArray(mask, template)
                            )
                            argument = template
                        elseif binding isa VarNamedTuple && template isa NamedTuple
                            binding = ModelValueTree(template, binding, Val(true))
                            argument = template
                        end
                    end
                    if argument isa Union{Nothing,Missing} &&
                        _has_binding_role(Condition, binding) &&
                        _has_binding_role(Fix, binding)
                        throw(
                            ArgumentError(
                                "Cannot make a partial binding of argument `$(maybe_prefix(vn, error_prefix))` with whole `$argument` storage; supply a concrete argument or a whole binding instead.",
                            ),
                        )
                    end
                    previous = _get_model_binding(model, vn)
                    argument = if previous === nothing
                        argument
                    else
                        _model_argument_value(previous, argument)
                    end
                    if inherited && !(
                        _has_binding_role(Condition, binding) &&
                        _has_binding_role(Fix, binding)
                    )
                        if binding isa VarNamedTuples.PartialArray &&
                            _inherits_shape(binding.mask)
                            argument = binding.mask.template
                        elseif binding isa ModelValueTree && _inherits_shape(binding)
                            argument = binding.template
                        end
                    end
                    if argument isa Union{Nothing,Missing}
                        _has_complete_model_data(binding) || throw(
                            ArgumentError(
                                "Cannot make a partial binding of argument `$(maybe_prefix(vn, error_prefix))` with whole `$argument` storage; supply a concrete argument or a whole binding instead.",
                            ),
                        )
                        binding = ModelValue{typeof(_model_role(binding, vn))}(
                            _model_data(binding)
                        )
                    else
                        binding = _prepare_bindings(
                            binding,
                            argument,
                            maybe_prefix(vn, error_prefix),
                            get(
                                model.values.storage.data,
                                $(QuoteNode(name)),
                                NoTemplate(),
                            ),
                        )
                    end
                    values = templated_setindex!!(
                        values, binding, vn, values.data[$(QuoteNode(name))]
                    )
                end
                if binding isa ModelValue
                    T = _declared_argument_type(
                        _binding_metadata(model), $(QuoteNode(stored_name))
                    )
                    binding.value isa T || throw(
                        ArgumentError(
                            "Bound value at `$(maybe_prefix(vn, error_prefix))` in model `$(nameof(model))` must be an instance of declared argument type $T; supplied $(typeof(binding.value)).",
                        ),
                    )
                end
            end
        end
    end
    return quote
        $(checks...)
        values
    end
end
_prepare_bindings(binding, storage, vn, retained=NoTemplate()) = binding
function _prepare_bindings(binding::ModelValueTree, storage, vn, retained=NoTemplate())
    return ModelValueTree(
        binding, _prepare_binding_fields(binding.values, storage, vn, retained)
    )
end
function _prepare_bindings(
    bindings::Union{VarNamedTuple,VarNamedTuples.PartialArray},
    storage,
    vn,
    retained=NoTemplate(),
)
    return _prepare_binding_fields(bindings, storage, vn, retained)
end
function _prepare_binding_fields(bindings, storage, vn, retained)
    if storage isa Union{Nothing,Missing,NoTemplate} && !(retained isa NoTemplate)
        return _prepare_binding_fields(bindings, retained, vn, NoTemplate())
    end
    return _fold_model_indices(copy(bindings), bindings) do result, binding, optic, template
        _check_partial_binding(
            ModelValue{Condition}(storage), optic, AbstractPPL.varname_to_optic(vn)
        )
        binding = _prepare_binding_selection(binding, storage, optic, vn, retained)
        return VarNamedTuples._setindex_optic!!(
            result, binding, optic, template, VarNamedTuples.AllowAll()
        )
    end
end
function _prepare_binding_selection(binding, storage, optic, vn, retained=NoTemplate())
    address = AbstractPPL.append_optic(vn, optic)
    _path_storage(storage, optic, vn, (vn,))
    child_storage = _storage_at(retained, optic)
    child_storage = child_storage === nothing ? NoTemplate() : child_storage
    VarNamedTuples._haskey_optic(storage, optic) ||
        return _prepare_bindings(binding, child_storage, address)
    if binding isa ModelValue
        if storage isa VarNamedTuple
            template = _storage_at(storage, optic)
            return _model_value_like(
                binding,
                _convert_resolved_value(
                    binding.value, template, AbstractPPL.Iden(), address, true
                ),
            )
        end
        return _convert_partial_argument_binding(binding, storage, optic, address)
    end
    return _prepare_bindings(
        binding, VarNamedTuples._getindex_optic(storage, optic, vn), address, child_storage
    )
end

function _keyword_binding_error(name)
    throw(
        ArgumentError(
            "Entries of keyword-splat argument `$name` cannot be bound; replace the whole argument with `condition` or `fix` instead.",
        ),
    )
end

"""
    condition(model::Model, values...[, template]; kwargs...)

Condition model variables on supplied values. Accept NamedTuples, VarNamedTuples,
Symbol or VarName pairs, and keyword arguments. Observations contribute likelihood.

An optional `@of`/`of` binding template must be the last positional argument.
A template in any other position throws `ArgumentError`. Indexed partial bindings
require concrete storage at binding time; the model's arguments supply their own storage.

# Examples

```jldoctest
julia> using DynamicPPL.AbstractPPL: @of, of

julia> @model child() = x ~ Normal();

julia> @model function parent()
           z = zeros(3)
           for i in eachindex(z)
               z[i] ~ to_submodel(child())
           end
           return z
       end;

julia> m = condition(parent(), @varname(z[1].x) => 1.0, @of(z = of(Array, 3)));

julia> conditioned(m)[@varname(z[1].x)]
1.0
```
"""
AbstractPPL.condition(model::Model, inputs...; kwargs...) =
    _conditionfix(model, Condition, inputs; kwargs...)

"""
    fix(model::Model, values...[, template]; kwargs...)

Fix model variables to supplied values, excluding their densities from the model.
Accept the same binding inputs as [`condition`](@ref). An optional `@of`/`of`
template must be the last positional argument; otherwise throw `ArgumentError`.
Remove an overlapping observation with [`decondition`](@ref) before fixing it.
"""
fix(model::Model, inputs...; kwargs...) = _conditionfix(model, Fix, inputs; kwargs...)
function _expand_child_binding(model, input::Pair{<:VarName})
    vn, value = input
    AbstractPPL.getsym(vn) in
    (_argument_names(model.args)..., _argument_names(model.defaults)...) && return input

    if value isa NamedTuple &&
        any(p -> last(p) && first(p) == vn, _lhs_addresses(_binding_metadata(model)))
        return map(keys(value)) do name
            AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()) => value[name]
        end
    end
    return input
end
function _expand_child_binding(model, input::NamedTuple)
    addresses = _lhs_addresses(_binding_metadata(model))
    any(p -> last(p) && AbstractPPL.getsym(first(p)) in keys(input), addresses) ||
        return input
    return map(_whole_binding_pairs(input)) do pair
        if last(pair) isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            NamedTuple{(AbstractPPL.getsym(first(pair)),)}((last(pair),))
        else
            _expand_child_binding(model, pair)
        end
    end
end
_expand_child_binding(model, input) = input
function _expand_child_binding(model, input::VarNamedTuple)
    addresses = _lhs_addresses(_binding_metadata(model))
    arguments = (_argument_names(model.args)..., _argument_names(model.defaults)...)
    return VarNamedTuple(
        NamedTuple{keys(input.data)}(
            map(keys(input.data)) do name
                value = input.data[name]
                if value isa NamedTuple &&
                    !(name in arguments) &&
                    any(p -> last(p) && first(p) == VarName{name}(), addresses)
                    VarNamedTuple(value)
                else
                    value
                end
            end,
        ),
    )
end

@generated function _keyword_binding_names(model::Model)
    names = Tuple(
        unsplat_symbol(n) for
        n in fieldnames(fieldtype(model, :defaults)) if is_splat_symbol(n)
    )
    return QuoteNode(names)
end
_normalize_keyword_value(value) = (; value...)
_normalize_keyword_value(value::Base.Pairs) = value
_normalize_keyword_input(input, ::Tuple{}) = input
_normalize_keyword_input(input, names) = input
function _normalize_keyword_input(input::Pair{<:VarName}, names::Tuple)
    vn, value = input
    if AbstractPPL.getsym(vn) in names
        AbstractPPL.getoptic(vn) isa AbstractPPL.Iden ||
            _keyword_binding_error(AbstractPPL.getsym(vn))
        return vn => _normalize_keyword_value(value)
    end
    return input
end
function _normalize_keyword_input(input::NamedTuple, names::Tuple)
    return NamedTuple{keys(input)}(
        map(keys(input)) do name
            value = input[name]
            if name in names
                value isa Union{VarNamedTuple,VarNamedTuples.PartialArray} &&
                    _keyword_binding_error(name)
                _normalize_keyword_value(value)
            else
                value
            end
        end,
    )
end
function _normalize_keyword_input(input::VarNamedTuple, names::Tuple)
    return VarNamedTuple(_normalize_keyword_input(input.data, names))
end
_normalize_keyword_input(input::Pair{<:VarName}, ::Tuple{}) = input
_normalize_keyword_input(input::NamedTuple, ::Tuple{}) = input
_normalize_keyword_input(input::VarNamedTuple, ::Tuple{}) = input

function _conditionfix(model, ::Type{R}, inputs; kwargs...) where {R}
    flattened = _binding_inputs(inputs)
    template_count = count(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, flattened)
    template_count <= 1 || throw(ArgumentError("Only one binding template may be supplied"))
    if template_count == 1
        last(inputs) isa Type{<:AbstractPPL.OfNamedTuple} || throw(
            ArgumentError("The binding template must be the last positional argument")
        )
    end
    offered_template = template_count == 0 ? nothing : zero(last(inputs))
    inputs = flattened
    keyword_values = (; kwargs...)
    if _model_prefix(model) !== nothing
        inputs = map(input -> _local_binding_input(model, input), inputs)
        isempty(kwargs) || (keyword_values = _local_binding_input(model, keyword_values))
    end
    keyword_names = _keyword_binding_names(model)
    inputs = map(input -> _normalize_keyword_input(input, keyword_names), inputs)
    keyword_values = _normalize_keyword_input(keyword_values, keyword_names)
    expanded = _binding_inputs(
        map(input -> _expand_child_binding(model, input), (keyword_values, inputs...))
    )
    table = _bind_resolved(model.values, R, expanded...; error_prefix=_model_prefix(model))
    return _finish_binding(model, table, offered_template)
end

# Specialise validation and reconstruction on the resolved table's concrete type.
function _finish_binding(model, table, offered_template)
    _check_removal_storage(model, table.storage)
    values = _check_argument_bindings(
        model,
        table.values,
        false,
        _model_prefix(model),
        merge(table.observed_templates, table.fixed_templates),
    )
    _check_binding_addresses(model, values)
    table = ResolvedBindings(
        values,
        table.storage,
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        table.roots,
    )
    _check_shared_latent_storage(
        table, merge(model.args, model.defaults), _model_prefix(model)
    )
    result = _reconstruct_model(model; values=table)
    return if offered_template === nothing
        result
    else
        _record_binding_template(result, offered_template)
    end
end
"""
    model::Model | values

Condition `model` on `values`, as in [`condition`](@ref). Accept a NamedTuple,
VarNamedTuple, pair or tuple of binding inputs. Use `condition` directly when
supplying a storage template.
"""
function Base.:|(
    model::Model, values::Union{NamedTuple,VarNamedTuple,Pair,Tuple,AbstractDict}
)
    any(value -> value isa Type{<:AbstractPPL.OfNamedTuple}, _binding_inputs(values)) &&
        throw(ArgumentError("Binding templates require an explicit condition or fix call"))
    return condition(model, values)
end

"""
    conditioned(model::Model)

Return the model's observations, including observed arguments, as a VarNamedTuple.
"""
conditioned(model::Model) = _listed_bindings(model, Condition)
"""
    fixed(model::Model)

Return the model's fixed values as a VarNamedTuple.
"""
fixed(model::Model) = _listed_bindings(model, Fix)
function _listed_bindings(model, R)
    values = _listing_argument_children(model, model.values.values)
    values = _select_model_values(R, values)
    prefix = _model_prefix(model)
    return if prefix === nothing
        values
    else
        _prefix_values(
            values,
            prefix,
            _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
        )
    end
end

_resolve_known_namespaces(value, model, vn) = value
function _resolve_known_namespaces(value::ModelValue{<:Any,<:NamedTuple}, model, vn)
    if !(AbstractPPL.getsym(vn) in _args_on_lhs(_binding_metadata(model))) &&
        any(p -> last(p) && subsumes(vn, first(p)), _lhs_addresses(_binding_metadata(model)))
        return _resolve_known_namespaces(_submodel_namespace(value), model, vn)
    end
    return value
end
@generated function _resolve_known_namespaces(
    values::VarNamedTuple{names}, model, vn
) where {names}
    fields = map(names) do name
        :(_resolve_known_namespaces(
            values.data.$name, model, maybe_prefix($(VarName{name}()), vn)
        ))
    end
    return :(VarNamedTuple(NamedTuple{$names}(($(fields...),))))
end

function _replace_listing_subtree(values, child, optic)
    return VarNamedTuples._setindex_optic!!(
        copy(values), child, optic, NoTemplate(), VarNamedTuples.AllowAll()
    )
end
function _listing_at_prefix(values, model, ::AbstractPPL.Iden)
    namespace = _resolve_known_namespaces(_submodel_namespace(values), model, nothing)
    return _listing_argument_children(model, namespace)
end
function _listing_at_prefix(values, model, optic::AbstractPPL.AbstractOptic)
    optic isa AbstractPPL.Property && (values = _submodel_namespace(values))
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(values, head)
    child === nothing && return values
    return _replace_listing_subtree(
        values, _listing_at_prefix(child, model, optic.child), head
    )
end
function _listing_argument_children(model, values)
    sites = _submodel_arguments(_binding_metadata(model))
    arguments = merge(model.args, model.defaults)
    for (lhs, name) in sites
        name isa Symbol || continue
        lhs === nothing && continue
        count(site -> first(site) == lhs, sites) == 1 || continue
        child = get(arguments, name, nothing)
        child isa Model || continue
        optic = AbstractPPL.varname_to_optic(lhs)
        node = _model_argument_binding(values, optic)
        node === nothing && continue
        prefix = _model_prefix(child)
        prefix_optic =
            prefix === nothing ? AbstractPPL.Iden() : AbstractPPL.varname_to_optic(prefix)
        namespace = _listing_at_prefix(node, child, prefix_optic)
        values = _replace_listing_subtree(values, namespace, optic)
    end
    return values
end

"""
    decondition(model::Model, names...)

Remove observations at the supplied Symbol or VarName addresses from this model and
its submodels, or all observations when no names are supplied. This also removes the
called model's own argument observations. Submodels take observations through `condition`.
"""
AbstractPPL.decondition(model::Model, names::Union{Symbol,VarName}...) =
    _decondition_unfix(model, names, Condition, Val(true))
"""
    unfix(model::Model, names...)

Remove fixed values at the supplied Symbol or VarName addresses from this model and
its submodels, or all fixed values when no names are supplied.
"""
unfix(model::Model, names::Union{Symbol,VarName}...) =
    _decondition_unfix(model, names, Fix, Val(true))
_removals(::Type{Condition}, model::Model) = model.observation_removals
_removals(::Type{Fix}, model::Model) = model.fixed_removals
function _decondition_unfix(model, names, R, ::Val{recursive}=Val(false)) where {recursive}
    names = map(names) do name
        vn = name isa Symbol ? VarName{name}() : name
        if _model_prefix(model) === nothing
            vn
        else
            _public_binding_name(model, vn; operation="remove")
        end
    end
    names = map(names) do vn
        vn === nothing && return nothing
        name, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
        if name in _keyword_binding_names(model) &&
            optic isa AbstractPPL.Index{Tuple{Symbol}}
            optic = AbstractPPL.Property{only(optic.ix)}(optic.child)
        end
        if AbstractPPL.is_dynamic(optic)
            storage = get(model.values.storage.data, name, NoTemplate())
            optic = _collapse_binding_selection(
                storage, optic, vn; operation="remove", location=VarName{name}()
            )
        end
        return VarName{name}(optic)
    end
    any(isnothing, names) && (names = ())
    for vn in names
        lhs = _lhs_names(_binding_metadata(model))
        lhs === nothing ||
            AbstractPPL.getsym(vn) in lhs ||
            throw(
                ArgumentError(
                    "Cannot remove `$vn`: it is not an LHS top symbol of this model."
                ),
            )
    end
    table = if isempty(names)
        _unbind_resolved(model.values, R)
    else
        foldl((t, vn) -> _remove_resolved_at(t, R, vn), names; init=model.values)
    end
    markers = _removals(R, model)
    if recursive
        for vn in (isempty(names) ? (nothing,) : names)
            marker = ModelRemoval(vn, _has_removable(R, model.values.values, vn))
            if vn === nothing
                markers = (marker,)
            elseif !any(r -> _removal_covers(r, vn), markers)
                markers = (markers..., marker)
            end
        end
    end
    result = _reconstruct_model(
        model;
        values=table,
        observation_removals=R === Condition ? markers : model.observation_removals,
        fixed_removals=R === Fix ? markers : model.fixed_removals,
    )
    _check_shared_latent_storage(
        table, merge(model.args, model.defaults), _model_prefix(model)
    )
    return _check_latent_arguments(result)
end
function _check_removal_storage(model, storage)
    for marker in (model.observation_removals..., model.fixed_removals...)
        marker.name === nothing && continue
        _check_removal_storage(
            model.values.storage,
            storage,
            AbstractPPL.varname_to_optic(marker.name),
            nothing,
        )
    end
    return nothing
end
_check_removal_storage(previous, supplied, ::AbstractPPL.Iden, vn) = nothing
function _check_removal_storage(previous, supplied, optic, vn)
    (previous === nothing || supplied === nothing) && return nothing
    if optic isa AbstractPPL.Index &&
        previous isa AbstractArray &&
        supplied isa AbstractArray
        axes(previous) == axes(supplied) || throw(
            ArgumentError(
                "Cannot change storage for namespace root `$vn` while recursive removals remain; rebuild the model with the new storage",
            ),
        )
    end
    head = AbstractPPL.ohead(optic)
    address = if vn === nothing
        AbstractPPL.optic_to_varname(head)
    else
        AbstractPPL.append_optic(vn, head)
    end
    return _check_removal_storage(
        _storage_at(previous, head), _storage_at(supplied, head), optic.child, address
    )
end

function _has_removable(R, values, vn)
    return _has_binding_role(
        R,
        if vn === nothing
            values
        else
            _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
        end,
    )
end
_has_binding_role(R, value) = _matches_model_role(R, value)
_has_binding_role(R, tree::ModelValueTree) = _has_binding_role(R, tree.values)
function _has_binding_role(R, values::Union{VarNamedTuple,VarNamedTuples.PartialArray})
    return _fold_model_indices(false, values) do found, value, _, _
        found || _has_binding_role(R, value)
    end
end
_removal_covers(r, vn) = r.name === nothing || subsumes(r.name, vn)

_check_removal_index_counts(removal, prefix, name, error_prefix) = nothing
function _check_removal_index_counts(
    removal::AbstractPPL.Property{S}, prefix::AbstractPPL.Property{S}, name, error_prefix
) where {S}
    return _check_removal_index_counts(removal.child, prefix.child, name, error_prefix)
end
function _check_removal_index_counts(
    removal::AbstractPPL.Index, prefix::AbstractPPL.Index, name, error_prefix
)
    length(removal.ix) == length(prefix.ix) || throw(
        ArgumentError(
            "Cannot apply recursive removal `$(maybe_prefix(name, error_prefix))`: its index count must match the parent submodel LHS.",
        ),
    )
    removal.ix == prefix.ix || return nothing
    return _check_removal_index_counts(removal.child, prefix.child, name, error_prefix)
end

function _select_removal(r, prefix, error_prefix=nothing)
    prefix === nothing && return (r,)
    r.name === nothing || _check_removal_index_counts(
        AbstractPPL.varname_to_optic(r.name),
        AbstractPPL.varname_to_optic(prefix),
        r.name,
        error_prefix,
    )
    name = if r.name === nothing || subsumes(r.name, prefix)
        nothing
    elseif subsumes(prefix, r.name)
        AbstractPPL.unprefix(r.name, prefix)
    else
        return ()
    end
    return (ModelRemoval(name, r.matched, r.token),)
end

_record_removal_use(context, role, marker) = nothing
function _record_removal_use(context::AbstractParentContext, role, marker)
    return _record_removal_use(childcontext(context), role, marker)
end

function _submodel_removals(R, model, prefix)
    return mapreduce(
        r -> _select_removal(r, prefix, _model_prefix(model)),
        (a, b) -> (a..., b...),
        _removals(R, model);
        init=(),
    )
end
function _apply_parent_removals(R, table::ResolvedBindings, markers, context, prefix)
    for marker in markers
        _has_removable(R, table.values, marker.name) &&
            _record_removal_use(context, R, marker)
        table = if marker.name === nothing
            _unbind_resolved(table, R)
        else
            _remove_resolved_at(table, R, marker.name, prefix)
        end
    end
    return table
end
function _check_assigned_binding_storage(array, optic, vn)
    eltype(array) <: VarNamedTuple && return nothing
    if checkbounds(Bool, array, optic.ix...; optic.kw...) &&
        !VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...) &&
        !isassigned(array, optic.ix...; optic.kw...)
        throw(ArgumentError("Cannot resolve `$vn`: child storage is unassigned."))
    end
    return nothing
end

_binding_read_context(context, values) = context
function _binding_read_context(context::AbstractParentContext, values)
    return setchildcontext(context, _binding_read_context(childcontext(context), values))
end

function _remove_model_values(R, values::VarNamedTuple, names)
    for vn in names
        values = _prune_model_bindings(
            _remove_model_binding(R, values, AbstractPPL.varname_to_optic(vn))
        )
    end
    return values
end

function _local_binding_input(model, ::Type{T}) where {T<:AbstractPPL.OfNamedTuple}
    storage = zero(T)
    selected = _storage_at(storage, AbstractPPL.varname_to_optic(_model_prefix(model)))
    if selected === nothing || !(selected isa NamedTuple)
        for name in _binding_template_names(storage)
            absolute = maybe_prefix(name, _model_prefix(model))
            throw(
                ArgumentError(
                    "Binding template entry `$name` uses a local name; use the absolute name `$absolute`.",
                ),
            )
        end
    end
    return AbstractPPL.of(selected === nothing ? (;) : selected)
end

_storage_at(storage, ::AbstractPPL.Iden) = storage
function _storage_at(storage, optic)
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(storage, head) || return nothing
    child = _binding_child_template(storage, head)
    child isa NoTemplate && return nothing
    return _storage_at(child, optic.child)
end

function _remove_resolved_at(table, R, vn, error_prefix=nothing)
    root = AbstractPPL.getsym(vn)
    storage = get(table.storage.data, root, NoTemplate())
    if storage isa Union{NoTemplate,Nothing,Missing} &&
        !_contains_model_binding(get(table.values.data, root, nothing))
        AbstractPPL.is_dynamic(AbstractPPL.getoptic(vn)) && throw(
            ArgumentError(
                "Cannot remove `$(maybe_prefix(vn, error_prefix))`: dynamic indices require concrete storage",
            ),
        )
        return table
    end
    return _unbind_resolved(table, R, vn; error_prefix)
end

function _binding_template_names(schema::NamedTuple, prefix=nothing)
    return mapreduce((a, b) -> (a..., b...), pairs(schema); init=()) do (name, storage)
        vn = maybe_prefix(VarName{name}(), prefix)
        storage isa NamedTuple ? _binding_template_names(storage, vn) : (vn,)
    end
end
function _record_binding_template(model, schema)
    prefix = _model_prefix(model)
    names = map(_binding_template_names(schema)) do name
        local_name = prefix === nothing || name != prefix && subsumes(prefix, name)
        address =
            local_name && prefix !== nothing ? AbstractPPL.unprefix(name, prefix) : name
        (local_name, address)
    end
    metadata = _with_binding_template_names(_binding_metadata(model), names)
    return Model{requires_threadsafe(model)}(
        model.f,
        model.args,
        model.defaults,
        model.context,
        model.values;
        args_on_lhs=metadata,
        observation_removals=model.observation_removals,
        fixed_removals=model.fixed_removals,
    )
end

function _check_latent_arguments(model)
    arguments = merge(model.args, model.defaults)
    lhs = _args_on_lhs(_binding_metadata(model))
    map(keys(arguments), values(arguments)) do stored_name, value
        _argument_may_be_unwritable(typeof(value)) ||
            _argument_may_be_unwritable(typeof(model.values)) ||
            return nothing
        vn = VarName{unsplat_symbol(stored_name)}()
        AbstractPPL.getsym(vn) in lhs || return nothing
        binding = _get_model_binding(model, vn)
        storage = binding === nothing ? value : _argument_storage(binding, value)
        _check_latent_storage(storage, maybe_prefix(vn, _model_prefix(model)))
    end
    return model
end
