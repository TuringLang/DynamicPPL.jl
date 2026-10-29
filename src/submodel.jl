"""
    Submodel{M}

Wrap a model whose variables are prefixed by its tilde address.
"""
struct Submodel{M}
    model::M
end

"""
    to_distribution(model)

Convert `model` to a distribution for use on the right-hand side of `~`.

In `variables ~ to_distribution(model)`, `variables` receives the latent variables
represented by the resulting distribution. The concrete representation and density depend
on the method for `typeof(model)`. This differs from [`to_submodel`](@ref), which assigns a
wrapped model's return value to the left-hand side and records its latent variables
separately.
"""
function to_distribution end

abstract type _StanDifferentiableFunction end
(f::_StanDifferentiableFunction)(x::AbstractArray{<:Real}) = _stan_value(f, x)

function _stan_value end
function _stan_value_and_pushforward end
function _stan_value_and_pullback end

# ----------------------
# Constructing submodels
# ----------------------

"""
    to_submodel(model::Model[, auto_prefix::Bool])

Wrap `model` for use on the right-hand side of `~`.

In `value ~ to_submodel(model)`, `model` is evaluated, its return value is assigned to
`value`, and its latent variables are recorded separately in the surrounding trace. Their
names are prefixed with the left-hand side: a latent variable `x` becomes `value.x`. This
differs from [`to_distribution`](@ref), which assigns the represented latent variables
themselves to the left-hand side.

Conceptually, `to_submodel(model)` is a `returned_value(model)` wrapper: its value is the
model's return value, not its latent variables.

Submodels cannot observe through arguments. Use [`condition`](@ref) to supply
observations and retain their likelihood; [`fix`](@ref) omits that term.
The model called directly by the user can still observe through its arguments.
See [Migration to model-owned bindings](@ref) for migration instructions.

`Submodel` is not a `Distribution`; it provides this tilde behavior but no standalone
`logpdf` method.

The optional second argument accepts `true` for compatibility. Passing `false` throws
`ArgumentError`; choose the child name on the left-hand side of `~` instead.

# Examples

```jldoctest submodel-to_submodel
julia> using DynamicPPL, Distributions

julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y)
            a ~ to_submodel(demo1(x))
            return y ~ Uniform(0, a)
       end;
```

When sampling from `demo2(missing, 0.4)`, the latent variable `x` is prefixed with `a`, the
left-hand side of the tilde:

```jldoctest submodel-to_submodel
julia> model = demo2(missing, 0.4);

julia> haskey(rand(model), @varname(a.x))
true
```

The variable `a` receives the return value of `demo1` and can be used in subsequent lines,
as in the definition of `y` above.

We can verify that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-to_submodel
julia> accs = setacc!!(VarInfo(), RawValueAccumulator(false));

julia> _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll());

julia> x = get_raw_values(accs)[@varname(a.x)];

julia> getlogjoint(accs) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```

An explicit prefix composes with the tilde address: `a ~ to_submodel(prefix(child,
@varname(b)))` records a child variable `x` as `a.b.x`.
"""
to_submodel(m::Model) = Submodel(m)
function to_submodel(m::Model, auto_prefix::Bool)
    auto_prefix || throw(
        ArgumentError(
            "Unprefixed submodels are unsupported; use `name ~ to_submodel(model)` and update child variable addresses to `name.x`.",
        ),
    )
    return to_submodel(m)
end

_submodel_namespace(values::VarNamedTuple) = values
_submodel_namespace(value::ModelValueTree{<:NamedTuple}) = value.values
function _submodel_namespace(
    value::ModelValue{R,<:NamedTuple}
) where {R<:Union{Condition,Fix}}
    return _tag_model_values(R, VarNamedTuple(value.value))
end
function _submodel_namespace(
    value::ModelValue{R,<:VarNamedTuple}
) where {R<:Union{Condition,Fix}}
    return _tag_model_values(R, value.value)
end
function _submodel_namespace(::Union{ModelValue,ModelValueTree})
    throw(
        ArgumentError(
            "Cannot explicitly bind a submodel return value. Remove the explicit binding, " *
            "or bind the child's variables by prefixed name (e.g. `@varname(a.z)`).",
        ),
    )
end

_submodel_namespace(::NoModelBinding) = VarNamedTuple()
_submodel_values(model::Model, prefix) = _submodel_values(model.values.values, prefix)
_submodel_values(values::VarNamedTuple, ::Nothing) = values
function _submodel_values(values::VarNamedTuple, prefix::VarName)
    binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(prefix))
    binding === nothing && return VarNamedTuple()
    return _submodel_namespace(binding)
end

function _child_storage(values::VarNamedTuple, prefix)
    value = _storage_at(values, AbstractPPL.varname_to_optic(prefix))
    value isa Union{Nothing,Number,NoTemplate} && return VarNamedTuple()
    value isa VarNamedTuple && return value
    value isa NamedTuple && return VarNamedTuple(value)
    throw(ArgumentError("Child namespace `$prefix` requires named storage"))
end
function _check_child_storage(storage, values, inherited, vn=nothing)
    if values isa VarNamedTuples.PartialArray && storage isa AbstractArray
        axes(values) == axes(storage) || throw(
            ArgumentError(
                "Binding shape at `$vn` conflicts with existing storage in the child model",
            ),
        )
    end
    _fold_model_indices(nothing, values) do _, binding, optic, _
        address = if vn === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(vn, optic)
        end
        incoming = _model_argument_binding(inherited, optic)
        incoming === nothing && return nothing
        owner = _storage_at(storage, optic)
        owner === nothing && return nothing
        if binding isa ModelValue
            _same_storage_shape(owner, binding.value) || throw(
                ArgumentError(
                    "Binding shape at `$address` conflicts with existing storage in the child model",
                ),
            )
        elseif binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray,ModelValueTree}
            _check_child_storage(
                owner,
                binding isa ModelValueTree ? binding.values : binding,
                incoming,
                address,
            )
        end
        nothing
    end
end
function _prepare_child_bindings(table, values::VarNamedTuple, prefix)
    return VarNamedTuple(
        NamedTuple{keys(values.data)}(
            map(keys(values.data)) do name
                value = values.data[name]
                previous = get(table.values.data, name, nothing)
                _contains_model_binding(previous) || return value
                vn = VarName{name}()
                if !(value isa ModelValue)
                    value = VarNamedTuples._map_pairs_recursive!!(copy(value), vn) do pair
                        address, binding = pair
                        _model_value_like(
                            binding,
                            _convert_binding_edit(
                                binding.value,
                                previous,
                                AbstractPPL.getoptic(address),
                                maybe_prefix(address, prefix),
                                typeof(_model_role(binding)),
                            ),
                        )
                    end
                end
                template = get(
                    table.observed_templates.data,
                    name,
                    get(table.fixed_templates.data, name, NoTemplate()),
                )
                template isa NoTemplate && return value
                if value isa ModelValue
                    _convert_resolved_value(
                        value.value,
                        template,
                        AbstractPPL.Iden(),
                        maybe_prefix(vn, prefix),
                        true,
                    )
                    value
                else
                    _prepare_namespace_bindings(value, template, maybe_prefix(vn, prefix))
                end
            end,
        ),
    )
end

_prepare_namespace_bindings(value::VarNamedTuple, template::Number, vn) = value
_prepare_namespace_bindings(value::ModelValue, template, vn) = value
function _prepare_namespace_bindings(values, template, vn)
    return _fold_model_indices(copy(values), values) do result, binding, optic, _
        address = AbstractPPL.append_optic(vn, optic)
        storage = _storage_at(template, optic)
        storage === nothing && return result
        binding = if binding isa ModelValue
            _prepare_binding_selection(binding, template, optic, vn)
        else
            _prepare_namespace_bindings(binding, storage, address)
        end
        VarNamedTuples._setindex_optic!!(
            result, binding, optic, template, VarNamedTuples.AllowAll()
        )
    end
end

@inline function tilde_assume!!(
    parent_model::Model,
    context::AbstractContext,
    submodel::Submodel{M},
    left_vn::VarName,
    template,
    vi::AbstractVarInfo,
) where {M<:Model}
    name = AbstractPPL.getsym(left_vn)
    name in
    (_argument_names(parent_model.args)..., _argument_names(parent_model.defaults)...) &&
        throw(
            ArgumentError(
                "Submodel tilde `$left_vn` is rooted at model argument `$name`; use a local LHS variable and condition the child's variables by their prefixed names.",
            ),
        )
    left_vn = _concretize_prefix(left_vn, template; prefix=Val(true))
    local_prefix = maybe_prefix(_model_prefix(submodel.model), left_vn)
    observations = _submodel_removals(Condition, parent_model, local_prefix)
    fixed = _submodel_removals(Fix, parent_model, local_prefix)
    return _tilde_assume_submodel!!(
        parent_model,
        context,
        submodel,
        left_vn,
        template,
        vi,
        local_prefix,
        observations,
        fixed,
    )
end
@inline function _tilde_assume_submodel!!(
    parent_model,
    context,
    submodel,
    left_vn,
    template,
    vi,
    local_prefix,
    observation_removals,
    fixed_removals,
)
    child = submodel.model
    inherited = _inherited_model_values(_submodel_values(parent_model, local_prefix))
    isempty(inherited.data) ||
        template isa NoTemplate ||
        _check_bound_shape(
            _get_model_binding(parent_model, VarName{AbstractPPL.getsym(left_vn)}()),
            template,
            AbstractPPL.getoptic(left_vn),
            maybe_prefix(left_vn, _model_prefix(parent_model)),
            Condition;
            submodel=true,
        )
    return _enter_submodel!!(
        parent_model,
        context,
        child,
        left_vn,
        template,
        vi,
        local_prefix,
        observation_removals,
        fixed_removals,
        inherited,
    )
end
_same_binding_template(a, b) = typeof(a) === typeof(b) && _same_storage_shape(a, b)
function _same_binding_template(
    a::Union{NamedTuple,VarNamedTuple}, b::Union{NamedTuple,VarNamedTuple}
)
    a = a isa VarNamedTuple ? a.data : a
    b = b isa VarNamedTuple ? b.data : b
    return all(keys(a)) do name
        !haskey(b, name) || _same_binding_template(a[name], b[name])
    end
end
@noinline function _check_child_templates(table, observations, fixed, prefix)
    for root in table.roots
        optic = AbstractPPL.varname_to_optic(root)
        for previous in (table.observed_templates, table.fixed_templates),
            supplied in (observations, fixed)

            a, b = _storage_at(previous, optic), _storage_at(supplied, optic)
            (a === nothing || b === nothing) && continue
            _same_binding_template(a, b) || throw(
                ArgumentError(
                    "Binding template at `$(maybe_prefix(root, prefix))` conflicts with existing storage in the child model",
                ),
            )
        end
    end
    return nothing
end
@noinline function _check_retained_child_storage(storage, table, inherited, prefix)
    for root in table.roots
        optic = AbstractPPL.varname_to_optic(root)
        binding = _model_argument_binding(inherited, optic)
        binding isa Union{Nothing,ModelValue} && continue
        previous = _storage_at(table.storage, optic)
        supplied = _storage_at(storage, optic)
        (previous === nothing || supplied === nothing) && continue
        inherited_shape = if binding isa VarNamedTuples.PartialArray
            _inherits_shape(binding.mask)
        else
            _inherits_shape(binding)
        end
        if !inherited_shape
            _fold_model_indices(
                nothing, binding isa ModelValueTree ? binding.values : binding
            ) do _, _, child_optic, _
                _check_namedtuple_index(
                    ModelValue{Condition}(previous),
                    child_optic,
                    AbstractPPL.varname_to_optic(maybe_prefix(root, prefix));
                    check_container=true,
                )
            end
        end
        inherited_shape ||
            _same_storage_shape(previous, supplied) ||
            throw(
                ArgumentError(
                    "Inherited storage at `$(maybe_prefix(root, prefix))` disagrees with the child's retained shape",
                ),
            )
    end
    return nothing
end
@noinline function _submodel_argument_observation_error(child, prefix)
    names = filter(keys(child.values.values.data)) do name
        _has_binding_role(ArgumentCondition, child.values.values.data[name])
    end
    addresses = join(("`$(maybe_prefix(VarName{name}(), prefix))`" for name in names), ", ")
    arguments = join(("`$name`" for name in names), ", ")
    positional = map(keys(child.args)) do stored
        name = unsplat_symbol(stored)
        if is_splat_symbol(stored)
            name in names ? "map(_ -> missing, $name)..." : "$name..."
        else
            name in names ? "missing" : string(name)
        end
    end
    keywords = map(keys(child.defaults)) do stored
        name = unsplat_symbol(stored)
        if is_splat_symbol(stored)
            name in names ? "map(_ -> missing, NamedTuple($name))..." : "$name..."
        else
            value = name in names ? "missing" : string(name)
            "$name=$value"
        end
    end
    call = string(
        nameof(child),
        "(",
        join(positional, ", "),
        isempty(keywords) ? "" : "; " * join(keywords, ", "),
        ")",
    )
    bindings = join(("$name=$name" for name in names), ", ") * ","
    plural = length(names) == 1 ? "argument" : "arguments"
    throw(
        ArgumentError(
            "Submodel `$prefix` observes $addresses through its $plural $arguments; submodels cannot take observations as arguments. Pass `missing` for $arguments and condition the submodel on the value instead, e.g. `$prefix ~ to_submodel(condition($call, ($bindings)))`. Use `condition` rather than `fix`: `condition` keeps the likelihood term for $arguments, while `fix` sets $arguments without one.",
        ),
    )
end

@inline function _enter_submodel!!(
    parent_model,
    context,
    child,
    left_vn,
    template,
    vi,
    local_prefix,
    observation_removals,
    fixed_removals,
    inherited,
)
    prefix = maybe_prefix(local_prefix, _model_prefix(parent_model))
    _has_binding_role(ArgumentCondition, child.values.values) &&
        _submodel_argument_observation_error(child, prefix)
    table = _apply_parent_removals(
        Condition, child.values, observation_removals, context, prefix
    )
    table = _apply_parent_removals(Fix, table, fixed_removals, context, prefix)
    _check_binding_addresses(child, inherited, prefix)
    inherited = _resolve_known_namespaces(inherited, child, nothing)
    storage = _child_storage(parent_model.values.storage, local_prefix)
    observed_templates = _child_storage(
        parent_model.values.observed_templates, local_prefix
    )
    fixed_templates = _child_storage(parent_model.values.fixed_templates, local_prefix)
    _check_child_templates(table, observed_templates, fixed_templates, prefix)
    _check_retained_child_storage(storage, table, inherited, prefix)
    _check_child_storage(storage, table.values, inherited, prefix)
    inherited = _prepare_child_bindings(table, inherited, prefix)
    inherited = _check_argument_bindings(child, inherited, true, prefix)
    _check_cross_roles(table.values, inherited, prefix; disjoint=true)
    values = _merge_model_values(table.values, inherited)
    stored = merge(table.storage, storage)
    _check_removal_storage(child, stored)
    table = ResolvedBindings(
        values,
        stored,
        merge(table.observed_templates, observed_templates),
        merge(table.fixed_templates, fixed_templates),
        table.arguments,
        table.roots,
    )
    # Julia 1.10 allocates keyword tuples when child removal-marker types differ.
    model = _reconstruct_model(
        child,
        child.context,
        table,
        Val(requires_threadsafe(child)),
        (child.observation_removals..., observation_removals...),
        (child.fixed_removals..., fixed_removals...),
    )
    isempty(observation_removals) && isempty(fixed_removals) ||
        _check_shared_latent_storage(table, merge(model.args, model.defaults), prefix)
    context = _binding_read_context(context, child.values.values)
    return _evaluate_submodel!!(parent_model, context, model, left_vn, template, vi)
end
@inline function _evaluate_submodel!!(parent_model, context, model, left_vn, template, vi)
    vn, template = _prefix_varname_and_template(left_vn, template, parent_model)
    model = _prefix_model(model, vn, template, model.values)
    model = setleafcontext(model, context)
    args, kwargs = make_evaluate_args_and_kwargs(model, vi)
    return @inline model.f(args...; kwargs...)
end
function tilde_observe!!(prefix, prefix_template, ::Submodel, left, ::Nothing, template, vi)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is a literal"))
end
