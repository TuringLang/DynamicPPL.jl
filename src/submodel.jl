"""
    Submodel{M,AutoPrefix}

A wrapper around a model, plus a flag indicating whether it should be automatically
prefixed with the left-hand variable in a `~` statement.
"""
struct Submodel{M,AutoPrefix}
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
`value`, and its latent LHS variables are recorded separately in the surrounding trace. By
default, their names are prefixed with the left-hand side: a latent LHS variable `x` becomes
`value.x`. This differs from [`to_distribution`](@ref), which assigns the represented latent
LHS variable values to the left-hand side.

Conceptually, `to_submodel(model)` is a `returned_value(model)` wrapper: its value is the
model's return value, not its latent LHS variable values.

Condition or fix a submodel through its namespace in the parent: for example,
`@varname(a.x)` in `a ~ to_submodel(child())`. With `auto_prefix=false`, use the child's
names unchanged. Parent explicit bindings override the child's explicit bindings and
argument-supplied observations at the same address; parent argument-supplied observations never reach a child.

Binding the return value throws `ArgumentError` at the submodel tilde during evaluation.
A submodel tilde whose LHS is rooted at a model argument throws `ArgumentError` when
it runs, including indexed and field addresses. Use a local LHS variable; bind the
child's LHS variables with `condition`.
To remove a child's argument-supplied observations from the parent, use
`decondition(parent, DynamicPPL.Recursive(), @varname(a.x))`.
See [Binding rules](@ref).

`Submodel` is not a `Distribution`; it provides this tilde behavior but no standalone
`logpdf` method.

!!! warning
    Keep `auto_prefix=true` unless the wrapped model has been explicitly prefixed. Disabling
    automatic prefixing can make latent LHS variable addresses collide.

# Arguments

- `model::Model`: the model to wrap.
- `auto_prefix::Bool=true`: whether to prefix the model's latent LHS variables with the
  left-hand side of `~`.

# Examples

```jldoctest submodel-to_submodel
julia> using DynamicPPL, Distributions

julia> @model function demo1()
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(y)
            a ~ to_submodel(demo1())
            return y ~ Uniform(0, a)
       end;
```

When sampling from `demo2(0.4)`, the latent LHS variable `x` is prefixed with `a`, the
left-hand side of the tilde:

```jldoctest submodel-to_submodel
julia> model = demo2(0.4);

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

## Without automatic prefixing

If `auto_prefix=false`, the submodel's latent LHS variable addresses are unchanged.
```jldoctest submodel-to_submodel-prefix; setup=:(using Distributions)
julia> @model function demo1()
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2_no_prefix(z)
            a ~ to_submodel(demo1(), false)
            return z ~ Uniform(-a, 1)
       end;

julia> model = demo2_no_prefix(0.4);

julia> haskey(rand(model), @varname(x))  # here we just use `x` instead of `a.x`
true
```
However, not using prefixing is generally not recommended as it can lead to variable name
clashes unless one is careful. For example, if the same submodel is used multiple times in a
model, not using prefixing will lead to variable name clashes.

One can manually specify a prefix using [`prefix(::Model, prefix_varname)`](@ref):

```jldoctest submodel-to_submodel-prefix
julia> @model function demo2(z)
            a ~ to_submodel(prefix(demo1(), @varname(sub1)), false)
            b ~ to_submodel(prefix(demo1(), @varname(sub2)), false)
            return z ~ Uniform(-a, b)
       end;

julia> model = demo2(0.4);

julia> haskey(rand(model), @varname(sub1.x))
true

julia> haskey(rand(model), @varname(sub2.x))
true
```
"""
to_submodel(m::Model, auto_prefix::Bool=true) = Submodel{typeof(m),auto_prefix}(m)

# ---------------------------
# Submodels in tilde-pipeline
# ---------------------------

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

function _submodel_values(model::Model, prefix)
    prefix = _model_value_varname(model.values, prefix, getprefix(model))
    return _submodel_values(_model_values(model.values), prefix)
end
_submodel_values(values::VarNamedTuple, ::Nothing) = values
function _submodel_values(values::VarNamedTuple, prefix::VarName)
    binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(prefix))
    binding === nothing && return VarNamedTuple()
    return _submodel_namespace(binding)
end

# Shape ownership survives selecting a child's namespace, including whole namespace bindings.
function _submodel_fixed_owners(model::Model, prefix)
    prefix = _model_value_varname(model.values, prefix, getprefix(model))
    owners = _fixed_owners(model.values)
    prefix === nothing && return owners
    return mapreduce((a, b) -> (a..., b...), owners; init=()) do owner
        if subsumes(owner, prefix)
            names = keys(_submodel_values(_model_values(model.values), prefix).data)
            map(name -> VarName{name}(), names)
        elseif subsumes(prefix, owner)
            (AbstractPPL.unprefix(owner, prefix),)
        else
            ()
        end
    end
end

"""
    DynamicPPL.tilde_assume!!(
        parent_model::Model,
        context::AbstractContext,
        submodel::DynamicPPL.Submodel,
        left_vn::VarName,
        template,
        vi::AbstractVarInfo
    )

Evaluate `submodel` under `parent_model`.
"""
@inline function tilde_assume!!(
    parent_model::Model,
    context::AbstractContext,
    submodel::Submodel{M,AutoPrefix},
    left_vn::VarName,
    template,
    vi::AbstractVarInfo,
) where {M<:Model,AutoPrefix}
    name = AbstractPPL.getsym(left_vn)
    if name in
        (_argument_names(parent_model.args)..., _argument_names(parent_model.defaults)...)
        throw(
            ArgumentError(
                "Submodel tilde `$left_vn` is rooted at model argument `$name`. " *
                "Use a local LHS variable, or bind the child's variables with `condition`.",
            ),
        )
    end
    !AutoPrefix &&
        getprefix(submodel.model) === nothing &&
        _check_shared_removals(parent_model, submodel.model)
    left_vn = AutoPrefix ? _concretize_prefix(left_vn, template; prefix=Val(true)) : left_vn
    local_prefix = if AutoPrefix
        maybe_prefix(getprefix(submodel.model), left_vn)
    else
        getprefix(submodel.model)
    end
    inherited_templates = _submodel_binding_templates(parent_model, local_prefix)
    observation_removals = _submodel_removals(Condition, parent_model, local_prefix)
    fixed_removals = _submodel_removals(Fix, parent_model, local_prefix)
    observations = _apply_parent_removals(
        Condition,
        _submodel_layer(Condition, submodel.model),
        observation_removals,
        context,
        submodel.model,
    )
    fixed = _apply_parent_removals(
        Fix, _submodel_layer(Fix, submodel.model), fixed_removals, context, submodel.model
    )
    child_owners = filter(_submodel_fixed_owners(submodel.model, nothing)) do owner
        !any(r -> _removal_covers(r, owner), fixed_removals)
    end
    child_layers = ModelBindingLayers(observations, fixed, child_owners)
    child_values = _model_values(child_layers)
    child_model = _reconstruct_model(submodel.model; values=LocalModelValues(child_values))
    parent_values = _check_argument_bindings(
        child_model,
        _prepare_inherited_templates(
            child_model,
            _submodel_inherited_values(parent_model, local_prefix),
            inherited_templates,
        ),
        maybe_prefix(local_prefix, getprefix(parent_model)),
    )
    # Shared unprefixed names may belong to the parent or another child.
    if AutoPrefix || getprefix(submodel.model) !== nothing
        _check_binding_addresses(child_model, parent_values)
    end
    owners = (child_owners..., _submodel_fixed_owners(parent_model, local_prefix)...)
    values = LocalModelValues(
        _with_removals(
            _merge_model_values(
                child_values,
                parent_values,
                maybe_prefix(local_prefix, getprefix(parent_model)),
            ),
            (
                _submodel_removals(Condition, submodel.model, nothing)...,
                observation_removals...,
            ),
            (_submodel_removals(Fix, submodel.model, nothing)..., fixed_removals...),
        ),
        owners,
    )
    child_templates = _submodel_binding_templates(submodel.model, nothing)
    templates = if isempty(child_templates)
        ()
    else
        filter(child_templates) do t
            role = t isa ModelBindingTemplate{Condition} ? Condition : Fix
            layer = _binding_layer(role, parent_values)
            removals = role === Condition ? observation_removals : fixed_removals
            return !any(r -> _removal_covers(r, t.name), removals) &&
                   _has_removable(role, _binding_layer(role, child_layers), t.name) &&
                   !any(vn -> subsumes(vn, t.name), keys(layer)) &&
                   !any(
                       p -> p isa ModelBindingTemplate{role} && p.name == t.name,
                       inherited_templates,
                   )
        end
    end
    templates = (templates..., inherited_templates...)
    values = _with_binding_templates(values, templates)
    isempty(observation_removals) && isempty(fixed_removals) ||
        _check_shared_latent_storage(
            _reconstruct_model(submodel.model; values),
            maybe_prefix(local_prefix, getprefix(parent_model)),
        )
    return _evaluate_submodel!!(
        parent_model, context, submodel, left_vn, template, vi, values
    )
end

# Specialize child evaluation on the selected submodel namespace bindings.
@inline function _evaluate_submodel!!(
    parent_model::Model,
    context::AbstractContext,
    submodel::Submodel{M,AutoPrefix},
    left_vn::VarName,
    template,
    vi::AbstractVarInfo,
    values::LocalModelValues,
) where {M,AutoPrefix}
    model = if AutoPrefix
        vn, template = _prefix_varname_and_template(left_vn, template, parent_model)
        _prefix_model(submodel.model, vn, template, values)
    else
        prefix = _compose_prefix_templates(parent_model.prefix, submodel.model.prefix)
        _reconstruct_model(submodel.model; prefix, values)
    end
    # Calling model.f directly avoids the inference recursion limit as nested prefixes
    # change the Model type; routing through _evaluate!! widens it to Any (Turing.jl#2844).
    model = setleafcontext(model, context)
    args, kwargs = make_evaluate_args_and_kwargs(model, vi)
    return model.f(args...; kwargs...)
end

function tilde_observe!!(prefix, ::DynamicPPL.Submodel, left, ::Nothing, template, vi)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is a literal"))
end
