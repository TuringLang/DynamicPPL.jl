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
`value`, and its latent variables are recorded separately in the surrounding trace. By
default, their names are prefixed with the left-hand side: a latent variable `x` becomes
`value.x`. This differs from [`to_distribution`](@ref), which assigns the represented latent
variables themselves to the left-hand side.

Conceptually, `to_submodel(model)` is a `returned_value(model)` wrapper: its value is the
model's return value, not its latent variables.

Condition a submodel through its internal variable names, not its return value. If an
argument provides storage for submodel return values, remove its default observation with
[`decondition`](@ref) before evaluation.

`Submodel` is not a `Distribution`; it provides this tilde behavior but no standalone
`logpdf` method.

!!! warning
    Operations normally associated with `left ~ right`, such as [`condition`](@ref), do not
    necessarily work with `to_submodel`.

!!! warning
    Keep `auto_prefix=true` unless the wrapped model has been explicitly prefixed. Disabling
    automatic prefixing can make latent-variable names collide.

# Arguments

- `model::Model`: the model to wrap.
- `auto_prefix::Bool=true`: whether to prefix the model's latent variables with the
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

When sampling from `demo2(0.4)`, the latent variable `x` is prefixed with `a`, the
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
julia> accs = setacc!!(OnlyAccsVarInfo(), RawValueAccumulator(false));

julia> _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll());

julia> x = get_raw_values(accs)[@varname(a.x)];

julia> getlogjoint(accs) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```

## Without automatic prefixing

If `auto_prefix=false`, the submodel's latent-variable names are unchanged.
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

"""
    DynamicPPL.tilde_assume!!(
        model::Model,
        context::AbstractContext,
        right::DynamicPPL.Submodel,
        vn::VarName,
        ::Any,
        vi::AbstractVarInfo
    )

Evaluate the submodel under the parent `model`.
"""
function tilde_assume!!(
    model::Model,
    context::AbstractContext,
    right::DynamicPPL.Submodel,
    vn::VarName,
    template,
    vi::AbstractVarInfo,
)
    return _evaluate!!(right, context, vi, model, vn, template)
end

_submodel_namespace(values::VarNamedTuple) = values
_submodel_namespace(value::ModelValueTree{<:NamedTuple}) = value.values
function _submodel_namespace(
    value::ModelValue{R,<:NamedTuple}
) where {R<:Union{Condition,Fix}}
    return _tag_model_values(R, VarNamedTuple(value.value))
end
function _submodel_namespace(::Union{ModelValue,ModelValueTree})
    throw(
        ArgumentError(
            "Cannot condition or fix a submodel's return value. Supply its internal variable names instead; decondition arguments used only as return-value buffers.",
        ),
    )
end

_submodel_values(values::VarNamedTuple, ::Nothing, template) = values
function _submodel_values(values::VarNamedTuple, prefix::VarName, template)
    binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(prefix))
    binding === nothing && return VarNamedTuple()
    return _prefix_values(_submodel_namespace(binding), prefix, template)
end

# When automatic prefixing is used, the submodel itself doesn't carry the
# prefix, as the prefix is obtained from the LHS of `~` (whereas the submodel
# is on the RHS). The prefix can only be obtained in `tilde_assume!!`, and then
# passed into this function.
function _evaluate!!(
    submodel::Submodel{M,AutoPrefix},
    context::AbstractContext,
    vi::AbstractVarInfo,
    parent_model::Model,
    left_vn::VarName,
    template,
) where {M<:Model,AutoPrefix}
    parent_prefix = parent_model.prefix
    model = if AutoPrefix
        vn, template = _prefix_varname_and_template(left_vn, template, parent_model)
        prefix(submodel.model, vn; template)
    elseif parent_prefix === nothing
        submodel.model
    else
        model = prefix(
            submodel.model,
            parent_prefix;
            template=_apply_prefix_template(parent_model.prefix_template, NoTemplate()),
        )
        if parent_model.prefix_template === nothing
            model
        else
            inner = if submodel.model.prefix_template === nothing
                submodel.model.prefix
            else
                submodel.model.prefix_template
            end
            prefix_template = _compose_prefix_templates(parent_model.prefix_template, inner)
            _reconstruct_model(model; prefix_template)
        end
    end
    values = _merge_model_values(
        model.values,
        _submodel_values(
            parent_model.values,
            model.prefix,
            _apply_prefix_template(model.prefix_template, NoTemplate()),
        ),
    )
    model = _reconstruct_model(model; values)

    # Calling model.f directly avoids the inference recursion limit as nested prefixes
    # change the Model type; routing through _evaluate!! widens it to Any (Turing.jl#2844).
    args, kwargs = make_evaluate_args_and_kwargs(model, context, vi)
    return model.f(args...; kwargs...)
end

function tilde_observe!!(
    ::Model, right::DynamicPPL.Submodel, left, ::Nothing, template, vi::AbstractVarInfo
)
    return _tilde_observe!!(nothing, nothing, right, left, nothing, template, vi)
end
function _tilde_observe!!(
    prefix, prefix_template, ::DynamicPPL.Submodel, left, ::Nothing, template, vi
)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is a literal"))
end
