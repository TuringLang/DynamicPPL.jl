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

Submodels cannot observe through arguments. Use [`condition`](@ref) to supply
observations and retain their likelihood; [`fix`](@ref) omits that term.
The model called directly by the user can still observe through its arguments.
See [Migration to model-owned bindings](@ref) for migration instructions.

`Submodel` is not a `Distribution`; it provides this tilde behavior but no standalone
`logpdf` method.

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

## Without automatic prefixing

If `auto_prefix=false`, the submodel's latent-variable names are unchanged.
```jldoctest submodel-to_submodel-prefix; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2_no_prefix(x, z)
            a ~ to_submodel(demo1(x), false)
            return z ~ Uniform(-a, 1)
       end;

julia> model = demo2_no_prefix(missing, 0.4);

julia> haskey(rand(model), @varname(x))  # here we just use `x` instead of `a.x`
true
```
However, not using prefixing is generally not recommended as it can lead to variable name
clashes unless one is careful. For example, if the same submodel is used multiple times in a
model, not using prefixing will lead to variable name clashes.

One can manually specify a prefix using [`prefix(::Model, prefix_varname)`](@ref):

```jldoctest submodel-to_submodel-prefix
julia> @model function demo2(x, y, z)
            a ~ to_submodel(prefix(demo1(x), @varname(sub1)), false)
            b ~ to_submodel(prefix(demo1(y), @varname(sub2)), false)
            return z ~ Uniform(-a, b)
       end;

julia> model = demo2(missing, missing, 0.4);

julia> haskey(rand(model), @varname(sub1.x))
true

julia> haskey(rand(model), @varname(sub2.x))
true
```
"""
to_submodel(m::Model, auto_prefix::Bool=true) = Submodel{typeof(m),auto_prefix}(m)

_relative_removal(a, b) = (nothing, false)
_relative_removal(a, ::AbstractPPL.Iden) = (a, true)
_relative_removal(::AbstractPPL.Iden, b::AbstractPPL.AbstractOptic) = (nothing, true)
_relative_removal(::AbstractPPL.Iden, ::AbstractPPL.Iden) = (nothing, true)
function _relative_removal(a::AbstractPPL.Property{S}, b::AbstractPPL.Property{S}) where {S}
    return _relative_removal(a.child, b.child)
end
function _relative_removal(a::AbstractPPL.Index, b::AbstractPPL.Index)
    child, active = _relative_removal(a.child, b.child)
    return child, active && subsumes(AbstractPPL.ohead(a), AbstractPPL.ohead(b))
end
_child_removals(markers, ::Nothing, template) = markers
function _child_removals(markers, prefix, template=NoTemplate())
    canonical_prefix = _expand_cartesian(
        _concretize_prefix(prefix, template; canonical=true)
    )
    return map(markers) do marker
        name = marker.name
        name === nothing && return marker
        name = _expand_cartesian(
            _concretize_prefix(
                name,
                template;
                depth=optic_skip_length(AbstractPPL.getoptic(prefix)),
                canonical=true,
            ),
        )
        a, b = AbstractPPL.varname_to_optic(name),
        AbstractPPL.varname_to_optic(canonical_prefix)
        child, active = _relative_removal(a, b)
        relative = if child isa Union{Nothing,AbstractPPL.Iden}
            nothing
        else
            AbstractPPL.optic_to_varname(child)
        end
        return (name=relative, active=marker.active && active)
    end
end
function _apply_removals(table, R, markers, arguments)
    for marker in markers
        marker.active || continue
        table = _remove_role(
            table,
            R,
            marker.name === nothing ? () : (marker.name,);
            arguments,
            validate=false,
        )
    end
    return table
end
_apply_removals(table, R, ::Tuple{}, arguments) = table
_as_namespace(x::VarNamedTuple, address) = x
_as_namespace(::Union{Nothing,NoBinding}, address) = VarNamedTuple()
function _as_namespace(x::BoundValue{R,<:NamedTuple}, address) where {R}
    return VarNamedTuple(map(BoundValue{R}, x.value))
end
function _as_namespace(x, address)
    return _throw_binding_error(address, "submodel binding requires named values")
end
function _child_bindings(parent, prefix, site)
    values = VarNamedTuple(map(r -> r.value, parent.values.roots.data))
    prefix === nothing && return values
    if prefix != site
        node = _binding_node_at(values, AbstractPPL.varname_to_optic(site))
        VarNamedTuples._mapreduce_recursive((a, b) -> nothing, node, site, nothing) do pair
            location, value = pair
            subsumes(prefix, location) ||
                !_has_bound_value(value) ||
                _throw_binding_error(
                    maybe_prefix(location, _model_prefix(parent)),
                    "address is outside child prefix $prefix",
                )
        end
    end
    return _as_namespace(
        _binding_node_at(values, AbstractPPL.varname_to_optic(prefix)),
        maybe_prefix(prefix, _model_prefix(parent)),
    )
end
function _inherited_constraints(parent, prefix, ::Val{name}) where {name}
    address = maybe_prefix(VarName{name}(), prefix)
    root = _binding_root(parent, address)
    root === nothing && return ()
    return mapreduce((a, b) -> (a..., b...), root.constraints; init=()) do constraint
        (; owner, storage) = constraint
        R = constraint.role
        mode = constraint.mode
        if subsumes(owner, address)
            relative = AbstractPPL._unprefix_optic(
                AbstractPPL.getoptic(address), AbstractPPL.getoptic(owner)
            )
            selected = _binding_node_at(storage, relative)
            # A number has no shape, and keeping its constraint makes the result type unstable.
            return if selected === nothing || mode === Val(:shape) && selected isa Number
                ()
            else
                (BindingConstraint(VarName{name}(), R, selected, mode),)
            end
        elseif subsumes(address, owner)
            return (
                BindingConstraint(
                    prefix === nothing ? owner : AbstractPPL.unprefix(owner, prefix),
                    R,
                    storage,
                    mode,
                ),
            )
        end
        return ()
    end
end
function _merge_child_root(old, incoming, parent, prefix, ::Val{name}, child) where {name}
    vn = VarName{name}()
    site = maybe_prefix(vn, prefix)
    address = maybe_prefix(site, _model_prefix(parent))
    _check_binding_address(child, vn, address)
    if incoming isa BoundValue && _is_namespace(child, vn)
        incoming = _as_namespace(incoming, address)
    end
    _check_role_overlap(old.value, incoming, address; disjoint=true)
    if incoming isa _PositionalNode &&
        old.storage isa Union{AbstractArray,Tuple} &&
        !(old.value isa NoBinding)
        axes(_unwrap_storage(incoming)) == axes(old.storage) ||
            _throw_binding_error(address, "child storage shape conflicts")
    end
    supplied = _binding_node_at(
        _binding_root(parent, site).storage, AbstractPPL.getoptic(site)
    )
    if !(supplied isa Union{Nothing,NoTemplate}) &&
        !(old.storage isa Union{Nothing,Missing,NoTemplate}) &&
        name in _lhs_arguments(child) &&
        !(incoming isa BoundValue)
        # A `VarNamedTuple` lists only bound fields; `_same_shape` compares those.
        (
            if supplied isa VarNamedTuple
                _same_shape(old.storage, supplied)
            else
                _matches_local_shape(old.storage, _unwrap_storage(supplied))
            end
        ) || _throw_binding_error(address, "child argument storage shape conflicts")
    end
    storage = incoming isa BoundValue ? incoming.value : old.storage
    if old.value isa NoBinding && !(name in _lhs_arguments(child)) && supplied !== nothing
        storage = supplied
    end
    storage isa Union{NoTemplate,Missing,Nothing} &&
        supplied !== nothing &&
        (storage = supplied)
    previous_constraints = filter(_effective_constraints(old, vn)) do c
        c.mode === Val(:shape) ||
            _has_bound_value(_binding_node_at(old.value, AbstractPPL.getoptic(c.owner)))
    end
    inherited = _inherited_constraints(parent, prefix, Val(name))
    for c in inherited, old_c in previous_constraints
        c.owner == old_c.owner || continue
        _same_shape(c.storage, old_c.storage) ||
            _throw_binding_error(address, "inherited template conflicts with storage shape")
        c.role === old_c.role &&
            c.mode === old_c.mode === Val(:type) &&
            typeof(c.storage) !== typeof(old_c.storage) &&
            _throw_binding_error(address, "inherited template conflicts with storage type")
    end
    constraints = (previous_constraints..., inherited...)
    for constraint in constraints
        owner = constraint.owner
        template = constraint.storage
        typed = constraint.mode === Val(:type)
        if owner == vn && (
            storage isa Union{NoTemplate,Missing,Nothing} ||
            old.value isa NoBinding && !typed
        )
            storage = template
        end
        incoming isa BoundValue || continue
        selected = _binding_node_at(incoming.value, AbstractPPL.getoptic(owner))
        selected === nothing && continue
        _same_shape(template, selected) ||
            _throw_binding_error(address, "child storage shape conflicts")
        typed && _convert_to_storage(selected, template, address)
    end
    incoming = VarNamedTuples._map_pairs_recursive!!(
        if incoming isa _InnerNode
            copy(incoming)
        else
            incoming
        end,
        vn,
    ) do pair
        location, wrapped = pair
        error_address = maybe_prefix(maybe_prefix(location, prefix), _model_prefix(parent))
        value = _apply_constraints(
            wrapped.value,
            AbstractPPL.getoptic(location),
            location,
            typeof(_binding_role(wrapped)),
            previous_constraints;
            error_address,
        )
        if name in _lhs_arguments(child) &&
            !(AbstractPPL.getoptic(location) isa AbstractPPL.Iden)
            old.storage isa Union{Missing,Nothing,NoTemplate} || _check_partial_storage(
                old.storage, AbstractPPL.getoptic(location), error_address
            )
            value = _convert_partial(
                value, old.storage, AbstractPPL.getoptic(location), error_address
            )
        end
        _rewrap_role(wrapped, value)
    end
    previous = old.value
    node = if previous isa Union{Nothing,NoBinding} || incoming isa BoundValue
        incoming
    else
        previous isa BoundValue && (previous = _split_bound_value(previous, @varname(_)))
        VarNamedTuples._merge(copy(previous), copy(incoming), Val(true))
    end
    return BindingRoot(node, storage, constraints)
end
@generated function _merge_child_table(
    table::BindingTable{<:VarNamedTuple{A}},
    incoming::VarNamedTuple{B},
    parent,
    prefix,
    child,
) where {A,B}
    names = Tuple(union(A, B))
    fields = map(names) do name
        name in B || return :(table.roots.data.$name)
        old = if name in A
            :(table.roots.data.$name)
        else
            :(BindingRoot(NoBinding(), NoTemplate(), ()))
        end
        :(_merge_child_root(
            $old, incoming.data.$name, parent, prefix, Val($(QuoteNode(name))), child
        ))
    end
    return :(BindingTable(VarNamedTuple(NamedTuple{$names}(($(fields...),)))))
end
# A child sharing its parent's root receives bindings only at the roots it may own. The
# parent's argument observations belong to its own tildes, as do bindings of the parent's
# own variables that the child takes as covariates; those roots are also returned as
# reserved. Selecting the names from types keeps the result inferred under Mooncake, which
# does not fold a run-time filter here.
@generated function _owned_bindings(
    bindings::VarNamedTuple{K,V}, parent::Model, child::Model
) where {K,V}
    names = _lhs_names(_model_metadata(child))
    arguments = _lhs_arguments(_model_metadata(parent))
    own = _own_lhs_names(_model_metadata(parent))
    inputs = map(
        unsplat_symbol,
        (
            fieldnames(fieldtype(child, :args))...,
            fieldnames(fieldtype(child, :defaults))...,
        ),
    )
    covariates = setdiff(inputs, _lhs_arguments(_model_metadata(child)))
    bound = Tuple(
        k for (k, T) in zip(K, fieldtypes(V)) if !(T <: NoBinding) && k ∉ arguments
    )
    reserved = Tuple(k for k in bound if k in covariates && k in own)
    owned = Tuple(k for k in bound if (names === nothing || k in names) && k ∉ reserved)
    return :(VarNamedTuple(bindings.data[$owned]), $reserved)
end
_with_parent_prefix(child, ::Nothing) = child
function _with_parent_prefix(child, prefix::VarName)
    return _with_prefix(child, prefix, NoTemplate(), child.values)
end
function _with_parent_prefix(child, prefix::PrefixTemplate)
    child = _with_parent_prefix(child, prefix.inner)
    return _with_prefix(child, prefix.prefix, prefix.template, child.values)
end

# Argument observations are not bindings of the tilde's left-hand side.
_has_explicit_binding(x) = x isa BoundValue{ConditionRole} || x isa BoundValue{FixRole}
function _has_explicit_binding(x::_InnerNode)
    return VarNamedTuples._mapreduce_recursive(
        p -> _has_explicit_binding(last(p)), |, x, @varname(_), false
    )
end
function tilde_assume!!(
    parent::Model,
    context::AbstractContext,
    submodel::Submodel{M,AutoPrefix},
    vn::VarName,
    template,
    vi::AbstractVarInfo,
) where {M<:Model,AutoPrefix}
    child = submodel.model
    if AutoPrefix
        AbstractPPL.getsym(vn) in
        (_argument_names(parent.args)..., _argument_names(parent.defaults)...) &&
            _throw_binding_error(vn, "submodel LHS cannot be an argument")
        vn = _concretize_prefix(vn, template; prefix=Val(true))
        _check_local_shape(parent, vn, template, ConditionRole; submodel=true)
        _check_local_shape(parent, vn, template, FixRole; submodel=true)
        site = vn
        local_prefix = maybe_prefix(_model_prefix(child), vn)
    else
        AbstractPPL.getsym(vn) in _unprefixed_roots(_model_metadata(parent)) || throw(
            ArgumentError(
                "`$vn`: an unprefixed submodel must be written as `~ to_submodel(m, false)` in the model body",
            ),
        )
        site = local_prefix = _model_prefix(child)
        template = NoTemplate()
        root = _binding_root(parent, vn)
        roots = if local_prefix === nothing
            _lhs_names(_model_metadata(child))
        else
            (AbstractPPL.getsym(local_prefix),)
        end
        root === nothing ||
            !_has_explicit_binding(root.value) ||
            roots === nothing ||
            AbstractPPL.getsym(vn) in roots ||
            _throw_binding_error(
                maybe_prefix(VarName{AbstractPPL.getsym(vn)}(), _model_prefix(parent)),
                "an unprefixed submodel's return value cannot be bound",
            )
    end
    observations = _child_removals(parent.observation_removals, local_prefix, template)
    fixes = _child_removals(parent.fixed_removals, local_prefix, template)
    table = _apply_removals(
        child.values, ConditionRole, observations, merge(child.args, child.defaults)
    )
    table = _apply_removals(table, FixRole, fixes, merge(child.args, child.defaults))
    bindings = _child_bindings(parent, local_prefix, site)
    reserved = ()
    if local_prefix === nothing
        bindings, dropped = _owned_bindings(bindings, parent, child)
        reserved = (parent.values.reserved..., dropped...)
        isempty(reserved) || for name in _own_lhs_names(_model_metadata(child))
            name in reserved && _throw_binding_error(
                maybe_prefix(VarName{name}(), _model_prefix(parent)),
                "an enclosing model and an unprefixed submodel both contain a tilde for this name",
            )
        end
    end
    table = _merge_child_table(table, bindings, parent, local_prefix, child)
    isempty(reserved) || (table = BindingTable(table.roots, reserved))
    for (name, root) in pairs(table.roots.data)
        _has_role(ArgumentRole, root.value) && _throw_binding_error(
            maybe_prefix(
                maybe_prefix(VarName{name}(), local_prefix), _model_prefix(parent)
            ),
            "submodels cannot observe through arguments",
        )
    end
    child = _reconstruct_model(
        child;
        values=table,
        observation_removals=(child.observation_removals..., observations...),
        fixed_removals=(child.fixed_removals..., fixes...),
    )
    child = if AutoPrefix
        vn, template = _apply_model_prefix(vn, template, parent)
        _with_prefix(child, vn, template, child.values)
    else
        prefix_template = _model_prefix_template(parent)
        _with_parent_prefix(
            child, prefix_template === nothing ? _model_prefix(parent) : prefix_template
        )
    end
    _check_argument_bindings(child)
    _check_argument_aliasing(child)
    child = setleafcontext(child, context)
    args, kwargs = make_evaluate_args_and_kwargs(child, vi)
    return child.f(args...; kwargs...)
end
function tilde_observe!!(prefix, prefix_template, ::Submodel, left, ::Nothing, template, vi)
    throw(ArgumentError("A submodel requires a named LHS"))
end
