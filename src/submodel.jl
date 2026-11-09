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
        return (name=relative, active=marker.active && active, matched=marker.matched)
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
    if prefix != site
        node = _binding_node_at(values, AbstractPPL.varname_to_optic(site))
        VarNamedTuples._mapreduce_recursive((a, b) -> nothing, node, site, nothing) do pair
            location, value = pair
            subsumes(prefix, location) ||
                !_has_bound_value(value) ||
                _throw_binding_error(
                    maybe_prefix(location, getprefix(parent)),
                    "address is outside child prefix $prefix",
                )
        end
    end
    return _as_namespace(
        _binding_node_at(values, AbstractPPL.varname_to_optic(prefix)),
        maybe_prefix(prefix, getprefix(parent)),
    )
end
function _inherited_constraints(parent, prefix, ::Val{name}) where {name}
    root = _binding_root(parent, prefix)
    root === nothing && return ()
    address = AbstractPPL.append_optic(prefix, AbstractPPL.Property{name}())
    return mapreduce((a, b) -> (a..., b...), root.constraints; init=()) do constraint
        (; owner, storage) = constraint
        R = constraint.role
        typed = constraint.typed === Val(true)
        if subsumes(owner, address)
            relative = AbstractPPL._unprefix_optic(
                AbstractPPL.getoptic(address), AbstractPPL.getoptic(owner)
            )
            selected = _binding_node_at(storage, relative)
            return if selected === nothing
                ()
            else
                (BindingConstraint(VarName{name}(), R, selected, typed),)
            end
        elseif subsumes(address, owner)
            return (
                BindingConstraint(AbstractPPL.unprefix(owner, prefix), R, storage, typed),
            )
        end
        return ()
    end
end
function _merge_child_root(old, incoming, parent, prefix, ::Val{name}, child) where {name}
    vn = VarName{name}()
    address = maybe_prefix(maybe_prefix(vn, prefix), getprefix(parent))
    _check_binding_address(child, vn, address)
    if incoming isa BoundValue && _is_namespace(child, vn)
        incoming = _as_namespace(incoming, address)
    end
    _check_role_overlap(old.value, incoming, address; disjoint=true)
    if incoming isa VarNamedTuples.PartialArray &&
        old.storage isa AbstractArray &&
        !(old.value isa NoBinding)
        axes(incoming.data) == axes(old.storage) ||
            _throw_binding_error(address, "child storage shape conflicts")
    end
    supplied = _binding_root(parent, prefix)
    supplied = if supplied === nothing
        NoTemplate()
    else
        _binding_node_at(
            supplied.storage,
            AbstractPPL.Property{name}() ∘ AbstractPPL.getoptic(prefix),
        )
    end
    if !(supplied isa Union{Nothing,NoTemplate}) &&
        !(old.storage isa Union{Nothing,Missing,NoTemplate}) &&
        name in _lhs_arguments(child) &&
        !(incoming isa BoundValue)
        _matches_local_shape(old.storage, _unwrap_storage(supplied)) ||
            _throw_binding_error(address, "child argument storage shape conflicts")
    end
    storage = incoming isa BoundValue ? incoming.value : old.storage
    if old.value isa NoBinding && !(name in _lhs_arguments(child)) && supplied !== nothing
        storage = supplied
    end
    storage isa Union{NoTemplate,Missing,Nothing} &&
        supplied !== nothing &&
        (storage = supplied)
    previous_constraints = filter(_effective_constraints(old, vn)) do c
        c.typed === Val(false) ||
            _has_bound_value(_binding_node_at(old.value, AbstractPPL.getoptic(c.owner)))
    end
    inherited = _inherited_constraints(parent, prefix, Val(name))
    for c in inherited, old_c in previous_constraints
        c.owner == old_c.owner || continue
        _same_shape(c.storage, old_c.storage) ||
            _throw_binding_error(address, "inherited template conflicts with storage shape")
        c.role === old_c.role &&
            c.typed === old_c.typed === Val(true) &&
            typeof(c.storage) !== typeof(old_c.storage) &&
            _throw_binding_error(address, "inherited template conflicts with storage type")
    end
    constraints = (previous_constraints..., inherited...)
    for constraint in constraints
        owner = constraint.owner
        template = constraint.storage
        typed = constraint.typed === Val(true)
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
        if incoming isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            copy(incoming)
        else
            incoming
        end,
        vn,
    ) do pair
        location, wrapped = pair
        error_address = maybe_prefix(maybe_prefix(location, prefix), getprefix(parent))
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
function tilde_assume!!(
    parent::Model,
    context::AbstractContext,
    submodel::Submodel{M},
    vn::VarName,
    template,
    vi::AbstractVarInfo,
) where {M<:Model}
    AbstractPPL.getsym(vn) in
    (_argument_names(parent.args)..., _argument_names(parent.defaults)...) &&
        _throw_binding_error(vn, "submodel LHS cannot be an argument")
    vn = _concretize_prefix(vn, template; prefix=Val(true))
    _check_local_shape(parent, vn, template, ConditionRole; submodel=true)
    _check_local_shape(parent, vn, template, FixRole; submodel=true)
    child = submodel.model
    local_prefix = maybe_prefix(getprefix(child), vn)
    context = _binding_check_child(context, parent, child, local_prefix, template)
    observations = _child_removals(parent.observation_removals, local_prefix, template)
    fixes = _child_removals(parent.fixed_removals, local_prefix, template)
    for (name, root) in pairs(child.values.roots.data)
        _has_role(ArgumentRole, root.value) && _throw_binding_error(
            maybe_prefix(maybe_prefix(VarName{name}(), local_prefix), getprefix(parent)),
            "submodels cannot observe through arguments",
        )
    end
    table = _apply_removals(
        child.values, ConditionRole, observations, merge(child.args, child.defaults)
    )
    table = _apply_removals(table, FixRole, fixes, merge(child.args, child.defaults))
    table = _merge_child_table(
        table, _child_bindings(parent, local_prefix, vn), parent, local_prefix, child
    )
    child = _reconstruct_model(
        child;
        values=table,
        observation_removals=(child.observation_removals..., observations...),
        fixed_removals=(child.fixed_removals..., fixes...),
    )
    vn, template = _apply_model_prefix(vn, template, parent)
    child = _with_prefix(child, vn, template, child.values)
    _check_argument_bindings(child)
    _check_argument_aliasing(child)
    child = setleafcontext(child, context)
    args, kwargs = make_evaluate_args_and_kwargs(child, vi)
    return child.f(args...; kwargs...)
end
function tilde_observe!!(prefix, ::Submodel, left, ::Nothing, template, vi)
    throw(ArgumentError("A submodel requires a named LHS"))
end
