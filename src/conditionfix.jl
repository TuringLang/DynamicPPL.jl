function _model_value_like(value::ModelValue{R}, x) where {R}
    return ModelValue{R}(x)
end
_inherits_binding(::ModelValue{R}) where {R} = R !== ArgumentCondition
function _inherited_model_values(values::VarNamedTuple)
    return _prune_model_bindings(
        VarNamedTuples._map_values_recursive!!(
            v -> v isa ModelValue && !_inherits_binding(v) ? NoModelBinding() : v,
            copy(values),
        ),
    )
end

_contains_missing(value) = _contains_placeholder(value, Missing, nothing)
_contains_nothing(value) = _contains_placeholder(value, Nothing, nothing)

function _contains_placeholder(value, ::Type{P}, seen) where {P}
    value isa P && return true
    if ismutabletype(typeof(value))
        seen === nothing && (seen = Base.IdSet{Any}())
        value in seen && return false
        push!(seen, value)
    end
    return _contains_placeholder_children(value, P, seen)
end
_contains_placeholder(::Union{Number,AbstractString,Symbol,Type}, ::Type, seen) = false
_contains_placeholder(::AbstractArray{<:Number}, ::Type, seen) = false

function _contains_placeholder_children(value, ::Type{P}, seen) where {P}
    return any(1:fieldcount(typeof(value))) do i
        isdefined(value, i) && _contains_placeholder(getfield(value, i), P, seen)
    end
end
function _contains_placeholder_children(value::Base.Pairs, ::Type{P}, seen) where {P}
    return _contains_placeholder(values(value), P, seen)
end
function _contains_placeholder_children(value::TransformedValue, ::Type{P}, seen) where {P}
    return _contains_placeholder(get_internal_value(value), P, seen)
end
function _contains_placeholder_children(values::AbstractArray, ::Type{P}, seen) where {P}
    return any(eachindex(values)) do i
        isassigned(values, i) && _contains_placeholder(values[i], P, seen)
    end
end
function _contains_placeholder_children(
    values::Union{Tuple,NamedTuple}, ::Type{P}, seen
) where {P}
    return any(value -> _contains_placeholder(value, P, seen), values)
end

function ModelBindingLayers(
    observations,
    fixed,
    owners::Tuple=Tuple(keys(fixed)),
    observation_removals::Tuple=(),
    fixed_removals::Tuple=(),
    templates::Tuple=(),
)
    return ModelBindingLayers(
        observations,
        fixed,
        _overlay_model_values(observations, fixed, owners),
        owners,
        observation_removals,
        fixed_removals,
        templates,
    )
end
_model_values(values::ModelBindingLayers) = values.values
_model_value_varname(::ModelBindingLayers, vn, prefix) = maybe_prefix(vn, prefix)
_observation_values(values::ModelBindingLayers) = values.observations
_fixed_values(values::ModelBindingLayers) = values.fixed
_fixed_owners(values::ModelBindingLayers) = values.owners
_fixed_owners(values) = ()
_observation_values(values) = _remove_model_values(Fix, _model_values(values))
_fixed_values(values) = _remove_model_values(Condition, _model_values(values))

LocalModelValues(values) = LocalModelValues(values, _fixed_owners(values))

_model_values(values::UnprefixedArgumentValues) = values.values
_model_value_varname(::UnprefixedArgumentValues, vn, prefix) = vn

_model_values(values::VarNamedTuple) = values
_model_values(values::LocalModelValues) = _model_values(values.values)
_observation_values(values::LocalModelValues) = _observation_values(values.values)
_fixed_values(values::LocalModelValues) = _fixed_values(values.values)
_fixed_owners(values::LocalModelValues) = values.owners
_model_value_varname(::VarNamedTuple, vn, prefix) = maybe_prefix(vn, prefix)
_model_value_varname(::LocalModelValues, vn, prefix) = vn

@generated function _overlay_model_values(
    observations::VarNamedTuple{O}, fixed::VarNamedTuple{F}, owners, prefix=nothing
) where {O,F}
    names = Tuple(union(O, F))
    fields = map(names) do name
        if name in O && name in F
            quote
                _check_model_binding(
                    observations.data.$name,
                    fixed.data.$name,
                    maybe_prefix($(VarName{name}()), prefix);
                    check_bounds=false,
                )
                _overlay_model_node(
                    observations.data.$name,
                    fixed.data.$name,
                    owners,
                    maybe_prefix($(VarName{name}()), prefix),
                )
            end
        elseif name in F
            :(fixed.data.$name)
        else
            :(observations.data.$name)
        end
    end
    return :(VarNamedTuple(NamedTuple{$names}(($(fields...),))))
end
_overlay_model_node(previous, fixed, owners, vn) = _merge_model_node(previous, fixed)
function _overlay_model_node(previous::VarNamedTuple, fixed::VarNamedTuple, owners, vn)
    return _overlay_model_values(previous, fixed, owners, vn)
end
function _overlay_model_node(previous, fixed::VarNamedTuples.PartialArray, owners, vn)
    previous isa NoModelBinding && return _copy_model_node(fixed)
    owns_shape = _inherits_shape(fixed.mask) || any(owner -> subsumes(owner, vn), owners)
    owns_shape &&
        eltype(fixed) <: ModelValue &&
        _has_complete_model_data(fixed) &&
        return _copy_model_node(fixed)
    if owns_shape
        previous = _read_shadowed_tuple(previous, fixed)
    end
    previous isa ModelValue && (previous = _expand_model_binding(previous))
    if previous isa VarNamedTuples.PartialArray &&
        !(fixed.data isa VarNamedTuples.GrowableArray) &&
        axes(previous) != axes(fixed) &&
        (
            owns_shape || any(
                i -> fixed.mask[i] && !checkbounds(Bool, previous.data, i),
                CartesianIndices(fixed.mask),
            )
        )
        # Extending one layer must not discard observations on another axis.
        merged_axes = if owns_shape
            axes(fixed.data)
        else
            ntuple(max(ndims(previous.data), ndims(fixed.data))) do d
                lo = min(first(axes(previous.data, d)), first(axes(fixed.data, d)))
                hi = max(last(axes(previous.data, d)), last(axes(fixed.data, d)))
                lo == 1 ? Base.OneTo(hi) : (lo:hi)
            end
        end
        previous = _resize_model_binding(previous, merged_axes)
    end
    previous = _inherit_model_array_owner(previous, fixed)
    return _fold_model_indices(
        _copy_model_node(previous), fixed
    ) do result, update, optic, template
        address = AbstractPPL.append_optic(vn, optic)
        _check_binding_template_bounds(result, optic, address)
        child = _model_argument_binding(result, optic)
        value = if child === nothing
            update
        else
            _overlay_model_node(child, update, owners, address)
        end
        VarNamedTuples._setindex_optic!!(
            result, value, optic, template, VarNamedTuples.AllowAll()
        )
    end
end
function _overlay_model_node(
    previous::VarNamedTuple, fixed::ModelValue{<:Any,<:NamedTuple}, owners, vn
)
    if any(name -> !hasproperty(fixed.value, name), keys(previous.data))
        return _overlay_model_values(previous, _submodel_namespace(fixed), owners, vn)
    end
    return fixed
end
_inherits_shape(::Any) = false
_inherits_shape(value::ModelValue) = _inherits_binding(value)
_inherits_shape(::ModelValueTree{T,V,S}) where {T,V,S} = S
function _overlay_model_node(previous, fixed::ModelValueTree, owners, vn)
    owns_shape = _inherits_shape(fixed) || any(owner -> subsumes(owner, vn), owners)
    inherited_shape = _inherits_shape(fixed) || (!owns_shape && _inherits_shape(previous))
    template = if !owns_shape && previous isa ModelValue
        previous.value
    elseif !owns_shape && previous isa ModelValueTree
        previous.template
    else
        fixed.template
    end
    if previous isa ModelValue
        # Read the shadowed value through the replacement's fields without rebuilding it.
        previous = VarNamedTuple(
            _merge_model_fields(
                previous, _empty_model_tree(fixed), Val(propertynames(template))
            ),
        )
    end
    previous = previous isa ModelValueTree ? previous.values : previous
    values = if previous isa VarNamedTuple
        if owns_shape
            names = filter(name -> hasproperty(template, name), keys(previous.data))
            previous = VarNamedTuple(NamedTuple{names}(previous.data))
        end
        _overlay_model_values(previous, fixed.values, owners, vn)
    else
        fixed.values
    end
    if !all(name -> hasproperty(template, name), keys(values.data))
        template = fixed.template
    end
    return ModelValueTree(template, values, Val(inherited_shape))
end

function Base.:(==)(a::ModelValueTree, b::ModelValueTree)
    return _inherits_shape(a) == _inherits_shape(b) &&
           a.template == b.template &&
           a.values == b.values
end
function Base.isequal(a::ModelValueTree, b::ModelValueTree)
    return _inherits_shape(a) == _inherits_shape(b) &&
           isequal(a.template, b.template) &&
           isequal(a.values, b.values)
end
function Base.hash(value::ModelValueTree, h::UInt)
    return hash(
        _inherits_shape(value),
        hash(value.template, hash(value.values, hash(:ModelValueTree, h))),
    )
end

function VarNamedTuples._getindex_optic(
    value::ModelValue{R}, optic::AbstractPPL.AbstractOptic, vn
) where {R}
    return _model_value_like(value, VarNamedTuples._getindex_optic(value.value, optic, vn))
end
function VarNamedTuples._getindex_optic(
    value::ModelValue{R}, ::AbstractPPL.Iden, vn
) where {R}
    return value
end
function VarNamedTuples._haskey_optic(
    value::ModelValue{R}, optic::AbstractPPL.AbstractOptic
) where {R<:Union{Condition,ArgumentCondition,Fix}}
    # Keep the wrapper at every step so nested tuples use binding-aware bounds checks.
    head = AbstractPPL.ohead(optic)
    VarNamedTuples._haskey_optic(value.value, head) || return false
    child = VarNamedTuples._getindex_optic(value.value, head, @varname(_))
    return VarNamedTuples._haskey_optic(_model_value_like(value, child), optic.child)
end
function VarNamedTuples._haskey_optic(
    value::ModelValue{R,<:AbstractArray}, optic::AbstractPPL.Index
) where {R<:Union{Condition,ArgumentCondition,Fix}}
    array = value.value
    optic = AbstractPPL.concretize_top_level(optic, array)
    checkbounds(Bool, array, optic.ix...; optic.kw...) || return false
    if !VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...)
        isassigned(array, optic.ix...; optic.kw...) || return false
    end
    child = VarNamedTuples.index_template(array, optic)
    return VarNamedTuples._haskey_optic(_model_value_like(value, child), optic.child)
end
function VarNamedTuples._getindex_optic(
    value::ModelValue{R,<:AbstractArray}, optic::AbstractPPL.Index, vn
) where {R}
    optic = AbstractPPL.concretize_top_level(optic, value.value)
    child = VarNamedTuples.index_template(value.value, optic)
    return VarNamedTuples._getindex_optic(_model_value_like(value, child), optic.child, vn)
end
VarNamedTuples._haskey_optic(::ModelValue, ::AbstractPPL.Iden) = true
function VarNamedTuples._haskey_optic(
    value::ModelValue{R,<:Tuple}, optic::AbstractPPL.Index
) where {R<:Union{Condition,ArgumentCondition,Fix}}
    optic = AbstractPPL.concretize_top_level(optic, value.value)
    isempty(optic.kw) && checkbounds(Bool, Base.OneTo(length(value.value)), optic.ix...) ||
        return false
    return VarNamedTuples._haskey_optic(
        _model_value_like(value, getindex(value.value, optic.ix...)), optic.child
    )
end

_model_role(::ModelValue{R}) where {R} = R()
_model_role(::ModelValue{ArgumentCondition}) = Condition()
_model_role(value::ModelValue, vn) = _model_role(value)
_matches_model_role(::Type{R}, value) where {R} = value isa ModelValue{R}
_matches_model_role(::Type{Condition}, ::ModelValue{ArgumentCondition}) = true
_model_role(::Nothing, ::VarName) = nothing
_model_role(::NoModelBinding, ::VarName) = nothing
function _model_role(value::VarNamedTuples.ArrayLikeBlock, vn::VarName)
    return _model_role(value.block, vn)
end
function _model_role(tree::ModelValueTree, vn::VarName)
    return if VarNamedTuples._haskey_optic(tree, AbstractPPL.Iden())
        values = tree.values.data
        _model_role(values, vn)
    else
        _model_role(tree.values, vn)
    end
end
_model_role(::VarNamedTuple, vn::VarName) = _partial_binding_error(vn)
function _partial_binding_error(vn)
    return throw(
        ArgumentError(
            "LHS variable `$vn` has both bound and unbound subvariables, such as observed " *
            "and `missing` elements; write element-wise tildes (`x[i] ~ ...`), or bind or " *
            "decondition all of `$vn`.",
        ),
    )
end
function _untemplated_parts_error(vn)
    return throw(
        ArgumentError(
            "LHS variable `$vn` is bound by parts, which cannot form a multivariate value " *
            "in storage without a template. To bind all of `$vn`, bind its whole value in " *
            "one binding, or supply storage with a model argument or a binding schema; to " *
            "leave some elements unbound, write element-wise tildes (`x[i] ~ ...`).",
        ),
    )
end
function _model_role(values::VarNamedTuples.PartialArray, vn::VarName)
    !(values.data isa VarNamedTuples.GrowableArray) &&
        all(values.mask) &&
        return _model_role(values.data, vn)
    any(values.mask) || return nothing
    role = _model_role(values.data[values.mask], vn)
    role === nothing && return nothing
    # Growable storage knows only the supplied indices, so without a hole it cannot show
    # that the LHS variable is partly unbound.
    values.data isa VarNamedTuples.GrowableArray &&
        all(values.mask) &&
        _untemplated_parts_error(vn)
    return _partial_binding_error(vn)
end
function _model_role(values::Union{AbstractArray,Tuple,NamedTuple}, vn::VarName)
    isempty(values) && throw(ArgumentError("Cannot determine the role of empty `$vn`"))
    role = nothing
    unbound = false
    for value in values
        next_role = _model_role(value, vn)
        if next_role === nothing
            unbound = true
            continue
        end
        role === nothing ||
            typeof(next_role) === typeof(role) ||
            throw(
                ArgumentError(
                    "Cannot condition and fix different parts of the same LHS variable `$vn`",
                ),
            )
        role = next_role
    end
    unbound && role !== nothing && _partial_binding_error(vn)
    return role
end

_model_role_at(values, ::AbstractPPL.Iden, vn) = _model_role(values, vn)
function _model_role_at(values::VarNamedTuple, optic::AbstractPPL.Property{S}, vn) where {S}
    return if haskey(values.data, S)
        _model_role_at(values.data[S], optic.child, vn)
    else
        nothing
    end
end
@inline function _model_role_at(value::ModelValue, optic::AbstractPPL.AbstractOptic, vn)
    VarNamedTuples._haskey_optic(value, optic) && return _model_role(value)
    value isa ModelValue{Fix} &&
        _fixed_shape_error(vn, "coverage with a static size and shape.")
    return nothing
end
@inline function _model_role_at(value::ModelValue, ::AbstractPPL.Iden, vn)
    return _model_role(value)
end
function _model_role_at(values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index, vn)
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    if !VarNamedTuples._matches_ndims(values, optic.ix)
        any(values.mask) && _growable_ndims_error(values, optic, vn)
        return nothing
    end
    if !checkbounds(Bool, values.data, optic.ix...; optic.kw...)
        # An out-of-bounds LHS slice may still overlap supplied indices.
        for indices in Iterators.product(Base.to_indices(values.data, optic.ix)...)
            checkbounds(Bool, values.data, indices...; optic.kw...) || continue
            getindex(values.mask, indices...; optic.kw...) || continue
            value = getindex(values.data, indices...; optic.kw...)
            _model_role_at(value, optic.child, vn) === nothing || _partial_binding_error(vn)
        end
        return nothing
    end
    if VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
        selected = _model_argument_binding(values, AbstractPPL.Index(optic.ix, optic.kw))
        return _model_role_at(selected, optic.child, vn)
    end
    haskey(values, optic.ix...; optic.kw...) || return nothing
    return _model_role_at(getindex(values.data, optic.ix...; optic.kw...), optic.child, vn)
end
@noinline function _growable_ndims_error(values, optic, vn)
    owner = _storage_address(vn, optic)
    stored = first(i for i in CartesianIndices(values.mask) if values.mask[i])
    bound = AbstractPPL.append_optic(owner, AbstractPPL.Index(Tuple(stored), (;)))
    indices(n) = n == 1 ? "1 index" : "$n indices"
    n = VarNamedTuples._index_ndims(optic.ix...)
    distinction = if n == 1 || ndims(values) == 1
        "; linear and Cartesian indices are distinct"
    else
        ""
    end
    throw(
        ArgumentError(
            "`$owner` has growable bindings with $(indices(ndims(values))) (e.g. `$bound`), " *
            "but tilde `$vn` uses $(indices(n))$distinction. " *
            "Bind addresses with the tilde's index count, or supply storage for `$owner` " *
            "with an argument or a binding schema (`@of`). In the schema, use the " *
            "binding call's local LHS top symbol and namespace path, omitting any " *
            "explicit model prefix.",
        ),
    )
end
function _get_model_role(model, vn, template=NoTemplate())
    root = _model_value_varname(
        model.values, VarName{AbstractPPL.getsym(vn)}(), _model_prefix(model)
    )
    if template isa NamedTuple
        binding = _model_argument_binding(
            _model_values(model.values), AbstractPPL.varname_to_optic(root)
        )
        if binding isa VarNamedTuples.PartialArray
            _fold_model_indices(nothing, binding) do _, _, optic, _
                _check_namedtuple_index(
                    template,
                    optic,
                    AbstractPPL.varname_to_optic(_binding_display_name(model, root)),
                )
            end
        end
    end
    vn = _model_value_varname(model.values, vn, _model_prefix(model))
    return _model_role_at(
        _model_values(model.values),
        AbstractPPL.varname_to_optic(vn),
        _binding_display_name(model, vn),
    )
end
function _get_model_binding(model, vn)
    vn = _model_value_varname(model.values, vn, _model_prefix(model))
    return _model_argument_binding(
        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
    )
end
function _get_argument_role(model, vn, argument)
    binding = _get_model_binding(model, argument)
    # A whole argument keeps its role when body computations change its shape or fields.
    return binding isa ModelValue ? _model_role(binding, vn) : _get_model_role(model, vn)
end
function _get_model_data(model, vn)
    vn = _model_value_varname(model.values, vn, _model_prefix(model))
    binding = _model_argument_binding(
        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
    )
    return if binding === nothing
        _model_data(VarNamedTuples._getindex_optic(_model_values(model.values), vn))
    else
        _model_data(binding)
    end
end
function _get_model_data(model, vn, argument, local_value)
    binding = _get_model_binding(model, argument)
    _check_fixed_shape(
        binding,
        local_value,
        AbstractPPL.getoptic(vn),
        vn,
        _fixed_owners(model.values),
        _model_value_varname(model.values, argument, _model_prefix(model)),
    )
    return _get_model_data(model, vn)
end

# Storage for a partial binding is not itself a fixed shape owner.
function _check_fixed_shape(binding, local_value, optic, vn, owners=(), address=vn)
    if optic isa AbstractPPL.Iden || any(owner -> subsumes(owner, address), owners)
        template = if binding isa VarNamedTuples.PartialArray
            binding.data
        elseif binding isa ModelValueTree
            binding.template
        else
            nothing
        end
        template === nothing || _check_fixed_shape(
            ModelValue{Fix}(template), local_value, AbstractPPL.Iden(), vn
        )
    end
    return _check_fixed_shape_child(binding, local_value, optic, vn, owners, address)
end
function _check_fixed_shape(
    binding::ModelValue{Fix}, local_value, optic, vn, owners=(), address=vn
)
    value = binding.value
    if value isa Tuple
        local_value isa Tuple && length(value) == length(local_value) || _fixed_shape_error(
            vn, "a static size and shape; the model body changed its argument's length."
        )
    elseif value isa Union{NamedTuple,Base.Pairs}
        local_value isa Union{NamedTuple,Base.Pairs} &&
            length(value) == length(local_value) &&
            all(name -> haskey(local_value, name), keys(value)) || _fixed_shape_error(
            vn, "a static size and shape; the model body changed its argument's fields."
        )
    elseif value isa AbstractArray || local_value isa AbstractArray
        value isa AbstractArray &&
            local_value isa AbstractArray &&
            axes(value) == axes(local_value) || _fixed_shape_error(
            vn, "a static size and shape; the model body changed its argument's shape."
        )
    end
    return _check_fixed_shape_child(binding, local_value, optic, vn, owners, address)
end
# Keep error construction out of the hot path so the shape checks can inline.
@noinline function _fixed_shape_error(vn, message)
    return throw(ArgumentError("Fixed LHS variable `$vn` requires $message"))
end
function _check_fixed_shape_child(
    binding, local_value, ::AbstractPPL.Iden, vn, owners, address
)
    return nothing
end
function _check_fixed_shape_child(
    binding, local_value, optic::AbstractPPL.Property{S}, vn, owners, address
) where {S}
    child = _model_argument_binding(binding, AbstractPPL.Property{S}())
    child !== nothing && hasproperty(local_value, S) ||
        _fixed_shape_error(vn, "coverage with a static size and shape.")
    return _check_fixed_shape(
        child,
        getproperty(local_value, S),
        optic.child,
        vn,
        owners,
        AbstractPPL.append_optic(address, AbstractPPL.Property{S}()),
    )
end
function _check_fixed_shape_child(
    binding, local_value, optic::AbstractPPL.Index, vn, owners, address
)
    optic = AbstractPPL.concretize_top_level(optic, local_value)
    child = _model_argument_binding(binding, AbstractPPL.Index(optic.ix, optic.kw))
    child === nothing && _fixed_shape_error(vn, "coverage with a static size and shape.")
    selected = if VarNamedTuples._is_multiindex(local_value, optic.ix...; optic.kw...)
        Base.maybeview(local_value, optic.ix...; optic.kw...)
    else
        getindex(local_value, optic.ix...; optic.kw...)
    end
    return _check_fixed_shape(
        child,
        selected,
        optic.child,
        vn,
        owners,
        AbstractPPL.append_optic(address, AbstractPPL.Index(optic.ix, optic.kw)),
    )
end

function _tag_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return map_pairs!!(pair -> ModelValue{R}(pair.second), copy(values))
end

# The mask records inherited whole-binding shape ownership.
struct ModelBindingArray{T,N,A<:AbstractArray{T,N},V<:AbstractArray} <: AbstractArray{T,N}
    data::A
    template::V
end
_inherits_shape(::ModelBindingArray) = true
Base.size(value::ModelBindingArray) = size(value.data)
Base.axes(value::ModelBindingArray) = axes(value.data)
Base.getindex(value::ModelBindingArray, indices...) = getindex(value.data, indices...)
function Base.isassigned(value::ModelBindingArray, indices::Vararg{Int})
    return isassigned(value.data, indices...)
end
function Base.setindex!(value::ModelBindingArray, child, indices...)
    setindex!(value.data, child, indices...)
    return value
end
function Base.copy(value::ModelBindingArray)
    return ModelBindingArray(copy(value.data), value.template)
end

function _expand_model_binding(previous::ModelValue{R,<:AbstractArray}) where {R}
    value = previous.value
    T = if isconcretetype(eltype(value))
        ModelValue{R,eltype(value)}
    else
        ModelValue{R}
    end
    data = similar(value, T)
    mask = similar(value, Bool)
    for i in eachindex(value)
        mask[i] = isassigned(value, i)
        if mask[i]
            data[i] = _model_value_like(previous, value[i])
        end
    end
    # The mask carries recursive ownership without hiding the data's array type.
    _inherits_binding(previous) && (mask = ModelBindingArray(mask, value))
    return VarNamedTuples.PartialArray(data, mask)
end
function _unsupported_partial_binding(value, address, owner; operation="bind")
    kind = if value isa Tuple
        "tuple"
    elseif value isa AbstractArray
        "array"
    elseif value isa AbstractDict
        "dictionary"
    else
        "struct"
    end
    location = "$kind owner `$owner` at `$address`"
    message = if operation == "remove"
        "Cannot remove `$address`: cannot partially edit $location through container type $(typeof(value)); remove the whole value `$owner` instead."
    else
        "Cannot partially bind $location through container type $(typeof(value)); bind or decondition the whole value `$owner` instead."
    end
    value isa AbstractArray &&
        (message *= " Or use collect(v) if losing axes or metadata is acceptable.")
    value isa AbstractDict && (message *= " Or use a NamedTuple/array argument.")
    throw(ArgumentError(message))
end
function _expand_model_binding(previous::ModelValue{R,<:NamedTuple}) where {R}
    fields = previous.value
    return ModelValueTree(
        previous.value,
        _tag_model_values(R, VarNamedTuple(fields)),
        Val(_inherits_binding(previous)),
    )
end
function VarNamedTuples._setindex_optic!!(
    previous::ModelValue{R,<:AbstractArray},
    value,
    optic::AbstractPPL.Index,
    template,
    permissions,
) where {R}
    return VarNamedTuples._setindex_optic!!(
        _expand_model_binding(previous), value, optic, template, permissions
    )
end
function VarNamedTuples._setindex_optic!!(
    previous::ModelValue{R}, value, optic::AbstractPPL.Property{S}, template, permissions
) where {R,S}
    hasproperty(previous.value, S) || throw(
        ArgumentError(
            "Cannot override nonexistent property `$S` of $(typeof(previous.value))"
        ),
    )
    expanded = _expand_model_binding(previous)
    return VarNamedTuples._setindex_optic!!(expanded, value, optic, template, permissions)
end

function VarNamedTuples._getindex_optic(
    tree::ModelValueTree, optic::AbstractPPL.Property, vn
)
    return VarNamedTuples._getindex_optic(tree.values, optic, vn)
end
function VarNamedTuples._haskey_optic(tree::ModelValueTree, optic::AbstractPPL.Property)
    return VarNamedTuples._haskey_optic(tree.values, optic)
end
function VarNamedTuples._haskey_optic(tree::ModelValueTree, ::AbstractPPL.Iden)
    return all(propertynames(tree.template)) do name
        haskey(tree.values.data, name) &&
            VarNamedTuples._haskey_optic(tree.values.data[name], AbstractPPL.Iden())
    end
end

function VarNamedTuples._setindex_optic!!(
    tree::ModelValueTree, value, optic::AbstractPPL.Property, template, permissions
)
    template = template isa ModelValueTree ? template.values : tree.template
    values = VarNamedTuples._setindex_optic!!(
        copy(tree.values), value, optic, template, permissions
    )
    return ModelValueTree(tree, values)
end
function VarNamedTuples._mapreduce_recursive(
    f, op, tree::ModelValueTree{T,<:VarNamedTuple}, vn, init
) where {T}
    return VarNamedTuples._mapreduce_recursive(f, op, tree.values, vn, init)
end
function VarNamedTuples._map_values_recursive!!(f, tree::ModelValueTree)
    return ModelValueTree(tree, map_values!!(f, copy(tree.values)))
end
function VarNamedTuples._map_pairs_recursive!!(f, tree::ModelValueTree, vn)
    return ModelValueTree(
        tree, VarNamedTuples._map_pairs_recursive!!(f, copy(tree.values), vn)
    )
end

_empty_model_tree(tree::ModelValueTree) = ModelValueTree(tree, VarNamedTuple())
function VarNamedTuples.make_leaf(
    value,
    optic::AbstractPPL.Index,
    template::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}},
) where {T,N,D<:AbstractArray{T,N}}
    leaf = invoke(
        VarNamedTuples.make_leaf,
        Tuple{Any,AbstractPPL.Index,VarNamedTuples.PartialArray},
        value,
        optic,
        template,
    )
    return VarNamedTuples.PartialArray(
        leaf.data, ModelBindingArray(leaf.mask, template.mask.template)
    )
end
function VarNamedTuples.make_leaf(value, optic::AbstractPPL.Index, template::ModelValue)
    return VarNamedTuples.make_leaf(value, optic, _expand_model_binding(template))
end
function VarNamedTuples.make_leaf(value, optic::AbstractPPL.Property, template::ModelValue)
    return VarNamedTuples.make_leaf(value, optic, _expand_model_binding(template))
end
function VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Property, template::ModelValueTree
)
    return VarNamedTuples._setindex_optic!!(
        _empty_model_tree(template), value, optic, template, VarNamedTuples.AllowAll()
    )
end
function (::VarNamedTuples.SharedGetProperty{S})(value::ModelValue) where {S}
    return VarNamedTuples.SharedGetProperty{S}()(value.value)
end
function (::VarNamedTuples.SharedGetProperty{S})(tree::ModelValueTree) where {S}
    return VarNamedTuples.SharedGetProperty{S}()(tree.template)
end
function _model_role_at(tree::ModelValueTree, optic::AbstractPPL.Property, vn)
    return _model_role_at(tree.values, optic, vn)
end
# Bindings stored by field cannot be read by index, nor bindings stored by index by field.
function _model_role_at(::Union{VarNamedTuple,ModelValueTree}, ::AbstractPPL.Index, vn)
    return _binding_kind_error(vn, "an index", "fields")
end
function _model_role_at(::VarNamedTuples.PartialArray, ::AbstractPPL.Property, vn)
    return _binding_kind_error(vn, "a field", "indices")
end
@noinline function _binding_kind_error(vn, read, stored)
    message = "Cannot read LHS variable `$vn`: its tilde uses $read, but its binding is stored by $stored; bind it with the address the tilde uses, or bind the whole value."
    throw(ArgumentError(message))
end
function _model_argument_binding(tree::ModelValueTree, optic::AbstractPPL.Property)
    return _model_argument_binding(tree.values, optic)
end
# Keyword splats are Base.Pairs in the body, but their binding tables use fields.
function _model_argument_binding(
    values::Union{VarNamedTuple,ModelValueTree{<:NamedTuple}},
    optic::AbstractPPL.Index{Tuple{Symbol},NamedTuple{(),Tuple{}}},
)
    return _model_argument_binding(
        values, AbstractPPL.Property{only(optic.ix)}(optic.child)
    )
end
function _model_role_at(
    values::Union{VarNamedTuple,ModelValueTree{<:NamedTuple}},
    optic::AbstractPPL.Index{Tuple{Symbol},NamedTuple{(),Tuple{}}},
    vn,
)
    return _model_role_at(values, AbstractPPL.Property{only(optic.ix)}(optic.child), vn)
end

# Partial edits rebuild only explicitly supported storage families. Package extensions
# admit metadata wrappers only when their backing storage is an Array.
_partial_binding_array(::AbstractArray) = false
_partial_binding_array(::Array) = true
_partial_binding_selector(template, optic) = false

# Binding preparation validates before expansion, while the full address is available.
# The compiler's evaluation-time NamedTuple check does not validate containers.
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

@generated function _merge_model_values(
    previous::VarNamedTuple{P}, updates::VarNamedTuple{U}, prefix=nothing
) where {P,U}
    names = Tuple(union(P, U))
    fields = map(names) do name
        if name in P && name in U
            quote
                _check_model_binding(
                    previous.data.$name,
                    updates.data.$name,
                    maybe_prefix($(VarName{name}()), prefix),
                )
                _merge_model_node(previous.data.$name, updates.data.$name)
            end
        elseif name in U
            :(_copy_model_node(updates.data.$name))
        else
            :(previous.data.$name)
        end
    end
    return :(VarNamedTuple(NamedTuple{$names}(($(fields...),))))
end
function _incompatible_partial_binding(value, vn)
    throw(
        ArgumentError(
            "Cannot bind parts below `$vn` with value of type $(typeof(value)). " *
            "For other bindings, use `decondition(model, @varname($vn))` first.",
        ),
    )
end

_check_model_binding(previous, updates, vn; check_bounds=true) = nothing
function _check_model_binding(
    previous,
    updates::Union{VarNamedTuple,VarNamedTuples.PartialArray},
    vn;
    check_bounds=true,
)
    _fold_model_indices(nothing, updates) do _, update, optic, _
        _check_partial_binding(previous, optic, AbstractPPL.varname_to_optic(vn))
        if previous isa ModelValue
            compatible = if updates isa VarNamedTuples.PartialArray
                previous.value isa AbstractArray
            else
                previous.value isa NamedTuple ||
                    all(name -> hasproperty(previous.value, name), keys(updates.data))
            end
            compatible || _incompatible_partial_binding(previous.value, vn)
        end
        child = _model_argument_binding(previous, optic)
        if check_bounds &&
            child === nothing &&
            (
                (
                    previous isa ModelValue &&
                    previous.value isa AbstractArray &&
                    !VarNamedTuples._haskey_optic(previous.value, optic)
                ) || (
                    previous isa VarNamedTuples.PartialArray &&
                    optic isa AbstractPPL.Index &&
                    !(previous.data isa VarNamedTuples.GrowableArray) &&
                    !checkbounds(Bool, previous.data, optic.ix...; optic.kw...)
                )
            )
            address = AbstractPPL.append_optic(vn, optic)
            throw(
                ArgumentError(
                    "Cannot bind `$address`: index is outside the binding at `$vn`"
                ),
            )
        end
        child === nothing || _check_model_binding(
            child, update, AbstractPPL.append_optic(vn, optic); check_bounds
        )
        return nothing
    end
    return nothing
end

_copy_model_node(value) = value
_copy_model_node(value::Union{VarNamedTuple,VarNamedTuples.PartialArray}) = copy(value)
function _copy_model_node(value::ModelValueTree)
    values = copy(value.values)
    return ModelValueTree(value, values)
end
_merge_model_node(previous, updates) = updates
_merge_model_node(previous, ::NoModelBinding) = previous
function _merge_model_node(previous, updates::VarNamedTuple)
    previous isa NoModelBinding && return copy(updates)
    if (previous isa ModelValue && previous.value isa NamedTuple) ||
        (previous isa ModelValueTree && previous.template isa NamedTuple)
        template = previous isa ModelValue ? previous.value : previous.template
        if !all(name -> hasproperty(template, name), keys(updates.data))
            # Named tuples can also supply extensible submodel namespaces.
            return _merge_model_values(_submodel_namespace(previous), updates)
        end
    end
    previous isa ModelValue &&
        return _merge_model_node(_expand_model_binding(previous), updates)
    previous isa VarNamedTuple && return _merge_model_values(previous, updates)
    if previous isa ModelValueTree
        return ModelValueTree(previous, _merge_model_values(previous.values, updates))
    end
    return _merge_model_indices(previous, updates)
end
function _merge_model_node(previous, updates::VarNamedTuples.PartialArray)
    return if previous isa NoModelBinding
        copy(updates)
    else
        _merge_model_indices(previous, updates)
    end
end
# A whole binding keeps its extent after partial edits, both within a layer
# and across a child boundary. Partial masks alone never establish an owner.
function _resize_model_binding(previous::VarNamedTuples.PartialArray, axes)
    data = similar(previous.data, eltype(previous), axes)
    previous.data isa VarNamedTuples.GrowableArray &&
        (data = VarNamedTuples.GrowableArray(data))
    mask = fill!(similar(data, Bool), false)
    for i in CartesianIndices(data)
        if checkbounds(Bool, previous.data, i) && previous.mask[i]
            data[i] = previous.data[i]
            mask[i] = true
        end
    end
    return VarNamedTuples.PartialArray(data, mask)
end
function _inherit_model_array_owner(previous, updates)
    previous isa VarNamedTuples.PartialArray || return previous
    data, mask = previous.data, previous.mask
    if _inherits_shape(updates.mask)
        mask = mask isa ModelBindingArray ? mask.data : mask
        mask = ModelBindingArray(mask, updates.mask.template)
    end
    return VarNamedTuples.PartialArray(data, mask)
end
# A replacement owns its array storage; a shadowed tuple only supplies observations.
_read_shadowed_tuple(previous, replacement) = previous
function _read_shadowed_tuple(previous::ModelValue{<:Any,<:Tuple}, replacement)
    data = map(CartesianIndices(replacement.mask)) do i
        _previous_model_child(previous, AbstractPPL.Index(Tuple(i), (;)))
    end
    return VarNamedTuples.PartialArray(data, map(x -> !(x isa NoModelBinding), data))
end
function _merge_model_node(
    previous, updates::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}}
) where {T,N,D<:AbstractArray{T,N}}
    previous isa NoModelBinding && return copy(updates)
    previous = _read_shadowed_tuple(previous, updates)
    previous isa ModelValue && (previous = _expand_model_binding(previous))
    if previous isa VarNamedTuples.PartialArray && axes(previous) != axes(updates)
        previous = _resize_model_binding(previous, axes(updates))
    end
    return _merge_model_indices(_inherit_model_array_owner(previous, updates), updates)
end
function _check_model_binding(
    previous,
    updates::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}},
    vn;
    check_bounds=true,
) where {T,N,D<:AbstractArray{T,N}}
    # Complete replacements read no shadowed children. Incomplete array replacements
    # can also read tuple observations through their own storage, including in children.
    !check_bounds &&
        eltype(updates) <: ModelValue &&
        _has_complete_model_data(updates) &&
        return nothing
    previous isa ModelValue && previous.value isa Tuple && return nothing
    return invoke(
        _check_model_binding,
        Tuple{Any,Union{VarNamedTuple,VarNamedTuples.PartialArray},Any},
        previous,
        updates,
        vn;
        check_bounds=false,
    )
end
function _prepare_argument_fields(
    template, bindings::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}}, vn
) where {T,N,D<:AbstractArray{T,N}}
    return invoke(
        _prepare_argument_fields,
        Tuple{Any,Union{VarNamedTuple,VarNamedTuples.PartialArray},Any},
        bindings.mask.template,
        bindings,
        vn,
    )
end

function _merge_model_index(previous, update, optic, template)
    child = _model_argument_binding(previous, optic)
    value = child === nothing ? _copy_model_node(update) : _merge_model_node(child, update)
    return VarNamedTuples._setindex_optic!!(
        previous, value, optic, template, VarNamedTuples.AllowAll()
    )
end
function _merge_model_indices(previous, updates)
    return _fold_model_indices(_merge_model_index, _copy_model_node(previous), updates)
end
function _fold_model_indices(f, result, updates::VarNamedTuples.PartialArray)
    mask =
        if eltype(updates) <: VarNamedTuples.ArrayLikeBlock ||
            VarNamedTuples.ArrayLikeBlock <: eltype(updates)
            copy(updates.mask)
        else
            updates.mask
        end
    for i in CartesianIndices(mask)
        mask[i] || continue
        update = updates.data[i]
        optic = if update isa VarNamedTuples.ArrayLikeBlock
            mask[update.ix..., update.kw...] .= false
            AbstractPPL.Index(update.ix, update.kw)
        else
            AbstractPPL.Index(Tuple(i), (;))
        end
        value = update isa VarNamedTuples.ArrayLikeBlock ? update.block : update
        result = f(result, value, optic, updates)
    end
    return result
end
@generated function _fold_model_indices(
    f, result, updates::VarNamedTuple{names}
) where {names}
    exs = map(names) do name
        :(
            result = f(
                result,
                updates.data.$name,
                AbstractPPL.Property{$(QuoteNode(name))}(),
                updates,
            )
        )
    end
    return quote
        $(exs...)
        result
    end
end
function _previous_model_child(previous, optic)
    value = _model_argument_binding(previous, optic)
    return value === nothing ? NoModelBinding() : value
end
function _merge_model_node(previous, updates::ModelValueTree)
    fields = _merge_model_fields(previous, updates, Val(propertynames(updates.template)))
    return ModelValueTree(updates, VarNamedTuple(fields))
end
function VarNamedTuples._merge(
    previous::ModelValueTree, updates::ModelValueTree, ::Val{true}
)
    return _merge_model_node(previous, updates)
end

_merge_model_fields(previous, updates, ::Val{()}) = NamedTuple()
function _merge_model_fields(previous, updates, ::Val{names}) where {names}
    name = first(names)
    child = _previous_model_child(previous, AbstractPPL.Property{name}())
    if haskey(updates.values.data, name)
        update = updates.values.data[name]
        child = child isa NoModelBinding ? update : _merge_model_node(child, update)
    end
    rest = _merge_model_fields(previous, updates, Val(Base.tail(names)))
    return child isa NoModelBinding ? rest : merge(NamedTuple{(name,)}((child,)), rest)
end

function BangBang.setindex!!(
    pa::VarNamedTuples.PartialArray{<:ModelValue},
    value::Union{ModelValue,NoModelBinding},
    inds::Vararg{Any};
    kw...,
)
    if !(value isa eltype(pa))
        # Static arrays do not widen their element type on insertion.
        data = similar(pa.data, Union{eltype(pa),typeof(value)})
        for i in eachindex(pa.mask)
            pa.mask[i] && (data[i] = pa.data[i])
        end
        pa = VarNamedTuples.PartialArray(data, pa.mask)
    end
    return invoke(
        setindex!!,
        Tuple{VarNamedTuples.PartialArray,Any,Vararg{Any}},
        pa,
        value,
        inds...;
        kw...,
    )
end

# A finite union keeps mixed binding tags visible to inference instead of typejoining them.
function VarNamedTuples._concretise_eltype!!(
    pa::VarNamedTuples.PartialArray{<:ModelValue{R,T} where {R}}
) where {T}
    isconcretetype(eltype(pa)) && return pa
    isconcretetype(T) || return invoke(
        VarNamedTuples._concretise_eltype!!, Tuple{VarNamedTuples.PartialArray}, pa
    )
    ET = Union{ModelValue{Condition,T},ModelValue{ArgumentCondition,T},ModelValue{Fix,T}}
    eltype(pa) === ET && return pa
    data = similar(pa.data, ET)
    for i in eachindex(pa.mask)
        pa.mask[i] && (data[i] = pa.data[i])
    end
    return VarNamedTuples.PartialArray(data, pa.mask)
end

function VarNamedTuples._prepare_indexed_value(
    value::ModelValue{R,<:AbstractArray}, data, inds...; kw...
) where {R}
    return if VarNamedTuples._is_multiindex(data, inds...; kw...)
        map(x -> _model_value_like(value, x), value.value)
    else
        value
    end
end

_model_data(value) = value
_model_data(value::ModelValue) = value.value
_model_data(values::AbstractArray) = map(_model_data, values)
function _model_data(values::Array{<:ModelValue{<:Any,T}}) where {T}
    # The roles may form a union even when every payload has the same type.
    # Allocate from that payload type; map's inferred eltype can widen on Julia 1.10.
    isconcretetype(T) || return map(_model_data, values)
    return map!(_model_data, similar(values, T), values)
end
# Payload types may also differ; promote them, as `map` takes their typejoin.
function _model_data(values::Array{<:ModelValue})
    data = map(_model_data, values)
    (isempty(data) || isconcretetype(eltype(data))) && return data
    return convert(Array{mapreduce(typeof, promote_type, data)}, data)
end
_model_data(values::VarNamedTuple) = map(_model_data, values.data)
function _model_data(values::VarNamedTuples.PartialArray)
    return _model_data(VarNamedTuples.unwrap_internal_array(values))
end
_model_data(tree::ModelValueTree) = _model_argument_value(tree, tree.template)
function VarNamedTuples.unwrap_internal_array(tree::ModelValueTree)
    VarNamedTuples._haskey_optic(tree, AbstractPPL.Iden()) ||
        throw(ArgumentError("Cannot extract a partially supplied model value"))
    return _model_data(tree)
end

_has_complete_model_data(::Any) = true
_has_complete_model_data(::NoModelBinding) = false
# A namespace does not replace the submodel return value.
_has_complete_model_data(::Union{VarNamedTuple,ModelValueTree}) = false
_has_complete_model_data(::VarNamedTuples.ArrayLikeBlock) = false
function _has_complete_model_data(values::VarNamedTuples.PartialArray)
    # Growable arrays describe supplied indices, not the extent of the argument.
    return !(values.data isa VarNamedTuples.GrowableArray) &&
           all(values.mask) &&
           all(_has_complete_model_data, values.data)
end

# Julia owns allocation, aliases, cycles, views, const fields and undefined slots.
# The wrapper only prevents copying numerical leaves (notably AD tape connections).
struct ModelArgumentCopy{T}
    value::T
    adapt::Bool
end
ModelArgumentCopy(value) = ModelArgumentCopy(value, false)
# Extensions can adapt non-writable AD containers after Julia copies the graph.
function _adapt_copied_argument end
function Base.deepcopy_internal(value::ModelArgumentCopy, memo::IdDict)
    _retain_argument_leaves!(memo, value.value, Base.IdSet{Any}())
    _retain_bound_argument_storage!(memo, value.value)
    return ModelArgumentCopy(
        Base.deepcopy_internal(value.value, memo), get(memo, ModelArgumentCopy, false)
    )
end
function _argument_opaque_value(value)
    return isbits(value) ||
           value isa Union{BigFloat,BigInt} ||
           value isa Number &&
           parentmodule(typeof(value)) === Base &&
           all(i -> _argument_opaque_value(getfield(value, i)), 1:fieldcount(typeof(value)))
end
function _retain_argument_value!(memo, value, seen)
    memo[value] = value
    return nothing
end
function _retain_argument_leaves!(memo, value, seen)
    isbits(value) && return nothing
    value isa Union{Type,Symbol,AbstractString,Module} && return nothing
    value in seen && return nothing
    push!(seen, value)
    if _argument_opaque_value(value) || value isa Number && ismutable(value)
        _retain_argument_value!(memo, value, seen)
    else
        _foreach_argument_child(value, Val(:latent), nothing) do child
            _retain_argument_leaves!(memo, child, seen)
        end
    end
    return nothing
end
# Resolve the optional adapter from type metadata so ordinary preparation remains
# inferred even when ReverseDiff's type-changing facade adapter is loaded.
_argument_ad_storage(::Type) = false
function _argument_storage_expr(query, T, seen)
    T <: Union{Number,Type,Symbol,AbstractString} && return false
    T in seen && return false
    push!(seen, T)
    children = if T isa Union
        Base.uniontypes(T)
    elseif !isconcretetype(T)
        return true
    elseif T <: Array
        (eltype(T),)
    elseif T <: AbstractArray
        # Elements need not be stored in fields, as in `Memory` or a lazy array.
        (eltype(T), fieldtypes(T)...)
    else
        fieldtypes(T)
    end
    result = :($query($T))
    for child in children
        result = :($result || $(_argument_storage_expr(query, child, seen)))
    end
    return result
end
@generated function _argument_may_need_adapter(::Type{T}) where {T}
    return _argument_storage_expr(_argument_ad_storage, T, Set{Any}())
end
function _copy_model_argument(value)
    copied = deepcopy(ModelArgumentCopy(value))
    return if _argument_may_need_adapter(typeof(value)) && copied.adapt
        _adapt_copied_argument(copied.value)
    else
        copied.value
    end
end
_copy_model_argument(value::Type) = value
function _copy_model_argument(value::Number)
    (isbits(value) || _argument_opaque_value(value)) && return value
    return invoke(_copy_model_argument, Tuple{Any}, value)
end
function _copy_model_argument(value::Array{<:Number})
    isbitstype(eltype(value)) && return copy(value)
    return invoke(_copy_model_argument, Tuple{Any}, value)
end
function _copy_model_argument(value, vn)
    _check_latent_storage(value, vn)
    _check_argument_key_storage(value, vn)
    return _copy_model_argument(value)
end

# Latent draws are written into the argument's own storage, which ranges, static and fill
# arrays cannot take. SparseArrays types are immutable structs over mutable buffers, and the
# argument copy adapts AD storage such as a ReverseDiff tracked array.
_writable_storage(::Type{T}) where {T} = ismutabletype(T) || _argument_ad_storage(T)
function _writable_storage(
    ::Type{<:Union{SparseArrays.SparseVector,SparseArrays.SparseMatrixCSC}}
)
    return true
end
_unwritable_storage_type(::Type) = false
_unwritable_storage_type(::Type{T}) where {T<:AbstractArray} = !_writable_storage(T)
# Skips the walk below, on every evaluation, for argument types that hold no such storage.
@generated function _argument_may_be_unwritable(::Type{T}) where {T}
    return _argument_storage_expr(_unwritable_storage_type, T, Set{Any}())
end
# The innermost parent of an array, which holds its elements.
_array_storage(value) = value
function _array_storage(value::AbstractArray)
    storage = parent(value)
    return storage === value ? value : _array_storage(storage)
end
# Returns the address and value of the first unwritable storage below `vn`. A read-only
# wrapper over a mutable parent is not detected, and neither are struct fields or `Dict`s.
_unwritable_storage(value, vn, seen) = nothing
function _unwritable_storage(value::Union{Tuple,NamedTuple}, vn, seen)
    for (key, child) in pairs(value)
        optic =
            key isa Symbol ? AbstractPPL.Property{key}() : AbstractPPL.Index((key,), (;))
        found = _unwritable_storage(child, AbstractPPL.append_optic(vn, optic), seen)
        found === nothing || return found
    end
    return nothing
end
function _unwritable_storage(value::AbstractArray, vn, seen)
    _writable_storage(typeof(_array_storage(value))) || return vn => value
    eltype(value) <: Number && return nothing
    # A self-referential argument would otherwise be walked without end.
    value in seen && return nothing
    push!(seen, value)
    for i in CartesianIndices(value)
        isassigned(value, i) || continue
        optic = AbstractPPL.Index(Tuple(i), (;))
        found = _unwritable_storage(value[i], AbstractPPL.append_optic(vn, optic), seen)
        found === nothing || return found
    end
    return nothing
end
function _check_latent_storage(value, vn)
    _argument_may_be_unwritable(typeof(value)) || return nothing
    found = _unwritable_storage(value, vn, Base.IdSet{Any}())
    found === nothing && return nothing
    address, storage = found
    # Model arguments keep their types: never convert one to make room for latent draws.
    throw(
        ArgumentError(
            "Argument `$address` is a `$(typeof(storage))`, which cannot hold latent " *
            "draws; pass a mutable copy of the array at `$address`, e.g. made with `collect`.",
        ),
    )
end
# Applies the check above when a removal is made rather than when the model runs.
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

# A partial binding can contain nested shape owners. Copy their templates with the
# argument in one graph so overlapping templates use the same identity table.
# One bits type keeps dense arrays of leaf plans cheap for Julia to deepcopy.
struct ModelArgumentLeaf
    bound::Bool
end
struct ModelArgumentStorage{T,C}
    value::T
    children::C
end
@inline function _unwritable_storage(storage::ModelArgumentStorage, vn, seen)
    # Binding listings materialize values without making their gaps latent.
    vn === nothing && return nothing
    value = storage.value
    value isa Union{Tuple,NamedTuple,AbstractArray} || return nothing
    # Partial binding validation already rejects immutable numeric array storage.
    value isa AbstractArray && eltype(value) <: Number && return nothing
    indices = value isa AbstractArray ? CartesianIndices(value) : keys(value)
    return _first_unwritable_storage(indices) do key
        value isa AbstractArray && !isassigned(value, key) && return nothing
        plan = get(storage.children, key, ModelArgumentLeaf(false))
        plan isa ModelArgumentLeaf && plan.bound && return nothing
        optic = if key isa Symbol
            AbstractPPL.Property{key}()
        else
            AbstractPPL.Index(key isa CartesianIndex ? Tuple(key) : (key,), (;))
        end
        child = plan isa ModelArgumentStorage ? plan : value[key]
        return _unwritable_storage(child, AbstractPPL.append_optic(vn, optic), seen)
    end
end
@inline function _first_unwritable_storage(f, indices::Tuple)
    isempty(indices) && return nothing
    found = f(first(indices))
    return found === nothing ? _first_unwritable_storage(f, Base.tail(indices)) : found
end
function _first_unwritable_storage(f, indices)
    for key in indices
        found = f(key)
        found === nothing || return found
    end
    return nothing
end
# Fully bound branches need no private storage. Mark latent reachability first:
# a bound branch may alias a latent one, which still needs an independent copy.
_retain_bound_argument_storage!(memo, value) = nothing
function _retain_bound_argument_storage!(memo, storage::ModelArgumentStorage)
    latent, bound = Base.IdSet{Any}(), Base.IdSet{Any}()
    _argument_storage_policy!(latent, bound, storage)
    for value in bound
        value in latent || (memo[value] = value)
    end
    return nothing
end
const ArgumentMemory = @static if isdefined(Core, :GenericMemory)
    Core.GenericMemory
else
    Union{}
end
_argument_memory_ids(value) = ()
function _argument_memory_ids(value::Union{Array,ArgumentMemory})
    return isempty(value) ? () : Base.dataids(value)
end

# Owning arrays expose elements; wrappers expose fields, never a second logical edge.
function _foreach_argument_field(f::F, value) where {F}
    if value isa Array && hasfield(typeof(value), :ref)
        isempty(value) && return nothing
        return f(value.ref.mem)
    elseif value isa Union{Array,ArgumentMemory,Core.SimpleVector}
        value isa Core.SimpleVector || isbitstype(eltype(value)) && return nothing
        for i in eachindex(value)
            isassigned(value, i) || continue
            result = f(value[i])
            result === nothing || return result
        end
    else
        for i in 1:fieldcount(typeof(value))
            isbitstype(fieldtype(typeof(value), i)) && continue
            isdefined(value, i) || continue
            result = f(getfield(value, i))
            result === nothing || return result
        end
    end
    return nothing
end
_foreach_argument_child(f::F, value) where {F} = _foreach_argument_field(f, value)
function _foreach_argument_child(f::F, value, mode::Val, bound) where {F}
    return if mode isa Val{:retained}
        _foreach_argument_field(f, value)
    else
        _foreach_argument_child(f, value)
    end
end
function _foreach_argument_child(f::F, value::AbstractDict, mode::Val, bound) where {F}
    mode isa Union{Val{:all},Val{:retained}} && return _foreach_argument_field(f, value)
    for (key, child) in value
        bound === nothing || push!(bound, key)
        result = f(child)
        result === nothing || return result
    end
    return nothing
end
function _argument_graph!(seen, value, repeated=nothing, mode=Val(:all), bound=nothing)
    if mode isa Val{:retained}
        isbits(value) && return nothing
        value isa Union{Type,Symbol,AbstractString,Module} && return nothing
    else
        mode isa Val{:latent} && _argument_opaque_value(value) && return nothing
        _argument_may_alias(typeof(value)) || return nothing
    end
    if ismutable(value)
        if value in seen
            repeated === nothing || push!(repeated, value)
            return nothing
        end
        push!(seen, value)
    end
    ids =
        value isa Array && hasfield(typeof(value), :ref) ? () : _argument_memory_ids(value)
    for id in ids
        if id in seen
            repeated === nothing || push!(repeated, id)
            return nothing
        end
        push!(seen, id)
    end
    _foreach_argument_child(value, mode, bound) do child
        _argument_graph!(seen, child, repeated, mode, bound)
    end
    return nothing
end

_has_latent_argument(leaf::ModelArgumentLeaf) = !leaf.bound
_has_latent_argument(value) = true
function _has_latent_argument(storage::ModelArgumentStorage)
    value = storage.value
    n = value isa Union{Tuple,AbstractArray} ? length(value) : fieldcount(typeof(value))
    return length(storage.children) < n || any(_has_latent_argument, storage.children)
end
function _check_argument_key_storage(value, vn)
    vn === nothing && return nothing
    _has_latent_argument(value) || return nothing
    retained, latent = Base.IdSet{Any}(), Base.IdSet{Any}()
    _check_argument_leaves(value, vn, Base.IdSet{Any}(), true, retained)
    isempty(retained) && return nothing
    if value isa ModelArgumentStorage
        _argument_branches!(latent, nothing, value, nothing, nothing, Val(:latent))
    else
        _argument_graph!(latent, value, nothing, Val(:latent))
    end
    any(x -> x in retained, latent) && throw(
        ArgumentError(
            "Cannot copy model argument `$vn`: retained storage is also reached by a latent path. Pass independent storage for the latent path.",
        ),
    )
    return nothing
end
function _check_argument_leaves(value, vn, seen, latent, retained)
    isbits(value) && return nothing
    value isa Union{Type,Symbol,AbstractString,Module} && return nothing
    _argument_opaque_value(value) &&
        return _argument_graph!(retained, value, nothing, Val(:retained))
    value in seen && return nothing
    push!(seen, value)
    if latent && value isa Number
        storage = _foreach_argument_field(_mutable_argument_storage, value)
        storage === nothing || throw(
            ArgumentError(
                "Cannot copy model argument `$vn`: number `$(typeof(value))` reaches mutable storage. Keep the storage outside the number.",
            ),
        )
    end
    value isa Number &&
        ismutable(value) &&
        _argument_graph!(retained, value, nothing, Val(:retained))
    if value isa AbstractDict
        for (key, child) in value
            isbits(key) ||
                key isa Union{Symbol,String} ||
                throw(
                    ArgumentError(
                        "Cannot copy model argument `$vn`: dictionary key `$(typeof(key))` is not an isbits value, Symbol or String. Use an isbits, Symbol or String key, or pass the dictionary as a separate covariate argument.",
                    ),
                )
            _check_argument_leaves(child, vn, seen, latent, retained)
        end
    else
        _foreach_argument_field(value) do child
            _check_argument_leaves(child, vn, seen, latent, retained)
        end
    end
    return nothing
end
function _mutable_argument_storage(value)
    isbits(value) && return nothing
    value isa Union{Type,Symbol,AbstractString,Module} && return nothing
    ismutable(value) && return true
    return _foreach_argument_field(_mutable_argument_storage, value)
end
function _check_argument_leaves(storage::ModelArgumentStorage, vn, seen, latent, retained)
    _check_argument_leaves(storage.value, vn, Base.IdSet{Any}(), false, retained)
    value, children = storage.value, storage.children
    indices = children isa NamedTuple ? fieldnames(typeof(value)) : eachindex(value)
    for i in indices
        (i isa Symbol ? isdefined(value, i) : value isa Tuple || isassigned(value, i)) ||
            continue
        child = i isa Symbol ? getfield(value, i) : value[i]
        plan = get(children, i, ModelArgumentLeaf(false))
        plan isa ModelArgumentLeaf && plan.bound && continue
        _check_argument_leaves(
            plan isa ModelArgumentStorage ? plan : child, vn, seen, latent, retained
        )
    end
    return nothing
end

# A partial owner contributes its fields and backing storage, with elements handled
# by the binding plan exactly once.
function _argument_owner_graph!(
    seen, value, repeated=nothing, mode=Val(:shared), bound=nothing
)
    mode isa Val{:latent} && _argument_opaque_value(value) && return nothing
    if value isa Union{Array,ArgumentMemory}
        if value in seen
            repeated === nothing || push!(repeated, value)
            return nothing
        end
        push!(seen, value)
        if value isa Array && hasfield(typeof(value), :ref)
            isempty(value) && return nothing
            return _argument_owner_graph!(seen, value.ref.mem, repeated, mode, bound)
        end
        for id in _argument_memory_ids(value)
            id in seen && repeated !== nothing && push!(repeated, id)
            push!(seen, id)
        end
    elseif value isa AbstractArray
        _foreach_argument_child(value) do child
            if child === parent(value)
                _argument_owner_graph!(seen, child, repeated, mode, bound)
            else
                _argument_graph!(seen, child, repeated, mode, bound)
            end
        end
    elseif ismutable(value)
        value in seen && repeated !== nothing && push!(repeated, value)
        push!(seen, value)
    end
    return nothing
end

function _argument_branches!(
    latent, bound, storage::ModelArgumentStorage, binding, repeated, mode=Val(:shared)
)
    value, children = storage.value, storage.children
    mode isa Val{:latent} && _argument_opaque_value(value) && return nothing
    _argument_owner_graph!(latent, value, repeated, mode, bound)
    indices = children isa NamedTuple ? fieldnames(typeof(value)) : eachindex(value)
    for i in indices
        (i isa Symbol ? isdefined(value, i) : value isa Tuple || isassigned(value, i)) ||
            continue
        child = i isa Symbol ? getfield(value, i) : value[i]
        plan = if i isa Symbol
            get(children, i, ModelArgumentLeaf(false))
        elseif checkbounds(Bool, children, i)
            children[i]
        else
            ModelArgumentLeaf(false)
        end
        optic = if i isa Symbol
            AbstractPPL.Property{i}()
        else
            AbstractPPL.Index(i isa CartesianIndex ? Tuple(i) : (i,), (;))
        end
        childbinding = _model_argument_binding(binding, optic)
        if plan isa ModelArgumentStorage
            _argument_branches!(latent, bound, plan, childbinding, repeated, mode)
        elseif plan.bound
            bound === nothing ||
                childbinding isa ModelValue{ArgumentCondition} && push!(bound, child)
        else
            _argument_graph!(latent, child, repeated, mode, bound)
        end
    end
    return nothing
end

function _argument_alias_type(T, seen)
    T <: Union{Type,Symbol,Module,String,BigFloat,BigInt} && return false
    isbitstype(T) && return false
    T in seen && return false
    push!(seen, T)
    T isa Union && return any(t -> _argument_alias_type(t, seen), Base.uniontypes(T))
    isconcretetype(T) || return true
    ismutabletype(T) && !(T <: Number) && return true
    return any(t -> _argument_alias_type(t, seen), fieldtypes(T))
end
@generated function _argument_may_alias(::Type{T}) where {T}
    return _argument_alias_type(T, Set{Any}())
end
function _check_shared_latent_storage(model::Model, prefix=_model_prefix(model))
    lhs = _args_on_lhs(model)
    latent, repeated = Base.IdSet{Any}(), Any[]
    roots, bound = Tuple{Any,Any,Any,Union{Bool,Nothing}}[], Any[]
    for (stored_name, value) in pairs(merge(model.args, model.defaults))
        name = unsplat_symbol(stored_name)
        _argument_may_alias(typeof(value)) || continue
        vn = VarName{name}()
        binding = name in lhs ? _get_model_binding(model, vn) : nothing
        role = nothing
        if !(name in lhs) || binding isa ModelValue{ArgumentCondition}
            push!(bound, value)
            role = false
        elseif binding === nothing
            _argument_graph!(latent, value, repeated, Val(:shared), bound)
            role = true
        elseif binding isa Union{ModelValueTree,VarNamedTuple,VarNamedTuples.PartialArray}
            storage = _argument_storage(binding, value)
            storage isa ModelArgumentStorage || continue
            value = storage.value
            _argument_branches!(latent, bound, storage, binding, repeated)
        else
            continue
        end
        push!(roots, (maybe_prefix(vn, prefix), value, binding, role))
    end
    isempty(latent) && return nothing
    shared = isempty(repeated) ? nothing : first(repeated)
    if shared === nothing
        visited = Base.IdSet{Any}()
        memory = Set{UInt}(node for node in latent if node isa UInt)
        for value in bound
            shared = _reached_latent(latent, value, visited, memory)
            shared === nothing || break
        end
    end
    shared === nothing && return nothing
    selected = _shared_argument_names(roots, shared)
    throw(
        ArgumentError(
            "Storage reachable from a latent argument in model `$(nameof(model))` is shared by $(join(map(name -> "`$name`", selected), " and ")). A latent draw may replace storage and leave another reference stale, even if that reference is only read; pass independent storage, for example `(a=v, b=copy(v))`.",
        ),
    )
end
function _reached_latent(latent, value, seen, memory)
    _argument_may_alias(typeof(value)) || return nothing
    for id in _argument_memory_ids(value)
        id in memory && return id
    end
    (!(value isa Union{Array,ArgumentMemory}) || isempty(value)) &&
        ismutable(value) &&
        value in latent &&
        return value
    value isa Union{Array,ArgumentMemory} && isbitstype(eltype(value)) && return nothing
    if ismutable(value)
        value in seen && return nothing
        push!(seen, value)
    end
    return _foreach_argument_child(value) do child
        _reached_latent(latent, child, seen, memory)
    end
end
function _shared_argument_names(roots, shared)
    involved, latent_names = Any[], Any[]
    for (name, value, binding, role) in roots
        graph, repeated, bound = Base.IdSet{Any}(), Any[], Any[]
        if role === false
            push!(bound, value)
        elseif role === true
            _argument_graph!(graph, value, repeated, Val(:shared), bound)
        else
            _argument_branches!(
                graph, bound, _argument_storage(binding, value), binding, repeated
            )
        end
        internal = any(node -> node === shared, repeated)
        if shared in graph
            push!(latent_names, name)
            internal && return (name, name)
        end
        bound_graph = Base.IdSet{Any}()
        for value in bound
            _argument_graph!(bound_graph, value)
        end
        name in latent_names && shared in bound_graph && return (name, name)
        (shared in graph || shared in bound_graph) && push!(involved, name)
    end
    name = first(latent_names)
    other = findfirst(other -> other != name, involved)
    return (name, other === nothing ? name : involved[other])
end
function _argument_storage_policy!(latent, bound, storage)
    value, children = storage.value, storage.children
    push!(latent, value)
    value isa Array && union!(latent, _argument_memory_ids(value))
    # Array headers and their backing memory must follow the same copy policy.
    # Julia 1.10 stores the memory directly; later versions expose a MemoryRef.
    if value isa Array && hasfield(typeof(value), :ref)
        push!(latent, value.ref)
        push!(latent, value.ref.mem)
    end
    # Partial views still need private backing storage, even when their parent is
    # also a fully bound branch elsewhere in the argument graph.
    if value isa AbstractArray && parent(value) !== value
        _argument_graph!(latent, parent(value))
    end
    if children isa NamedTuple
        for name in fieldnames(typeof(value))
            isdefined(value, name) || continue
            child = getfield(value, name)
            plan = get(children, name, ModelArgumentLeaf(false))
            _argument_child_policy!(latent, bound, child, plan)
        end
    else
        for i in eachindex(value)
            (value isa Tuple || isassigned(value, i)) || continue
            plan = if (children isa Tuple || checkbounds(Bool, children, i))
                children[i]
            else
                ModelArgumentLeaf(false)
            end
            _argument_child_policy!(latent, bound, value[i], plan)
        end
    end
    return nothing
end
function _argument_child_policy!(latent, bound, value, leaf::ModelArgumentLeaf)
    leaf.bound && bound === nothing && return nothing
    return _argument_graph!(leaf.bound ? bound : latent, value)
end
function _argument_child_policy!(latent, bound, value, storage::ModelArgumentStorage)
    # A nested owner can replace the original child, including its size and type.
    value === storage.value || bound === nothing || _argument_graph!(bound, value)
    return _argument_storage_policy!(latent, bound, storage)
end
_argument_storage(value, template) = ModelArgumentLeaf(true)
function _argument_storage(tree::ModelValueTree, template)
    return _argument_storage(tree.values, tree.template)
end
# Keep property names static so heterogeneous sibling types do not get merged.
@generated function _argument_storage(values::VarNamedTuple{names}, template) where {names}
    children = map(names) do name
        :(_argument_property_storage(values.data.$name, template, Val($(QuoteNode(name)))))
    end
    return quote
        template isa NoTemplate && return nothing
        ModelArgumentStorage(template, NamedTuple{$names}(($(children...),)))
    end
end
function _argument_property_storage(value, template, ::Val{name}) where {name}
    hasproperty(template, name) || throw(
        ArgumentError(
            "Cannot override nonexistent property `$name` of $(typeof(template))."
        ),
    )
    return _argument_child_storage(value, template, AbstractPPL.Property{name}())
end
_argument_child_storage(::ModelValue, template, optic) = ModelArgumentLeaf(true)
_argument_child_storage(::NoModelBinding, template, optic) = ModelArgumentLeaf(false)
function _argument_child_storage(value, template, optic)
    child = if optic isa AbstractPPL.Property
        getproperty(template, _binding_property_name(optic))
    elseif template isa Tuple
        getindex(template, optic.ix...; optic.kw...)
    else
        VarNamedTuples.maybe_index_template(template, optic)
    end
    return _argument_storage(value, child)
end
function _argument_storage(values::VarNamedTuples.PartialArray, template)
    _has_complete_model_data(values) && return ModelArgumentLeaf(true)
    original = template
    _inherits_shape(values.mask) && (template = values.mask.template)
    original = original isa AbstractArray ? original : template
    if template isa AbstractArray &&
        !(values.data isa VarNamedTuples.GrowableArray) &&
        (template !== original || axes(template) != axes(values.data))
        T = promote_type(eltype(template), eltype(original))
        resized = similar(template, T, axes(values.data))
        # A new extent still keeps stale argument values at surviving latent indices.
        for i in eachindex(original)
            if checkbounds(Bool, resized, i) &&
                !haskey(values, i) &&
                isassigned(original, i)
                resized[i] = original[i]
            end
        end
        template = resized
    end
    # Capture the final template by value, rather than boxing the reassigned argument.
    children = let template = template
        map(CartesianIndices(values.mask)) do i
            values.mask[i] || return ModelArgumentLeaf(false)
            value = values.data[i]
            if value isa VarNamedTuples.ArrayLikeBlock
                _argument_child_storage(
                    value.block, template, AbstractPPL.Index(value.ix, value.kw)
                )
            else
                _argument_child_storage(value, template, AbstractPPL.Index(Tuple(i), (;)))
            end
        end
    end
    return ModelArgumentStorage(template, children)
end

_model_argument_value(value, template) = value
_model_argument_value(::Nothing, template) = _copy_model_argument(template)
_model_argument_value(value, template, vn) = _model_argument_value(value, template)
_model_argument_value(::Nothing, template, vn) = _copy_model_argument(template, vn)
_model_argument_value(value::ModelValue, template) = value.value
_model_argument_value(values::AbstractArray, template) = _model_data(values)
function _model_argument_value(
    values::Union{ModelValueTree,VarNamedTuple,VarNamedTuples.PartialArray},
    template,
    vn=nothing,
)
    storage = _argument_storage(values, template)
    return _apply_model_bindings(values, _copy_model_argument(storage, vn))
end

_apply_model_bindings(value, storage) = value
_apply_model_bindings(value::ModelValue, storage) = value.value
_apply_model_bindings(values::VarNamedTuple, ::Nothing) = _model_data(values)
function _apply_model_bindings(values::VarNamedTuples.PartialArray, ::ModelArgumentLeaf)
    return _model_data(values)
end
function _apply_model_bindings(tree::ModelValueTree, storage)
    return _apply_model_bindings(tree.values, storage)
end
@generated function _apply_model_bindings(
    values::VarNamedTuple{names}, storage
) where {names}
    updates = map(names) do name
        :(
            result = _set_argument_property(
                result,
                Val($(QuoteNode(name))),
                _apply_model_bindings(values.data.$name, storage.children.$name),
            )
        )
    end
    return quote
        result = storage.value
        $(updates...)
        result
    end
end
function _set_argument_property(result, ::Val{name}, value) where {name}
    if ismutabletype(typeof(result))
        if hasfield(typeof(result), name) && isconst(typeof(result), name)
            getproperty(result, name) === value && return result
            throw(
                ArgumentError(
                    "Cannot replace const property `$name` of argument type $(typeof(result)); bind the whole value instead.",
                ),
            )
        end
        setproperty!(result, name, value)
        return result
    end
    # An immutable path whose mutable child was updated needs no reconstruction.
    if (ismutable(value) || value isa AbstractArray) &&
        isdefined(result, name) &&
        getproperty(result, name) === value
        return result
    end
    return _rebuild_argument_property(result, Val(name), value)
end
# Check dispatch, not inference: custom setters own their reconstruction protocol.
function _argument_reconstructible(value::T, patch::P) where {T,P<:NamedTuple}
    setter = ConstructionBase.setproperties
    hasmethod(setter, Tuple{T,P}) || return false
    # NamedTuples always support replacing existing properties.
    T <: NamedTuple && all(name -> hasfield(T, name), fieldnames(P)) && return true
    which(setter, Tuple{T,P}) !== which(setter, Tuple{Any,NamedTuple}) && return true
    isempty(fieldnames(P)) && return true

    # ConstructionBase's fallback uses properties in field order, with the patch
    # taking precedence, then calls constructorof(T). Declared field types can be
    # abstract, so inspect the actual unpatched property types as dispatch does.
    names = fieldnames(T)
    propertynames(value) === names || return false
    all(name -> name in names, fieldnames(P)) || return false
    types = map(names) do name
        Core.Typeof(getproperty(hasfield(P, name) ? patch : value, name))
    end
    return hasmethod(ConstructionBase.constructorof(T), Tuple{types...})
end
function _rebuild_argument_property(result, ::Val{name}, value) where {name}
    patch = NamedTuple{(name,)}((value,))
    _argument_reconstructible(result, patch) || throw(
        ArgumentError(
            "Cannot rebuild argument of type $(typeof(result)) when binding property `$name`; provide a ConstructionBase.setproperties method or bind the whole value.",
        ),
    )
    return ConstructionBase.setproperties(result, patch)
end
function _apply_model_bindings(values::VarNamedTuples.PartialArray, storage)
    result = storage.value
    mask =
        if eltype(values) <: VarNamedTuples.ArrayLikeBlock ||
            VarNamedTuples.ArrayLikeBlock <: eltype(values)
            copy(values.mask)
        else
            values.mask
        end
    for i in CartesianIndices(mask)
        mask[i] || continue
        binding = values.data[i]
        if binding isa VarNamedTuples.ArrayLikeBlock
            mask[binding.ix..., binding.kw...] .= false
            value = _apply_model_bindings(binding.block, storage.children[i])
            result = _set_argument_index(result, value, binding.ix...; binding.kw...)
        else
            value = _apply_model_bindings(binding, storage.children[i])
            result = _set_argument_index(result, value, Tuple(i)...)
        end
    end
    return result
end
function _set_argument_index(result, value, indices...; kwargs...)
    if result isa AbstractArray &&
        BangBang.implements(setindex!, result) &&
        value isa eltype(result)
        setindex!(result, value, indices...; kwargs...)
        return result
    end
    return _setindex!!(result, value, indices...; kwargs...)
end
function _rebuild_argument_property(result::NamedTuple, ::Val{name}, value) where {name}
    return ConstructionBase.setproperties(result, NamedTuple{(name,)}((value,)))
end

@generated function _select_model_values(
    ::Type{R}, values::VarNamedTuple{names}
) where {R,names}
    fields = map(names) do name
        :(
            let selected = _select_model_node(R, values.data.$name)
                if selected isa NoModelBinding
                    (;)
                else
                    NamedTuple{($(QuoteNode(name)),)}((selected,))
                end
            end
        )
    end
    return :(VarNamedTuple(merge((;), $(fields...))))
end
function _select_model_node(::Type{R}, value::ModelValue) where {R}
    return _matches_model_role(R, value) ? value.value : NoModelBinding()
end
function _select_model_node(::Type{R}, values::VarNamedTuple) where {R}
    selected = _select_model_values(R, values)
    return isempty(selected) ? NoModelBinding() : selected
end
function _select_model_node(
    ::Type{R}, values::VarNamedTuples.PartialArray{<:ModelValue}
) where {R}
    values.data isa VarNamedTuples.GrowableArray &&
        return _select_model_node_recursive(R, values)
    T = Core.Compiler.return_type(_model_data, Tuple{eltype(values)})
    isconcretetype(T) || return _select_model_node_recursive(R, values)
    data = similar(values.data, T)
    mask = copy(values.mask)
    found = false
    for i in eachindex(mask)
        mask[i] || continue
        value = values.data[i]
        mask[i] = _matches_model_role(R, value)
        if mask[i]
            data[i] = value.value
            found = true
        end
    end
    return found ? VarNamedTuples.PartialArray(data, mask) : NoModelBinding()
end
_select_model_node(::Type{R}, value) where {R} = _select_model_node_recursive(R, value)
function _select_model_node_recursive(::Type{R}, value) where {R}
    selected = _select_model_values_recursive(R, VarNamedTuple(; _=value))
    return get(selected.data, :_, NoModelBinding())
end
function _select_model_values_recursive(::Type{R}, values::VarNamedTuple) where {R}
    selected = mapfoldl(
        identity,
        function (selected, pair)
            vn, value = pair
            return if _matches_model_role(R, value)
                templated_setindex!!(
                    selected, value.value, vn, values.data[AbstractPPL.getsym(vn)]
                )
            else
                selected
            end
        end,
        values;
        init=VarNamedTuple(),
    )
    return _plain_model_values(selected)
end

# Keep whole supplied values intact, but expose incomplete bindings as ordinary partial values.
_plain_model_values(value) = value
function _plain_model_values(values::VarNamedTuple)
    return VarNamedTuple(map(_plain_model_values, values.data))
end
function _plain_model_values(tree::ModelValueTree)
    VarNamedTuples._haskey_optic(tree, AbstractPPL.Iden()) && return _model_data(tree)
    return _plain_model_values(tree.values)
end
function _plain_model_values(values::VarNamedTuples.PartialArray)
    return _fold_model_indices(empty(values), values) do selected, value, optic, template
        return VarNamedTuples._setindex_optic!!(
            selected, _plain_model_values(value), optic, template, VarNamedTuples.AllowAll()
        )
    end
end

function _materialize_argument_values(model::Model)
    model.values isa UnprefixedArgumentValues || return model
    values = _prefix_values(
        _model_values(model.values),
        _model_prefix(model),
        _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
    )
    return _reconstruct_model(model; values)
end

"""
    model | (x = 1.0, ...)

Return a `Model` which now treats variables on the right-hand side as observations.

See [`condition`](@ref) for more information and examples.
"""
Base.:|(model::Model, values::Union{NamedTuple,Pair,Tuple,VarNamedTuple}) =
    _bind_ordered_inputs(Condition, model, _binding_inputs(values))

function _check_binding_addresses(model, values)
    metadata = _binding_metadata(model)
    names = _lhs_names(metadata)
    names === nothing && return nothing
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    if prefix !== nothing
        for name in keys(values.data)
            name === AbstractPPL.getsym(prefix) || throw(
                ArgumentError(
                    "Cannot bind `$name`: it is outside this model's prefix `$prefix`."
                ),
            )
        end
    end
    _has_unprefixed_submodel(metadata) && return nothing
    for name in keys(_submodel_values(values, prefix).data)
        name in names || _binding_name_error(name, "bind")
    end
    return nothing
end
@noinline function _binding_name_error(vn, operation)
    message =
        "Cannot $operation `$vn`: it is not an LHS top symbol of this model. Only " *
        "a literal `to_submodel(child, false)` tilde lets an unprefixed child own other names."
    throw(ArgumentError(message))
end

@generated function _check_argument_bindings(model::Model, values, prefix=nothing)
    names = (
        fieldnames(fieldtype(model, :args))..., fieldnames(fieldtype(model, :defaults))...
    )
    checks = map(names) do stored_name
        name = unsplat_symbol(stored_name)
        quote
            name = $(QuoteNode(name))
            stored_name = $(QuoteNode(stored_name))
            vn = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
            binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
            if name in _args_on_lhs(model)
                if binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
                    if $(
                        is_splat_symbol(stored_name) &&
                        stored_name in fieldnames(fieldtype(model, :defaults))
                    )
                        throw(
                            ArgumentError(
                                "Entries of keyword-splat argument `$name` cannot be bound; replace the whole argument with `condition` or `fix` instead.",
                            ),
                        )
                    end
                    argument = get(merge(model.args, model.defaults), stored_name, nothing)
                    previous = _model_argument_binding(
                        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
                    )
                    argument = prepare_model_argument(previous, argument)
                    if argument isa Union{Nothing,Missing}
                        _has_complete_model_data(binding) || throw(
                            ArgumentError(
                                "Cannot make a partial binding of argument `$vn` with whole `$argument` storage; supply a concrete argument (e.g. `$(nameof(model))(zeros(n))`) or a whole binding instead.",
                            ),
                        )
                        binding = ModelValue{typeof(_model_role(binding, vn))}(
                            _model_data(binding)
                        )
                    else
                        binding = _prepare_argument_fields(
                            argument, binding, maybe_prefix(vn, prefix)
                        )
                    end
                    values = templated_setindex!!(
                        values, binding, vn, values.data[AbstractPPL.getsym(vn)]
                    )
                end
                if binding isa ModelValue
                    value = binding.value
                    T = _declared_argument_type(_binding_metadata(model), stored_name)
                    value isa T || throw(
                        ArgumentError(
                            "Bound value at `$vn` in model `$(nameof(model))` must be an instance of declared argument type $T; supplied $(typeof(value)).",
                        ),
                    )
                end
            else
                binding === nothing || throw(
                    ArgumentError(
                        "Argument `$name` does not occur on the left-hand side of `~` and cannot be conditioned or fixed; construct the model with a new argument value instead. If `$name` names a variable of an unprefixed submodel, rename the argument.",
                    ),
                )
            end
        end
    end
    return quote
        isempty(values) && return values
        $(checks...)
        return values
    end
end

_prepare_argument_fields(template, binding, vn) = binding
function _prepare_argument_fields(
    template, bindings::Union{VarNamedTuple,VarNamedTuples.PartialArray}, vn
)
    return _fold_model_indices(copy(bindings), bindings) do result, binding, optic, storage
        _check_partial_binding(
            ModelValue{Condition}(template), optic, AbstractPPL.varname_to_optic(vn)
        )
        _check_binding_template_bounds(
            template, optic, AbstractPPL.append_optic(vn, optic); check_fields=true
        )
        if VarNamedTuples._haskey_optic(template, optic)
            address = AbstractPPL.append_optic(vn, optic)
            if binding isa ModelValue
                # Conversion needs the owner's type, not a possibly unassigned value.
                binding = _convert_partial_argument_binding(
                    binding, template, optic, address
                )
            else
                child = VarNamedTuples._getindex_optic(template, optic, vn)
                binding = _prepare_argument_fields(child, binding, address)
            end
        end
        return VarNamedTuples._setindex_optic!!(
            result, binding, optic, storage, VarNamedTuples.AllowAll()
        )
    end
end

_binding_property_name(::AbstractPPL.Property{S}) where {S} = S

function _convert_partial_argument_binding(
    binding::ModelValue{R}, template, optic, vn
) where {R}
    value = binding.value
    multi =
        optic isa AbstractPPL.Index &&
        VarNamedTuples._is_multiindex(template, optic.ix...; optic.kw...)
    T = if template isa AbstractArray && optic isa AbstractPPL.Index
        eltype(template)
    elseif optic isa AbstractPPL.Property
        name = _binding_property_name(optic)
        if hasfield(typeof(template), name)
            fieldtype(typeof(template), name)
        else
            child = getproperty(template, name)
            multi = child isa AbstractArray
            multi ? eltype(child) : typeof(child)
        end
    else
        Any
    end
    converted = multi ? map(v -> convert(T, v), value) : convert(T, value)
    isequal(converted, value) || throw(
        ArgumentError(
            "Cannot exactly represent partial binding at `$vn` in storage element type $T",
        ),
    )
    return _model_value_like(binding, converted)
end

# The address of the storage that `optic`, a suffix of `vn`'s optic, indexes into.
function _storage_address(vn, optic)
    owner = AbstractPPL.varname_to_optic(vn)
    for _ in 1:_optic_length(optic)
        owner = AbstractPPL.oinit(owner)
    end
    return AbstractPPL.optic_to_varname(owner)
end
_optic_length(::AbstractPPL.Iden) = 0
_optic_length(optic::AbstractPPL.AbstractOptic) = 1 + _optic_length(optic.child)
@noinline function _no_index_storage_error(optic, vn, operation)
    owner = _storage_address(vn, optic)
    key = length(optic.ix) == 1 ? only(optic.ix) : nothing
    advice = if key isa Symbol
        field = AbstractPPL.Property{key}(optic.child) ∘ AbstractPPL.varname_to_optic(owner)
        "NamedTuple fields use `$(AbstractPPL.optic_to_varname(field))`"
    elseif operation == "remove"
        "use integer indices"
    else
        "use integer indices, or supply storage with a whole binding or a binding template"
    end
    message = "Cannot $operation `$vn`: no storage at `$owner` in the layer being edited to resolve this index; $advice."
    throw(ArgumentError(message))
end

function _check_binding_template_bounds(
    template, ::AbstractPPL.Iden, vn; operation="bind", check_fields=false
)
    return nothing
end
function _check_binding_template_bounds(
    template, optic::AbstractPPL.Property{S}, vn; operation="bind", check_fields=false
) where {S}
    template isa Union{ModelValue,ModelValueTree} && (template = _binding_storage(template))
    if check_fields && template isa NamedTuple && !haskey(template, S)
        throw(
            ArgumentError(
                "Cannot $operation `$vn`: nonexistent field `$S` in storage at `$(_storage_address(vn, optic))`.",
            ),
        )
    end
    optic.child isa AbstractPPL.Iden && return nothing
    child = VarNamedTuples.SharedGetProperty{S}()(template)
    return _check_binding_template_bounds(child, optic.child, vn; operation, check_fields)
end
function _check_binding_template_bounds(
    template, optic::AbstractPPL.Index, vn; operation="bind", check_fields=false
)
    template isa Union{ModelValue,ModelValueTree} && (template = _binding_storage(template))
    array = if template isa VarNamedTuples.PartialArray
        template.data
    else
        VarNamedTuples.template_array(template)
    end
    # Without storage, `end`, `:` and Symbol indices cannot be resolved.
    array isa Union{NoTemplate,VarNamedTuples.SkipTemplate,Missing} &&
        !all(i -> i isa Union{Integer,AbstractVector{<:Integer}}, optic.ix) &&
        _no_index_storage_error(optic, vn, operation)
    coptic = AbstractPPL.concretize_top_level(optic, array)
    inbounds = if array isa VarNamedTuples.GrowableArray
        true
    elseif array isa Tuple
        checkbounds(Bool, Base.OneTo(length(array)), coptic.ix...)
    elseif array isa AbstractArray
        checkbounds(Bool, array, coptic.ix...; coptic.kw...)
    elseif array isa Union{NoTemplate,VarNamedTuples.SkipTemplate,Missing}
        dims = VarNamedTuples.get_maximum_size_from_indices(coptic.ix...; coptic.kw...)
        all(>=(0), dims) || throw(ArgumentError("invalid Array dimensions"))
        Base.checkbounds_indices(Bool, map(Base.OneTo, dims), coptic.ix)
    else
        true
    end
    inbounds || _outside_storage_error(optic, vn, operation)
    if !(coptic.child isa AbstractPPL.Iden)
        child = if template isa VarNamedTuples.PartialArray
            _model_argument_binding(template, AbstractPPL.ohead(coptic))
        elseif template isa AbstractArray
            _check_assigned_binding_storage(template, coptic, vn)
            VarNamedTuples.index_template(template, coptic)
        else
            VarNamedTuples.index_template(template, coptic)
        end
        # Without storage below it, a slice still bounds the next index by its shape.
        next = coptic.child
        if next isa AbstractPPL.Index &&
            VarNamedTuples._is_multiindex(array, coptic.ix...; coptic.kw...) &&
            VarNamedTuples.template_array(child) isa
            Union{NoTemplate,VarNamedTuples.SkipTemplate,Missing} &&
            all(i -> i isa Union{Integer,AbstractVector{<:Integer}}, next.ix)
            shape = CartesianIndices(VarNamedTuples._selected_index_shape(coptic.ix...))
            checkbounds(Bool, shape, next.ix...) ||
                _outside_storage_error(next, vn, operation)
        end
        _check_binding_template_bounds(child, next, vn; operation, check_fields)
    end
    return nothing
end
function _check_assigned_binding_storage(array, optic, vn)
    if checkbounds(Bool, array, optic.ix...; optic.kw...) &&
        !VarNamedTuples._is_multiindex(array, optic.ix...; optic.kw...) &&
        !isassigned(array, optic.ix...; optic.kw...)
        throw(ArgumentError("Cannot resolve `$vn`: child storage is unassigned."))
    end
    return nothing
end
@noinline function _outside_storage_error(optic, vn, operation)
    message = "Cannot $operation `$vn`: index is outside the storage at `$(_storage_address(vn, optic))`"
    throw(ArgumentError(message))
end

"""
    condition(model::Model; values...)
    condition(model::Model, values..., [template])

Return a `Model` which treats the LHS variables bound by `values` as observations: they replace
sampling and contribute to the likelihood.

See also: [`decondition`](@ref), [`conditioned`](@ref)

Bindings are stored on this model and reach LHS variables at or below their addresses,
including child addresses such as `a.x`. Outermost explicit bindings win. A binding
at a name shared with an unprefixed child binds both models. `|` follows the same rule.

Fixed bindings shadow observations. Within each layer, later bindings replace earlier
ones where they overlap. Subvariables of one LHS variable must have the same role.
A local LHS variable reads its binding at its tilde. For an argument LHS variable,
every binding replaces the argument before the body runs; observations read its current
value in the body, while fixed LHS variables reset to their bound value at the tilde.
Observe raw data under a separate name if the body transforms the argument first.

Use NamedTuples/keywords for whole top-level values, `VarName` pairs for any address
(`:x => v` abbreviates `@varname(x) => v`), or a [`VarNamedTuple`](@ref) produced by
DynamicPPL. Positional inputs and tuples apply left to right. `condition` rejects every
`AbstractDict` with `ArgumentError`; `model | dict` has no method. One positional binding
template, e.g. `@of(z = of(Array, 3))`, supplies storage for partially bound local LHS variables
without binding values (`using AbstractPPL: of, @of`).
Existing owners in the observation layer take precedence over templates; conflicts throw.
An `of` type fixed before evaluation fixes its element type: runtime bindings under
ForwardDiff/ReverseDiff need `@of(z = of(Array, typeof(m), n))` or a whole value.

Addresses and independently declared types are checked at binding time; submodel checks
wait until reached, and unused bindings are ignored. Whole argument values must also fit
the full signature (shared type constraints are checked during evaluation); whole bindings of local LHS
variables must fit their storage type. Partial values convert to the replaced
element/field type: Julia conversion errors propagate (`1.5` into `Int` raises
`InexactError`); successful but lossy conversions (`0.1` into `Float32`) raise `ArgumentError`.
A binding owns shape at its address within its layer; partial edits preserve that owner.
Binding an argument does not recompute construction-time defaults or select another method.

Bound values are not copied; the body must not mutate them, including through aliases.
Partial array bindings and removals require `Array` or Array-backed `OffsetArray`,
`ComponentArray` or `DimArray` owners along the edited path. For other arrays, bind or
decondition the whole value, or use `collect` if losing axes or metadata is acceptable.
Partial bindings take a shallow snapshot when made: the owner's container is copied one
level deep, but nested mutable values remain shared and must not be mutated either.
Bound values containing `missing` or `nothing` throw when the binding is made; use
[`decondition`](@ref) to make observations latent; see [Missing data](@ref).
Partial bindings or removals that rebuild tuple or struct owners
throw at any depth; bind or remove the enclosing owner whole. See [Binding rules](@ref).
Incomplete partial bindings into whole `missing`/`nothing` arguments throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
See [Binding rules](@ref) for the argument contract, binding contract and submodel rules,
and [Performance](@ref binding-performance) for the cost of rebuilding partially bound array arguments.

# Examples
## Simple univariate model
```jldoctest condition
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> m, x = model(); (m != 1.0 && x != 100.0)
true

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       conditioned_model = condition(model, x=100.0, m=1.0);

julia> m, x = conditioned_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only condition on `x = 100.0`.
       conditioned_model = condition(model, x = 100.0);

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true

julia> # We can also use the nicer `|` syntax.
       conditioned_model = model | (x = 100.0, );

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true
```

In the above we have specified the LHS variables to observe via keyword arguments. You can also
provide a `NamedTuple`, a `VarNamedTuple`, or `VarName` pairs. Tuples of these inputs
are applied left to right. Other inputs to `condition`, including every `AbstractDict`,
throw `ArgumentError`; `|` has no `AbstractDict` method.

For example, here we use a `VarName` pair:

```jldoctest condition
julia> conditioned_model_pair = condition(model, @varname(x) => 100.0);

julia> m, x = conditioned_model_pair(); (m != 1.0 && x == 100.0)
true

julia> # There's also an option using `|` by letting the right-hand side be a tuple
       # with elements of type `Pair{<:VarName}`, i.e. `vn => value` with `vn isa VarName`.
       conditioned_model_pairs = model | (@varname(x) => 100.0);

julia> m, x = conditioned_model_pairs(); (m != 1.0 && x == 100.0)
true
```

## Condition individual indexed LHS variables

Supply only the indices to observe; omitted LHS variables remain latent.

However, note that in this case each index must address a separate LHS variable. If we write
`m ~ MvNormal(...)`, then we cannot
condition on only `m[1]`. Partly bound LHS variables throw `ArgumentError` during evaluation.
(In principle, for some distributions this can be possible, specifically when the
distribution can be factorised into independent components, like an MvNormal with a
diagonal covariance matrix. However, this is not currently implemented.)

```jldoctest condition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> using AbstractPPL: of, @of

julia> conditioned_model = condition(model, @varname(m[2]) => 1.0, @of(m = of(Array, 2)));

julia> # `m[1]` is sampled while `m[2]` is observed.
       m = conditioned_model(); (m[1] != 1.0 && m[2] == 1.0)
true
```

Intuitively one might also expect to be able to write `model | (m[2] = 1.0, )`. You cannot
do this with a `NamedTuple` because the `VarName` `m[2]` cannot be represented as a `Symbol`
(i.e., `Symbol("m[2]")` is not the same as `@varname(m[2])`).

```jldoctest condition
julia> try
           condition(model, var"m[2]" = 1.0)
       catch err
           err isa ArgumentError
       end
true
```

Use `VarName` pairs with a binding template as above. A template supplies storage without
observing or fixing any values. It is an AbstractPPL `OfNamedTuple` type, equivalently
written `of((m=of(Array, 2),))`, and may appear anywhere among the positional inputs,
at most once per call. Keywords remain binding data; `|` does not take a template.
An existing owner within the layer being edited takes precedence; conflicting storage
throws `ArgumentError`. Template entries for arguments and names the call does not bind
are ignored. Templates use absolute names, as in `rand(model)` output: for a model
prefixed with `p`, use `@of(p = @of(z = of(Array, 3)))`. Resolve symbolic sizes before binding.
Use whole bindings for custom arrays and structs that `of` cannot describe.
Values DynamicPPL produces, such as `rand(model)` and `conditioned(model)`, are
[`VarNamedTuple`](@ref)s and can be passed back for round trips.

## Nested models

`condition` also supports the use of nested models through the use of [`to_submodel`](@ref).

A submodel tilde whose LHS is rooted at a model argument throws `ArgumentError` when
it runs. Use a local LHS variable and bind the child through its namespace, as below.

```jldoctest condition
julia> @model demo_inner() = m ~ Normal()
demo_inner (generic function with 2 methods)

julia> @model function demo_outer()
           # By default, `to_submodel` prefixes the LHS variables using the left-hand side of `~`.
           inner ~ to_submodel(demo_inner())
           return inner
       end
demo_outer (generic function with 2 methods)

julia> @model argument_outer(inner) = inner ~ to_submodel(demo_inner());

julia> try
           argument_outer(0.0)()
       catch err
           err isa ArgumentError
       end
true

julia> model = demo_outer();

julia> model() ≠ 1.0
true

julia> # To condition the LHS variable inside `demo_inner` we need to refer to it as `inner.m`.
       conditioned_model = condition(model, @varname(inner.m) => 1.0);

julia> conditioned_model()
1.0

julia> # Binding `inner` supplies a submodel namespace. For example, this will work:
       conditioned_model2 = condition(model; inner=(m=1.0,));

julia> conditioned_model2()
1.0

julia> # Conditioning a submodel's return value is not supported.
       conditioned_model_fail = condition(model; inner="something else");

julia> try
           conditioned_model_fail()
       catch err
           err isa ArgumentError
       end
true
```
"""
function AbstractPPL.condition(model::Model, inputs...; values...)
    inputs = isempty(values) ? inputs : (inputs..., NamedTuple(values))
    return _bind_inputs(Condition, model, inputs)
end

# Binding preparation consults owners in the layer being edited.
_binding_layer(::Type{Condition}, values) = _observation_values(values)
_binding_layer(::Type{Fix}, values) = _fixed_values(values)
function _binding_layer_model(::Type{R}, model) where {R}
    layer = _binding_layer(R, model.values)
    values = model.values isa LocalModelValues ? LocalModelValues(layer) : layer
    return _reconstruct_model(model; values)
end

_binding_owner(::Type{R}, value) where {R} = NoTemplate()
_binding_owner(::Type{R}, value::ModelValue) where {R} = value.value
_binding_owner(::Type{R}, value::ModelValueTree) where {R} = _model_data(value)
function _binding_owner(::Type{R}, value::VarNamedTuples.PartialArray) where {R}
    value.data isa VarNamedTuples.GrowableArray && return NoTemplate()
    # Selection unwraps the role tags while retaining storage for unbound entries.
    selected = _select_model_node(R, value)
    return selected isa VarNamedTuples.PartialArray ? selected.data : selected
end

function _prepare_local_binding_type(::Type{R}, previous, update, vn, addresses) where {R}
    if previous isa Union{
        ModelValue{<:Any,<:Union{NamedTuple,VarNamedTuple}},ModelValueTree{<:NamedTuple}
    } && any(address -> last(address) && first(address) == vn, addresses)
        previous = _submodel_namespace(previous)
    end
    owner = _binding_owner(R, previous)
    owner isa NoTemplate &&
        return _prepare_local_binding_children(R, previous, update, vn, addresses)
    if update isa ModelValue
        update.value isa typeof(owner) || throw(
            ArgumentError(
                "Bound value at `$vn` must be an instance of template type $(typeof(owner)); supplied $(typeof(update.value)).",
            ),
        )
        return update
    end
    return _prepare_argument_fields(owner, update, vn)
end
function _prepare_local_binding_children(
    ::Type{R}, previous, update, vn, addresses
) where {R}
    return update
end
function _prepare_local_binding_children(
    ::Type{R},
    previous,
    updates::Union{VarNamedTuple,VarNamedTuples.PartialArray},
    vn,
    addresses,
) where {R}
    previous === nothing && return updates
    # Namespace nodes supply no storage themselves; owners live at their children.
    return _fold_model_indices(copy(updates), updates) do result, update, optic, storage
        child = _model_argument_binding(previous, optic)
        prepared = _prepare_local_binding_type(
            R, child, update, AbstractPPL.append_optic(vn, optic), addresses
        )
        VarNamedTuples._setindex_optic!!(
            result, prepared, optic, storage, VarNamedTuples.AllowAll()
        )
    end
end
function _prepare_local_binding_types(::Type{R}, model, values) where {R}
    prefix = _model_prefix(model)
    local_values = if model.values isa LocalModelValues || prefix === nothing
        values
    else
        _submodel_values(values, prefix)
    end
    local_previous = _submodel_values(model, nothing)
    for name in keys(local_values.data)
        name in _args_on_lhs(model) && continue
        previous = get(local_previous.data, name, nothing)
        previous === nothing && continue
        update = _prepare_local_binding_type(
            R,
            previous,
            local_values.data[name],
            VarName{name}(),
            _lhs_addresses(_binding_metadata(model)),
        )
        local_values = VarNamedTuple(
            merge(local_values.data, NamedTuple{(name,)}((update,)))
        )
    end
    return if model.values isa LocalModelValues || prefix === nothing
        local_values
    else
        _prefix_values(
            local_values,
            prefix,
            _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
        )
    end
end

_binding_templates(values) = ()
_binding_templates(values::ModelBindingLayers) = values.templates
_binding_templates(values::LocalModelValues) = _binding_templates(values.values)
function _with_binding_templates(values, templates::Tuple)
    isempty(templates) && return values
    return ModelBindingLayers(
        _observation_values(values),
        _fixed_values(values),
        _model_values(values),
        _fixed_owners(values),
        _removals(Condition, values),
        _removals(Fix, values),
        templates,
    )
end
function _with_binding_templates(values::LocalModelValues, templates::Tuple)
    return LocalModelValues(
        _with_binding_templates(values.values, templates), values.owners
    )
end
function _prefix_binding_template(t::ModelBindingTemplate{R}, prefix) where {R}
    return ModelBindingTemplate{R}(maybe_prefix(t.name, prefix), t.storage)
end
function _binding_template_entries(::Type{R}, schema::NamedTuple, prefix=nothing) where {R}
    return mapreduce((a, b) -> (a..., b...), pairs(schema); init=()) do (name, storage)
        address = maybe_prefix(VarName{name}(), prefix)
        if storage isa NamedTuple
            _binding_template_entries(R, storage, address)
        else
            (ModelBindingTemplate{R}(address, storage),)
        end
    end
end
function _surviving_binding_templates(previous, values, ::Type{R}, replaced=()) where {R}
    return filter(_binding_templates(previous)) do t
        !(t isa ModelBindingTemplate{R}) && return true
        return !any(vn -> subsumes(vn, t.name), replaced) &&
               _has_removable(R, _binding_layer(R, values), t.name)
    end
end
function _submodel_binding_templates(model, prefix)
    prefix = _model_value_varname(model.values, prefix, _model_prefix(model))
    return mapreduce((a, b) -> (a..., b...), _binding_templates(model.values); init=()) do t
        prefix === nothing && return (t,)
        subsumes(prefix, t.name) || return ()
        return (_unprefix_binding_template(t, prefix),)
    end
end
function _unprefix_binding_template(t::ModelBindingTemplate{R}, prefix) where {R}
    return ModelBindingTemplate{R}(AbstractPPL.unprefix(t.name, prefix), t.storage)
end
struct ConflictingBindingTemplates end
function _template_schema(schema, t::ModelBindingTemplate{R}, values) where {R}
    binding = _model_argument_binding(
        _binding_layer(R, values), AbstractPPL.varname_to_optic(t.name)
    )
    owner = _binding_owner(R, binding)
    storage = if owner isa NoTemplate || _same_schema_storage(owner, t.storage)
        t.storage
    else
        ConflictingBindingTemplates()
    end
    previous = haskey(schema, t.name) ? schema[t.name] : storage
    storage =
        _same_schema_storage(previous, storage) ? storage : ConflictingBindingTemplates()
    return setindex!!(schema, storage, t.name)
end
function _filter_binding_schema(model, schema::NamedTuple, prefix=nothing; owned_here::Bool)
    metadata = _binding_metadata(model)
    if _lhs_names(metadata) === nothing || !any(last, _lhs_addresses(metadata))
        return owned_here ? schema : (;)
    end
    names = filter(keys(schema)) do name
        address = maybe_prefix(VarName{name}(), prefix)
        if schema[name] isa NamedTuple
            return !isempty(_filter_binding_schema(model, schema[name], address; owned_here))
        end
        owned =
            _lhs_names(metadata) === nothing ||
            any(_lhs_addresses(metadata)) do (lhs, submodel)
                !submodel && (subsumes(lhs, address) || subsumes(address, lhs))
            end
        return owned == owned_here
    end
    return NamedTuple{names}(
        map(names) do name
            value = schema[name]
            if value isa NamedTuple
                _filter_binding_schema(
                    model, value, maybe_prefix(VarName{name}(), prefix); owned_here
                )
            else
                value
            end
        end,
    )
end
function _prepare_inherited_templates(model, values, templates)
    isempty(templates) && return values
    result = VarNamedTuple()
    arguments = map(unsplat_symbol, keys(merge(model.args, model.defaults)))
    for role in (Condition, Fix)
        schema = VarNamedTuple()
        for t in templates
            t isa ModelBindingTemplate{role} || continue
            AbstractPPL.getsym(t.name) in arguments && continue
            schema = _template_schema(schema, t, values)
        end
        schema = _filter_binding_schema(model, _model_data(schema); owned_here=true)
        selected = _select_model_values(role, values)
        prepared = if isempty(schema)
            selected
        else
            _prepare_schema_input(role, model, selected, schema)
        end
        result = _merge_model_values(result, _tag_model_values(role, prepared))
    end
    return result
end

# Flatten ordered input groups without interpreting keyword values as templates.
function _binding_inputs(values::Tuple)
    return mapreduce(_binding_inputs, (a, b) -> (a..., b...), values; init=())
end
_binding_inputs(value) = (value,)
_binding_inputs((name, value)::Pair{Symbol}) = (VarName{name}() => value,)

function _bind_ordered_inputs(::Type{R}, model, values) where {R}
    bound = foldl((m, v) -> _bind_model(R, m, v), values; init=model)
    _check_shared_latent_storage(bound)
    return bound
end

function _bind_inputs(::Type{R}, model::Model, inputs::Tuple) where {R}
    inputs = _binding_inputs(inputs)
    schemas = filter(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, inputs)
    length(schemas) <= 1 ||
        throw(ArgumentError("At most one binding template is allowed per call."))
    values = filter(x -> !(x isa Type{<:AbstractPPL.OfNamedTuple}), inputs)
    isempty(schemas) && return _bind_ordered_inputs(R, model, values)
    model = _materialize_argument_values(model)
    schema = VarNamedTuples.materialize_template(only(schemas))
    model = _record_binding_template(model, schema)
    schema = _local_binding_schema(model, schema, values)
    arguments = map(unsplat_symbol, keys(merge(model.args, model.defaults)))
    schema = NamedTuple{filter(name -> !(name in arguments), keys(schema))}(schema)
    schema = _bound_binding_schema(model, schema, values)
    deferred = _filter_binding_schema(model, schema; owned_here=false)
    schema = _filter_binding_schema(model, schema; owned_here=true)
    for value in values
        model = _bind_schema_input(R, model, value, schema)
    end
    _check_shared_latent_storage(model)
    isempty(deferred) && return model
    templates = _binding_template_entries(
        R, deferred, model.values isa LocalModelValues ? nothing : _model_prefix(model)
    )
    bindings = _with_binding_templates(
        model.values, _merge_binding_templates(_binding_templates(model.values), templates)
    )
    return _reconstruct_model(model; values=bindings)
end
function _bound_binding_schema(model, schema::NamedTuple, values, prefix=nothing)
    selected = map(keys(schema)) do name
        address = maybe_prefix(VarName{name}(), prefix)
        storage = schema[name]
        if storage isa NamedTuple
            _bound_binding_schema(model, storage, values, address)
        else
            root = _model_value_varname(model.values, address, _model_prefix(model))
            any(v -> _input_binds_root(v, root, model), values) ? storage : nothing
        end
    end
    selected_schema = NamedTuple{keys(schema)}(selected)
    names = filter(keys(schema)) do name
        storage = selected_schema[name]
        storage !== nothing && !(storage isa NamedTuple && isempty(storage))
    end
    return NamedTuple{names}(selected_schema)
end
_same_binding_template_name(a, b) = false
function _same_binding_template_name(
    a::ModelBindingTemplate{R}, b::ModelBindingTemplate{R}
) where {R}
    return a.name == b.name
end
function _merge_binding_templates(previous::Tuple, updates::Tuple)
    return foldl(updates; init=previous) do templates, update
        index = findfirst(templates) do t
            _same_binding_template_name(t, update)
        end
        index === nothing && return (templates..., update)
        return Base.setindex(
            templates, _merge_binding_template(templates[index], update), index
        )
    end
end
function _merge_binding_template(a::ModelBindingTemplate{R}, b) where {R}
    storage = if _same_schema_storage(a.storage, b.storage)
        a.storage
    else
        ConflictingBindingTemplates()
    end
    return ModelBindingTemplate{R}(a.name, storage)
end
function _binding_template_names(template::NamedTuple, prefix=nothing)
    return mapreduce((a, b) -> (a..., b...), pairs(template); init=()) do (name, storage)
        address = maybe_prefix(VarName{name}(), prefix)
        storage isa NamedTuple ? _binding_template_names(storage, address) : (address,)
    end
end
function _record_binding_template(model::Model, template)
    prefix = _model_prefix(model)
    names = map(_binding_template_names(template)) do name
        local_name = prefix === nothing || (name != prefix && subsumes(prefix, name))
        address =
            local_name && prefix !== nothing ? AbstractPPL.unprefix(name, prefix) : name
        return (local_name, address)
    end
    metadata = _with_binding_template_names(_binding_metadata(model), names)
    return Model{requires_threadsafe(model)}(
        model.f,
        model.args,
        model.defaults,
        model.context,
        model.values;
        args_on_lhs=metadata,
    )
end

function _local_binding_schema(model, schema, values)
    prefix = _model_prefix(model)
    prefix === nothing && return schema
    local_schema = _schema_namespace(schema, AbstractPPL.varname_to_optic(prefix))
    for name in keys(schema)
        root = maybe_prefix(VarName{name}(), prefix)
        if (name !== AbstractPPL.getsym(prefix) || !(local_schema isa NamedTuple)) &&
            any(v -> _input_binds_root(v, root, model), values)
            throw(
                ArgumentError(
                    "Binding template entry `$name` uses a local name; use the absolute name `$root`.",
                ),
            )
        end
    end
    return local_schema isa NamedTuple ? local_schema : NamedTuple()
end

_schema_namespace(value, ::AbstractPPL.Iden) = value
_schema_namespace(value, ::AbstractPPL.AbstractOptic) = nothing
function _schema_namespace(value::NamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return haskey(value, S) ? _schema_namespace(value[S], optic.child) : nothing
end

function _schema_namespace(value::VarNamedTuple, optic::AbstractPPL.Property)
    return _schema_namespace(value.data, optic)
end

function _binding_display_name(model, vn)
    return model.values isa LocalModelValues ? maybe_prefix(vn, _model_prefix(model)) : vn
end
function _schema_binding_address(model, vn)
    vn = _expand_cartesian(vn)
    prefix = _model_prefix(model)
    if !(model.values isa LocalModelValues) &&
        prefix !== nothing &&
        AbstractPPL.getsym(prefix) === AbstractPPL.getsym(vn)
        template = _apply_prefix_template(_model_prefix_template(model), NoTemplate())
        return _concretize_prefix(
            vn, template; depth=optic_skip_length(AbstractPPL.getoptic(prefix))
        )
    end
    return vn
end
function _input_binds_root(value::Pair{<:VarName}, root, model)
    address = _schema_binding_address(model, first(value))
    subsumes(root, address) && return true
    subsumes(address, root) || return false
    storage = last(value)
    storage isa Union{NamedTuple,VarNamedTuple} || return true
    relative = AbstractPPL.unprefix(root, address)
    return _schema_namespace(storage, AbstractPPL.varname_to_optic(relative)) !== nothing
end
function _input_binds_root(value::NamedTuple, root, model)
    return any(pairs(value)) do (name, data)
        _input_binds_root(VarName{name}() => data, root, model)
    end
end
function _input_binds_root(value::VarNamedTuple, root, model)
    return any(pair -> _input_binds_root(pair, root, model), pairs(value))
end
_input_binds_root(value, root, model) = false

_schema_storage(value::ModelValue) = value.value
_schema_storage(value::ModelValueTree) = _model_data(value)
_schema_storage(value) = value

function _same_schema_storage(a::AbstractArray, b::AbstractArray)
    return eltype(a) === eltype(b) && axes(a) == axes(b)
end
function _same_schema_storage(a::NamedTuple, b::NamedTuple)
    return keys(a) == keys(b) && all(_same_schema_storage(x, y) for (x, y) in zip(a, b))
end
_same_schema_storage(a, b) = typeof(a) === typeof(b)
function _same_schema_storage(a::VarNamedTuples.PartialArray, b::AbstractArray)
    return _same_schema_storage(a.data, b)
end
function _same_schema_storage(a::VarNamedTuple, b::NamedTuple)
    return all(
        name -> haskey(b, name) && _same_schema_storage(a.data[name], b[name]), keys(a.data)
    )
end

function _schema_has_address(model, schema::NamedTuple, vn::VarName)
    local_name = _local_removal_name(model, vn; operation="bind")
    local_name === nothing && return !isempty(schema)
    return _schema_has_optic(schema, AbstractPPL.varname_to_optic(local_name))
end
_schema_has_optic(value, ::AbstractPPL.AbstractOptic) = true
_schema_has_optic(value::NamedTuple, ::AbstractPPL.Iden) = true
function _schema_has_optic(value::NamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return haskey(value, S) && _schema_has_optic(value[S], optic.child)
end

_check_deferred_schema(storage, root, input, model) = nothing
function _check_deferred_schema(::ConflictingBindingTemplates, root, input, model)
    throw(
        ArgumentError(
            "Binding template storage for `$(_binding_display_name(model, root))` conflicts with its existing owner.",
        ),
    )
end
function _check_deferred_schema(storage::NamedTuple, root, input, model)
    for (name, child) in pairs(storage)
        child_root = AbstractPPL.append_optic(root, AbstractPPL.Property{name}())
        _input_binds_root(input, child_root, model) || continue
        _check_deferred_schema(child, child_root, input, model)
    end
    return nothing
end

function _check_schema_owner(owner, storage, root, input, model)
    _same_schema_storage(owner, storage) || throw(
        ArgumentError(
            "Binding template storage for `$root` conflicts with its existing owner."
        ),
    )
    return nothing
end
function _check_schema_owner(
    owner::Union{NamedTuple,VarNamedTuple}, storage::NamedTuple, root, input, model
)
    for (name, child_storage) in pairs(storage)
        child_root = AbstractPPL.append_optic(root, AbstractPPL.Property{name}())
        _input_binds_root(input, child_root, model) || continue
        data = owner isa VarNamedTuple ? owner.data : owner
        haskey(data, name) || continue
        _check_schema_owner(
            _schema_storage(data[name]), child_storage, child_root, input, model
        )
    end
    return nothing
end

function _bind_schema_input(::Type{R}, model, input, schema) where {R}
    layer_model = _binding_layer_model(R, model)
    prepared = _prepare_schema_input(R, model, input, schema, layer_model)
    return _bind_model(R, model, prepared; preparation_model=layer_model)
end

function _prepare_schema_input(
    ::Type{R}, model, input, schema, layer_model=_binding_layer_model(R, model)
) where {R}
    layer = _model_values(layer_model.values)
    templates = VarNamedTuple()
    for (name, storage) in pairs(schema)
        root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        _input_binds_root(input, root, model) || continue
        _check_deferred_schema(storage, root, input, model)
        previous = _model_argument_binding(layer, AbstractPPL.varname_to_optic(root))
        if previous !== nothing
            owner = if previous isa Union{VarNamedTuples.PartialArray,VarNamedTuple}
                _select_model_node(R, previous)
            else
                _schema_storage(previous)
            end
            _check_schema_owner(owner, storage, root, input, model)
            storage =
                owner isa Union{VarNamedTuple,VarNamedTuples.PartialArray} ? storage : owner
        end
        templates = templated_setindex!!(
            templates, storage, root, _binding_template(model, VarNamedTuple(), root)
        )
    end
    is_pair = input isa Pair{<:VarName}
    plain = is_pair ? VarNamedTuple() : _make_condfix_values(layer_model, input)
    entries = is_pair ? (input,) : pairs(plain)
    prepared = VarNamedTuple()
    for (vn, value) in entries
        vn = _schema_binding_address(model, vn)
        uses_schema = _schema_has_address(model, schema, vn)
        if !uses_schema
            is_pair && return _make_condfix_values(layer_model, input)
            prepared = templated_setindex!!(
                prepared, value, vn, _binding_template(model, plain, vn)
            )
            continue
        end
        template = _binding_template(model, templates, vn)
        optic = AbstractPPL.getoptic(vn)
        _check_binding_template_bounds(template, optic, vn)
        value = _convert_binding_template(value, template, optic, vn)
        prepared = templated_setindex!!(prepared, value, vn, template)
    end
    return prepared
end

function _convert_binding_template(value, template, ::AbstractPPL.Iden, vn)
    template isa Union{NoTemplate,VarNamedTuples.SkipTemplate} && return value
    value isa typeof(template) || throw(
        ArgumentError(
            "Bound value at `$vn` must be an instance of template type $(typeof(template)); supplied $(typeof(value)).",
        ),
    )
    _same_schema_storage(value, template) || throw(
        ArgumentError("Bound value at `$vn` conflicts with its binding template storage."),
    )
    return value
end
function _binding_child_template(template, optic)
    return VarNamedTuples.maybe_index_template(template, optic)
end
function _binding_child_template(template, ::AbstractPPL.Property{S}) where {S}
    return VarNamedTuples.SharedGetProperty{S}()(template)
end
function _convert_binding_template(value, template, optic::AbstractPPL.AbstractOptic, vn)
    template isa NoTemplate && return value
    head = AbstractPPL.ohead(optic)
    head =
        head isa AbstractPPL.Index ? AbstractPPL.concretize_top_level(head, template) : head
    # VarNamedTuple nodes describe namespaces, not fields of the bound value.
    # Descend to the local root before applying whole-value template checks.
    if optic.child isa AbstractPPL.Iden && !(template isa VarNamedTuple)
        return _convert_partial_argument_binding(
            ModelValue{Condition}(value), template, head, vn
        ).value
    end
    child = _binding_child_template(template, head)
    return _convert_binding_template(value, child, optic.child, vn)
end

function _bind_model(::Type{R}, model::Model, values; preparation_model=model) where {R}
    model = _materialize_argument_values(model)
    preparation_model = _binding_layer_model(
        R, _materialize_argument_values(preparation_model)
    )
    values = _make_condfix_values(preparation_model, values)
    _check_bound_placeholders(R, values)
    values = _tag_model_values(R, values)
    values = _check_argument_bindings(preparation_model, values)
    _check_binding_addresses(model, values)
    values = _prepare_local_binding_types(R, preparation_model, values)
    new_addresses = Tuple(keys(values))
    observations = _observation_values(model.values)
    fixed_values = _fixed_values(model.values)
    values = if R === Fix
        new_owners = Tuple(keys(values))
        fixed_owners = (
            filter(
                owner -> !any(new -> subsumes(new, owner), new_owners),
                _fixed_owners(model.values),
            )...,
            new_owners...,
        )
        # Original arguments supply storage where this layer has no owner yet.
        owners = _argument_defaults(
            merge(model.args, model.defaults), Val(_args_on_lhs(model))
        )
        owners = _prune_model_bindings(
            VarNamedTuple(
                map(owners.data) do owner
                    owner.value isa Union{Missing,Nothing} ? NoModelBinding() : owner
                end,
            ),
        )
        if !(model.values isa LocalModelValues) && _model_prefix(model) !== nothing
            owners = _prefix_values(
                owners,
                _model_prefix(model),
                _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
            )
        end
        owners = VarNamedTuple(merge(owners.data, fixed_values.data))
        fixed_values = _remove_model_values(Condition, _merge_model_values(owners, values))
        ModelBindingLayers(observations, fixed_values, fixed_owners)
    elseif isempty(fixed_values)
        _merge_model_values(observations, values)
    else
        ModelBindingLayers(
            _merge_model_values(observations, values),
            fixed_values,
            _fixed_owners(model.values),
        )
    end
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    values = _replace_removals(model.values, values, R, new_addresses)
    values = _with_binding_templates(
        values, _surviving_binding_templates(model.values, values, R, new_addresses)
    )
    return _reconstruct_model(model; values)
end
function AbstractPPL.condition(model::Model; values...)
    return condition(model, NamedTuple(values))
end

"""
    _make_condfix_values(model, values)

Convert normalised binding values to a `VarNamedTuple`.
Input ordering, keyword arguments and templates are handled by the binding entry points.
"""
function _make_condfix_values(model, values)
    throw(
        ArgumentError(
            "Bindings require a VarNamedTuple, NamedTuple, or VarName pairs (or an ordered tuple of these); use keywords for whole values or @varname(x) => value for an address. AbstractDict inputs are not supported.",
        ),
    )
end
_make_condfix_values(model, values::NamedTuple) = VarNamedTuple(values)
function _make_condfix_values(model, values::VarNamedTuple)
    for vn in keys(values)
        _binding_template(model, _model_values(model.values), vn) isa NoTemplate &&
            _check_local_property_index(model, vn)
        _check_partial_binding(
            _model_values(model.values), AbstractPPL.varname_to_optic(vn)
        )
    end
    return values
end

function _check_bound_placeholders(::Type{R}, values::VarNamedTuple) where {R}
    (_contains_missing(values) || _contains_nothing(values)) || return nothing
    vn = first(
        vn for
        (vn, value) in pairs(values) if _contains_missing(value) || _contains_nothing(value)
    )
    remove = R === Fix ? "unfix" : "decondition"
    throw(
        ArgumentError(
            "Cannot bind `$vn` to a value containing `missing` or `nothing`; bindings " *
            "cannot hold placeholders. To make it latent, leave it unbound or use " *
            "`$remove(model, @varname($vn))`.",
        ),
    )
end

_binding_storage(value) = value
_binding_storage(::Union{Nothing,Missing,Number}) = NoTemplate()
_binding_storage(value::ModelValue) = _binding_storage(value.value)
_binding_storage(value::ModelValueTree) = _binding_storage(_model_data(value))
_binding_storage(value::NamedTuple) = map(_binding_storage, value)
_binding_storage(value::VarNamedTuple) = VarNamedTuple(map(_binding_storage, value.data))
function _binding_storage(value::VarNamedTuples.PartialArray)
    # Inferred, growable storage has no owner to constrain subsequent indices.
    value.data isa VarNamedTuples.GrowableArray && return NoTemplate()
    return VarNamedTuples._map_values_recursive!!(_binding_storage, copy(value))
end
# Whole bindings own their shape even inside growable storage or its slices.
_binding_address_storage(value, optic) = _binding_storage(value)
function _binding_address_storage(
    ::VarNamedTuples.PartialArray{<:Any,<:Any,<:SubArray},
    ::Union{AbstractPPL.Iden,AbstractPPL.Property},
)
    return NoTemplate()
end
function _binding_address_storage(
    value::VarNamedTuples.PartialArray, optic::AbstractPPL.Index
)
    value.data isa Union{VarNamedTuples.GrowableArray,SubArray} &&
        all(i -> i isa Union{Integer,AbstractVector{<:Integer}}, optic.ix) ||
        return _binding_storage(value)
    child = _model_argument_binding(value, AbstractPPL.ohead(optic))
    child === nothing && return NoTemplate()
    return VarNamedTuples.SkipTemplate{1}(_binding_address_storage(child, optic.child))
end
function _binding_address_storage(
    value::VarNamedTuple, optic::AbstractPPL.Property{S}
) where {S}
    storage = _binding_storage(value)
    haskey(value.data, S) || return storage
    child = _binding_address_storage(value.data[S], optic.child)
    return VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((child,))))
end

@generated function _binding_address_template(
    model::Model, values, address; operation="bind"
)
    fields = (
        fieldnames(fieldtype(model, :args))..., fieldnames(fieldtype(model, :defaults))...
    )
    arguments = map(fields) do stored_name
        name = unsplat_symbol(stored_name)
        quote
            root = _model_value_varname(model.values, $(VarName{name}()), _model_prefix(model))
            if subsumes(root, address)
                argument = merge(model.args, model.defaults)[$(QuoteNode(stored_name))]
                previous = _model_argument_binding(
                    values, AbstractPPL.varname_to_optic(root)
                )
                template = if previous === nothing
                    argument
                else
                    prepare_model_argument(previous, argument)
                end
                _check_partial_binding(
                    ModelValue{Condition}(template),
                    AbstractPPL._unprefix_optic(
                        AbstractPPL.getoptic(address), AbstractPPL.getoptic(root)
                    ),
                    AbstractPPL.varname_to_optic(root);
                    operation,
                )
                return VarNamedTuples.make_leaf(
                    _binding_storage(template),
                    AbstractPPL.getoptic(root),
                    _binding_template(model, values, root),
                )
            end
        end
    end
    return quote
        $(arguments...)
        return _binding_address_storage(
            _binding_template(model, values, address), AbstractPPL.getoptic(address)
        )
    end
end

function _make_condfix_values(model, pair::Pair{<:VarName})
    vn, template = _check_binding_address(model, _model_values(model.values), first(pair))
    return templated_setindex!!(VarNamedTuple(), last(pair), vn, template)
end

# Property LHS addresses describe named binding storage even when local Julia storage
# does not exist yet. Use that template only for spelling checks, never as a shape owner.
function _check_local_property_index(model, vn; operation="bind")
    local_name = _local_removal_name(model, vn; operation)
    local_name === nothing && return nothing
    optic = AbstractPPL.getoptic(local_name)
    optic isa AbstractPPL.Index || return nothing
    name = AbstractPPL.getsym(local_name)
    name in _args_on_lhs(model) && return nothing
    fields = ()
    for (address, submodel) in _lhs_addresses(_binding_metadata(model))
        AbstractPPL.getsym(address) === name || continue
        head = AbstractPPL.getoptic(address)
        (submodel || !(head isa AbstractPPL.Property)) && return nothing
        field = _binding_property_name(head)
        field in fields || (fields = (fields..., field))
    end
    isempty(fields) && return nothing
    template = NamedTuple{fields}(map(_ -> NoTemplate(), fields))
    # Keep the namespace, replacing only the local root's index path.
    root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
    return _check_namedtuple_index(
        template, optic, AbstractPPL.varname_to_optic(root); operation
    )
end

@inline function _check_binding_address(model, values, vn; operation="bind")
    vn = _schema_binding_address(model, vn)
    local_name = _local_removal_name(model, vn; operation)
    metadata = _binding_metadata(model)
    names = _lhs_names(metadata)
    if operation == "remove" &&
        local_name !== nothing &&
        names !== nothing &&
        !_has_unprefixed_submodel(metadata) &&
        AbstractPPL.getsym(local_name) ∉ names
        _binding_name_error(vn, operation)
    end
    for stored_name in keys(model.defaults)
        is_splat_symbol(stored_name) || continue
        argument = unsplat_symbol(stored_name)
        root = _model_value_varname(model.values, VarName{argument}(), _model_prefix(model))
        if root != vn && subsumes(root, vn)
            if operation == "remove"
                optic = AbstractPPL._unprefix_optic(
                    AbstractPPL.getoptic(vn), AbstractPPL.getoptic(root)
                )
                if optic isa AbstractPPL.Index{Tuple{Symbol}}
                    vn = AbstractPPL.append_optic(
                        root, AbstractPPL.Property{only(optic.ix)}(optic.child)
                    )
                end
                continue
            end
            message = "Entries of keyword-splat argument `$argument` cannot be bound; replace the whole argument with `condition` or `fix` instead."
            throw(ArgumentError(message))
        end
    end
    _check_partial_binding(values, AbstractPPL.varname_to_optic(vn); operation)
    template = _binding_address_template(model, values, vn; operation)
    template isa NoTemplate && _check_local_property_index(model, vn; operation)
    _check_partial_binding(
        ModelValue{Condition}(template),
        AbstractPPL.getoptic(vn),
        AbstractPPL.Property{AbstractPPL.getsym(vn)}();
        operation,
    )
    check_fields =
        operation == "remove" &&
        !any(_lhs_addresses(metadata)) do (address, submodel)
            submodel && local_name !== nothing && subsumes(address, local_name)
        end
    _check_binding_template_bounds(
        template, AbstractPPL.getoptic(vn), vn; operation, check_fields
    )
    return vn, template
end
function _binding_template(
    model, templates::VarNamedTuple, vn::VarName; prefix=_model_prefix(model)
)
    template = get(templates.data, AbstractPPL.getsym(vn), NoTemplate())
    if template isa NoTemplate &&
        prefix !== nothing &&
        AbstractPPL.getsym(vn) === AbstractPPL.getsym(prefix)
        return _apply_prefix_template(_model_prefix_template(model), NoTemplate())
    end
    return template
end

"""
    decondition(model::Model)
    decondition(model::Model, [DynamicPPL.Recursive()], names...)

Remove this model's conditioned bindings at `names...`, or all conditioned bindings
if no names are supplied.

Unlike [`unfix`](@ref), `decondition(m, :x)` removes explicit and argument-supplied
observations, even beneath a fixed binding. `x` becomes latent unless still fixed.
A latent argument retains its old value before its tilde; its sampled value replaces
that value at the tilde and is used by subsequent body statements.

A name matches when it equals, contains, or is contained in a stored binding's address.
NamedTuple indices are rejected: use `x.a` instead of `x[1]` or `x[:a]`.
Tuple observations and complete local LHS variables keep integer indices.
Only the matching conditioned parts are removed. Removing an already latent or only fixed
LHS variable is a no-op. A name with no LHS variable throws `ArgumentError` when the model
can decide, using the same address checks as bindings: argument fields, storage bounds,
and the model prefix. `check_model` warns about recursive removals that no reached model uses; a local
removal at a child address that names nothing is a silent no-op.

By default, only bindings stored on this model are removed, at any address. This includes
a child address bound on this model; the child's own binding there then
applies again, because bindings at one address resolve outermost first. With
`DynamicPPL.Recursive()`, removal also reaches enclosed models, including argument-supplied
observations and runtime bindings. With no names it clears the observation layer at every depth. Fixed bindings remain.
Removing the same valid address twice is a no-op, both at the call and at evaluation.
`check_model` warns about recursive removals unused by all reached models.

The removal belongs to this model, moves with `prefix`, and cannot remove an enclosing
model's bindings. Its next observation at that address replaces the removal. Removals hold
no value or shape and are omitted from `conditioned`.

Named recursive removals shared by an own LHS and an unprefixed submodel throw; prefix the
child or remove without `Recursive()`. See [Binding rules](@ref).

Removal preserves the shape of the owner in the observation layer; it does not restore
an observation overwritten by an earlier [`condition`](@ref).

# Examples
```jldoctest decondition
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> conditioned_model = condition(demo(), m = 1.0, x = 10.0);

julia> conditioned_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `decondition`.
       model = decondition(conditioned_model, @varname(m));

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` also accepts symbols, although VarNames are preferable for
       # type stability reasons.
       model = decondition(conditioned_model, :m);

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` multiple at once:
       (m, x) = decondition(conditioned_model, :m, :x)(); (m ≠ 1.0 && x ≠ 10.0)
true

julia> # `decondition` without any symbols will `decondition` all LHS variables.
       (m, x) = decondition(model)(); (m ≠ 1.0 && x ≠ 10.0)
true
```

Part of a whole binding can be deconditioned when that part is a separate LHS variable.

```jldoctest decondition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> conditioned_model = condition(model, @varname(m) => [1.0, 2.0]);

julia> conditioned_model()
2-element Vector{Float64}:
 1.0
 2.0

julia> deconditioned_model = decondition(conditioned_model, @varname(m[1]));

julia> m = deconditioned_model(); (m[1] != 1.0 && m[2] == 2.0)
true
```
"""
function AbstractPPL.decondition(model::Model, syms::Union{Symbol,VarName}...)
    return _local_remove(Condition, model, syms)
end

function _check_removal_addresses(values, names)
    for vn in names
        _check_partial_binding(values, AbstractPPL.varname_to_optic(vn); operation="remove")
    end
    return nothing
end

function _remove_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return _prune_model_bindings(_remove_model_binding(R, values, AbstractPPL.Iden()))
end

@generated function _prune_model_bindings(values::VarNamedTuple{names,T}) where {names,T}
    if all(t -> t <: ModelValue || t <: NoModelBinding, fieldtypes(T))
        kept = Tuple(
            name for (name, t) in zip(names, fieldtypes(T)) if !(t <: NoModelBinding)
        )
        fields = map(name -> :(values.data.$name), kept)
        return :(VarNamedTuple(NamedTuple{$kept}(($(fields...),))))
    end
    return :(_prune_model_bindings_recursive(values))
end
function _prune_model_bindings_recursive(values::VarNamedTuple)
    return mapfoldl(
        identity,
        function (kept, pair)
            vn, value = pair
            return if value isa NoModelBinding
                kept
            else
                templated_setindex!!(kept, value, vn, values.data[AbstractPPL.getsym(vn)])
            end
        end,
        values;
        init=VarNamedTuple(),
    )
end

function _remove_model_values(::Type{R}, values::VarNamedTuple, names) where {R}
    # Element names `x[i]`/`x[i, j]` under a whole array binding are removed after the
    # prune, in one pass over the array instead of one table edit per name. Without
    # element names, `()` keeps the other removals inferable.
    elements = if any(vn -> _is_element_index(AbstractPPL.getoptic(vn)), names)
        _element_symbols(R, values, names)
    else
        ()
    end
    # Copy each top-level array once per call, not once per name.
    owned = Symbol[]
    for vn in names
        AbstractPPL.getsym(vn) in elements && continue
        if _model_argument_binding(values, AbstractPPL.varname_to_optic(vn)) === nothing
            values = mapfoldl(
                identity,
                function (remaining, pair)
                    return if subsumes(vn, pair.first)
                        _remove_model_binding(
                            R, remaining, AbstractPPL.varname_to_optic(pair.first)
                        )
                    else
                        remaining
                    end
                end,
                values;
                init=values,
            )
        else
            sym, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
            previous = values.data[sym]
            if sym in owned &&
                previous isa VarNamedTuples.PartialArray &&
                optic isa AbstractPPL.Index
                child = _remove_model_binding(R, previous, optic, true)
                child === previous || (
                    values = VarNamedTuple(merge(values.data, NamedTuple{(sym,)}((child,))))
                )
            else
                values = _remove_model_binding(R, values, AbstractPPL.varname_to_optic(vn))
                values.data[sym] === previous || push!(owned, sym)
            end
        end
    end
    values = _prune_model_bindings(values)
    for sym in elements
        child = _remove_model_elements(values.data[sym], sym, names)
        values = VarNamedTuple(
            if child isa NoModelBinding
                Base.structdiff(values.data, NamedTuple{(sym,)})
            else
                merge(values.data, NamedTuple{(sym,)}((child,)))
            end,
        )
    end
    return values
end

function _element_symbols(::Type{R}, values, names) where {R}
    syms = Symbol[]
    for vn in names
        sym = AbstractPPL.getsym(vn)
        sym in syms || push!(syms, sym)
    end
    return filter!(syms) do sym
        previous = get(values.data, sym, nothing)
        previous isa ModelValue{<:Any,<:AbstractArray} &&
            _matches_model_role(R, previous) &&
            all(names) do vn
                optic = AbstractPPL.getoptic(vn)
                AbstractPPL.getsym(vn) !== sym || (
                    _is_element_index(optic) &&
                    length(optic.ix) == ndims(previous.value) &&
                    checkbounds(Bool, previous.value, optic.ix...)
                )
            end
    end
end
_is_element_index(optic) = false
function _is_element_index(
    ::AbstractPPL.Index{<:Tuple{Vararg{Int}},NamedTuple{(),Tuple{}},AbstractPPL.Iden}
)
    return true
end
# Matches expanding, unmasking and pruning: the element type is the join of the kept elements.
function _remove_model_elements(
    previous::ModelValue{R,<:AbstractArray}, sym, names
) where {R}
    value = previous.value
    mask = similar(value, Bool)
    for i in eachindex(mask, value)
        mask[i] = isassigned(value, i)
    end
    for vn in names
        AbstractPPL.getsym(vn) === sym && (mask[AbstractPPL.getoptic(vn).ix...] = false)
    end
    # Always `typejoin`: on Julia 1.11-1.13 a `t <: T || (T = t)` guard leaves `T == Union{}`.
    T = Union{}
    for i in eachindex(mask, value)
        mask[i] && (T = typejoin(T, typeof(_model_value_like(previous, value[i]))))
    end
    T === Union{} && return NoModelBinding()
    data = _fill_model_elements!(similar(value, T), previous, mask)
    _inherits_binding(previous) && (mask = ModelBindingArray(mask, value))
    return VarNamedTuples.PartialArray(data, mask)
end
function _fill_model_elements!(data, previous, mask)
    for i in eachindex(data, mask)
        mask[i] && (data[i] = _model_value_like(previous, previous.value[i]))
    end
    return data
end

function _remove_model_binding(::Type{R}, value, optic::AbstractPPL.AbstractOptic) where {R}
    if optic isa AbstractPPL.Iden
        return VarNamedTuples._map_values_recursive!!(
            v -> _matches_model_role(R, v) ? NoModelBinding() : v, _copy_model_node(value)
        )
    elseif _matches_model_role(R, value) && VarNamedTuples._haskey_optic(value, optic)
        return _remove_model_binding(R, _expand_model_binding(value), optic)
    end
    return value
end
function _remove_model_binding(
    ::Type{R}, values::VarNamedTuple, optic::AbstractPPL.Property{S}
) where {R,S}
    haskey(values.data, S) || return values
    child = _remove_model_binding(R, values.data[S], optic.child)
    return VarNamedTuple(merge(values.data, NamedTuple{(S,)}((child,))))
end
function _remove_model_binding(
    ::Type{R}, tree::ModelValueTree, optic::AbstractPPL.Property
) where {R}
    return ModelValueTree(tree, _remove_model_binding(R, tree.values, optic))
end
function _remove_model_binding(
    ::Type{R}, values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index, owned=false
) where {R}
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    original = values
    values = _model_slice_storage(values, optic)
    checkbounds(Bool, values, optic.ix...; optic.kw...) || return original
    multiindex = VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
    selected = if multiindex
        VarNamedTuples._subset_partialarray(values, optic.ix...; optic.kw...)
    elseif haskey(values, optic.ix...; optic.kw...)
        getindex(values, optic.ix...; optic.kw...)
    else
        return original
    end
    values === original || (selected = _resize_model_binding(selected, axes(selected)))
    child = _remove_model_binding(R, selected, optic.child)
    child === selected && return original
    template, values = values, owned ? values : copy(values)
    # Removed elements are unmasked in place, so many removals share one copy.
    if child isa NoModelBinding &&
        !(getindex(values.data, optic.ix...; optic.kw...) isa VarNamedTuples.ArrayLikeBlock)
        setindex!(values.mask, false, optic.ix...; optic.kw...)
        return values
    end
    values = VarNamedTuples._setindex_optic!!(
        values,
        child,
        AbstractPPL.Index(optic.ix, optic.kw),
        template,
        VarNamedTuples.AllowAll(),
    )
    # Setting a slice copies only its masked entries, so unmask those removed below it.
    if multiindex && child isa VarNamedTuples.PartialArray
        mask = view(values.mask, optic.ix...; optic.kw...)
        mask .&= child.mask
    end
    return if axes(values) == axes(original)
        values
    else
        _resize_model_binding(values, axes(original))
    end
end

"""
    conditioned(model::Model)

Return this model's conditioned values as plain values, independent of binding history.

Bindings held by children are not listed.
The result lists only bindings stored on `model`. Observations held by
submodels, including their argument-supplied observations and bindings made inside the
model body, are not listed, because those submodels exist only during evaluation.

The result is a `VarNamedTuple` containing ordinary or partial values. After partial
removal or mixed roles, containers become plain partial values (`VarNamedTuple` or
`PartialArray`), not the original container type.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: conditioned, prefix

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns the addresses and values of conditioned bindings.
       conditioned(condition(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # Prefixing also prefixes values already stored on the model.
       cm = condition(m, m=1.0) |> model -> prefix(model, @varname(a));

julia> conditioned(cm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Since we conditioned on `a.m`, it is an observed LHS variable.
       # However, `a.x` is still a latent LHS variable.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x

julia> # Values added after prefixing use their supplied names unchanged.
       cm = condition(prefix(m, @varname(a)), (@varname(a.m) => 1.0));

julia> conditioned(cm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x
```
"""
conditioned(model::Model) = _select_model_values(
    Condition, _model_values(_materialize_argument_values(model).values)
)

"""
    fix(model::Model; values...)
    fix(model::Model, values..., [template])

Return a `Model` which treats the LHS variables bound by `values` as constants: they replace
sampling and contribute no log probability. Fixed argument LHS variables reset to their bound
value when their tilde statement runs, even if the body has computed a different value.

Fixed bindings shadow observations. [`unfix`](@ref) uncovers the observation below,
or leaves the LHS variable latent if none remains; [`decondition`](@ref) removes
observations even beneath a fixed binding.

Bindings reach child addresses too; outermost explicit bindings win.
Inputs, unsupported partial bindings, conversion errors, aliasing and argument preparation
follow [`condition`](@ref).
Incomplete partial bindings into whole `missing`/`nothing` arguments throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
Partial bindings copy the owner's container one level deep; nested mutable values remain
shared and must not be mutated.
For example, `fix(model, @varname(z[2]) => 1.0, @of(z = of(Array, 3)))` supplies local
storage (`using AbstractPPL: of, @of`). Owners in the fixed layer take precedence over
the template. Runtime bindings under ForwardDiff/ReverseDiff need storage compatible with
AD values, e.g. `@of(z = of(Array, typeof(m), n))`, or a whole value.

Fixed values must cover their LHS variables with a static size and shape. Changing a
fixed argument's size or shape by replacement in the body throws `ArgumentError` naming the
LHS variable. In-place mutation of a whole fixed argument is not detected: it also mutates
the stored binding, and the body must not do it.
Shape validation stops at the LHS variable's address; it does not inspect nested values
inside a whole structured LHS variable.
A multivariate draw is one LHS variable: its subvariables cannot have different roles.
Use separate LHS variables (`x[i] ~ ...`) to fix indices independently.

See also: [`unfix`](@ref), [`fixed`](@ref), [Binding rules](@ref).

# Examples
## Simple univariate model
```jldoctest fix
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> m, x = model(); (m ≠ 1.0 && x ≠ 100.0)
true

julia> # Create a new instance which treats `x` as fixed
       # with value `100.0`, and similarly for `m=1.0`.
       fixed_model = fix(model, x=100.0, m=1.0);

julia> m, x = fixed_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only fix on `x = 100.0`.
       fixed_model = fix(model, x = 100.0);

julia> m, x = fixed_model(); (m != 1.0 && x == 100.0)
true
```

## Other ways of specifying fixed values

Specifying fixed values can be done exactly in the same way as for [`condition`](@ref);
please see its docstring for more examples.

## Difference from `condition`

Fixing omits the bound variable's log probability:

```jldoctest; setup=:(using DynamicPPL, Distributions)
julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> model_fixed = fix(model, m = 1.0);

julia> model_conditioned = condition(model, m = 1.0);

julia> logjoint(model_fixed, (x=1.0,))
-0.9189385332046728

julia> logjoint(model_conditioned, (x=1.0,))
-2.3378770664093453

julia> # The difference is the missing log-probability of `m`:
       logpdf(Normal(), 1.0)
-1.4189385332046727
```
"""
function fix(model::Model, inputs...; values...)
    inputs = isempty(values) ? inputs : (inputs..., NamedTuple(values))
    return _bind_inputs(Fix, model, inputs)
end
function fix(model::Model; values...)
    return fix(model, NamedTuple(values))
end

"""
    unfix(model::Model)
    unfix(model::Model, [DynamicPPL.Recursive()], names...)

Remove this model's fixed bindings at `names...`, or all fixed bindings if no names
are supplied. NamedTuple indices are rejected: use `x.a` instead of `x[1]` or `x[:a]`.
Tuple observations and complete local LHS variables keep integer indices.
Matching and the unsupported partial and shared-address cases follow [`decondition`](@ref).
Removing a valid address with no stored fixed binding is a no-op.
Pass `DynamicPPL.Recursive()` to remove fixed bindings from enclosed models too, including
runtime bindings. With no names it clears the fixed layer at every depth, uncovering
observations below. Named removals, prefixing, and later replacement follow [`decondition`](@ref).
Removals hold no value or shape and are omitted from `fixed`.

Removal uncovers the explicit or argument-supplied observation below the fixed binding,
or makes the LHS variable latent when no observation remains.
Observations removed with `decondition` stay removed. Removing a binding returns its
address to the shape of its next owner. For `@model f(x) = x ~ Normal()`,
`unfix(fix(f(1.0); x=5.0), :x)` and
`unfix(fix(decondition(f(1.0), :x); x=5.0), :x)` respectively observe `1.0` and leave `x` latent.

See also: [`fix`](@ref), [Binding rules](@ref).

# Examples
```jldoctest unfix
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> fixed_model = fix(demo(), m = 1.0, x = 10.0);

julia> fixed_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `unfix`.
       model = unfix(fixed_model, @varname(m));

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # When `NamedTuple` is used as the underlying, you can also provide
       # the symbol directly (though the `@varname` approach is preferable if
       # the LHS variable is known at compile-time).
       model = unfix(fixed_model, :m);

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # `unfix` multiple at once:
       (m, x) = unfix(fixed_model, :m, :x)(); (m != 1.0 && x != 10.0)
true

julia> # `unfix` without any symbols will `unfix` all LHS variables.
       (m, x) = unfix(model)(); (m != 1.0 && x != 10.0)
true
```
"""
function unfix(model::Model, syms::Union{Symbol,VarName}...)
    return _local_remove(Fix, model, syms)
end

# An empty `names` removes every binding of role `R`.
function _local_remove(::Type{R}, model, names) where {R}
    # Child addresses bound here are stored here, so they are removed here.
    model = _materialize_argument_values(model)
    layer = _binding_layer(R, model.values)
    names = let layer = layer
        map(vn -> _check_removal_name(model, layer, vn isa Symbol ? VarName{vn}() : vn), names)
    end
    _check_removal_addresses(layer, names)
    layer = if isempty(names)
        _remove_model_values(R, layer)
    else
        _remove_model_values(R, layer, names)
    end
    observations = R === Condition ? layer : _observation_values(model.values)
    fixed_values = R === Fix ? layer : _fixed_values(model.values)
    values = if isempty(fixed_values)
        observations
    else
        owners = _fixed_owners(model.values)
        if R === Fix
            owners = filter(owners) do owner
                any(vn -> subsumes(owner, vn) || subsumes(vn, owner), keys(fixed_values))
            end
        end
        ModelBindingLayers(observations, fixed_values, owners)
    end
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    values = _with_removals(
        values, _removals(Condition, model.values), _removals(Fix, model.values)
    )
    values = _with_binding_templates(
        values, _surviving_binding_templates(model.values, values, R)
    )
    removed = _reconstruct_model(model; values)
    _check_shared_latent_storage(removed)
    return _check_latent_arguments(removed)
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

"""
    fixed(model::Model)

Return this model's fixed values as plain values, independent of binding history.

Bindings held by children are not listed.
The result lists only bindings stored on `model`. Fixed bindings held by
submodels, including bindings made inside the model body, are not listed, because those
submodels exist only during evaluation.

The result is a `VarNamedTuple` containing ordinary or partial values. After partial
removal or mixed roles, containers become plain partial values (`VarNamedTuple` or
`PartialArray`), not the original container type.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: fixed, prefix

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns the addresses and values of fixed bindings.
       fixed(fix(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # Prefixing also prefixes values already stored on the model.
       fm = prefix(fix(m, m=1.0), @varname(a));

julia> fixed(fm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x

julia> # Values added after prefixing use their supplied names unchanged.
       fm = fix(prefix(m, @varname(a)), (@varname(a.m) => 1.0));

julia> fixed(fm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x
```
"""
fixed(model::Model) =
    _select_model_values(Fix, _model_values(_materialize_argument_values(model).values))

# Recursive removals are stored separately from values and therefore own no shape.
_removals(::Type, values) = ()
_removals(::Type{Condition}, values::ModelBindingLayers) = values.observation_removals
_removals(::Type{Fix}, values::ModelBindingLayers) = values.fixed_removals
_removals(R::Type, values::LocalModelValues) = _removals(R, values.values)

function _with_removals(values, observations::Tuple, fixed::Tuple)
    isempty(observations) && isempty(fixed) && return values
    return ModelBindingLayers(
        _observation_values(values),
        _fixed_values(values),
        _fixed_owners(values),
        observations,
        fixed,
        _binding_templates(values),
    )
end
function _with_removals(values::LocalModelValues, observations::Tuple, fixed::Tuple)
    return LocalModelValues(
        _with_removals(values.values, observations, fixed), values.owners
    )
end
function _prefix_removal(r::ModelRemoval, prefix)
    return ModelRemoval(
        maybe_prefix(r.name, prefix),
        map(vn -> maybe_prefix(vn, prefix), r.exceptions),
        r.matched,
        r.token,
    )
end
function _removal_covers(r, vn)
    return (r.name === nothing || subsumes(r.name, vn)) &&
           !any(ex -> subsumes(ex, vn), r.exceptions)
end
function _replace_removal(r, addresses)
    any(vn -> r.name !== nothing && subsumes(vn, r.name), addresses) && return ()
    exceptions = (
        r.exceptions...,
        filter(vn -> r.name === nothing || subsumes(r.name, vn), addresses)...,
    )
    return (ModelRemoval(r.name, exceptions, r.matched, r.token),)
end
function _replace_removals(previous, values, ::Type{R}, addresses) where {R}
    edited = mapreduce(
        r -> _replace_removal(r, addresses),
        (a, b) -> (a..., b...),
        _removals(R, previous);
        init=(),
    )
    observations = R === Condition ? edited : _removals(Condition, previous)
    fixed = R === Fix ? edited : _removals(Fix, previous)
    return _with_removals(values, observations, fixed)
end

function _has_removable(::Type{R}, values, vn) where {R}
    binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
    VarNamedTuples._mapreduce_recursive(
        pair -> _matches_model_role(R, pair.second), |, binding, vn, false
    ) && return true
    return mapreduce(
        pair -> subsumes(vn, pair.first) && _matches_model_role(R, pair.second),
        |,
        values;
        init=false,
    )
end
function _remove_marked(::Type{R}, values, r::ModelRemoval) where {R}
    removed = if r.name === nothing
        _remove_model_values(R, values)
    else
        _check_removal_addresses(values, (r.name,))
        _remove_model_values(R, values, (r.name,))
    end
    # Later bindings carve exceptions out of a broad removal. Keep their original
    # storage when restoring a partial subtree.
    for ex in r.exceptions
        binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(ex))
        binding === nothing && continue
        restored = templated_setindex!!(
            VarNamedTuple(), binding, ex, values.data[AbstractPPL.getsym(ex)]
        )
        removed = _merge_model_values(removed, restored)
    end
    return removed
end

function _local_removal_name(model, vn; operation="remove")
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
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

@inline function _check_removal_name(model, layer, vn)
    vn, template = _check_binding_address(model, layer, vn; operation="remove")
    AbstractPPL.is_dynamic(AbstractPPL.getoptic(vn)) || return vn
    return _concretize_prefix(vn, template)
end

function _check_shared_removals(parent, child)
    names = _lhs_names(_binding_metadata(child))
    names === nothing && return nothing
    for role in (Condition, Fix), r in _removals(role, parent.values)
        r.name === nothing && continue
        vn = _local_removal_name(parent, r.name)
        if vn !== nothing &&
            any(
                pair ->
                    !last(pair) && (subsumes(vn, first(pair)) || subsumes(first(pair), vn)),
                _lhs_addresses(_binding_metadata(parent)),
            ) &&
            any(
                pair -> subsumes(vn, first(pair)) || subsumes(first(pair), vn),
                _lhs_addresses(_binding_metadata(child)),
            )
            throw(
                ArgumentError(
                    "Cannot recursively remove `$(_binding_display_name(parent, r.name))`: it names both this model's LHS and an unprefixed submodel address. Prefix the submodel or remove without Recursive().",
                ),
            )
        end
    end
    return nothing
end

function _recursive_remove(::Type{R}, model, names) where {R}
    model = _materialize_argument_values(model)
    values = _binding_layer(R, model.values)
    markers = _removals(R, model.values)
    requested =
        isempty(names) ? (nothing,) : map(n -> n isa Symbol ? VarName{n}() : n, names)
    for vn in requested
        vn = vn === nothing ? nothing : _check_removal_name(model, values, vn)
        matched = vn === nothing ? !isempty(values) : _has_removable(R, values, vn)
        marker = ModelRemoval(vn, (), matched)
        matched && (values = _remove_marked(R, values, marker))
        # A covering marker already removes `vn`; another would only change the model type.
        if vn === nothing
            markers = (marker,)
        elseif !any(
            let vn = vn
                r -> _removal_covers(r, vn)
            end,
            markers,
        )
            markers = (markers..., marker)
        end
    end
    observations = R === Condition ? values : _observation_values(model.values)
    fixed = R === Fix ? values : _fixed_values(model.values)
    owners = filter(_fixed_owners(model.values)) do owner
        any(vn -> subsumes(owner, vn) || subsumes(vn, owner), keys(fixed))
    end
    layers = ModelBindingLayers(
        observations,
        fixed,
        owners,
        R === Condition ? markers : _removals(Condition, model.values),
        R === Fix ? markers : _removals(Fix, model.values),
    )
    layers = model.values isa LocalModelValues ? LocalModelValues(layers) : layers
    layers = _with_binding_templates(
        layers, _surviving_binding_templates(model.values, layers, R)
    )
    removed = _reconstruct_model(model; values=layers)
    _check_shared_latent_storage(removed)
    return _check_latent_arguments(removed)
end
function AbstractPPL.decondition(model::Model, ::Recursive, names::Union{Symbol,VarName}...)
    return _recursive_remove(Condition, model, names)
end
function unfix(model::Model, ::Recursive, names::Union{Symbol,VarName}...)
    return _recursive_remove(Fix, model, names)
end

function _select_removal(r, prefix)
    prefix === nothing && return (r,)
    any(ex -> subsumes(ex, prefix), r.exceptions) && return ()
    name = if r.name === nothing || subsumes(r.name, prefix)
        nothing
    elseif subsumes(prefix, r.name)
        AbstractPPL.unprefix(r.name, prefix)
    else
        return ()
    end
    exceptions = map(
        ex -> AbstractPPL.unprefix(ex, prefix),
        filter(ex -> subsumes(prefix, ex), r.exceptions),
    )
    return (ModelRemoval(name, exceptions, r.matched, r.token),)
end
function _submodel_removals(::Type{R}, model, prefix) where {R}
    prefix = _model_value_varname(model.values, prefix, _model_prefix(model))
    return mapreduce(
        r -> _select_removal(r, prefix),
        (a, b) -> (a..., b...),
        _removals(R, model.values);
        init=(),
    )
end
function _submodel_layer(::Type{R}, model) where {R}
    prefix = if model.values isa Union{LocalModelValues,UnprefixedArgumentValues}
        nothing
    else
        _model_prefix(model)
    end
    return _submodel_values(_binding_layer(R, model.values), prefix)
end

# DebugUtils can record use through its context without storing mutable state in a model.
_record_removal_use(context, role, marker) = nothing
function _record_removal_use(context::AbstractParentContext, role, marker)
    return _record_removal_use(childcontext(context), role, marker)
end
_apply_parent_removals(::Type{R}, values, ::Tuple{}, context, model) where {R} = values
function _apply_parent_removals(
    ::Type{R}, values, markers::Tuple{ModelRemoval,Vararg{ModelRemoval}}, context, model
) where {R}
    r = first(markers)
    if r.name !== nothing
        # An absent binding still has to satisfy the argument's storage rules.
        # Both the selected layer and the removal address are local to this child.
        local_model = _reconstruct_model(model; values=LocalModelValues(values))
        _binding_address_template(local_model, values, r.name; operation="remove")
    end
    matched = r.name === nothing ? !isempty(values) : _has_removable(R, values, r.name)
    matched && _record_removal_use(context, R, r)
    matched && (values = _remove_marked(R, values, r))
    return _apply_parent_removals(R, values, Base.tail(markers), context, model)
end

# Filter each layer before overlaying: a local fix must not hide an observation
# that is recursive in the child namespace.
function _submodel_inherited_values(model, prefix)
    prefix = _model_value_varname(model.values, prefix, _model_prefix(model))
    return _inherited_namespace(model.values, prefix)
end
function _inherited_namespace(values, prefix)
    return _inherited_model_values(_submodel_values(_model_values(values), prefix))
end
function _inherited_namespace(values::LocalModelValues, prefix)
    return _inherited_namespace(values.values, prefix)
end
function _inherited_namespace(values::ModelBindingLayers, prefix)
    observations = _inherited_model_values(_submodel_values(values.observations, prefix))
    fixed = _inherited_model_values(_submodel_values(values.fixed, prefix))
    return _overlay_model_values(observations, fixed, ())
end
