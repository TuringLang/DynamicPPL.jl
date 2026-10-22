_valid_model_binding(value) = value isa ModelValue
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

# Base.any carries missing and stops at true, giving Missing precedence.
const _NoPlaceholder = false
const _NothingPlaceholder = missing
const _MissingPlaceholder = true

function _classify_placeholder(value, seen=nothing)
    value === missing && return _MissingPlaceholder
    value === nothing && return _NothingPlaceholder
    if ismutabletype(typeof(value))
        seen === nothing && (seen = Base.IdSet{Any}())
        value in seen && return _NoPlaceholder
        push!(seen, value)
    end
    return _classify_placeholder_children(value, seen)
end
_classify_placeholder(::Union{Number,AbstractString,Symbol,Type}, seen) = _NoPlaceholder
_classify_placeholder(::AbstractArray{<:Number}, seen) = _NoPlaceholder

function _classify_placeholder_children(value, seen)
    return any(1:fieldcount(typeof(value))) do i
        isdefined(value, i) && _classify_placeholder(getfield(value, i), seen)
    end
end
function _classify_placeholder_children(value::Base.Pairs, seen)
    return _classify_placeholder(values(value), seen)
end
function _classify_placeholder_children(value::TransformedValue, seen)
    return _classify_placeholder(get_internal_value(value), seen)
end
function _classify_placeholder_children(values::AbstractArray, seen)
    return any(eachindex(values)) do i
        isassigned(values, i) && _classify_placeholder(values[i], seen)
    end
end
function _classify_placeholder_children(values::Union{Tuple,NamedTuple}, seen)
    return any(value -> _classify_placeholder(value, seen), values)
end

function ModelBindingLayers(
    observations,
    fixed,
    owners::Tuple=Tuple(keys(fixed)),
    observation_removals::Tuple=(),
    fixed_removals::Tuple=(),
    templates::Tuple=();
    shapes=ModelBindingShapes(),
)
    return ModelBindingLayers(
        observations,
        fixed,
        _overlay_model_values(observations, fixed, owners),
        owners,
        observation_removals,
        fixed_removals,
        templates,
        shapes,
    )
end
function _model_values(values::ModelBindingLayers)
    return values.values === nothing ? values.observations : values.values
end
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
    isempty(fixed) &&
        previous isa ModelValue{<:Any,<:AbstractArray} &&
        axes(previous.value) == axes(fixed.data) &&
        return previous
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
        # Preserve every occupied position when combining binding storage.
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
        # Read surviving entries through the replacement fields without rebuilding the previous value.
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
           a.observed == b.observed &&
           a.template == b.template &&
           a.values == b.values
end
function Base.isequal(a::ModelValueTree, b::ModelValueTree)
    return _inherits_shape(a) == _inherits_shape(b) &&
           a.observed == b.observed &&
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
function _model_role(values::VarNamedTuple, vn::VarName)
    bound = VarNamedTuples._mapreduce_recursive(
        pair -> pair.second isa ModelValue, |, values, vn, false
    )
    return bound ? _partial_binding_error(vn) : nothing
end
function _partial_binding_error(vn)
    return throw(
        ArgumentError(
            "LHS variable `$vn` has both bound and unbound subvariables, such as observed " *
            "and `missing` elements; write element-wise tildes (`x[i] ~ ...`), or bind or " *
            "decondition all of `$vn`.",
        ),
    )
end
function _model_role(values::VarNamedTuples.PartialArray, vn::VarName)
    all(values.mask) && return _model_role(values.data, vn)
    any(values.mask) || return nothing
    _model_role(values.data[values.mask], vn) === nothing && return nothing
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

# Namespace reads validate the same path as ordinary tildes, but do not assign
# a role to the namespace itself: its children may have different roles.
_model_role_at(values, ::AbstractPPL.Iden, vn, leaf=_model_role) = leaf(values, vn)
function _model_role_at(
    values::VarNamedTuple, optic::AbstractPPL.Property{S}, vn, leaf=_model_role
) where {S}
    return if haskey(values.data, S)
        _model_role_at(values.data[S], optic.child, vn, leaf)
    else
        nothing
    end
end
@inline function _model_role_at(
    value::ModelValue, optic::AbstractPPL.AbstractOptic, vn, leaf=_model_role
)
    head = AbstractPPL.ohead(optic)
    if VarNamedTuples._haskey_optic(value, head)
        child = VarNamedTuples._getindex_optic(value, head, vn)
        return _model_role_at(child, optic.child, vn, leaf)
    end
    value isa ModelValue{Fix} &&
        !(value.value isa NamedTuple && optic isa AbstractPPL.Property) &&
        _bound_shape_error(vn, Fix, "coverage with a static size and shape.")
    return nothing
end
@inline function _model_role_at(value::ModelValue, ::AbstractPPL.Iden, vn, leaf=_model_role)
    return leaf(value, vn)
end
function _model_role_at(
    values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index, vn, leaf=_model_role
)
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    # Namespace leaves assign no role, so only their index count needs validation.
    leaf === _namespace_leaf && optic.child isa AbstractPPL.Iden && return nothing
    if !checkbounds(Bool, values.data, optic.ix...; optic.kw...)
        # An out-of-bounds LHS slice may still overlap supplied indices.
        for indices in Iterators.product(Base.to_indices(values.data, optic.ix)...)
            checkbounds(Bool, values.data, indices...; optic.kw...) || continue
            getindex(values.mask, indices...; optic.kw...) || continue
            value = getindex(values.data, indices...; optic.kw...)
            _model_role_at(value, optic.child, vn, leaf) === nothing ||
                _partial_binding_error(vn)
        end
        return nothing
    end
    if VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
        selected = _model_argument_binding(values, AbstractPPL.Index(optic.ix, optic.kw))
        return _model_role_at(selected, optic.child, vn, leaf)
    end
    haskey(values, optic.ix...; optic.kw...) || return nothing
    return _model_role_at(
        getindex(values.data, optic.ix...; optic.kw...), optic.child, vn, leaf
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
    # Whole argument observations must follow body computations, including size changes.
    binding isa Union{ModelValue{Condition},ModelValue{ArgumentCondition}} &&
        return _model_role(binding, vn)
    return _local_binding_role(binding, vn, NoTemplate(), model)
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
        vn,
        _fixed_owners(model.values),
        _model_value_varname(model.values, root, _model_prefix(model)),
    )
    return _get_model_data(model, vn)
end

# Fixed array bindings must match body storage along the address path.
function _check_bound_shape(
    binding, local_value, optic, vn, owners=(), address=vn, role=Fix
)
    if (
        role === Fix &&
        (optic isa AbstractPPL.Iden || any(owner -> subsumes(owner, address), owners))
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
            (),
            vn,
            role,
        )
    end
    return _check_bound_shape_child(binding, local_value, optic, vn, owners, address, role)
end
function _check_bound_shape(
    binding::ModelValue{R}, local_value, optic, vn, owners=(), address=vn, role=Fix
) where {R}
    R === role || return nothing
    value = binding.value
    _skip_bound_shape_check(value) && return _check_bound_shape_child(
        binding, local_value, optic, vn, owners, address, role
    )
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
    return _check_bound_shape_child(binding, local_value, optic, vn, owners, address, role)
end
_skip_bound_shape_check(::Any) = false

# Keep error construction out of the hot path so the shape checks can inline.
@noinline function _bound_shape_error(vn, role, message)
    label = role === Fix ? "Fixed" : "Observed"
    return throw(ArgumentError("$label LHS variable `$vn` requires $message"))
end
function _check_bound_shape_child(
    binding, local_value, ::AbstractPPL.Iden, vn, owners, address, role
)
    return nothing
end
function _check_bound_shape_child(
    binding, local_value, optic::AbstractPPL.Property{S}, vn, owners, address, role
) where {S}
    child = _model_argument_binding(binding, AbstractPPL.Property{S}())
    child === nothing && role === Condition && return nothing
    child !== nothing && hasproperty(local_value, S) ||
        _bound_shape_error(vn, role, "coverage with a static size and shape.")
    return _check_bound_shape(
        child,
        getproperty(local_value, S),
        optic.child,
        vn,
        owners,
        AbstractPPL.append_optic(address, AbstractPPL.Property{S}()),
        role,
    )
end
function _check_bound_shape_child(
    binding, local_value, optic::AbstractPPL.Index, vn, owners, address, role
)
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
    return _check_bound_shape(
        child,
        selected,
        optic.child,
        vn,
        owners,
        AbstractPPL.append_optic(address, AbstractPPL.Index(optic.ix, optic.kw)),
        role,
    )
end

function _tag_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return map_pairs!!(pair -> ModelValue{R}(pair.second), copy(values))
end

# The mask records inherited whole-binding shape ownership.
struct ModelBindingArray{T,N,A<:AbstractArray{T,N},V<:AbstractArray} <: AbstractArray{T,N}
    data::A
    template::V
    observed::Bool
end
ModelBindingArray(data, template) = ModelBindingArray(data, template, false)
_whole_observation(value) = false
_whole_observation(value::ModelValueTree) = value.observed
function _whole_observation(value::VarNamedTuples.PartialArray)
    return value.mask isa ModelBindingArray && value.mask.observed
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
    return ModelBindingArray(copy(value.data), value.template, value.observed)
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
    _inherits_binding(previous) && (mask = ModelBindingArray(mask, value, R === Condition))
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
        R === Condition,
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
        leaf.data,
        ModelBindingArray(leaf.mask, template.mask.template, template.mask.observed),
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
function _model_role_at(
    tree::ModelValueTree, optic::AbstractPPL.Property, vn, leaf=_model_role
)
    return _model_role_at(tree.values, optic, vn, leaf)
end
# Bindings stored by field cannot be read by index, nor bindings stored by index by field.
function _model_role_at(
    ::Union{VarNamedTuple,ModelValueTree}, ::AbstractPPL.Index, vn, leaf=_model_role
)
    return _binding_kind_error(vn, "an index", "fields")
end
function _model_role_at(
    ::VarNamedTuples.PartialArray, ::AbstractPPL.Property, vn, leaf=_model_role
)
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
    leaf=_model_role,
)
    return _model_role_at(
        values, AbstractPPL.Property{only(optic.ix)}(optic.child), vn, leaf
    )
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
    previous::VarNamedTuple{P},
    updates::VarNamedTuple{U},
    prefix=nothing,
    merge::F=_merge_model_node,
) where {P,U,F}
    names = Tuple(union(P, U))
    fields = map(names) do name
        if name in P && name in U
            :(merge(
                previous.data.$name,
                updates.data.$name,
                maybe_prefix($(VarName{name}()), prefix),
            ))
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
function _merge_model_node(previous, updates, vn)
    _check_model_binding(previous, updates, vn)
    return _merge_model_node(previous, updates)
end
_merge_model_node(previous, updates) = updates
_merge_model_node(previous, ::NoModelBinding) = previous
function _merge_model_node(previous, updates::VarNamedTuple)
    previous isa NoModelBinding && return copy(updates)
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
        mask = ModelBindingArray(mask, updates.mask.template, updates.mask.observed)
    end
    return VarNamedTuples.PartialArray(data, mask)
end
# Preserve surviving tuple entries and their roles when the replacement supplies array storage.
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
    # Whole replacements must not read discarded entries; partial replacements may need
    # surviving tuple entries.
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
function _prepare_bindings(
    bindings::VarNamedTuples.PartialArray{T,N,D,<:ModelBindingArray{Bool,N}}, template, vn
) where {T,N,D<:AbstractArray{T,N}}
    return _prepare_binding_fields(bindings, bindings.mask.template, vn)
end

function _merge_model_indices(previous, updates, merge=(a, b, _) -> _merge_model_node(a, b))
    result = _fold_model_indices(
        _copy_model_node(previous), updates
    ) do result, update, optic, template
        child = _model_argument_binding(result, optic)
        value =
            child === nothing ? _copy_model_node(update) : merge(child, update, optic)
        return VarNamedTuples._setindex_optic!!(
            result, value, optic, template, VarNamedTuples.AllowAll()
        )
    end
    result isa PendingBindings &&
        updates isa VarNamedTuples.PartialArray &&
        !(updates.data isa VarNamedTuples.GrowableArray) &&
        return _prepare_bindings(result, _shape_storage(updates), result.name)
    return result
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

# SparseArrays use mutable buffers; argument copies adapt supported AD storage.
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
    # Latent copies may widen element types, but do not convert immutable array storage.
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
function _check_argument_key_storage(value::Array{T}, vn) where {T<:Number}
    isbitstype(T) && return nothing
    return invoke(_check_argument_key_storage, Tuple{Any,Any}, value, vn)
end
function _check_argument_key_storage(value, vn)
    (vn === nothing || isbits(value)) && return nothing
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
    value isa Union{Missing,Nothing} && return nothing
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
    _argument_may_alias(typeof(merge(model.args, model.defaults))) || return nothing
    lhs = _args_on_lhs(model)
    latent, repeated = Base.IdSet{Any}(), Any[]
    roots, bound = Tuple{Any,Any,Any,Union{Bool,Nothing}}[], Any[]
    for (stored_name, value) in pairs(merge(model.args, model.defaults))
        _argument_may_alias(typeof(value)) || continue
        name = unsplat_symbol(stored_name)
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

function _select_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return _map_model_nodes(v -> _select_model_node(R, v), values)
end
@generated function _map_model_nodes(f, values::VarNamedTuple{names}) where {names}
    fields = map(names) do name
        :(
            let child = f(values.data.$name)
                if child isa NoModelBinding
                    (;)
                else
                    NamedTuple{($(QuoteNode(name)),)}((child,))
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
    T = Core.Compiler.return_type(_model_data, Tuple{eltype(values)})
    if values.data isa VarNamedTuples.GrowableArray || !isconcretetype(T)
        return _select_model_array(R, values)
    end
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
_select_model_node(::Type{R}, value) where {R} = NoModelBinding()
function _select_model_node(::Type{R}, tree::ModelValueTree) where {R}
    selected = _select_model_values(R, tree.values)
    isempty(selected) && return NoModelBinding()
    return _plain_model_values(ModelValueTree(tree, selected))
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
    return _map_model_nodes(_plain_model_values, values)
end
function _map_model_nodes(f, values::VarNamedTuples.PartialArray)
    return _fold_model_indices(empty(values), values) do result, value, optic, template
        child = f(value)
        child isa NoModelBinding && return result
        return VarNamedTuples._setindex_optic!!(
            result, child, optic, template, VarNamedTuples.AllowAll()
        )
    end
end
function _select_model_node(::Type{R}, values::VarNamedTuples.PartialArray) where {R}
    return _select_model_array(R, values)
end
function _select_model_array(::Type{R}, values) where {R}
    return _prune_model_node(_map_model_nodes(v -> _select_model_node(R, v), values))
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
function Base.:|(model::Model, values::Union{NamedTuple,Pair,Tuple,VarNamedTuple})
    inputs = _binding_inputs(values)
    any(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, inputs) && throw(
        ArgumentError(
            "Binding templates require `condition` or `fix`; use `condition(model, values, template)` instead of `|`.",
        ),
    )
    return _conditionfix(model, inputs, Condition)
end

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

@generated function _check_argument_bindings(
    model::Model, values, prefix=nothing, storage_model=model
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
            name = $(QuoteNode(name))
            stored_name = $(QuoteNode(stored_name))
            vn = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
            binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
            if name in _args_on_lhs(model)
                previous = _get_model_binding(model, VarName{name}())
                stored = _get_model_binding(storage_model, VarName{name}())
                if stored !== nothing &&
                    binding isa Union{ModelValue,VarNamedTuple,VarNamedTuples.PartialArray}
                    storage = if stored isa ModelValue
                        stored.value
                    elseif prefix !== nothing
                        _shape_storage(stored)
                    else
                        first(_base_argument_storage(storage_model, vn))
                    end
                    value = binding
                    if binding isa ModelValue && $keyword
                        storage = (; storage...)
                        value = _model_value_like(binding, (; binding.value...))
                    end
                    _check_argument_shape(storage, value, vn)
                end
                if binding isa
                    Union{VarNamedTuple,VarNamedTuples.PartialArray,PendingBindings}
                    if $keyword
                        _keyword_binding_error(name)
                    end
                    argument = get(merge(model.args, model.defaults), stored_name, nothing)
                    argument = if previous === nothing
                        argument
                    else
                        prepare_model_argument(previous, argument)
                    end
                    if argument isa Union{Nothing,Missing}
                        if prefix !== nothing &&
                            binding isa VarNamedTuples.PartialArray &&
                            _inherits_shape(binding.mask)
                            argument = _shape_storage(binding)
                        else
                            _has_complete_model_data(binding) || throw(
                                ArgumentError(
                                    "Cannot make a partial binding of argument `$vn` with whole `$argument` storage; supply a concrete argument (e.g. `$(nameof(model))(zeros(n))`) or a whole binding instead.",
                                ),
                            )
                            binding = ModelValue{typeof(_model_role(binding, vn))}(
                                _model_data(binding)
                            )
                        end
                    else
                        binding = _prepare_bindings(binding, argument, maybe_prefix(vn, prefix))
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

_prepare_bindings(binding, storage, vn) = binding
function _prepare_bindings(
    bindings::Union{VarNamedTuple,VarNamedTuples.PartialArray}, storage, vn
)
    return _prepare_binding_fields(bindings, storage, vn)
end
function _prepare_binding_fields(bindings, storage, vn)
    return _fold_model_indices(copy(bindings), bindings) do result, binding, optic, template
        _check_partial_binding(
            ModelValue{Condition}(storage), optic, AbstractPPL.varname_to_optic(vn)
        )
        binding = _prepare_binding_selection(binding, storage, optic, vn)
        return VarNamedTuples._setindex_optic!!(
            result, binding, optic, template, VarNamedTuples.AllowAll()
        )
    end
end
function _prepare_binding_selection(binding, storage, optic, vn)
    address = AbstractPPL.append_optic(vn, optic)
    _check_binding_template_bounds(storage, optic, address; check_fields=true)
    VarNamedTuples._haskey_optic(storage, optic) || return binding
    binding isa ModelValue &&
        return _convert_partial_argument_binding(binding, storage, optic, address)
    return _prepare_bindings(
        binding, VarNamedTuples._getindex_optic(storage, optic, vn), address
    )
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
        _shape_payload_type(eltype(template))
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
    converted = if multi
        map(v -> _convert_partial_binding_value(T, v, vn), value)
    else
        _convert_partial_binding_value(T, value, vn)
    end
    isequal(converted, value) || throw(
        ArgumentError(
            "Cannot exactly represent partial binding at `$vn` in storage element type $T",
        ),
    )
    return _model_value_like(binding, converted)
end

function _has_partial_binding_conversion(::Type{T}, ::Type{V}) where {T,V}
    return hasmethod(T, Tuple{V}) ||
           which(convert, Tuple{Type{T},V}) !== which(convert, Tuple{Type{T},Number})
end

function _convert_partial_binding_value(::Type{T}, value::V, vn) where {T,V}
    if T <: Union{AbstractFloat,Integer} &&
        V <: Real &&
        !(V <: T) &&
        !(V <: Union{Integer,AbstractFloat,Rational,Irrational}) &&
        !_has_partial_binding_conversion(T, V)
        throw(
            ArgumentError(
                "Cannot represent partial binding at `$vn` in storage element type $T for value type $V; allocate storage with an element type that can hold $V.",
            ),
        )
    end
    return convert(T, value)
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
@noinline function _no_index_storage_error(optic, vn, operation; child_name=false)
    owner = _storage_address(vn, optic)
    key = length(optic.ix) == 1 ? only(optic.ix) : nothing
    advice = if key isa Symbol
        field = AbstractPPL.Property{key}(optic.child) ∘ AbstractPPL.varname_to_optic(owner)
        "NamedTuple fields use `$(AbstractPPL.optic_to_varname(field))`"
    elseif child_name
        if operation == "remove"
            "a `begin`, `end` or `:` index in a child's name resolves only against storage in the layer being edited; use integer indices"
        else
            "a `begin`, `end` or `:` index in a child's name resolves only against storage in the layer being edited; use integer indices or bind the whole value"
        end
    elseif operation == "remove"
        "use integer indices"
    else
        "use integer indices, or supply storage with a whole binding or a binding template"
    end
    message = "Cannot $operation `$vn`: no storage at `$owner` in the layer being edited to resolve this index; $advice."
    throw(ArgumentError(message))
end

function _check_binding_template_bounds(
    template, ::AbstractPPL.Iden, vn; operation="bind", check_fields=false, child_name=false
)
    return nothing
end
function _check_binding_template_bounds(
    template,
    optic::AbstractPPL.Property{S},
    vn;
    operation="bind",
    check_fields=false,
    child_name=false,
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
    return _check_binding_template_bounds(
        child, optic.child, vn; operation, check_fields, child_name
    )
end
function _check_binding_template_bounds(
    template,
    optic::AbstractPPL.Index,
    vn;
    operation="bind",
    check_fields=false,
    child_name=false,
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
        _no_index_storage_error(optic, vn, operation; child_name)
    coptic = AbstractPPL.concretize_top_level(optic, array)
    inbounds = if array isa VarNamedTuples.GrowableArray
        true
    elseif array isa Tuple
        checkbounds(Bool, Base.OneTo(length(array)), coptic.ix...)
    elseif array isa AbstractArray
        checkbounds(Bool, array, coptic.ix...; coptic.kw...)
    elseif array isa Union{NoTemplate,VarNamedTuples.SkipTemplate,Missing} && !child_name
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
        _check_binding_template_bounds(child, next, vn; operation, check_fields, child_name)
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

Bindings belong to this model and reach LHS variables at or below their addresses, including
child addresses such as `a.x`. Each position has at most one binding across roles and models.
Same-model observations, including argument data, may replace each other; other roles or
child bindings must first be removed, using `Recursive()` for child bindings.
A shared unprefixed name reaches both models under this rule. `|` follows it too.

Within one call, addresses must be disjoint after resolution and cannot select a position twice.
Disjoint bindings may jointly cover an LHS variable, but every executed tilde takes one role.
A bound local LHS variable reads its binding at its tilde. A whole observation of a local
must match the size and shape supplied by the body at each executed tilde; a mismatch throws.
Argument bindings prepare the argument before the body runs. Whole argument observations
follow its current value; partial observations keep their selected positions. Fixed values
reset at the tilde. See [Size checks at executed tildes](@ref binding-size-checks) for the
checks and their limits.
Observe raw data under a separate name if the body transforms the argument first.

Use NamedTuples/keywords for whole top-level values, `VarName` pairs for any address
(`:x => v` abbreviates `@varname(x) => v`), or a [`VarNamedTuple`](@ref) produced by
DynamicPPL. Positional inputs and nested tuples are flattened into one call; overlapping
addresses throw. `condition` rejects every
`AbstractDict` with `ArgumentError`; `model | dict` has no method. One positional binding
template, e.g. `@of(z = of(Array, 3))`, supplies storage for partially bound local LHS variables
without binding values (`using AbstractPPL: of, @of`).
An `of` type fixed before evaluation fixes its element type: runtime bindings under
ForwardDiff/ReverseDiff need `@of(z = of(Array, typeof(m), n))` or a whole value.

Addresses and independently declared types are checked as soon as storage allows.
A child-local partial binding needs storage before the child's body; a skipped local branch
does not defer this check. An unreached child performs no child-dependent checks.
Child templates remain candidates until entry and are ignored for arguments. Whole argument
values must fit the full signature (shared type constraints are checked during evaluation).
Whole local bindings must fit a supplied template type; otherwise only geometry constrains
a whole replacement. Partial values convert to the replaced
element/field type: Julia conversion errors propagate (`1.5` into `Int` raises
`InexactError`); successful but lossy conversions (`0.1` into `Float32`) raise `ArgumentError`.
AD real values that cannot fit ordinary numeric storage raise `ArgumentError` with advice
to allocate storage whose element type can hold the value.
Both layers share one binding shape, retained after removal except at argument roots, which revert to the argument’s own shape after whole removal of both layers. Change it only at its root,
with a whole value or template after clearing both layers with `decondition` and `unfix`.
Binding an argument does not recompute construction-time defaults or select another method.

Bound values are not copied; the body must not mutate them, including through aliases.
Partial array bindings and removals require `Array` or Array-backed `OffsetArray`,
`ComponentArray` or `DimArray` owners along the edited path. For other arrays, bind or
decondition the whole value, or use `collect` if losing axes or metadata is acceptable.
A partial binding copies the containers on its path when storage first becomes available:
at binding for ancestor-held storage, or afresh at each child entry for live child storage.
Deferred checks preserve existing snapshots. Values off the edited path remain shared.
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
provide a `NamedTuple`, a `VarNamedTuple`, or `VarName` pairs. Nested tuples of inputs
are flattened; addresses within the call must be disjoint. Other inputs to `condition`,
including every `AbstractDict`, throw `ArgumentError`; `|` has no `AbstractDict` method.

For example, here we use a `VarName` pair:

```jldoctest condition
julia> conditioned_model_pair = condition(model, @varname(x) => 100.0);

julia> m, x = conditioned_model_pair(); (m != 1.0 && x == 100.0)
true

julia> # There's also an option using `|` with a single `Pair{<:VarName}` on the
       # right-hand side, i.e. `vn => value` with `vn isa VarName`.
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
Existing storage in either role constrains the shared shape, including wholly masked
storage. A used template checks that storage; a conflict throws `ArgumentError`. A template
can replace a root shape only after both roles have been cleared at that root. A template
shape persists until a new template replaces it. Entries for arguments and names the call
does not bind are ignored. Templates use absolute names, as in `rand(model)` output: for a model
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
    return _conditionfix(model, inputs, Condition)
end

# Binding preparation consults owners in the layer being edited.
_binding_layer(::Type{Condition}, values) = _observation_values(values)
_binding_layer(::Type{Fix}, values) = _fixed_values(values)
function _binding_layer_model(::Type{R}, model) where {R}
    layer = _binding_layer(R, model.values)
    layer = ModelBindingLayers(
        R === Condition ? layer : VarNamedTuple(),
        R === Fix ? layer : VarNamedTuple(),
        (),
        (),
        (),
        _binding_templates(model.values);
        shapes=_binding_shape_store(model.values),
    )
    values = model.values isa LocalModelValues ? LocalModelValues(layer) : layer
    return _reconstruct_model(model; values)
end

_binding_templates(values) = ()
_binding_templates(values::ModelBindingLayers) = values.templates
_binding_templates(values::LocalModelValues) = _binding_templates(values.values)
function _prefix_binding_template(t::ModelBindingTemplate{R}, prefix) where {R}
    return ModelBindingTemplate{R}(maybe_prefix(t.name, prefix), t.storage, t.namespace)
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
    isempty(_binding_templates(model.values)) && return ()
    prefix = _model_value_varname(model.values, prefix, _model_prefix(model))
    prefix === nothing || (prefix = _resolve_binding_address(model, prefix))
    return mapreduce((a, b) -> (a..., b...), _binding_templates(model.values); init=()) do t
        prefix === nothing && return (t,)
        subsumes(prefix, t.name) || return ()
        return (_unprefix_binding_template(t, prefix),)
    end
end
function _unprefix_binding_template(t::ModelBindingTemplate{R}, prefix) where {R}
    return ModelBindingTemplate{R}(
        AbstractPPL.unprefix(t.name, prefix), t.storage, t.namespace
    )
end
function _prepare_deferred_template(t::ModelBindingTemplate{R}) where {R}
    namespace = if t.storage isa AbstractArray
        VarNamedTuples.PartialArray(
            similar(t.storage, VarNamedTuple{(),Tuple{}}),
            fill!(similar(t.storage, Bool), false),
        )
    else
        nothing
    end
    return ModelBindingTemplate{R}(t.name, t.storage, namespace)
end
function _prepare_namespace_bindings(model, values, shapes, templates)
    for t in templates
        _is_namespace_storage(model, t.name) || continue
        optic = AbstractPPL.varname_to_optic(t.name)
        node = _model_argument_binding(values, optic)
        node isa PendingBindings || continue
        t.storage isa ConflictingBindingTemplates &&
            _prepare_bindings(node, t.storage, t.name)
        t.namespace === nothing && _binding_storage_error(t.name)
        previous = _model_argument_binding(_local_binding_shape_store(model).roots, optic)
        ShapeOp{false}(model, (t.name,))(previous, t.namespace, t.name)
        for (index, _) in node.entries
            _check_binding_template_bounds(
                t.storage, index, AbstractPPL.append_optic(t.name, index)
            )
        end
        values = _namespace_binding(values, optic, t.namespace)
        shapes = _namespace_binding(shapes, optic, t.namespace)
    end
    return values, shapes
end
struct ConflictingBindingTemplates end
function _binding_schema_owned(model, address)
    metadata = _binding_metadata(model)
    if _lhs_names(metadata) === nothing || !any(last, _lhs_addresses(metadata))
        return true
    end
    return any(_lhs_addresses(metadata)) do (lhs, submodel)
        !submodel && (subsumes(lhs, address) || subsumes(address, lhs))
    end
end
# Flatten ordered input groups without interpreting keyword values as templates.
function _binding_inputs(values::Tuple)
    return mapreduce(_binding_inputs, (a, b) -> (a..., b...), values; init=())
end
_binding_inputs(value) = (value,)
_binding_inputs((name, value)::Pair{Symbol}) = (VarName{name}() => value,)

# Only storage enclosing a submodel prefix belongs to this model. Templates for
# variables owned by the child still need its arguments and LHS metadata.
function _is_namespace_storage(model, name)
    address = _local_removal_name(model, name; operation="bind")
    address === nothing && return false
    AbstractPPL.getsym(address) in _args_on_lhs(model) && return false
    return any(_lhs_addresses(_binding_metadata(model))) do (lhs, submodel)
        submodel && subsumes(address, lhs)
    end
end
function _check_namespace_index_counts(value, optic::AbstractPPL.AbstractOptic, vn, model)
    value isa Union{ModelValue,ModelValueTree} && return nothing
    child = _model_argument_binding(value, AbstractPPL.ohead(optic))
    child === nothing || _check_namespace_index_counts(child, optic.child, vn, model)
    return nothing
end
_check_namespace_index_counts(value, ::AbstractPPL.Iden, vn, model) = nothing
function _binding_entries(model, pair::Pair{<:VarName}, supplied=NoTemplate())
    vn, value = pair
    if value isa VarNamedTuples.PartialArray &&
        AbstractPPL.getoptic(vn) isa AbstractPPL.Iden
        return _binding_entries(
            model, VarNamedTuple(NamedTuple{(AbstractPPL.getsym(vn),)}((value,)))
        )
    end
    if value isa NamedTuple &&
        (_is_namespace_storage(model, vn) || vn == _model_prefix(model))
        return _binding_entries(model, value, vn, supplied)
    end
    return ((vn, value, supplied),)
end
@generated function _binding_entries(
    model, input::NamedTuple{names}, prefix=nothing, supplied=NoTemplate()
) where {names}
    return foldl(names; init=:(())) do entries, name
        :(_merge_binding_entries(
            $entries,
            _binding_entries(
                model, maybe_prefix($(VarName{name}()), prefix) => input.$name, supplied
            ),
        ))
    end
end
_merge_binding_entries(a::Tuple, b::Tuple) = (a..., b...)
_merge_binding_entries(a, b) = vcat(collect(a), collect(b))
function _binding_entries(model, input::VarNamedTuple)
    return _binding_entries(model, input, nothing, NoTemplate())
end
function _binding_entries(model, input, prefix, supplied)
    input isa VarNamedTuples.PartialArray &&
        isempty(input) &&
        return ((prefix, copy(input), supplied),)
    return _fold_model_indices(Tuple[], input) do entries, value, optic, _
        vn = if prefix === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(prefix, optic)
        end
        storage = prefix === nothing ? _deferred_input_storage(value) : supplied
        if prefix === nothing &&
            value isa VarNamedTuples.PartialArray &&
            !(value.data isa VarNamedTuples.GrowableArray) &&
            !_is_namespace_storage(model, vn)
            storage = _shape_storage(value)
        end
        children = if value isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            _binding_entries(model, value, vn, storage)
        else
            _binding_entries(model, vn => value, storage)
        end
        append!(entries, children)
    end
end
function _binding_entries(model, input)
    throw(
        ArgumentError(
            "Bindings require a VarNamedTuple, NamedTuple, or VarName pairs (or an ordered tuple of these); use keywords for whole values or @varname(x) => value for an address. AbstractDict inputs are not supported.",
        ),
    )
end

_numeric_binding_fields(::Type, leaf=Number) = false
function _numeric_binding_fields(
    ::Type{<:VarNamedTuple{names,T}}, ::Type{L}=Number
) where {names,T,L}
    return all(t -> t <: L && _shape_payload_type(t) <: Number, fieldtypes(T))
end
_prepare_bulk_binding(model, layer, value, vn) = nothing
function _prepare_bulk_binding(
    model, layer, value::VarNamedTuples.PartialArray{T}, vn
) where {T}
    isempty(value) && return nothing
    index = first(i for i in CartesianIndices(value.mask) if value.mask[i])
    optic = AbstractPPL.Index(Tuple(index), (;))
    address = AbstractPPL.append_optic(vn, optic)
    if _is_namespace_storage(model, vn)
        value.data isa VarNamedTuples.GrowableArray || return nothing
        _numeric_binding_fields(T) || return nothing
        previous = _model_argument_binding(
            _model_values(model.values), AbstractPPL.varname_to_optic(vn)
        )
        previous === nothing ||
            (
                previous isa PendingBindings &&
                previous.base === nothing &&
                all(
                    e -> _numeric_binding_fields(typeof(last(e)), ModelValue),
                    previous.entries,
                )
            ) ||
            return nothing
        shape = get(
            _binding_shape_store(model.values).roots.data, AbstractPPL.getsym(vn), nothing
        )
        shape === nothing ||
            (shape isa PendingBindings && shape.base === nothing) ||
            return nothing
        _resolve_binding(model, layer, address)
        # Whole numeric child roots need no conversion before the child runs.
        entries = Dict(
            AbstractPPL.Index(Tuple(i), (;)) => value.data[i] for
            i in CartesianIndices(value.mask) if value.mask[i]
        )
        return PendingBindings(vn, length(entries) == 1 ? (only(entries),) : entries)
    end
    T <: Number && isconcretetype(T) || return nothing
    value.data isa VarNamedTuples.GrowableArray && return nothing
    _partial_binding_array(value.data) || return nothing
    resolved = _resolve_binding(model, layer, address; supplied=value.data)
    _check_binding_storage(model, resolved.template, resolved.name; supplied=value.data)
    storage = resolved.template
    storage isa AbstractArray || return nothing
    axes(storage) == axes(value) && eltype(storage) === T || return nothing
    # Matching numeric storage makes the scalar checks common to the entire batch.
    return value
end
function _prepare_bulk_bindings(model, inputs, ::Type{R}, schema) where {R}
    length(inputs) == 1 && only(inputs) isa Union{NamedTuple,VarNamedTuple} ||
        return nothing
    isempty(schema) && isempty(_binding_templates(model.values)) || return nothing
    _model_prefix(model) === nothing || return nothing
    input = only(inputs) isa NamedTuple ? VarNamedTuple(only(inputs)) : only(inputs)
    layer = _binding_layer(R, model.values)
    prepared = map(keys(input.data)) do name
        _prepare_bulk_binding(model, layer, input.data[name], VarName{name}())
    end
    any(isnothing, prepared) && return nothing
    return VarNamedTuple(NamedTuple{keys(input.data)}(prepared))
end
function _prepare_binding_batch(model, inputs, ::Type{R}, schema) where {R}
    bulk = _prepare_bulk_bindings(model, inputs, R, schema)
    bulk === nothing || return bulk
    values = _model_values(model.values)
    entries = map(input -> _binding_entries(model, input), inputs)
    raw_seen = _binding_overlap_state(inputs)
    # Whole/part overlap must fail before missing storage can hide the conflict.
    for batch in entries, (vn, value, _) in batch
        _check_namespace_index_counts(values, AbstractPPL.varname_to_optic(vn), vn, model)
        value isa VarNamedTuples.PartialArray && isempty(value) && continue
        if _defer_binding_overlap(model, vn, value)
            _check_deferred_overlap!(raw_seen, model, vn, value)
        else
            _check_binding_overlap!(raw_seen, model, vn, NoTemplate())
        end
    end
    supplied_values = foldl(entries; init=VarNamedTuple()) do sources, batch
        foldl(batch; init=sources) do sources, (vn, _, supplied)
            name = AbstractPPL.getsym(vn)
            if supplied isa NoTemplate || haskey(sources.data, name)
                sources
            else
                VarNamedTuple(merge(sources.data, NamedTuple{(name,)}((supplied,))))
            end
        end
    end
    layer_model = _binding_layer_model(R, model)
    layer = _model_values(layer_model.values)
    seen = _binding_overlap_state(inputs)
    resolved = map(entries) do batch
        map(batch) do (vn, value, supplied)
            supplied = get(supplied_values.data, AbstractPPL.getsym(vn), supplied)
            address = _resolve_binding(model, layer, vn; supplied)
            template = address.template
            name = address.name
            _check_binding_storage(layer_model, template, name; supplied)
            if !(value isa VarNamedTuples.PartialArray && isempty(value)) &&
                !_defer_binding_overlap(model, name, value)
                _check_binding_overlap!(seen, layer_model, name, template)
            end
            (name, value, template)
        end
    end
    group =
        if any(
            batch -> any(entry -> _pending_binding(model, first(entry)), batch), resolved
        )
            Ref(nothing)
        else
            nothing
        end
    deferred_roots = foldl(entries; init=()) do roots, batch
        foldl(batch; init=roots) do roots, (name, value, _)
            _defer_binding_overlap(model, name, value) ? (roots..., name) : roots
        end
    end
    return foldl(resolved; init=VarNamedTuple()) do prepared, batch
        f =
            (p, entry) ->
                _prepare_binding_entry(layer_model, schema, p, entry, group, deferred_roots)
        if batch isa Tuple
            Base.afoldl(f, prepared, batch...)
        else
            foldl(f, batch; init=prepared)
        end
    end
end
function _prepare_binding_entry(
    model, schema, prepared, (name, value, template), group, deferred_roots
)
    optic = AbstractPPL.getoptic(name)
    _check_binding_extents(value, template, optic, name)
    pending = _pending_binding(model, name)
    whole =
        _schema_has_address(model, schema, name) || any(
            t -> t isa ModelBindingTemplate{Nothing} && subsumes(t.name, name),
            _binding_templates(model.values),
        )
    value = _convert_binding_template(value, template, optic, name; whole)
    _check_namespace_index_counts(prepared, AbstractPPL.varname_to_optic(name), name, model)
    if pending &&
        _property_binding_path(optic) &&
        (
            !isempty(deferred_roots) ||
            _has_deferred_binding(_binding_shape_store(model.values).roots)
        )
        if any(root -> subsumes(root, name) || subsumes(name, root), deferred_roots)
            value = DeferredBinding((DeferredWrite(value, group),))
        end
        update = templated_setindex!!(VarNamedTuple(), value, name, NoTemplate())
        return _merge_model_values(prepared, update)
    elseif pending && value isa NamedTuple && !_is_namespace_storage(model, name)
        value = DeferredBinding((DeferredWrite(value, group),))
    end
    template = if pending
        PendingBindingTemplate(template, VarName{AbstractPPL.getsym(name)}(), group)
    else
        template
    end
    if pending
        update = templated_setindex!!(VarNamedTuple(), value, name, template)
        return _merge_model_values(prepared, update)
    end
    return templated_setindex!!(prepared, value, name, template)
end

_property_binding_path(::AbstractPPL.Iden) = true
_property_binding_path(o::AbstractPPL.Property) = _property_binding_path(o.child)
_property_binding_path(::AbstractPPL.Index) = false

_scalar_binding_optic(::AbstractPPL.Iden) = true
_scalar_binding_optic(o::AbstractPPL.Property) = _scalar_binding_optic(o.child)
function _scalar_binding_optic(o::AbstractPPL.Index)
    return all(i -> i isa Integer, o.ix) && _scalar_binding_optic(o.child)
end
_binding_overlap_state(::Tuple{<:Union{NamedTuple,VarNamedTuple}}) = nothing
_binding_overlap_state(inputs) = (Set{VarName}(), Set{VarName}(), Dict{VarName,Any}())
function _binding_overlap_state(inputs::Tuple{<:Pair{<:VarName}})
    return if _scalar_binding_optic(AbstractPPL.getoptic(first(only(inputs))))
        nothing
    else
        _binding_overlap_state(nothing)
    end
end
@noinline function _binding_overlap_error(vn)
    throw(
        ArgumentError(
            "Binding address `$vn` overlaps another address or selects a position more than once; use disjoint addresses, or separate calls to overwrite values.",
        ),
    )
end
_check_binding_overlap!(::Nothing, model, name, template) = nothing
function _check_binding_overlap!(seen, model, vn, template)
    leaves, parents, cache = seen
    selections = _binding_selections(
        template, AbstractPPL.getoptic(vn), VarName{AbstractPPL.getsym(vn)}(), cache
    )
    for name in selections
        name in parents && _binding_overlap_error(_binding_display_name(model, vn))
        prefix, optic = VarName{AbstractPPL.getsym(name)}(), AbstractPPL.getoptic(name)
        while true
            prefix in leaves && _binding_overlap_error(_binding_display_name(model, vn))
            optic isa AbstractPPL.Iden && break
            push!(parents, prefix)
            prefix = AbstractPPL.append_optic(prefix, AbstractPPL.ohead(optic))
            optic = optic.child
        end
        push!(leaves, name)
    end
    return nothing
end

_pending_selection(index::Union{Integer,CartesianIndex}) = index
_pending_selection(index::AbstractArray{Bool}) = findall(index)
_pending_selection(index::AbstractArray{<:Union{Integer,CartesianIndex}}) = index
_pending_selection(index) = nothing
function _pending_positions(indices)
    selectors = map(_pending_selection, indices)
    any(isnothing, selectors) && return nothing
    all(i -> i isa Union{Integer,CartesianIndex}, selectors) &&
        return CartesianIndex(_expand_cartesian(selectors))
    iterators = map(i -> i isa Union{Integer,CartesianIndex} ? fill(i) : i, selectors)
    selected = map(Iterators.product(iterators...)) do position
        CartesianIndex(_expand_cartesian(position))
    end
    return reshape(selected, map(length, VarNamedTuples._selected_index_shape(indices...)))
end
function _binding_selections(::NoTemplate, optic::AbstractPPL.Index, vn, cache)
    selected = _pending_positions(optic.ix)
    selected === nothing && return (AbstractPPL.append_optic(vn, optic),)
    return _binding_selection_names(
        NoTemplate(), selected, optic.child, vn, cache, AbstractPPL.append_optic(vn, optic)
    )
end

function _conditionfix_inputs(model, values, ::Type{R}, schema=()) where {R}
    model = _materialize_argument_values(model)
    prepared = _prepare_binding_batch(model, values, R, schema)
    bound = _bind_values(model, prepared, R)
    _check_shared_latent_storage(bound)
    return bound
end

function _conditionfix(model::Model, inputs::Tuple, ::Type{R}) where {R}
    inputs = _binding_inputs(inputs)
    if model.values isa LocalModelValues && _model_prefix(model) !== nothing
        inputs = let model = model
            map(x -> _local_binding_input(model, x), inputs)
        end
    end
    inputs = _binding_inputs(inputs)
    schemas = filter(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, inputs)
    length(schemas) <= 1 ||
        throw(ArgumentError("At most one binding template is allowed per call."))
    values = filter(x -> !(x isa Type{<:AbstractPPL.OfNamedTuple}), inputs)
    isempty(schemas) && return _conditionfix_inputs(model, values, R)
    model = _materialize_argument_values(model)
    schema = VarNamedTuples.materialize_template(only(schemas))
    model = _record_binding_template(model, schema)
    schema = _local_binding_schema(model, schema, values)
    arguments = map(unsplat_symbol, keys(merge(model.args, model.defaults)))
    schema = NamedTuple{filter(name -> !(name in arguments), keys(schema))}(schema)
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    templates = let model = model
        filter(_binding_template_entries(R, schema, prefix)) do t
            any(v -> _input_binds_root(v, t.name, model), values)
        end
    end
    owned, deferred = (), ()
    for t in templates
        local_name = prefix === nothing ? t.name : AbstractPPL.unprefix(t.name, prefix)
        if _binding_schema_owned(model, local_name)
            model = _bind_shape(model, t.name, t.storage)
            owned = (owned..., t)
        elseif _is_namespace_storage(model, t.name)
            model = _bind_shape(model, t.name, t.storage; namespace=true)
        else
            deferred = (deferred..., _prepare_deferred_template(t))
        end
    end
    model = _conditionfix_inputs(model, values, R, owned)
    isempty(deferred) && return model
    bindings = _with_binding_metadata(
        model.values;
        templates=_merge_binding_templates(_binding_templates(model.values), deferred),
    )
    return _reconstruct_model(model; values=bindings)
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
_merge_binding_template(::ModelBindingTemplate{Nothing}, b) = b
function _merge_binding_template(a::ModelBindingTemplate{R}, b) where {R}
    storage = if _same_schema_storage(a.storage, b.storage)
        a.storage
    else
        ConflictingBindingTemplates()
    end
    return ModelBindingTemplate{R}(
        a.name, storage, storage === a.storage ? a.namespace : nothing
    )
end
function _record_binding_template(model::Model, template)
    prefix = _model_prefix(model)
    names = map(_binding_template_entries(Nothing, template)) do entry
        name = entry.name
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
        root = _model_value_varname(model.values, VarName{name}(), prefix)
        if (name !== AbstractPPL.getsym(prefix) || !(local_schema isa NamedTuple)) &&
            any(v -> _input_binds_root(v, root, model), values)
            throw(
                ArgumentError(
                    "Binding template entry `$name` uses a local name; use the absolute name `$(_binding_display_name(model, root))`.",
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
            vn,
            template;
            depth=optic_skip_length(AbstractPPL.getoptic(prefix)),
            canonical=true,
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
    return eltype(a) === eltype(b) &&
           _same_binding_geometry(_binding_geometry(a), _binding_geometry(b))
end
function _same_schema_storage(a::NamedTuple, b::NamedTuple)
    return keys(a) == keys(b) && all(_same_schema_storage(x, y) for (x, y) in zip(a, b))
end
function _same_schema_storage(a::Tuple, b::Tuple)
    return typeof(a) === typeof(b) &&
           all(_same_schema_storage(x, y) for (x, y) in zip(a, b))
end
_same_schema_storage(a, b) = typeof(a) === typeof(b)
function _same_schema_storage(a::VarNamedTuples.PartialArray, b::AbstractArray)
    return _shape_payload_type(eltype(a.data)) === eltype(b) &&
           _same_binding_geometry(_shape_root(a).geometry, _binding_geometry(b))
end
function _same_schema_storage(a::VarNamedTuple, b::NamedTuple)
    return all(
        name -> haskey(b, name) && _same_schema_storage(a.data[name], b[name]), keys(a.data)
    )
end

function _schema_has_address(model, schema::Tuple, vn)
    return any(t -> subsumes(t.name, vn) || subsumes(vn, t.name), schema)
end
_check_binding_extents(value, template, ::AbstractPPL.Iden, vn) = nothing
function _check_binding_extents(value, template, optic::AbstractPPL.AbstractOptic, vn)
    head = AbstractPPL.ohead(optic)
    head = if head isa AbstractPPL.Index
        AbstractPPL.concretize_top_level(head, VarNamedTuples.template_array(template))
    else
        head
    end
    child = _binding_child_template(template, head)
    if optic.child isa AbstractPPL.Iden &&
        head isa AbstractPPL.Index &&
        value isa AbstractArray &&
        VarNamedTuples._is_multiindex(
            VarNamedTuples.template_array(template), head.ix...; head.kw...
        )
        array = VarNamedTuples.template_array(template)
        indices = array isa AbstractArray ? Base.to_indices(array, head.ix) : head.ix
        shape = map(length, VarNamedTuples._selected_index_shape(indices...))
        # Slice assignment ignores singleton dimensions and permits a flattened trailing extent.
        actual, selected = filter(!=(1), size(value)), filter(!=(1), shape)
        common = max(0, min(length(actual), length(selected)) - 1)
        length(value) == prod(shape) && all(i -> actual[i] == selected[i], 1:common) ||
            throw(
                ArgumentError(
                    "Cannot bind `$vn`: value size $(size(value)) does not match selection size $shape; supply a value matching the selected extent.",
                ),
            )
    end
    return _check_binding_extents(value, child, optic.child, vn)
end

function _convert_binding_template(value, template, ::AbstractPPL.Iden, vn; whole=true)
    (!whole || template isa Union{NoTemplate,VarNamedTuples.SkipTemplate}) && return value
    value isa typeof(template) || throw(
        ArgumentError(
            "Bound value at `$vn` must be an instance of template type $(typeof(template)); supplied $(typeof(value)).",
        ),
    )
    _same_schema_storage(value, template) || throw(
        ArgumentError(
            "Bound value at `$vn` conflicts with its binding template storage; supply a new template with the value.",
        ),
    )
    return value
end
function _binding_child_template(template, optic)
    return VarNamedTuples.maybe_index_template(template, optic)
end
function _binding_child_template(template, ::AbstractPPL.Property{S}) where {S}
    return VarNamedTuples.SharedGetProperty{S}()(template)
end
function _convert_binding_template(
    value, template, optic::AbstractPPL.AbstractOptic, vn; whole=true
)
    template isa NoTemplate && return value
    head = AbstractPPL.ohead(optic)
    head =
        head isa AbstractPPL.Index ? AbstractPPL.concretize_top_level(head, template) : head
    # VarNamedTuple nodes describe namespaces, not fields of the bound value.
    # Descend to the local root before applying whole-value template checks.
    if optic.child isa AbstractPPL.Iden && !(template isa VarNamedTuple)
        template isa VarNamedTuples.PartialArray && (template = template.data)
        template isa Union{VarNamedTuples.GrowableArray,BindingGeometryArray} &&
            return value
        if (template isa AbstractArray && eltype(template) <: NoTemplate) || (
            head isa AbstractPPL.Property &&
            _binding_child_template(template, head) isa
            Union{NoTemplate,BindingGeometryArray}
        )
            return value
        end
        return _convert_partial_argument_binding(
            ModelValue{Condition}(value), template, head, vn
        ).value
    end
    child = _binding_child_template(template, head)
    return _convert_binding_template(value, child, optic.child, vn; whole)
end

function _binding_names(values)
    names = VarName[]
    map_pairs!!(copy(values)) do pair
        push!(names, first(pair))
        last(pair)
    end
    return names
end
function _check_disjoint_bindings(previous, updates, vn=nothing; ancestor=false)
    previous isa Union{Nothing,NoModelBinding} && return nothing
    updates isa Union{Nothing,NoModelBinding} && return nothing
    if previous isa DeferredBinding || updates isa DeferredBinding
        return _check_disjoint_bindings(
            _deferred_projection(previous), _deferred_projection(updates), vn; ancestor
        )
    end
    updates isa ModelValueTree && (updates = updates.values)
    if previous isa PendingBindings && updates isa ModelValue
        return _check_disjoint_bindings(
            _prepare_bindings(previous, updates.value, vn), updates, vn; ancestor
        )
    elseif previous isa ModelValue && updates isa PendingBindings
        return _check_disjoint_bindings(
            previous, _prepare_bindings(updates, previous.value, vn), vn; ancestor
        )
    end
    if updates isa ModelValue
        if previous isa ModelValue
            origin = ancestor ? "another model" : "the other role"
            throw(
                ArgumentError(
                    "Cannot bind `$vn`: a position is already bound by $origin; remove that binding first, using `Recursive()` for a child binding.",
                ),
            )
        end
        return _check_disjoint_bindings(updates, previous, vn; ancestor)
    end
    return _fold_model_indices(nothing, updates) do _, update, optic, _
        address = if vn === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(vn, optic)
        end
        child = _model_argument_binding(previous, optic)
        _check_disjoint_bindings(child, update, address; ancestor)
    end
end

function _bind_values(model::Model, values, ::Type{R}) where {R}
    preparation_model = _binding_layer_model(R, model)
    _check_bound_placeholders(R, values)
    # Not copied: `deepcopy` would detach values bound in the body from ReverseDiff's tape.
    values = _tag_model_values(R, values)
    values = _check_argument_bindings(preparation_model, values, nothing, model)
    _check_binding_addresses(model, values)
    _check_disjoint_bindings(
        _binding_layer(R === Fix ? Condition : Fix, model.values), values
    )
    model = _bind_shape_batch(model, _local_shape_snapshot(model, values), values)
    new_addresses = if R === Fix || !isempty(_binding_templates(model.values))
        Tuple(_binding_names(values))
    else
        ()
    end
    observations = _observation_values(model.values)
    fixed_values = _fixed_values(model.values)
    values = if R === Fix
        new_owners = new_addresses
        fixed_owners = (
            filter(
                owner -> !any(new -> subsumes(new, owner), new_owners),
                _fixed_owners(model.values),
            )...,
            new_owners...,
        )
        fixed_values = _merge_model_values(fixed_values, values)
        ModelBindingLayers(observations, fixed_values, fixed_owners)
    else
        ModelBindingLayers(
            _merge_model_values(observations, values),
            fixed_values,
            _fixed_owners(model.values),
        )
    end
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    values = _with_binding_metadata(
        values;
        observation_removals=_removals(Condition, model.values),
        fixed_removals=_removals(Fix, model.values),
        templates=_surviving_binding_templates(model.values, values, R, new_addresses),
        shapes=_binding_shape_store(model.values),
    )
    return _reconstruct_model(model; values)
end
function AbstractPPL.condition(model::Model; values...)
    return condition(model, NamedTuple(values))
end

function _check_bound_placeholders(::Type{R}, values::VarNamedTuple) where {R}
    _classify_placeholder(values) === _NoPlaceholder && return nothing
    vn = first(
        vn for
        (vn, value) in pairs(values) if _classify_placeholder(value) !== _NoPlaceholder
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
_binding_storage(value::VarNamedTuple) = VarNamedTuple(map(_binding_storage, value.data))
function _binding_storage(value::VarNamedTuples.PartialArray)
    # Inferred, growable storage has no owner to constrain subsequent indices.
    value.data isa VarNamedTuples.GrowableArray && return NoTemplate()
    return _shape_storage(value)
end
_binding_address_storage(value, optic) = _binding_storage(value)
function _binding_address_storage(
    value::VarNamedTuple, optic::AbstractPPL.Property{S}
) where {S}
    storage = _binding_storage(value)
    haskey(value.data, S) || return storage
    child = _binding_address_storage(value.data[S], optic.child)
    child isa NoTemplate && return child
    return VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((child,))))
end

function _binding_address_template(model, values, address; operation="bind")
    shape = _binding_shape_template(model, address, values)
    shape isa NoTemplate && (shape = _binding_template(model, values, address))
    return _binding_address_storage(shape, AbstractPPL.getoptic(address))
end

function _pending_binding(model, vn)
    local_name = _local_removal_name(model, vn; operation="bind")
    local_name === nothing && return true
    metadata = _binding_metadata(model)
    _lhs_names(metadata) === nothing && return false
    name = AbstractPPL.getsym(local_name)
    name in _args_on_lhs(model) && return false
    addresses = filter(x -> AbstractPPL.getsym(first(x)) === name, _lhs_addresses(metadata))
    return isempty(addresses) ? _has_unprefixed_submodel(metadata) : any(last, addresses)
end

@noinline function _binding_storage_error(vn)
    throw(
        ArgumentError(
            "Cannot bind `$vn` without storage; supply an `@of` template, a whole value, a `VarNamedTuple` with storage, or storage on the child before prefixing.",
        ),
    )
end
_require_binding_storage(template, ::AbstractPPL.Iden, vn; kwargs...) = nothing
function _require_binding_storage(template, optic, vn; skip=0, supplied=NoTemplate())
    if optic isa AbstractPPL.Index && skip <= 0
        storage = VarNamedTuples.template_array(template)
        if !(storage isa AbstractArray) || storage isa VarNamedTuples.GrowableArray
            template = supplied
            storage = VarNamedTuples.template_array(template)
        end
        storage isa AbstractArray && !(storage isa VarNamedTuples.GrowableArray) ||
            _binding_storage_error(vn)
        optic = AbstractPPL.concretize_top_level(optic, storage)
        _check_binding_template_bounds(template, optic, vn)
    end
    return _require_binding_storage(
        _binding_child_template(template, optic),
        optic.child,
        vn;
        skip=skip - 1,
        supplied=_binding_child_template(supplied, optic),
    )
end

struct PendingBindings{N,E,B,G}
    name::N
    entries::E
    base::B
    groups::G
end
PendingBindings(name, entries, base=nothing) = PendingBindings(name, entries, base, nothing)
function _pending_group(p, optic)
    return p.groups isa AbstractDict ? get(p.groups, optic, nothing) : p.groups
end
# Bypass the OrderedDict try/catch wrapper on Julia 1.10 for runtime AD.
function _pending_dict(kv)
    @static if VERSION >= v"1.11"
        OrderedDict(kv)
    else
        Base.dict_with_eltype((K, V) -> OrderedDict{K,V}, kv, eltype(kv))
    end
end
_pending_entries_map(f, entries) = map(f, entries)
_pending_entries_map(f, entries::AbstractDict) = _pending_dict(f(e) for e in entries)
function _pending_map(f, p::PendingBindings)
    return PendingBindings(p.name, _pending_entries_map(f, p.entries), p.base, p.groups)
end
_pending_element(value, i) = value[i]
_pending_element(value::ModelValue, i) = _model_value_like(value, value.value[i])
function _pending_expand(f, result, p, optic::AbstractPPL.Property{S}, value) where {S}
    head = AbstractPPL.Property{S}()
    child = VarNamedTuples.make_leaf(
        value,
        optic.child,
        PendingBindingTemplate(
            NoTemplate(), AbstractPPL.append_optic(p.name, head), _pending_group(p, optic)
        ),
    )
    return f(result, child, head, p)
end
function _pending_selected(optic, storage=nothing, vn=nothing)
    selected = if storage isa VarNamedTuples.PartialArray
        _binding_select(CartesianIndices(axes(storage)), optic, vn)
    else
        _pending_positions(optic.ix)
    end
    next, path = optic.child, AbstractPPL.ohead(optic)
    while selected isa AbstractArray && next isa AbstractPPL.Index
        selected = _binding_select(selected, next, vn)
        path = AbstractPPL.ohead(next) ∘ path
        next = next.child
    end
    return selected, next, path
end
function _pending_expand(f, result, p, optic::AbstractPPL.Index, value)
    selected, next, _ = _pending_selected(
        optic, p.base, AbstractPPL.append_optic(p.name, optic)
    )
    selected === nothing && _no_index_storage_error(
        optic, AbstractPPL.append_optic(p.name, optic), "bind"; child_name=true
    )
    multi = selected isa AbstractArray
    for (i, position) in enumerate(multi ? selected : (selected,))
        multi && value isa VarNamedTuples.PartialArray && !haskey(value, i) && continue
        head = AbstractPPL.Index(Tuple(position), (;))
        template =
            if !multi && next isa AbstractPPL.Property && next.child isa AbstractPPL.Iden
                NoTemplate()
            else
                PendingBindingTemplate(
                    NoTemplate(),
                    AbstractPPL.append_optic(p.name, head),
                    _pending_group(p, optic),
                )
            end
        child = VarNamedTuples.make_leaf(
            multi ? _pending_element(value, i) : value, next, template
        )
        result = f(result, child, head, p)
    end
    return result
end
# A direct, inlined loop avoids fold callback allocations on Julia 1.10.
@inline function _fold_model_indices(f::F, result, p::PendingBindings) where {F}
    for (optic, value) in p.entries
        result = @inline _pending_expand(f, result, p, optic, value)
    end
    return result
end
function _binding_address_storage(p::PendingBindings, optic::AbstractPPL.Index)
    all(i -> _pending_selection(i) !== nothing, optic.ix) || return NoTemplate()
    child = _model_argument_binding(p, AbstractPPL.ohead(optic))
    return VarNamedTuples.SkipTemplate{1}(_binding_address_storage(child, optic.child))
end
_binding_storage(::PendingBindings) = NoTemplate()
function _classify_placeholder_children(p::PendingBindings, seen)
    return any(e -> _classify_placeholder(last(e), seen), p.entries)
end
Base.copy(p::PendingBindings) = _pending_map(e -> first(e) => _copy_model_node(last(e)), p)
_copy_model_node(p::PendingBindings) = copy(p)
Base.isempty(p::PendingBindings) = isempty(p.entries)
function VarNamedTuples._storage_template(p::PendingBindings)
    message = "Cannot make a storage template from pending bindings at `$(p.name)`; supply concrete storage with `@of` instead."
    throw(ArgumentError(message))
end
function VarNamedTuples._map_values_recursive!!(f, p::PendingBindings)
    entries = _pending_entries_map(
        e -> first(e) => VarNamedTuples._map_values_recursive!!(f, last(e)), p.entries
    )
    base = p.base === nothing ? nothing : f(p.base)
    return PendingBindings(
        p.name, entries, base isa NoModelBinding ? nothing : base, p.groups
    )
end
function VarNamedTuples._map_pairs_recursive!!(f, p::PendingBindings, vn)
    return _pending_map(
        e ->
            first(e) => VarNamedTuples._map_pairs_recursive!!(
                f, last(e), AbstractPPL.append_optic(vn, first(e))
            ),
        p,
    )
end
# Listings fold pending entries at their addresses; storage remains unchanged.
function VarNamedTuples._mapreduce_recursive(f, op, p::PendingBindings, vn, init)
    result = init
    p.base === nothing ||
        (result = VarNamedTuples._mapreduce_recursive(f, op, p.base, vn, result))
    for (optic, value) in p.entries
        result = VarNamedTuples._mapreduce_recursive(
            f, op, value, AbstractPPL.append_optic(vn, optic), result
        )
    end
    return result
end
function VarNamedTuples._merge(a::PendingBindings, b::PendingBindings, ::Val{true})
    return _merge_model_node(a, b)
end
function Base.show(io::IO, p::PendingBindings)
    print(io, "(")
    p.base === nothing || (show(io, p.base); print(io, ", "))
    for (i, (optic, value)) in enumerate(p.entries)
        i == 1 || print(io, ", ")
        AbstractPPL._pretty_print_optic(io, optic)
        print(io, " => ")
        show(io, value)
    end
    return print(io, ")")
end
function _pending_lookup(p, selection)
    optic = if p.base isa VarNamedTuples.PartialArray
        _resolve_binding_optic(p.base.data, selection)
    else
        selection
    end
    return _fold_model_indices(nothing, p) do result, value, head, _
        head == optic || return result
        result === nothing ? value : _merge_model_node(result, value)
    end
end
function Base.:(==)(a::PendingBindings, b::PendingBindings)
    return a.entries == b.entries && a.base == b.base
end
function Base.isequal(a::PendingBindings, b::PendingBindings)
    return isequal(a.entries, b.entries) && isequal(a.base, b.base)
end
Base.hash(p::PendingBindings, h::UInt) = hash(p.entries, hash(p.base, h))
function _model_argument_binding(p::PendingBindings, optic::AbstractPPL.AbstractOptic)
    child = _pending_lookup(p, AbstractPPL.ohead(optic))
    return if child === nothing
        _model_argument_binding(p.base, optic)
    else
        _model_argument_binding(child, optic.child)
    end
end
_model_argument_binding(p::PendingBindings, ::AbstractPPL.Iden) = p
function _model_argument_binding(p::PendingBindings, optic::AbstractPPL.Index)
    all(i -> i isa Union{Integer,CartesianIndex}, optic.ix) && return invoke(
        _model_argument_binding,
        Tuple{PendingBindings,AbstractPPL.AbstractOptic},
        p,
        _expand_cartesian(optic),
    )
    positions = _pending_positions(optic.ix)
    positions === nothing && return nothing
    children = map(
        position ->
            _model_argument_binding(p, AbstractPPL.Index(Tuple(position), optic.kw)),
        positions,
    )
    return _model_argument_binding(
        VarNamedTuples.PartialArray(children, map(!isnothing, children)), optic.child
    )
end
function VarNamedTuples._getindex_optic(
    p::PendingBindings, optic::Union{AbstractPPL.Property,AbstractPPL.Index}, vn
)
    child = _model_argument_binding(p, optic)
    child === nothing && throw(KeyError(vn))
    return VarNamedTuples._getindex_optic(child, AbstractPPL.Iden(), vn)
end
function VarNamedTuples._haskey_optic(p::PendingBindings, optic::AbstractPPL.AbstractOptic)
    child = _model_argument_binding(p, optic)
    return child !== nothing && VarNamedTuples._haskey_optic(child, AbstractPPL.Iden())
end
VarNamedTuples._haskey_optic(::PendingBindings, ::AbstractPPL.Iden) = false
function _model_role_at(
    p::PendingBindings, optic::AbstractPPL.AbstractOptic, vn, leaf=_model_role
)
    leaf === _namespace_leaf && _pending_index_style(p, optic, vn)
    return leaf(_model_argument_binding(p, optic), vn)
end
_model_role_at(p::PendingBindings, ::AbstractPPL.Iden, vn, leaf=_model_role) = leaf(p, vn)
_model_role(p::PendingBindings, vn) = isempty(p) ? nothing : _partial_binding_error(vn)
_model_data(p::PendingBindings) = VarNamedTuples._map_values_recursive!!(_model_data, p)
function _materialize_pending(p::PendingBindings)
    storage = p.base isa VarNamedTuples.PartialArray ? p.base.data : NoTemplate()
    return foldl(p.entries; init=_copy_model_node(p.base)) do result, (optic, value)
        if result === nothing
            VarNamedTuples.make_leaf(value, optic, storage)
        else
            VarNamedTuples._setindex_optic!!(
                result, value, optic, storage, VarNamedTuples.AllowAll()
            )
        end
    end
end
function _select_model_node(::Type{R}, p::PendingBindings) where {R}
    all(e -> first(e) isa AbstractPPL.Property, p.entries) &&
        return _select_model_node(R, _submodel_namespace(p))
    return _prune_model_node(
        _pending_map(e -> first(e) => _select_model_node(R, last(e)), p)
    )
end
function _prune_model_node(p::PendingBindings)
    eltype(p.entries) <: Pair{<:Any,<:ModelValue} && return p
    eltype(p.entries) <: Pair{<:Any,NoModelBinding} &&
        return something(p.base, NoModelBinding())
    entries = _pending_entries_map(e -> first(e) => _prune_model_node(last(e)), p.entries)
    keep = e -> !(last(e) isa NoModelBinding)
    entries = (entries isa AbstractDict ? filter! : filter)(keep, entries)
    isempty(entries) && return something(p.base, NoModelBinding())
    groups = if p.groups isa AbstractDict
        filter(e -> haskey(entries, first(e)), p.groups)
    else
        p.groups
    end
    return PendingBindings(p.name, entries, p.base, groups)
end
function _pending_index_style(p, optic, vn)
    optic isa AbstractPPL.Index || return nothing
    p.base isa VarNamedTuples.PartialArray && return nothing
    for (previous, _) in p.entries
        previous isa AbstractPPL.Index || continue
        length(previous.ix) == length(optic.ix) || throw(
            ArgumentError(
                "Pending binding at `$vn` uses inconsistent index counts; supply an index with the same number of dimensions or provide storage.",
            ),
        )
    end
    return nothing
end
function _merge_model_node(
    base::VarNamedTuples.PartialArray{<:VarNamedTuple}, update::PendingBindings
)
    return _merge_model_indices(base, update)
end
function _merge_model_node(
    base::Union{ModelValue,VarNamedTuples.PartialArray}, update::PendingBindings
)
    prepared = _prepare_bindings(
        update, base isa ModelValue ? base.value : _shape_storage(base), update.name
    )
    return _merge_model_node(base, prepared)
end
function _pending_set!!(entries::OrderedDict{K,V}, value, optic) where {K,V}
    if !(optic isa K && value isa V)
        entries = OrderedDict{Union{K,typeof(optic)},Union{V,typeof(value)}}(entries)
    end
    return setindex!(entries, value, optic)
end
function _pending_mask_prefix(optic)
    optic isa AbstractPPL.Iden && return nothing
    child = _pending_mask_prefix(optic.child)
    head = AbstractPPL.ohead(optic)
    child === nothing || return child ∘ head
    optic isa AbstractPPL.Index || return nothing
    return any(i -> i isa AbstractArray{Bool}, optic.ix) ? head : nothing
end
function _pending_remainder(optic, value, removal, replacement, R=Condition)
    if optic isa AbstractPPL.Iden
        child = _remove_model_binding(R, value, removal)
        replacement && (child = _remove_model_binding(Fix, child, removal))
        return child isa NoModelBinding ? nothing : optic => child
    elseif AbstractPPL.ohead(optic) == AbstractPPL.ohead(removal)
        child = _pending_remainder(optic.child, value, removal.child, replacement, R)
        child === nothing && return nothing
        return first(child) ∘ AbstractPPL.ohead(optic) => last(child)
    end
    return AbstractPPL.subsumes(removal, optic) ? nothing : optic => value
end
function _pending_remainder(
    optic::AbstractPPL.Index, value, removal::AbstractPPL.Index, replacement, R=Condition
)
    selected, next, path = _pending_selected(optic)
    removed, tail, _ = _pending_selected(removal)
    positions = removed isa AbstractArray ? removed : (removed,)
    if !(selected isa AbstractArray) &&
        selected in positions &&
        !(next isa AbstractPPL.Iden)
        child = _pending_remainder(next, value, tail, replacement, R)
        child === nothing || return first(child) ∘ path => last(child)
        value = NoModelBinding()
        next = tail = AbstractPPL.Iden()
    end
    edit = function (child, position)
        position in positions || return child
        relative = if AbstractPPL.subsumes(tail, next)
            AbstractPPL.Iden()
        elseif _scalar_binding_optic(next) && AbstractPPL.subsumes(next, tail)
            AbstractPPL._unprefix_optic(tail, next)
        else
            return child
        end
        child = _remove_model_binding(R, child, relative)
        return replacement ? _remove_model_binding(Fix, child, relative) : child
    end
    if selected isa AbstractArray
        data = map(CartesianIndices(selected)) do i
            child = if value isa VarNamedTuples.PartialArray && !haskey(value, i)
                NoModelBinding()
            else
                _pending_element(value, i)
            end
            edit(child, selected[i])
        end
        mask = map(x -> !(x isa NoModelBinding), data)
        any(mask) && return optic => VarNamedTuples.PartialArray(data, mask)
        replacement && !isempty(mask) && return nothing
    else
        child = edit(value, selected)
        child isa NoModelBinding || return optic => child
        replacement && return nothing
    end
    path = _pending_mask_prefix(optic)
    path === nothing && return nothing
    empty = if value isa ModelValue
        _model_value_like(value, Union{}[])
    else
        ModelValue{R}(Union{}[])
    end
    return AbstractPPL.Index((Int[],), (;)) ∘ path => empty
end
function _merge_model_node(a::PendingBindings, b::PendingBindings)
    a.base isa VarNamedTuples.PartialArray &&
        return _merge_model_node(_materialize_pending(a), b, a.name)
    if a.entries isa Tuple{Pair} && b.entries isa Tuple{Pair}
        (old, previous), (optic, value) = only(a.entries), only(b.entries)
        if old == optic
            entries = (optic => _merge_model_node(previous, value),)
            return PendingBindings(b.name, entries, a.base, _pending_group(b, optic))
        end
    end
    entries = _pending_dict(a.entries)
    groups = _pending_dict(o => _pending_group(a, o) for (o, _) in a.entries)
    for (optic, value) in b.entries
        haskey(entries, optic) && (value = _merge_model_node(entries[optic], value))
        _pending_index_style(a, optic, b.name)
        for (old, previous) in collect(entries)
            remainder = _pending_remainder(old, previous, optic, true)
            if remainder === nothing || first(remainder) != old
                group = pop!(groups, old)
                delete!(entries, old)
                if remainder !== nothing
                    entries = _pending_set!!(entries, last(remainder), first(remainder))
                    groups = _pending_set!!(groups, group, first(remainder))
                end
            elseif last(remainder) !== previous
                entries = _pending_set!!(entries, last(remainder), old)
            end
        end
        entries = _pending_set!!(entries, value, optic)
        groups = _pending_set!!(groups, _pending_group(b, optic), optic)
    end
    return PendingBindings(b.name, entries, a.base, groups)
end
function _pending_fields(values::VarNamedTuple, name)
    return PendingBindings(
        name,
        map(
            (n, v) -> AbstractPPL.Property{n}() => v, keys(values.data), Tuple(values.data)
        ),
    )
end
function _merge_model_node(a::VarNamedTuple, b::PendingBindings)
    return _merge_model_node(_pending_fields(a, b.name), b)
end
function _merge_model_node(a::PendingBindings, b::VarNamedTuple)
    return _merge_model_node(a, _pending_fields(b, a.name))
end
function VarNamedTuples._setindex_optic!!(
    p::PendingBindings, value, optic::AbstractPPL.AbstractOptic, template, permissions
)
    return _merge_model_node(p, PendingBindings(p.name, (optic => value,)))
end
function VarNamedTuples._setindex_optic!!(
    p::PendingBindings,
    value,
    optic::AbstractPPL.Iden,
    template,
    permissions::VarNamedTuples.SetPermissions,
)
    return invoke(
        VarNamedTuples._setindex_optic!!,
        Tuple{PendingBindings,Any,AbstractPPL.AbstractOptic,Any,Any},
        p,
        value,
        optic,
        template,
        permissions,
    )
end
function _remove_model_binding(
    ::Type{R}, p::PendingBindings, optic::Union{AbstractPPL.Index,AbstractPPL.Property}
) where {R}
    entries = empty(_pending_dict(p.entries))
    groups = Dict{AbstractPPL.AbstractOptic,Union{Nothing,Base.RefValue{Nothing}}}()
    for (old, value) in p.entries
        remainder = if _scalar_binding_optic(old) && AbstractPPL.subsumes(old, optic)
            old => _remove_model_binding(R, value, AbstractPPL._unprefix_optic(optic, old))
        elseif old isa AbstractPPL.Index && optic isa AbstractPPL.Index
            _pending_remainder(old, value, optic, false, R)
        elseif _matches_model_role(R, value)
            _pending_remainder(old, value, optic, false)
        elseif AbstractPPL.subsumes(AbstractPPL.ohead(optic), AbstractPPL.ohead(old))
            old => _remove_model_binding(R, value, optic.child)
        else
            old => value
        end
        remainder === nothing && continue
        entries = _pending_set!!(entries, last(remainder), first(remainder))
        groups[first(remainder)] = _pending_group(p, old)
    end
    return _prune_model_node(PendingBindings(p.name, entries, p.base, groups))
end
function _binding_entries(
    model, pair::Pair{<:VarName,<:VarNamedTuple}, supplied=NoTemplate()
)
    return _binding_entries(model, last(pair), first(pair), supplied)
end
function _valid_model_binding(p::PendingBindings)
    return all(e -> _valid_model_binding(last(e)), p.entries)
end
_valid_model_binding(v::VarNamedTuple) = all(_valid_model_binding, values(v.data))
_has_complete_model_data(::PendingBindings) = false
function _check_namespace_index_counts(
    p::PendingBindings, optic::AbstractPPL.AbstractOptic, vn, model
)
    _pending_index_style(p, optic, vn)
    child = _model_argument_binding(p, AbstractPPL.ohead(optic))
    child === nothing || _check_namespace_index_counts(child, optic.child, vn, model)
    return nothing
end
_check_namespace_index_counts(::PendingBindings, ::AbstractPPL.Iden, vn, model) = nothing
function _check_namedtuple_index(
    p::PendingBindings, optic::AbstractPPL.AbstractOptic, prefix; kwargs...
)
    child = _model_argument_binding(p, AbstractPPL.ohead(optic))
    child === nothing || _check_namedtuple_index(
        child, optic.child, AbstractPPL.ohead(optic) ∘ prefix; kwargs...
    )
    return nothing
end
_check_namedtuple_index(::PendingBindings, ::AbstractPPL.Iden, prefix; kwargs...) = nothing
struct PendingBindingTemplate{T,N,G}
    storage::T
    name::N
    group::G
end
PendingBindingTemplate(storage, name) = PendingBindingTemplate(storage, name, nothing)
function (get::VarNamedTuples.SharedGetProperty{S})(t::PendingBindingTemplate) where {S}
    return PendingBindingTemplate(
        get(t.storage), AbstractPPL.append_optic(t.name, AbstractPPL.Property{S}()), t.group
    )
end
function VarNamedTuples.template_array(t::PendingBindingTemplate)
    return VarNamedTuples.template_array(t.storage)
end
function VarNamedTuples.index_template(t::PendingBindingTemplate, optic)
    return PendingBindingTemplate(
        VarNamedTuples.maybe_index_template(t.storage, optic),
        AbstractPPL.append_optic(t.name, AbstractPPL.ohead(optic)),
        t.group,
    )
end
function VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Property{S}, t::PendingBindingTemplate
) where {S}
    child = VarNamedTuples.make_leaf(
        value, optic.child, VarNamedTuples.SharedGetProperty{S}()(t)
    )
    return if t.storage isa NoTemplate
        PendingBindings(t.name, (AbstractPPL.Property{S}() => child,), nothing, t.group)
    else
        VarNamedTuple(NamedTuple{(S,)}((child,)))
    end
end
function _submodel_namespace(p::PendingBindings)
    return _fold_model_indices(VarNamedTuple(), p) do result, value, optic, _
        optic isa AbstractPPL.Property || _binding_storage_error(p.name)
        VarNamedTuples._setindex_optic!!(
            result, value, optic, NoTemplate(), VarNamedTuples.AllowAll()
        )
    end
end
function _binding_address_storage(
    p::PendingBindings, optic::AbstractPPL.Property{S}
) where {S}
    child = _model_argument_binding(p, AbstractPPL.ohead(optic))
    storage = _binding_address_storage(child, optic.child)
    return storage isa NoTemplate ? storage : VarNamedTuple(NamedTuple{(S,)}((storage,)))
end
function VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Index, t::PendingBindingTemplate
)
    storage = VarNamedTuples.template_array(t)
    if storage isa AbstractArray && !(storage isa VarNamedTuples.GrowableArray)
        return VarNamedTuples.make_leaf(value, optic, storage)
    end
    _pending_positions(optic.ix) === nothing && _no_index_storage_error(
        optic, AbstractPPL.append_optic(t.name, optic), "bind"; child_name=true
    )
    return PendingBindings(t.name, (optic => value,), nothing, t.group)
end
function _prepare_bindings(p::PendingBindings, ::ConflictingBindingTemplates, vn)
    message = "Binding template storage for `$vn` conflicts with its existing owner; replace the template and binding on the child."
    throw(ArgumentError(message))
end
function _fold_pending(f, check, result, value, optic, group, cursor)
    cursor[] += 1
    return f(result, optic, value, group, cursor[])
end
function _fold_pending(f, check, result, value::PendingBindings, optic, group, cursor)
    return foldl(value.entries; init=result) do result, (selector, child)
        start = cursor[] + 1
        result = _pending_expand(result, value, selector, child) do result, child, head, _
            _fold_pending(
                f,
                check,
                result,
                child,
                head ∘ optic,
                _pending_group(value, selector),
                cursor,
            )
        end
        check(selector ∘ optic, start:cursor[])
        result
    end
end
function _fold_pending(
    f,
    check,
    result,
    value::Union{VarNamedTuple,VarNamedTuples.PartialArray,ModelValueTree},
    optic,
    group,
    cursor,
)
    value isa ModelValueTree && (value = value.values)
    return _fold_model_indices(result, value) do result, child, head, _
        _fold_pending(f, check, result, child, head ∘ optic, group, cursor)
    end
end
function _prepare_bindings(p::PendingBindings, storage, vn)
    storage isa Union{AbstractArray,NamedTuple} || _binding_storage_error(vn)
    groups = Dict{VarName,Union{Nothing,Base.RefValue{Nothing}}}()
    origins = Dict{VarName,Int}()
    _fold_pending(
        (args...) -> nothing, nothing, p, AbstractPPL.Iden(), nothing, Ref(0)
    ) do _, optic, value, group, i
        address = AbstractPPL.append_optic(vn, optic)
        _check_partial_binding(
            ModelValue{Condition}(storage), optic, AbstractPPL.varname_to_optic(vn)
        )
        for name in _binding_selections(storage, optic, vn)
            haskey(groups, name) &&
                group !== nothing &&
                groups[name] === group &&
                _binding_overlap_error(address)
            groups[name], origins[name] = group, i
        end
        nothing
    end
    check = function (optic, positions)
        (isempty(positions) || any(i -> i in positions, values(origins))) || return nothing
        return _check_binding_template_bounds(
            storage, optic, AbstractPPL.append_optic(vn, optic)
        )
    end
    prepared = _fold_pending(
        check, nothing, p, AbstractPPL.Iden(), nothing, Ref(0)
    ) do result, selector, value, _, i
        names = _binding_selections(storage, selector, vn)
        for (j, name) in enumerate(names)
            origins[name] == i || continue
            optic = AbstractPPL._unprefix_optic(
                AbstractPPL.getoptic(name), AbstractPPL.getoptic(vn)
            )
            item =
                if _partial_binding_selector(storage, selector) &&
                    value isa ModelValue{<:Any,<:AbstractArray}
                    _pending_element(value, j)
                else
                    value
                end
            if item isa ModelValue
                item = _model_value_like(
                    item, _convert_binding_template(item.value, storage, optic, name)
                )
            end
            result = if result === nothing
                VarNamedTuples.make_leaf(item, optic, storage)
            else
                VarNamedTuples._setindex_optic!!(
                    result, item, optic, storage, VarNamedTuples.AllowAll()
                )
            end
        end
        result
    end
    return something(prepared, NoModelBinding())
end
function _model_argument_value(p::PendingBindings, template, vn=p.name)
    return _model_argument_value(_prepare_bindings(p, template, vn), template, vn)
end
function _prepare_child_bindings(model, values, templates, prefix)
    observations = _prepare_child_layer(Condition, model, values, templates, prefix)
    fixed = _prepare_child_layer(Fix, model, values, templates, prefix)
    _check_disjoint_bindings(observations, fixed, prefix)
    return _merge_model_values(observations, fixed)
end
function _prepare_child_layer(::Type{R}, model, values, templates, prefix) where {R}
    layer_model = _binding_layer_model(R, model)
    layer = _resolve_deferred_binding(_binding_layer(R, values), model, nothing)
    layer = _check_argument_bindings(layer_model, layer, prefix, model)
    layer = _prepare_child_locals(layer_model, layer, model, templates, prefix)
    return _prune_model_bindings(layer)
end
function _check_inherited_template_storage(node, templates, vn)
    node isa VarNamedTuples.PartialArray && node.mask isa ModelBindingArray || return nothing
    for t in templates
        t.name == vn || continue
        storage = _shape_storage(node)
        storage isa AbstractArray &&
            t.storage isa AbstractArray &&
            !_same_schema_storage(storage, t.storage) &&
            throw(
                ArgumentError(
                    "Binding template at `$vn` conflicts with existing storage; clear the binding first.",
                ),
            )
    end
    return nothing
end
function _prepare_child_locals(layer_model, layer, model, templates, prefix)
    return _fold_model_indices(copy(layer), layer) do result, node, optic, template
        vn = AbstractPPL.optic_to_varname(optic)
        AbstractPPL.getsym(vn) in _args_on_lhs(model) && return result
        _check_inherited_template_storage(node, templates, vn)
        storage = _binding_address_template(
            layer_model, _model_values(layer_model.values), vn
        )
        node = _prepare_child_local(node, storage, vn, layer_model, templates, prefix)
        VarNamedTuples._setindex_optic!!(
            result, node, optic, template, VarNamedTuples.AllowAll()
        )
    end
end
function _prepare_child_local(node, storage, vn, model, templates, prefix)
    _binding_schema_owned(model, vn) || return node
    if node isa PendingBindings &&
        (storage isa NoTemplate || storage isa VarNamedTuple) &&
        all(e -> first(e) isa AbstractPPL.Property, node.entries)
        return _prepare_child_local(
            _submodel_namespace(node), storage, vn, model, templates, prefix
        )
    end
    for t in templates
        t.name == vn || continue
        if storage isa AbstractArray &&
            t.storage isa AbstractArray &&
            !_same_schema_storage(storage, t.storage)
            throw(
                ArgumentError(
                    "Binding template at `$vn` conflicts with existing storage; clear the binding first.",
                ),
            )
        end
        node = _prepare_bindings(node, t.storage, maybe_prefix(vn, prefix))
        storage isa NoTemplate && (storage = t.storage)
    end
    return _prepare_bindings(node, storage, maybe_prefix(vn, prefix))
end
function _prepare_child_local(
    node::VarNamedTuple{names}, storage, vn, model, templates, prefix
) where {names}
    _binding_schema_owned(model, vn) || return node
    return VarNamedTuple(
        NamedTuple{names}(
            map(names) do name
                child = AbstractPPL.append_optic(vn, AbstractPPL.Property{name}())
                value =
                    hasproperty(storage, name) ? getproperty(storage, name) : NoTemplate()
                _prepare_child_local(
                    node.data[name], value, child, model, templates, prefix
                )
            end,
        ),
    )
end
function _local_binding_role(node, vn, template, model)
    node === nothing && return nothing
    template isa NoTemplate ||
        _check_bound_shape(node, template, AbstractPPL.getoptic(vn), vn, (), vn, Condition)
    if template isa NamedTuple && node isa VarNamedTuples.PartialArray
        _fold_model_indices(nothing, node) do _, _, optic, _
            _check_namedtuple_index(
                template, optic, AbstractPPL.Property{AbstractPPL.getsym(vn)}()
            )
        end
    end
    return _model_role_at(
        node, AbstractPPL.getoptic(vn), maybe_prefix(vn, _model_prefix(model))
    )
end
function _check_binding_storage(model::Model, template, vn; supplied=NoTemplate())
    local_name = _local_removal_name(model, vn; operation="bind")
    local_name !== nothing &&
        AbstractPPL.getsym(local_name) in _args_on_lhs(model) &&
        return nothing
    _pending_binding(model, vn) && return nothing
    display_vn = _binding_display_name(model, vn)
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    skip = prefix === nothing ? 0 : optic_skip_length(AbstractPPL.getoptic(prefix)) + 1
    return _require_binding_storage(
        template, AbstractPPL.getoptic(vn), display_vn; skip, supplied
    )
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

function _check_binding_owner(primary, fallback, optic, prefix)
    primary === nothing && return _check_partial_binding(fallback, optic, prefix)
    optic isa AbstractPPL.Iden && return nothing
    primary isa Union{VarNamedTuple,VarNamedTuples.PartialArray,PendingBindings} ||
        return nothing
    head = AbstractPPL.ohead(optic)
    return _check_binding_owner(
        _model_argument_binding(primary, head),
        _model_argument_binding(fallback, head),
        optic.child,
        head ∘ prefix,
    )
end
struct ResolvedBinding{N,T}
    name::N
    template::T
end

@noinline function _keyword_binding_error(name)
    throw(
        ArgumentError(
            "Entries of keyword-splat argument `$name` cannot be bound; replace the whole argument with `condition` or `fix` instead.",
        ),
    )
end

@inline function _resolve_binding(
    model, values, vn; operation="bind", supplied=NoTemplate()
)
    input_vn = _schema_binding_address(model, vn)
    display_vn = _binding_display_name(model, input_vn)
    vn = _resolve_binding_address(model, input_vn)
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
            _keyword_binding_error(argument)
        end
    end
    operation == "bind" &&
        _check_namespace_index_counts(values, AbstractPPL.varname_to_optic(vn), vn, model)
    _check_partial_binding(values, AbstractPPL.varname_to_optic(vn); operation)
    operation == "bind" && _check_binding_owner(
        values,
        _model_values(model.values),
        AbstractPPL.varname_to_optic(vn),
        AbstractPPL.Iden(),
    )
    template = _binding_address_template(model, values, vn; operation)
    template isa NoTemplate && _check_local_property_index(model, vn; operation)
    template isa NoTemplate && (template = supplied)
    _check_partial_binding(
        ModelValue{Condition}(template),
        AbstractPPL.getoptic(vn),
        AbstractPPL.Property{AbstractPPL.getsym(vn)}();
        operation,
    )
    check_fields = !any(_lhs_addresses(metadata)) do (address, submodel)
        submodel && local_name !== nothing && subsumes(address, local_name)
    end
    _check_binding_template_bounds(
        template,
        AbstractPPL.getoptic(vn),
        display_vn;
        operation,
        check_fields,
        child_name=local_name !== nothing && !_binding_schema_owned(model, local_name),
    )
    return ResolvedBinding(vn, template)
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

`decondition(m, :x)` removes explicit and argument-supplied observations.
Removed positions become latent; fixed positions remain fixed.
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

By default, only bindings stored on this model are removed, including its bindings at child
addresses. With `DynamicPPL.Recursive()`, removal also reaches enclosed models, including
argument-supplied observations and runtime bindings. With no names it clears observations
at every depth. Fixed bindings remain. Removal never restores an earlier binding.
Removing the same valid address twice is a no-op, both at the call and at evaluation.

The removal belongs to this model, moves with `prefix`, and cannot remove an enclosing
model's bindings. A recursive removal still reaches children before later bindings apply. Removals hold
no value or shape and are omitted from `conditioned`.

Named recursive removals shared by an own LHS and an unprefixed submodel throw; prefix the
child or remove without `Recursive()`. See [Binding rules](@ref).

Removing parts retains storage, including shape and element type, even if every position
is removed. Whole removal discards the role's value storage. Local geometry persists; once
both roles are cleared at an argument root, its original argument storage applies again.
Template constraints persist until replaced by a new template. Removal never restores an
observation overwritten by an earlier [`condition`](@ref).

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
    return _decondition_unfix(model, syms, Condition)
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
_prune_model_bindings_recursive(values) = _map_model_nodes(_prune_model_node, values)
_prune_model_node(value) = value
_prune_model_node(value::VarNamedTuples.PartialArray{<:ModelValue}) = value
_prune_model_node(::VarNamedTuples.PartialArray{<:NoModelBinding}) = NoModelBinding()
function _prune_model_node(value::VarNamedTuple)
    value = _prune_model_bindings(value)
    return isempty(value.data) ? NoModelBinding() : value
end

function _prune_model_node(tree::ModelValueTree)
    return ModelValueTree(tree, _prune_model_bindings(tree.values))
end
function _prune_model_node(values::VarNamedTuples.PartialArray)
    result = _map_model_nodes(_prune_model_node, values)
    return isempty(result) ? NoModelBinding() : result
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
    T === Union{} && (T = ModelValue{R,eltype(value)})
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

function _remove_model_binding(
    ::Type{R}, values::VarNamedTuple, optic::AbstractPPL.Iden
) where {R}
    return VarNamedTuple(map(value -> _remove_model_binding(R, value, optic), values.data))
end
function _remove_model_binding(
    ::Type{R}, value::VarNamedTuples.PartialArray{<:ModelValue{S}}, ::AbstractPPL.Iden
) where {R,S}
    return if (R === S || R === Condition && S === ArgumentCondition)
        NoModelBinding()
    else
        value
    end
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
    values = _remove_model_binding(R, tree.values, optic)
    observed = tree.observed && (R !== Condition || isequal(values, tree.values))
    return ModelValueTree(tree.template, values, Val(_inherits_shape(tree)), observed)
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
    multiindex && isempty(selected) && return original
    values === original || (selected = _resize_model_binding(selected, axes(selected)))
    child = _remove_model_binding(R, selected, optic.child)
    child === selected && return original
    template, values = values, owned ? values : copy(values)
    if R === Condition && values.mask isa ModelBindingArray && values.mask.observed
        values = VarNamedTuples.PartialArray(
            values.data, ModelBindingArray(values.mask.data, values.mask.template)
        )
    end
    # Removed elements are unmasked in place, so many removals share one copy.
    if child isa NoModelBinding &&
        !(getindex(values.data, optic.ix...; optic.kw...) isa VarNamedTuples.ArrayLikeBlock)
        if multiindex
            view(values.mask, optic.ix...; optic.kw...) .= false
        else
            setindex!(values.mask, false, optic.ix...; optic.kw...)
        end
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

The result lists only observations stored on `model`, including argument-supplied
observations and explicit bindings at child addresses. It excludes bindings stored on
submodels, whose construction and runtime bindings may depend on evaluation.

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
    Condition, _listing_argument_children(model, _public_model_values(model))
)

"""
    fix(model::Model; values...)
    fix(model::Model, values..., [template])

Return a `Model` which treats the LHS variables bound by `values` as constants: they replace
sampling and contribute no log probability. Fixed argument LHS variables reset to their bound
value when their tilde statement runs, even if the body has computed a different value.

Each position has at most one binding across roles and models. Remove an observation
before fixing its position, and remove a child's binding before binding from an ancestor.
Later calls may replace this model's own fixed values. [`unfix`](@ref) makes removed
positions latent; it restores no observation. Keep the original model to restore a temporary fix.

Bindings reach child addresses too. Inputs, unsupported partial bindings, conversion errors,
aliasing, copy timing and argument preparation follow [`condition`](@ref).

Fixed values cover their contents; omitted NamedTuple fields remain latent. At an
executed fixed tilde, an array size or shape mismatch along the address path throws
`ArgumentError` naming the LHS variable. This covers arguments and locals, with whole or
partial fixed bindings. The check does not inspect inside a whole structured value or detect
in-place mutation of a bound value. Partial bindings at positions no executed tilde reads
are reported by [`check_model`](@ref), not rejected by evaluation. See
[Size checks at executed tildes](@ref binding-size-checks).

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
    return _conditionfix(model, inputs, Fix)
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
runtime bindings. With no names it clears fixed bindings at every depth.
Named removals, prefixing, and later replacement follow [`decondition`](@ref).
Removals hold no value or shape and are omitted from `fixed`.

Removed positions become latent. For `@model f(x) = x ~ Normal()`,
`fix(f(1.0); x=5.0)` throws because `x` is observed. Use
`fix(decondition(f(1.0), :x); x=5.0)`; removing that fix leaves `x` latent.
Shared binding shape persists, except that whole removal of both roles at an argument root
restores the argument's shape. An observed array with immutable storage cannot become latent or fixed
through removal; rebuild it with mutable storage, for example `f(collect(data))`.

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

julia> # You can also provide the symbol directly (though the `@varname`
       # approach is preferable if the LHS variable is known at compile-time).
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
    return _decondition_unfix(model, syms, Fix)
end

function _decondition_unfix(model, names, ::Type{R}, recursive...) where {R}
    if model.values isa LocalModelValues && _model_prefix(model) !== nothing
        names = map(
            n -> _public_binding_name(
                model, n isa Symbol ? VarName{n}() : n; operation="remove"
            ),
            names,
        )
        any(isnothing, names) && (names = ())
    end
    removed = _unbind_values(model, names, R, recursive...)
    _check_shared_latent_storage(removed)
    return _check_latent_arguments(removed)
end

# An empty `names` removes every binding of role `R`.
function _unbind_values(model, names, ::Type{R}) where {R}
    # Child addresses bound here are stored here, so they are removed here.
    model = _materialize_argument_values(model)
    layer = _binding_layer(R, model.values)
    names = _check_removal_names(model, layer, names)
    _check_removal_addresses(layer, names)
    layer = if isempty(names)
        _remove_model_values(R, layer)
    else
        _remove_model_values(R, layer, names)
    end
    observations = R === Condition ? layer : _observation_values(model.values)
    fixed_values = R === Fix ? layer : _fixed_values(model.values)
    values =
        if isempty(fixed_values.data) &&
            isempty(_binding_shape_store(model.values).roots.data)
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
    values = _with_binding_metadata(
        values;
        observation_removals=_removals(Condition, model.values),
        fixed_removals=_removals(Fix, model.values),
        templates=_surviving_binding_templates(model.values, values, R),
        shapes=_binding_shape_store(model.values),
    )
    return _reconstruct_model(model; values)
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

The result lists only fixed bindings stored on `model`, including explicit bindings at
child addresses. It excludes bindings stored on submodels, whose construction and runtime
bindings may depend on evaluation.

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
fixed(model::Model) = _select_model_values(
    Fix, _listing_argument_children(model, _public_model_values(model))
)

# Recursive removals are stored separately from values and therefore own no shape.
_removals(::Type, values) = ()
_removals(::Type{Condition}, values::ModelBindingLayers) = values.observation_removals
_removals(::Type{Fix}, values::ModelBindingLayers) = values.fixed_removals
_removals(R::Type, values::LocalModelValues) = _removals(R, values.values)

function _prefix_removal(r::ModelRemoval, prefix)
    return ModelRemoval(maybe_prefix(r.name, prefix), r.matched, r.token)
end
function _removal_covers(r, vn)
    return r.name === nothing || subsumes(r.name, vn)
end

_has_binding_role(R, value) = _matches_model_role(R, value)
function _has_binding_role(R, p::PendingBindings)
    return _has_binding_role(R, p.base) ||
           any(e -> _has_binding_role(R, last(e)), p.entries)
end
_has_binding_role(R, v::VarNamedTuple) = any(x -> _has_binding_role(R, x), values(v.data))
function _has_removable(::Type{R}, values, vn) where {R}
    binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
    VarNamedTuples._mapreduce_recursive(
        pair -> _has_binding_role(R, pair.second), |, binding, vn, false
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

    return removed
end

function _local_removal_name(model, vn; operation="remove")
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    return _binding_local_name(vn, prefix, operation)
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

function _check_removal_names(model, layer, names)
    return map(
        vn -> _check_removal_name(model, layer, vn isa Symbol ? VarName{vn}() : vn), names
    )
end
function _check_removal_names(model, layer, names::AbstractVector)
    return map!(similar(names, VarName), names) do vn
        _check_removal_name(model, layer, vn isa Symbol ? VarName{vn}() : vn)
    end
end

@inline function _check_removal_name(model, layer, vn)
    resolved = _resolve_binding(model, layer, vn; operation="remove")
    AbstractPPL.is_dynamic(AbstractPPL.getoptic(resolved.name)) || return resolved.name
    return _concretize_prefix(resolved.name, resolved.template)
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

function _unbind_values(model, names, ::Type{R}, ::Recursive) where {R}
    model = _materialize_argument_values(model)
    values = _binding_layer(R, model.values)
    markers = _removals(R, model.values)
    requested =
        isempty(names) ? (nothing,) : map(n -> n isa Symbol ? VarName{n}() : n, names)
    requested = mapreduce((a, b) -> (a..., b...), requested; init=()) do vn
        vn === nothing && return (nothing,)
        resolved = _check_removal_name(model, values, vn)
        return _resolved_removal_names(model, resolved)
    end
    for vn in requested
        vn = vn === nothing ? nothing : _check_removal_name(model, values, vn)
        matched = vn === nothing ? !isempty(values) : _has_removable(R, values, vn)
        marker = ModelRemoval(vn, matched)
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
    layers = _with_binding_metadata(
        layers; templates=_surviving_binding_templates(model.values, layers, R)
    )
    layers = _with_binding_shapes(layers, _binding_shape_store(model.values))
    return _reconstruct_model(model; values=layers)
end
function AbstractPPL.decondition(model::Model, ::Recursive, names::Union{Symbol,VarName}...)
    return _decondition_unfix(model, names, Condition, Recursive())
end
function unfix(model::Model, ::Recursive, names::Union{Symbol,VarName}...)
    return _decondition_unfix(model, names, Fix, Recursive())
end

function _select_removal(r, prefix)
    prefix === nothing && return (r,)
    name = if r.name === nothing || subsumes(r.name, prefix)
        nothing
    elseif subsumes(prefix, r.name)
        AbstractPPL.unprefix(r.name, prefix)
    else
        return ()
    end
    return (ModelRemoval(name, r.matched, r.token),)
end
function _submodel_removals(::Type{R}, model, prefix) where {R}
    isempty(_removals(R, model.values)) && return ()
    vn = _model_value_varname(model.values, prefix, _model_prefix(model))
    resolved = vn === nothing ? nothing : _resolve_binding_address(model, vn)
    return mapreduce(
        r -> _select_existing_removal(model, r, resolved),
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
        template = _binding_address_template(
            local_model, values, r.name; operation="remove"
        )
        _check_partial_binding(
            ModelValue{Condition}(template),
            AbstractPPL.getoptic(r.name),
            AbstractPPL.Property{AbstractPPL.getsym(r.name)}();
            operation="remove",
        )
    end
    matched = r.name === nothing ? !isempty(values) : _has_removable(R, values, r.name)
    matched && _record_removal_use(context, R, r)
    matched && (values = _remove_marked(R, values, r))
    return _apply_parent_removals(R, values, Base.tail(markers), context, model)
end

function _submodel_inherited_values(model, prefix)
    prefix = _model_value_varname(model.values, prefix, _model_prefix(model))
    values = model.values
    if prefix !== nothing
        _model_role_at(
            _model_values(values),
            AbstractPPL.varname_to_optic(prefix),
            prefix,
            _namespace_leaf,
        )
    end
    return _inherited_namespace(values, prefix)
end
_namespace_leaf(value, vn) = nothing
# Dense namespace arrays contain named children, never array-like blocks. Their
# data and mask have the same axes, so one bounds check suffices for a scalar read.
@inline function _model_argument_binding(
    values::VarNamedTuples.PartialArray{<:VarNamedTuple,N,<:Array,<:Array},
    optic::AbstractPPL.Index{<:Tuple{Vararg{Integer}},NamedTuple{(),Tuple{}}},
) where {N}
    checkbounds(Bool, values, optic.ix...) || return nothing
    @inbounds values.mask[optic.ix...] || return nothing
    selected = @inbounds values.data[optic.ix...]
    return _model_argument_binding(selected, optic.child)
end
function _inherited_namespace(values, prefix)
    return _inherited_model_values(_submodel_values(_model_values(values), prefix))
end
function _inherited_namespace(values::LocalModelValues, prefix)
    return _inherited_namespace(values.values, prefix)
end
@inline function _inherited_namespace(
    values::ModelBindingLayers{<:VarNamedTuple,VarNamedTuple{(),Tuple{}}}, prefix
)
    return _inherited_model_values(_submodel_values(values.observations, prefix))
end
function _inherited_namespace(values::ModelBindingLayers, prefix)
    observations = _inherited_model_values(_submodel_values(values.observations, prefix))
    fixed = _inherited_model_values(_submodel_values(values.fixed, prefix))
    return _overlay_model_values(observations, fixed, ())
end

struct ScalarBindingGeometry end
struct UnknownBindingGeometry end
struct ArrayBindingGeometry{A,C}
    axes::A
    children::C
end
struct FieldBindingGeometry{C<:NamedTuple}
    children::C
end

struct RecursiveBindingGeometry
    ancestor::Int
end

_binding_geometry(value) = _binding_geometry(value, ())
_geometry_fields(value::Union{Number,AbstractString,Symbol,Type,Nothing,Missing}) = nothing
_geometry_fields(value::NamedTuple) = keys(value)
function _geometry_fields(value)
    T = typeof(value)
    return parentmodule(T) in (Base, Core) || fieldcount(T) == 0 ? nothing : fieldnames(T)
end
function _binding_geometry(value, parents::Tuple)
    value isa Base.Pairs && return _binding_geometry(values(value), parents)
    names = _geometry_fields(value)
    names === nothing && return ScalarBindingGeometry()
    ancestor = findfirst(v -> v === value, parents)
    ancestor === nothing || return RecursiveBindingGeometry(ancestor)
    children = map(names) do i
        if isdefined(value, i)
            _binding_geometry(getfield(value, i), (parents..., value))
        else
            UnknownBindingGeometry()
        end
    end
    return FieldBindingGeometry(NamedTuple{names}(children))
end
function _binding_geometry(
    value::Union{AbstractArray,Tuple,VarNamedTuples.PartialArray}, parents::Tuple
)
    value isa Union{SubsamplingShape,VarNamedTuples.PartialArray} &&
        return _binding_geometry(value.data, parents)
    if _shape_payload_type(eltype(value)) <: Number
        return ArrayBindingGeometry(axes(value), nothing)
    end
    ancestor = findfirst(v -> v === value, parents)
    ancestor === nothing || return RecursiveBindingGeometry(ancestor)
    children = map(eachindex(value)) do i
        if value isa Tuple || isassigned(value, i)
            _binding_geometry(value[i], (parents..., value))
        else
            UnknownBindingGeometry()
        end
    end
    if !isempty(children) && all(g -> g === first(children), children)
        children = Fill(first(children), axes(children))
    end
    return ArrayBindingGeometry(axes(value), children)
end
_same_binding_geometry(::ScalarBindingGeometry, ::ScalarBindingGeometry) = true
_same_binding_geometry(::UnknownBindingGeometry, b) = true
_same_binding_geometry(a, ::UnknownBindingGeometry) = true
_same_binding_geometry(::UnknownBindingGeometry, ::UnknownBindingGeometry) = true
_same_binding_geometry(a, b) = false
function _same_binding_geometry(a::RecursiveBindingGeometry, b::RecursiveBindingGeometry)
    return a.ancestor == b.ancestor
end
function _same_binding_geometry(a::ArrayBindingGeometry, b::ArrayBindingGeometry)
    a.axes == b.axes || return false
    a.children === nothing && b.children === nothing && return true
    a.children === nothing &&
        return all(x -> _same_binding_geometry(ScalarBindingGeometry(), x), b.children)
    b.children === nothing &&
        return all(x -> _same_binding_geometry(x, ScalarBindingGeometry()), a.children)
    a.children isa Fill &&
        b.children isa Fill &&
        return _same_binding_geometry(first(a.children), first(b.children))
    return all(_same_binding_geometry(x, y) for (x, y) in zip(a.children, b.children))
end
function _same_binding_geometry(a::FieldBindingGeometry, b::FieldBindingGeometry)
    return keys(a.children) == keys(b.children) &&
           all(_same_binding_geometry(x, y) for (x, y) in zip(a.children, b.children))
end

_refine_binding_geometry(a, b) = a
_refine_binding_geometry(::UnknownBindingGeometry, b) = b
function _refine_binding_geometry(a::ArrayBindingGeometry, b::ArrayBindingGeometry)
    a.children === nothing && return a
    b.children === nothing && return ArrayBindingGeometry(
        a.axes,
        map(x -> _refine_binding_geometry(x, ScalarBindingGeometry()), a.children),
    )
    return ArrayBindingGeometry(
        a.axes, map(_refine_binding_geometry, a.children, b.children)
    )
end
function _refine_binding_geometry(a::FieldBindingGeometry, b::FieldBindingGeometry)
    return FieldBindingGeometry(map(_refine_binding_geometry, a.children, b.children))
end

# The roots are leaves in an existing VarNamedTuple tree. PartialArray nodes group
# namespace children; their mask records roots, never observation/fixed coverage.
struct BindingShape{G}
    geometry::G
end
function _submodel_namespace(shape::BindingShape{<:FieldBindingGeometry})
    return VarNamedTuple(map(BindingShape, shape.geometry.children))
end
struct ModelBindingShapes{R}
    roots::R
end
ModelBindingShapes() = ModelBindingShapes(VarNamedTuple())
_shape_root(x) = BindingShape(_binding_geometry(x))
_shape_payload_type(::Type{T}) where {T} = T
_shape_payload_type(::Type{<:ModelValue{R,T}}) where {R,T} = T
_binding_geometry(x::ModelValue, parents::Tuple) = _binding_geometry(x.value, parents)
function _binding_geometry(x::ModelValueTree, parents::Tuple)
    return _binding_geometry(x.template, parents)
end
_shape_storage(x, fallback=NoTemplate()) = _model_data(x)
function _shape_storage(x::VarNamedTuple, base::Union{NamedTuple,NoTemplate}=NoTemplate())
    base isa NoTemplate && (base = NamedTuple())
    storage = merge(base, x.data)
    return map(_shape_storage, storage, merge(storage, base))
end
function _shape_storage(x::VarNamedTuples.PartialArray, fallback=NoTemplate())
    T = _shape_payload_type(eltype(x))
    T <: Number && isbitstype(T) && return similar(x.data, T)
    storage = similar(x.data, Core.Compiler.return_type(_shape_storage, Tuple{eltype(x)}))
    for i in eachindex(x.data)
        child = if fallback isa AbstractArray && isassigned(fallback, i)
            fallback[i]
        else
            NoTemplate()
        end
        if isassigned(x.data, i)
            storage[i] = _shape_storage(x.data[i], child)
        elseif !(child isa NoTemplate)
            storage[i] = child
        end
    end
    return storage
end

struct BindingGeometryArray{N,G,O,P} <: AbstractArray{Any,N}
    geometry::G
    optic::O
    previous::P
end
function BindingGeometryArray(g, optic=nothing, previous=nothing)
    ax = _shape_axes(g)
    return BindingGeometryArray{length(ax),typeof(g),typeof(optic),typeof(previous)}(
        g, optic, previous
    )
end
function Base.axes(a::BindingGeometryArray)
    return _shape_axes(a.geometry)
end
Base.size(a::BindingGeometryArray) = map(length, axes(a))
Base.similar(a::BindingGeometryArray, ::Type{T}) where {T} = similar(Array{T}, axes(a))
function Base.getindex(a::BindingGeometryArray{N}, indices::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, indices...)
    g = a.geometry
    if g isa ArrayBindingGeometry
        return if g.children === nothing
            NoTemplate()
        else
            _geometry_template(g.children[LinearIndices(axes(a))[indices...]])
        end
    end
    return _shape_template(
        _model_argument_binding(g, AbstractPPL.Index(indices, (;))),
        a.optic.child,
        _model_argument_binding(a.previous, AbstractPPL.Index(indices, (;))),
    )
end
function Base.isassigned(a::BindingGeometryArray, indices::Vararg{Integer})
    checkbounds(Bool, a, indices...) || return false
    g = a.geometry
    return !(g isa ArrayBindingGeometry) ||
           g.children === nothing ||
           !(g.children[LinearIndices(axes(a))[indices...]] isa UnknownBindingGeometry)
end
_geometry_template(geometry) = NoTemplate()
function _geometry_template(geometry::ArrayBindingGeometry)
    if geometry.children === nothing &&
        !all(a -> a isa Union{Base.OneTo,UnitRange}, geometry.axes)
        return fill!(similar(Array{NoTemplate}, geometry.axes), NoTemplate())
    end
    return BindingGeometryArray(geometry)
end
function _geometry_template(geometry::FieldBindingGeometry)
    return map(_geometry_template, geometry.children)
end
_partial_binding_array(::BindingGeometryArray) = true
function VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Index, template::BindingGeometryArray
)
    child = VarNamedTuples.make_sub_value(value, optic, template)
    data = similar(template, typeof(child))
    array = VarNamedTuples.PartialArray(data, fill!(similar(data, Bool), false))
    return VarNamedTuples._setindex_optic!!(
        array, value, optic, template, VarNamedTuples.AllowAll()
    )
end

function _snapshot_shape(p::PendingBindings)
    entries = _fold_model_indices(OrderedDict{Union{},Union{}}(), p) do out, value, optic, _
        _pending_set!!(out, _snapshot_shape(value), optic)
    end
    shape = PendingBindings(p.name, entries)
    fields = all(e -> first(e) isa AbstractPPL.Property, entries)
    return fields ? _submodel_namespace(shape) : shape
end
_snapshot_shape(x) = _shape_root(x)
_snapshot_shape(::ModelValue{ArgumentCondition}) = NoModelBinding()
function _snapshot_shape(x::VarNamedTuple)
    return _prune_model_bindings(VarNamedTuple(map(_snapshot_shape, x.data)))
end
function _shape_template(g::PendingBindings, optic::AbstractPPL.Index, previous=nothing)
    g.base isa VarNamedTuples.PartialArray &&
        return BindingGeometryArray(g, optic, previous)
    g.base === nothing || return _shape_template(g.base, optic, previous)
    child = _model_argument_binding(g, AbstractPPL.ohead(optic))
    return VarNamedTuples.SkipTemplate{1}(
        _shape_template(
            child, optic.child, _model_argument_binding(previous, AbstractPPL.ohead(optic))
        ),
    )
end
struct ShapeOp{Build,M,W}
    model::M
    whole::W
end
function ShapeOp{Build}(model, whole) where {Build}
    return ShapeOp{Build,typeof(model),typeof(whole)}(model, whole)
end
@inline function (op::ShapeOp{Build})(
    a::PendingBindings, b::PendingBindings, vn
) where {Build}
    if a.base isa VarNamedTuples.PartialArray
        b.base === nothing && (b = _namespace_binding(b, AbstractPPL.Iden(), a.base))
        Build && return op(_materialize_pending(a), b, vn)
    end
    Build || return _check_shape_children(op, a, b, vn)
    entries = _pending_dict(a.entries)
    for (optic, value) in b.entries
        if haskey(entries, optic) && !(value isa BindingShape)
            value = op(entries[optic], value, AbstractPPL.append_optic(vn, optic))
        end
        entries = _pending_set!!(entries, value, optic)
    end
    return PendingBindings(a.name, entries, a.base)
end
function (op::ShapeOp)(a::PendingBindings, b::VarNamedTuple, vn)
    return op(a, _pending_fields(b, a.name), vn)
end
function (op::ShapeOp)(a::VarNamedTuple, b::PendingBindings, vn)
    return op(_pending_fields(a, b.name), b, vn)
end
@inline function (op::ShapeOp{Build})(a::PendingBindings, b::BindingShape, vn) where {Build}
    a.base === nothing || return op(a.base, b, vn)
    all(e -> last(e) isa BindingShape, a.entries) && return Build ? b : ()
    return _binding_root_error(vn)
end
function _snapshot_shape(x::VarNamedTuples.PartialArray)
    if eltype(x) <: VarNamedTuple
        T = Core.Compiler.return_type(_snapshot_shape, Tuple{eltype(x)})
        data = similar(x.data, T)
        for i in eachindex(x.mask)
            x.mask[i] && (data[i] = _snapshot_shape(x.data[i]))
        end
        return VarNamedTuples.PartialArray(data, copy(x.mask))
    end
    x.data isa VarNamedTuples.GrowableArray && return nothing
    return _shape_root(x)
end
_resolve_binding_optic(storage, optic::AbstractPPL.Iden) = optic
function _resolve_binding_optic(storage, optic::AbstractPPL.Property{S}) where {S}
    storage isa VarNamedTuple && (storage = storage.data)
    child = hasproperty(storage, S) ? getproperty(storage, S) : NoTemplate()
    return AbstractPPL.Property{S}(_resolve_binding_optic(child, optic.child))
end
function _resolve_binding_optic(
    storage::VarNamedTuples.SkipTemplate, optic::AbstractPPL.Index
)
    return AbstractPPL.Index(
        optic.ix,
        optic.kw,
        _resolve_binding_optic(
            VarNamedTuples.maybe_index_template(storage, optic), optic.child
        ),
    )
end
function _resolve_binding_optic(storage, optic::AbstractPPL.Index)
    storage isa AbstractArray || return optic
    storage isa VarNamedTuples.GrowableArray && return optic
    optic = AbstractPPL.concretize_top_level(optic, storage)
    checkbounds(Bool, storage, optic.ix...; optic.kw...) || return optic
    indices = if isempty(optic.kw) && all(i -> i isa Integer, optic.ix)
        Tuple(CartesianIndices(axes(storage))[optic.ix...])
    else
        optic.ix
    end
    child = VarNamedTuples.maybe_index_template(storage, optic)
    return AbstractPPL.Index(indices, optic.kw, _resolve_binding_optic(child, optic.child))
end

_check_argument_shape(storage, value, vn) = nothing
function _check_argument_shape(storage, value::ModelValue, vn)
    storage isa ModelValue && (storage = storage.value)
    storage isa Union{NoTemplate,Nothing,Missing} && return nothing
    _same_storage_geometry(storage, value.value) || _binding_shape_error(vn)
    return nothing
end
function _check_argument_shape(storage, value::VarNamedTuple, vn)
    for (name, child) in pairs(value.data)
        hasproperty(storage, name) && _check_argument_shape(
            getproperty(storage, name),
            child,
            AbstractPPL.append_optic(vn, AbstractPPL.Property{name}()),
        )
    end
    return nothing
end
function _check_argument_shape(storage, value::VarNamedTuples.PartialArray, vn)
    storage isa AbstractArray || return nothing
    axes(storage) == axes(value.data) || _binding_shape_error(vn)
    for i in eachindex(value.mask)
        value.mask[i] &&
            isassigned(storage, i) &&
            _check_argument_shape(storage[i], value.data[i], vn)
    end
    return nothing
end
function _same_storage_geometry(a, b, parents=())
    a isa Base.Pairs && return _same_storage_geometry(values(a), b, parents)
    b isa Base.Pairs && return _same_storage_geometry(a, values(b), parents)
    (a === missing || b === missing) && return true
    array = a isa Union{AbstractArray,Tuple}
    array == (b isa Union{AbstractArray,Tuple}) || return false
    if array
        axes(a) == axes(b) || return false
        eltype(a) <: Number && eltype(b) <: Number && return true
        names = eachindex(a)
    else
        names = _geometry_fields(a)
        names == _geometry_fields(b) || return false
        names === nothing && return true
    end
    i, j = findfirst(p -> first(p) === a, parents), findfirst(p -> last(p) === b, parents)
    (i !== nothing || j !== nothing) && return i == j
    return all(names) do name
        if array
            (!(a isa Tuple) && !isassigned(a, name)) ||
                (!(b isa Tuple) && !isassigned(b, name)) ||
                _same_storage_geometry(a[name], b[name], (parents..., (a, b)))
        else
            !isdefined(a, name) ||
                !isdefined(b, name) ||
                _same_storage_geometry(
                    getfield(a, name), getfield(b, name), (parents..., (a, b))
                )
        end
    end
end
_binding_address_exists(storage, ::AbstractPPL.Iden) = true
function _binding_address_exists(storage, optic::AbstractPPL.Property{S}) where {S}
    storage isa NoTemplate && return true
    storage isa VarNamedTuple && (storage = storage.data)
    hasproperty(storage, S) || return false
    return _binding_address_exists(getproperty(storage, S), optic.child)
end
function _binding_address_exists(storage, optic::AbstractPPL.Index)
    storage isa NoTemplate && return true
    storage isa VarNamedTuples.SkipTemplate &&
        return _binding_address_exists(_binding_child_template(storage, optic), optic.child)
    storage isa AbstractArray || return false
    checkbounds(Bool, storage, optic.ix...; optic.kw...) || return false
    return optic.child isa AbstractPPL.Iden || _binding_address_exists(
        VarNamedTuples.maybe_index_template(storage, optic), optic.child
    )
end
function _select_existing_removal(model, removal, prefix)
    vn = removal.name
    if vn !== nothing
        storage = _binding_shape_template(model, vn)
        _binding_address_exists(storage, AbstractPPL.getoptic(vn)) || return ()
        removal = ModelRemoval(
            _resolve_binding_address(model, vn), removal.matched, removal.token
        )
    end
    return _select_removal(removal, prefix)
end

function _resolved_removal_names(model, vn)
    storage = _binding_shape_template(model, vn)
    storage isa NoTemplate && return (vn,)
    return Tuple(
        _binding_selections(
            storage, AbstractPPL.getoptic(vn), VarName{AbstractPPL.getsym(vn)}()
        ),
    )
end
function _binding_index_storage(storage)
    data = copyto!(similar(storage, Int), LinearIndices(storage))
    return VarNamedTuples.PartialArray(data, fill!(similar(storage, Bool), true))
end
_binding_selections(storage, optic, vn) = _binding_selections(storage, optic, vn, nothing)
_binding_selections(storage, ::AbstractPPL.Iden, vn, cache) = (vn,)
function _binding_selections(storage, optic::AbstractPPL.Property{S}, vn, cache) where {S}
    storage = storage isa BindingShape ? _geometry_template(storage.geometry) : storage
    if _partial_binding_selector(storage, optic)
        indices = if cache === nothing
            _binding_index_storage(storage)
        else
            get!(() -> _binding_index_storage(storage), cache, vn)
        end
        _check_binding_template_bounds(storage, optic, AbstractPPL.append_optic(vn, optic))
        selected = VarNamedTuples._getindex_optic(indices, optic, vn)
        positions = selected isa Integer ? (selected,) : selected
        return (
            AbstractPPL.append_optic(
                vn, AbstractPPL.Index(Tuple(CartesianIndices(axes(storage))[index]), (;))
            ) for index in positions
        )
    end
    storage isa VarNamedTuple && (storage = storage.data)
    child = hasproperty(storage, S) ? getproperty(storage, S) : NoTemplate()
    return _binding_selections(
        child, optic.child, AbstractPPL.append_optic(vn, AbstractPPL.Property{S}()), cache
    )
end
function _binding_selections(storage, optic::AbstractPPL.Index, vn, cache)
    storage = storage isa BindingShape ? _geometry_template(storage.geometry) : storage
    storage === nothing && return _binding_selections(NoTemplate(), optic, vn, cache)
    storage isa AbstractArray || return (AbstractPPL.append_optic(vn, optic),)
    storage isa VarNamedTuples.GrowableArray &&
        return (AbstractPPL.append_optic(vn, optic),)
    address = AbstractPPL.append_optic(vn, optic)
    selected = _binding_select(CartesianIndices(axes(storage)), optic, address)
    return _binding_selection_names(storage, selected, optic.child, vn, cache, address)
end
function _binding_select(storage, optic, vn)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    checkbounds(Bool, storage, optic.ix...; optic.kw...) ||
        _outside_storage_error(optic, vn, "bind")
    return getindex(storage, optic.ix...; optic.kw...)
end
function _binding_selection_names(storage, selected, ::AbstractPPL.Iden, vn, cache, address)
    positions = selected isa CartesianIndex ? (selected,) : selected
    return map(positions) do index
        AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(index), (;)))
    end
end
function _binding_selection_names(storage, selected, child, vn, cache, address)
    while selected isa AbstractArray && child isa AbstractPPL.Index
        selected = _binding_select(selected, child, address)
        child = child.child
    end
    positions = selected isa CartesianIndex ? (selected,) : selected
    return let child = child
        mapreduce(vcat, positions; init=Union{}[]) do index
            step = AbstractPPL.Index(Tuple(index), (;))
            name = AbstractPPL.append_optic(vn, step)
            value =
                if storage isa AbstractArray &&
                    isassigned(storage, index) &&
                    !(storage isa VarNamedTuples.PartialArray && !storage.mask[index])
                    storage[index]
                else
                    NoTemplate()
                end
            collect(_binding_selections(value, child, name, cache))
        end
    end
end

# True replaces a released root; false keeps its parts pending.
struct ShapeRules{R<:Tuple{Vararg{Pair{<:VarName,Bool}}}}
    roots::R
end
_shape_occupied(r::ShapeRules, vn) = !any(p -> first(p) == vn, r.roots)
_whole_shape(r::ShapeRules, vn) = any(p -> first(p) == vn && last(p), r.roots)
struct InheritedShapes{A,B,N,R}
    child::A
    parent::B
    name::N
    rules::R
end
function _namespace_binding(p::PendingBindings, ::AbstractPPL.Iden, storage)
    p.base isa Union{BindingShape,ModelValue} && _binding_root_error(p.name)
    return PendingBindings(p.name, p.entries, storage, p.groups)
end
function _namespace_binding(
    x::VarNamedTuple, optic::AbstractPPL.Property{S}, storage
) where {S}
    haskey(x.data, S) || return x
    child = _namespace_binding(x.data[S], optic.child, storage)
    return VarNamedTuple(merge(x.data, NamedTuple{(S,)}((child,))))
end
function _namespace_binding(
    x::Union{BindingShape,VarNamedTuples.PartialArray,Nothing}, ::AbstractPPL.Iden, storage
)
    return x
end
function _namespace_binding(x::InheritedShapes, optic, storage)
    return _map_inherited_shapes(v -> _namespace_binding(v, optic, storage), x)
end
_shape_axes(g) = nothing
_shape_axes(g::Union{AbstractArray,VarNamedTuples.PartialArray}) = axes(g)
_shape_axes(g::ArrayBindingGeometry) = g.axes
_shape_axes(g::PendingBindings) = _shape_axes(g.base)
function _shape_axes(g::InheritedShapes)
    ax = _shape_axes(g.parent)
    return ax === nothing ? _shape_axes(g.child) : ax
end
_is_shape_root(x) = x isa BindingShape
_is_shape_root(x::InheritedShapes) = _is_shape_root(x.child) || _is_shape_root(x.parent)
function _shape_template(x::InheritedShapes, optic, previous=nothing)
    if !_is_shape_root(x)
        if optic isa AbstractPPL.Property
            return _property_shape_template(x, optic, previous)
        elseif optic isa AbstractPPL.Index && _shape_axes(x) !== nothing
            return BindingGeometryArray(x, optic, previous)
        end
    end
    return _map_inherited_shapes(v -> _shape_template(v, optic, previous), x)
end
function _resolve_inherited_optic(x, optic)
    return _resolve_binding_optic(x.parent, _resolve_binding_optic(x.child, optic))
end
_resolve_binding_optic(x::InheritedShapes, optic::AbstractPPL.Iden) = optic
function _resolve_binding_optic(x::InheritedShapes, optic::AbstractPPL.Property)
    return _resolve_inherited_optic(x, optic)
end
function _resolve_binding_optic(x::InheritedShapes, optic::AbstractPPL.Index)
    return _resolve_inherited_optic(x, optic)
end
_binding_address_exists(x::InheritedShapes, optic::AbstractPPL.Iden) = true
function _binding_address_exists(x::InheritedShapes, optic::AbstractPPL.Property)
    return _binding_address_exists(x.child, optic) &&
           _binding_address_exists(x.parent, optic)
end
function _binding_address_exists(x::InheritedShapes, optic::AbstractPPL.Index)
    return _binding_address_exists(x.child, optic) &&
           _binding_address_exists(x.parent, optic)
end
_materialize_shapes(x) = x
function _materialize_shapes(x::InheritedShapes)
    return _merge_binding_shapes(x.child, x.parent, x.name, x.rules, x.rules)
end
_prepared_shapes(values) = ModelBindingShapes()
_prepared_shapes(values::ModelBindingLayers) = values.shapes
_prepared_shapes(values::LocalModelValues) = _prepared_shapes(values.values)
function _binding_shape_store(values)
    return ModelBindingShapes(_materialize_shapes(_prepared_shapes(values).roots))
end
function _inherited_binding_shapes(a, b, vn, rules)
    a === b && return a
    a === nothing && return b
    b === nothing && return a
    a isa VarNamedTuple{(),Tuple{}} && return b
    b isa VarNamedTuple{(),Tuple{}} && return a
    if a isa PendingBindings &&
        b isa PendingBindings &&
        a.base isa VarNamedTuples.PartialArray &&
        b.base === nothing
        b = _namespace_binding(b, AbstractPPL.Iden(), a.base)
    end
    if a isa PendingBindings && b isa BindingShape
        a.base === nothing && return b
        return _inherited_binding_shapes(a.base, b, vn, rules)
    end
    if a isa BindingShape && b isa BindingShape && _whole_shape(rules, vn)
        return b
    end
    return InheritedShapes(a, b, vn, rules)
end
_shape_child_name(vn, optic) = AbstractPPL.append_optic(vn, optic)
_shape_child_name(::Nothing, optic) = AbstractPPL.optic_to_varname(optic)
function _map_inherited_shapes(f, x::InheritedShapes, vn=x.name)
    return _inherited_binding_shapes(f(x.child), f(x.parent), vn, x.rules)
end
_model_argument_binding(x::InheritedShapes, ::AbstractPPL.Iden) = x
function _model_argument_binding(x::InheritedShapes, optic::AbstractPPL.AbstractOptic)
    vn = _shape_child_name(x.name, optic)
    return _map_inherited_shapes(v -> _model_argument_binding(v, optic), x, vn)
end
function _fold_model_indices(f, result, x::InheritedShapes)
    step =
        (result, _, optic, template) ->
            f(result, _model_argument_binding(x, optic), optic, template)
    result = _fold_model_indices(step, result, x.child)
    return _fold_model_indices(result, x.parent) do result, child, optic, template
        _model_argument_binding(x.child, optic) === nothing || return result
        f(result, child, optic, template)
    end
end
_submodel_namespace(x::InheritedShapes) = _map_inherited_shapes(_submodel_namespace, x)
_submodel_values(x::InheritedShapes, ::Nothing) = x
function _submodel_values(x::InheritedShapes, prefix::VarName)
    binding = _model_argument_binding(x, AbstractPPL.varname_to_optic(prefix))
    return binding === nothing ? VarNamedTuple() : _submodel_namespace(binding)
end
function _local_shape_roots(model, x::InheritedShapes)
    return _map_inherited_shapes(v -> _local_shape_roots(model, v), x)
end
_with_binding_shapes(values, shapes) = _with_binding_metadata(values; shapes)
function _with_binding_metadata(
    values;
    observation_removals=_removals(Condition, values),
    fixed_removals=_removals(Fix, values),
    templates=_binding_templates(values),
    shapes=_prepared_shapes(values),
)
    inner = values isa LocalModelValues ? values.values : values
    if !(inner isa ModelBindingLayers) &&
        shapes.roots isa VarNamedTuple{(),Tuple{}} &&
        isempty(observation_removals) &&
        isempty(fixed_removals) &&
        isempty(templates)
        return values
    end
    updated = ModelBindingLayers(
        _observation_values(inner),
        _fixed_values(inner),
        _model_values(inner),
        _fixed_owners(inner),
        observation_removals,
        fixed_removals,
        templates,
        shapes,
    )
    return values isa LocalModelValues ? LocalModelValues(updated, values.owners) : updated
end
function _prefix_binding_shapes(shapes, vn, template)
    return ModelBindingShapes(_prefix_values(shapes.roots, vn, template))
end
function _local_binding_shape_store(model)
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    shapes = _prepared_shapes(model.values)
    prefix === nothing && return shapes
    return ModelBindingShapes(_submodel_values(shapes.roots, prefix))
end
# Masked storage occupies its root even when every position is masked.
function _shape_occupied(model, vn)
    optic = AbstractPPL.varname_to_optic(vn)
    return _model_argument_binding(_observation_values(model.values), optic) !== nothing ||
           _model_argument_binding(_fixed_values(model.values), optic) !== nothing
end

@noinline function _binding_shape_error(vn, detail="")
    message = "Binding shape at `$vn`$detail cannot change; clear the root with `decondition` and `unfix`, then bind a whole value or template. For an inherited binding, clear the root on the child model or supply the child argument with the right size."
    throw(ArgumentError(message))
end
@noinline function _binding_root_error(vn, root=vn)
    message = if vn == root
        "Cannot create binding root `$vn`: an existing binding root already occupies this address; rebuild the model to change its roots."
    else
        "Cannot create binding root `$vn` across existing root `$root`; rebuild the model to change its roots."
    end
    throw(ArgumentError(message))
end
_whole_shape(whole::Tuple, vn) = vn in whole
function _whole_shape(values::VarNamedTuple, vn)
    value = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
    return value isa ModelValue ||
           value isa PendingBindings && value.base isa VarNamedTuples.PartialArray
end
@noinline function _template_shape_error(vn)
    message = "Binding shape at `$vn` was set by a binding template and cannot change; supply a new template with the value."
    throw(ArgumentError(message))
end
@inline function (op::ShapeOp{Build})(a::BindingShape, b::BindingShape, vn) where {Build}
    a === b && return Build ? a : ()
    replaceable = _whole_shape(op.whole, vn) && !_shape_occupied(op.model, vn)
    templated =
        op.whole isa VarNamedTuple && any(
            t -> t isa ModelBindingTemplate{Nothing} && t.name == vn,
            _binding_templates(op.model.values),
        )
    if !_same_binding_geometry(a.geometry, b.geometry)
        replaceable || _binding_shape_error(vn)
        templated && _template_shape_error(vn)
        return Build ? b : (vn => true,)
    end
    replaceable && !templated && return Build ? b : (vn => true,)
    return Build ? BindingShape(_refine_binding_geometry(a.geometry, b.geometry)) : ()
end
@inline function (op::ShapeOp{Build})(
    a::VarNamedTuples.PartialArray,
    b::PendingBindings{<:Any,<:Any,<:VarNamedTuples.PartialArray},
    vn,
) where {Build}
    Build && return op(a, _materialize_pending(b), vn)
    rules = op(a, b.base, vn)
    return _check_shape_children(op, a, b, vn, rules)
end
@inline function (op::ShapeOp{Build})(
    a::PendingBindings, b::VarNamedTuples.PartialArray, vn
) where {Build}
    if a.base isa VarNamedTuples.PartialArray
        Build && return op(_materialize_pending(a), b, vn)
        return _check_shape_children(op, a, b, vn, op(a.base, b, vn))
    end
    return op(a.base, b, vn)
end
@inline function (op::ShapeOp{Build})(
    a::BindingShape, b::Union{VarNamedTuple,VarNamedTuples.PartialArray,PendingBindings}, vn
) where {Build}
    if b isa PendingBindings && !_shape_occupied(op.model, vn)
        return Build ? PendingBindings(b.name, b.entries, a) : (vn => false,)
    end
    geometry = _refine_shape_part(a.geometry, AbstractPPL.Iden(), b, vn, Build)
    return Build ? BindingShape(geometry) : ()
end
function _refine_shape_part(
    geometry,
    ::AbstractPPL.Iden,
    value::Union{VarNamedTuple,VarNamedTuples.PartialArray,PendingBindings},
    vn,
    build=true,
)
    return _fold_model_indices(geometry, value) do g, child, optic, _
        _refine_shape_part(g, optic, child, AbstractPPL.append_optic(vn, optic), build)
    end
end
function _refine_shape_part(
    geometry, ::AbstractPPL.Iden, value::BindingShape, vn, build=true
)
    _same_binding_geometry(geometry, value.geometry) || _binding_shape_error(vn)
    return build ? _refine_binding_geometry(geometry, value.geometry) : geometry
end
function _refine_shape_part(
    ::ScalarBindingGeometry,
    ::Union{AbstractPPL.Property,AbstractPPL.Index},
    value,
    vn,
    build=true,
)
    return _binding_shape_error(vn)
end
function _refine_shape_part(
    g::FieldBindingGeometry, optic::AbstractPPL.Property{S}, value, vn, build=true
) where {S}
    haskey(g.children, S) || _binding_shape_error(vn)
    child = _refine_shape_part(g.children[S], optic.child, value, vn, build)
    return build ? FieldBindingGeometry(Base.setindex(g.children, child, S)) : g
end
function _refine_shape_part(
    g::ArrayBindingGeometry, optic::AbstractPPL.Index, value, vn, build=true
)
    checkbounds(Bool, CartesianIndices(g.axes), optic.ix...) || _binding_shape_error(vn)
    g.children === nothing && return g
    index = LinearIndices(g.axes)[optic.ix...]
    child = _refine_shape_part(g.children[index], optic.child, value, vn, build)
    (!build || child === g.children[index]) && return g
    children = BangBang.setindex!!(copy(g.children), child, index)
    return ArrayBindingGeometry(g.axes, children)
end

@inline function (op::ShapeOp{Build})(
    a::Union{VarNamedTuple,VarNamedTuples.PartialArray}, b::BindingShape, vn
) where {Build}
    root = if a isa VarNamedTuple
        AbstractPPL.append_optic(vn, AbstractPPL.Property{first(keys(a.data))}())
    else
        vn
    end
    return _binding_root_error(vn, root)
end
(op::ShapeOp{Build})(a, ::Nothing, vn) where {Build} = Build ? a : ()
(op::ShapeOp{Build})(::Nothing, b, vn) where {Build} = Build ? b : ()
(op::ShapeOp{Build})(::Nothing, ::Nothing, vn) where {Build} = Build ? nothing : ()
@inline function (op::ShapeOp{Build})(a::VarNamedTuple, b::VarNamedTuple, vn) where {Build}
    a === b && return Build ? a : ()
    Build || return _check_shape_children(op, a, b, vn)
    return _merge_model_values(a, b, vn, (x, y, address) -> op(x, y, address))
end
@inline function (op::ShapeOp{Build})(
    a::VarNamedTuples.PartialArray, b::VarNamedTuples.PartialArray, vn
) where {Build}
    rules = _check_namespace_axes(op, a, b, vn)
    Build && !isempty(rules) && (a = _resize_model_binding(a, axes(b)))
    any(a.mask) || return Build ? b : rules
    any(b.mask) || return Build ? a : rules
    Build || return _check_shape_children(op, a, b, vn, rules)
    return _merge_model_indices(
        a, b, (x, y, optic) -> op(x, y, AbstractPPL.append_optic(vn, optic))
    )
end
function _merge_binding_shapes(a, b, vn, model, whole)
    return ShapeOp{true}(model, whole)(_materialize_shapes(a), _materialize_shapes(b), vn)
end
function _check_binding_shapes(a, b, model, whole)
    return ShapeOp{false}(model, whole)(a, b, nothing)
end
function _check_namespace_axes(op, a, b, vn)
    previous, incoming = _shape_axes(a), _shape_axes(b)
    (previous === nothing || incoming === nothing || previous == incoming) && return ()
    if op.model isa Model
        message = "Cannot change namespace root `$vn` while a recursive removal remains below it; rebuild the model."
        for role in (Condition, Fix), removal in _removals(role, op.model.values)
            removal.name !== nothing &&
                removal.name != vn &&
                subsumes(vn, removal.name) &&
                throw(ArgumentError(message))
        end
    end
    _whole_shape(op.whole, vn) && !_shape_occupied(op.model, vn) || _binding_shape_error(
        vn, " with size $(map(length, previous)) (supplied $(map(length, incoming)))"
    )
    return (vn => true,)
end
@inline function _check_shape_children(
    op, a, b, vn, rules=_check_namespace_axes(op, a, b, vn)
)
    return _fold_model_indices(rules, b) do rules, child, optic, _
        a isa PendingBindings &&
            b isa PendingBindings &&
            child isa BindingShape &&
            return rules
        extra = op(_model_argument_binding(a, optic), child, _shape_child_name(vn, optic))
        return (rules..., extra...)
    end
end
@inline function (op::ShapeOp{Build})(a::InheritedShapes, b, vn) where {Build}
    @assert !Build
    b isa BindingShape || return _check_shape_children(op, a, b, vn)
    first = op(a.child, b, vn)
    return (first..., op(a.parent, b, vn)...)
end
@inline function (op::ShapeOp{Build})(a, b::InheritedShapes, vn) where {Build}
    @assert !Build
    b.child isa BindingShape || return _check_shape_children(op, a, b, vn)
    first = op(a, b.child, vn)
    return (first..., op(a, b.parent, vn)...)
end
(op::ShapeOp{Build})(a::InheritedShapes, ::Nothing, vn) where {Build} = ()
(op::ShapeOp{Build})(::Nothing, b::InheritedShapes, vn) where {Build} = ()
@inline function (op::ShapeOp{Build})(
    a::InheritedShapes, b::InheritedShapes, vn
) where {Build}
    @assert !Build
    b.child isa BindingShape || return _check_shape_children(op, a, b, vn)
    first = op(a, b.child, vn)
    return (first..., op(a, b.parent, vn)...)
end
function _bind_shape_batch(model, batch, whole=())
    store = _binding_shape_store(model.values)
    roots = _merge_binding_shapes(store.roots, batch, nothing, model, whole)
    shapes = ModelBindingShapes(roots)
    return _reconstruct_model(model; values=_with_binding_shapes(model.values, shapes))
end
function _bind_shape(model::Model, vn::VarName, storage; namespace=false)
    vn = _resolve_binding_address(model, vn)
    value = if namespace
        VarNamedTuples.PartialArray(
            similar(storage, VarNamedTuple{(),Tuple{}}),
            fill!(similar(storage, Bool), false),
        )
    else
        BindingShape(_binding_geometry(storage))
    end
    previous = _model_argument_binding(
        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
    )
    if !namespace && previous !== nothing
        _same_schema_storage(_schema_storage(previous), storage) || throw(
            ArgumentError(
                "Binding template at `$vn` conflicts with existing storage; clear the binding first.",
            ),
        )
    end
    batch = templated_setindex!!(
        VarNamedTuple(), value, vn, _binding_shape_template(model, vn)
    )
    model = _bind_shape_batch(model, batch, (vn,))
    namespace && return model
    templates = _merge_binding_templates(
        _binding_templates(model.values), (ModelBindingTemplate{Nothing}(vn, storage),)
    )
    return _reconstruct_model(model; values=_with_binding_metadata(model.values; templates))
end
function _local_shape_roots(model, values)
    arguments = (_argument_names(model.args)..., _argument_names(model.defaults)...)
    return VarNamedTuple(Base.structdiff(values.data, NamedTuple{arguments}))
end
function _local_shape_snapshot(model::Model, values::VarNamedTuple)
    prefix = model.values isa LocalModelValues ? nothing : _model_prefix(model)
    locals = _submodel_values(values, prefix)
    shapes = _snapshot_shape(_local_shape_roots(model, locals))
    return if prefix === nothing
        shapes
    else
        _prefix_binding_shapes(
            ModelBindingShapes(shapes), prefix, _model_prefix_template(model)
        ).roots
    end
end

function _unbind_shape(model::Model, vn::VarName)
    vn = _resolve_binding_address(model, vn)
    store = _binding_shape_store(model.values)
    haskey(store.roots, vn) || return model
    _shape_occupied(model, vn) && _binding_shape_error(vn)
    roots = _delete_binding_root(copy(store.roots), AbstractPPL.varname_to_optic(vn))
    return _reconstruct_model(
        model; values=_with_binding_shapes(model.values, ModelBindingShapes(roots))
    )
end
function _delete_binding_root(x::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    optic.child isa AbstractPPL.Iden &&
        return VarNamedTuple(Base.structdiff(x.data, NamedTuple{(S,)}))
    return VarNamedTuple(
        merge(x.data, NamedTuple{(S,)}((_delete_binding_root(x.data[S], optic.child),)))
    )
end
function _delete_binding_root(x::VarNamedTuples.PartialArray, optic::AbstractPPL.Index)
    optic.child isa AbstractPPL.Iden &&
        return BangBang.delete!!(x, optic.ix...; optic.kw...)
    child = _delete_binding_root(x.data[optic.ix...], optic.child)
    data = BangBang.setindex!!(x.data, child, optic.ix...)
    return VarNamedTuples.PartialArray(data, x.mask)
end
function _binding_shape_template(model, vn, values=nothing)
    base = _base_argument_storage(model, vn, values)
    if base !== nothing
        value, root = base
        return VarNamedTuples.make_leaf(
            value,
            AbstractPPL.getoptic(root),
            _binding_template(model, VarNamedTuple(), root),
        )
    end
    for t in _binding_templates(model.values)
        t isa ModelBindingTemplate{Nothing} && subsumes(t.name, vn) || continue
        return VarNamedTuples.make_leaf(
            t.storage, AbstractPPL.getoptic(t.name), NoTemplate()
        )
    end
    return _shape_template(
        _model_argument_binding(
            _prepared_shapes(model.values).roots,
            AbstractPPL.Property{AbstractPPL.getsym(vn)}(),
        ),
        AbstractPPL.getoptic(vn),
        values === nothing ? nothing : get(values.data, AbstractPPL.getsym(vn), nothing),
    )
end
function _base_argument_storage(model::Model, vn, values=nothing)
    arguments = merge(model.args, model.defaults)
    for (name, stored_name) in zip(_argument_names(arguments), keys(arguments))
        root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        name in _args_on_lhs(model) && subsumes(root, vn) || continue
        binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(root))
        binding === nothing && (binding = _get_model_binding(model, VarName{name}()))
        value = if binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            _shape_storage(binding, arguments[stored_name])
        elseif binding === nothing
            arguments[stored_name]
        elseif binding isa ModelValue
            binding.value
        else
            _binding_storage(binding)
        end
        return value, root
    end
    return nothing
end
function _shape_template(root::BindingShape, optic, previous=nothing)
    previous === nothing && return _geometry_template(root.geometry)
    return _binding_storage(previous)
end
_shape_template(::Nothing, optic, previous=nothing) = NoTemplate()
_shape_template(value, optic, previous=nothing) = value
function _shape_template(
    value::VarNamedTuple, optic::AbstractPPL.Property{S}, previous=nothing
) where {S}
    return _property_shape_template(value, optic, previous)
end
function _property_shape_template(value, optic::AbstractPPL.Property{S}, previous) where {S}
    child = _shape_template(
        _model_argument_binding(value, AbstractPPL.ohead(optic)),
        optic.child,
        _model_argument_binding(previous, AbstractPPL.ohead(optic)),
    )
    return child isa NoTemplate ? child : VarNamedTuple(NamedTuple{(S,)}((child,)))
end
function _shape_template(
    value::VarNamedTuples.PartialArray, optic::AbstractPPL.Index, previous=nothing
)
    return BindingGeometryArray(value, optic, previous)
end

function _resolve_binding_address(model, vn)
    storage = _binding_shape_template(model, vn)
    storage isa NoTemplate && return vn
    return VarName{AbstractPPL.getsym(vn)}(
        _resolve_binding_optic(storage, AbstractPPL.getoptic(vn))
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
function _public_model_values(model)
    values = _model_values(model.values)
    if model.values isa LocalModelValues && _model_prefix(model) !== nothing
        template = _apply_prefix_template(_model_prefix_template(model), NoTemplate())
        return _prefix_values(values, _model_prefix(model), template)
    end
    return _model_values(_materialize_argument_values(model).values)
end
function _model_values_for_prefix(model)
    model.values isa UnprefixedArgumentValues && return model.values.values
    _model_prefix(model) === nothing && return model.values
    return ModelBindingLayers(
        _submodel_layer(Condition, model),
        _submodel_layer(Fix, model),
        _submodel_fixed_owners(model, nothing),
        _submodel_removals(Condition, model, nothing),
        _submodel_removals(Fix, model, nothing),
        _submodel_binding_templates(model, nothing);
        shapes=_local_binding_shape_store(model),
    )
end

_record_binding_read(context, model, vn, template) = context
function _record_binding_read(context::AbstractParentContext, model, vn, template)
    return setchildcontext(
        context, _record_binding_read(childcontext(context), model, vn, template)
    )
end

_binding_read_context(context, values) = context
function _binding_read_context(context::AbstractParentContext, values)
    return setchildcontext(context, _binding_read_context(childcontext(context), values))
end

include("deferred_bindings.jl")
