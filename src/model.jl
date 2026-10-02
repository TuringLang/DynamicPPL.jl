# ------------
# Model values
# ------------

struct Condition end
struct ArgumentCondition end
struct Fix end

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

struct ModelValue{R<:Union{Condition,ArgumentCondition,Fix},T}
    value::T
    function ModelValue{R}(value::T) where {R<:Union{Condition,ArgumentCondition,Fix},T}
        (R === Condition || R === ArgumentCondition || R === Fix) ||
            throw(ArgumentError("A model value must have one concrete role"))
        return new{R,T}(value)
    end
end

struct NoModelBinding end

# Fixed bindings preserve the observation layer they shadow.
struct ModelBindingLayers{O<:VarNamedTuple,F<:VarNamedTuple,V<:VarNamedTuple,A<:Tuple}
    observations::O
    fixed::F
    values::V
    # Whole fixed owners survive expansion into partial storage.
    owners::A
end
function ModelBindingLayers(observations, fixed, owners::Tuple=Tuple(keys(fixed)))
    return ModelBindingLayers(
        observations, fixed, _overlay_model_values(observations, fixed, owners), owners
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

# Child bindings selected from the parent's submodel namespace use the child's storage shape.
struct LocalModelValues{V<:Union{VarNamedTuple,ModelBindingLayers},A<:Tuple}
    values::V
    owners::A
end
LocalModelValues(values) = LocalModelValues(values, _fixed_owners(values))
# Default arguments need no enclosing namespace storage during evaluation.
struct UnprefixedArgumentValues{V<:VarNamedTuple}
    values::V
end
_model_values(values::UnprefixedArgumentValues) = values.values
_model_value_varname(::UnprefixedArgumentValues, vn, prefix) = vn

_model_values(values::VarNamedTuple) = values
_model_values(values::LocalModelValues) = _model_values(values.values)
_observation_values(values::LocalModelValues) = _observation_values(values.values)
_fixed_values(values::LocalModelValues) = _fixed_values(values.values)
_fixed_owners(values::LocalModelValues) = values.owners
_model_value_varname(::VarNamedTuple, vn, prefix) = maybe_prefix(vn, prefix)
_model_value_varname(::LocalModelValues, vn, prefix) = vn

# Partial bindings retain the container from which their fields or tuple elements came.
struct ModelValueTree{T,V<:Union{VarNamedTuple,Tuple}}
    template::T
    values::V
    function ModelValueTree(template::T, values::V) where {T,V<:Union{VarNamedTuple,Tuple}}
        if values isa Tuple
            template isa Tuple && length(template) == length(values) ||
                throw(ArgumentError("Tuple bindings must preserve their template's length"))
        else
            all(name -> hasproperty(template, name), keys(values.data)) ||
                throw(ArgumentError("Field bindings must belong to their template"))
        end
        return new{T,V}(template, values)
    end
end

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
                    $(VarName{name}());
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
    eltype(fixed) <: ModelValue &&
        _has_complete_model_data(fixed) &&
        return _copy_model_node(fixed)
    previous isa ModelValue && (previous = _expand_model_binding(previous))
    owns_shape = any(owner -> subsumes(owner, vn), owners)
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
        data = similar(previous.data, eltype(previous), merged_axes)
        mask = fill!(similar(data, Bool), false)
        for i in CartesianIndices(data)
            if checkbounds(Bool, previous.data, i) && previous.mask[i]
                data[i] = previous.data[i]
                mask[i] = true
            end
        end
        previous = VarNamedTuples.PartialArray(data, mask)
    end
    if previous isa VarNamedTuples.PartialArray && fixed.data isa ModelBindingArray
        data = previous.data isa ModelBindingArray ? previous.data.data : previous.data
        previous = VarNamedTuples.PartialArray(
            ModelBindingArray(data, fixed.data.template), previous.mask
        )
    end
    return _fold_model_indices(
        _copy_model_node(previous), fixed
    ) do result, update, optic, template
        child = _model_argument_binding(result, optic)
        value = if child === nothing
            update
        else
            _overlay_model_node(child, update, owners, AbstractPPL.append_optic(vn, optic))
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
function _overlay_model_node(previous, fixed::ModelValueTree, owners, vn)
    owns_shape = any(owner -> subsumes(owner, vn), owners)
    template = if !owns_shape && previous isa ModelValue
        previous.value
    elseif !owns_shape && previous isa ModelValueTree
        previous.template
    else
        fixed.template
    end
    values = if fixed.values isa Tuple
        template = length(template) < length(fixed.template) ? fixed.template : template
        ntuple(length(template)) do i
            optic = AbstractPPL.Index((i,), (;))
            child = _previous_model_child(previous, optic)
            _overlay_model_node(
                child,
                get(fixed.values, i, NoModelBinding()),
                owners,
                AbstractPPL.append_optic(vn, optic),
            )
        end
    else
        previous isa ModelValue && (previous = _expand_model_binding(previous))
        previous = previous isa ModelValueTree ? previous.values : previous
        if previous isa VarNamedTuple
            _overlay_model_values(previous, fixed.values, owners, vn)
        else
            fixed.values
        end
    end
    if values isa VarNamedTuple &&
        !all(name -> hasproperty(template, name), keys(values.data))
        template = fixed.template
    end
    return ModelValueTree(template, values)
end

function Base.:(==)(a::ModelValueTree, b::ModelValueTree)
    return a.template == b.template && a.values == b.values
end
function Base.isequal(a::ModelValueTree, b::ModelValueTree)
    return isequal(a.template, b.template) && isequal(a.values, b.values)
end
function Base.hash(value::ModelValueTree, h::UInt)
    return hash(value.template, hash(value.values, hash(:ModelValueTree, h)))
end

function VarNamedTuples._getindex_optic(
    value::ModelValue{R}, optic::AbstractPPL.AbstractOptic, vn
) where {R}
    return ModelValue{R}(VarNamedTuples._getindex_optic(value.value, optic, vn))
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
    return VarNamedTuples._haskey_optic(ModelValue{R}(child), optic.child)
end
VarNamedTuples._haskey_optic(::ModelValue, ::AbstractPPL.Iden) = true
function VarNamedTuples._haskey_optic(
    value::ModelValue{R,<:Tuple}, optic::AbstractPPL.Index
) where {R<:Union{Condition,ArgumentCondition,Fix}}
    optic = AbstractPPL.concretize_top_level(optic, value.value)
    isempty(optic.kw) && checkbounds(Bool, Base.OneTo(length(value.value)), optic.ix...) ||
        return false
    return VarNamedTuples._haskey_optic(
        ModelValue{R}(getindex(value.value, optic.ix...)), optic.child
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
        values = tree.values isa VarNamedTuple ? tree.values.data : tree.values
        _model_role(values, vn)
    else
        _model_role(tree.values, vn)
    end
end
_model_role(::VarNamedTuple, vn::VarName) = _partial_binding_error(vn)
function _partial_binding_error(vn)
    return throw(
        ArgumentError(
            "LHS variable `$vn` must be bound as a whole; bind all its subvariables or none.",
        ),
    )
end
function _model_role(values::VarNamedTuples.PartialArray, vn::VarName)
    !(values.data isa VarNamedTuples.GrowableArray) &&
        all(values.mask) &&
        return _model_role(values.data, vn)
    any(values.mask) || return nothing
    role = _model_role(values.data[values.mask], vn)
    return role === nothing ? nothing : _partial_binding_error(vn)
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
    if !checkbounds(Bool, values.data, optic.ix...; optic.kw...)
        # Storage bounds describe supplied indices, so an out-of-bounds LHS variable may overlap.
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
function _get_model_role(model, vn)
    vn = _model_value_varname(model.values, vn, _model_prefix(model))
    return _model_role_at(_model_values(model.values), AbstractPPL.varname_to_optic(vn), vn)
end
function _get_model_binding(model, vn)
    vn = _model_value_varname(model.values, vn, _model_prefix(model))
    return _model_argument_binding(
        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
    )
end
function _get_argument_role(model, vn, argument)
    binding = _get_model_binding(model, argument)
    # TODO: remove once users have migrated off the `x === missing; x = ...` placeholder idiom.
    if binding isa ModelValue{ArgumentCondition,<:Union{Missing,Nothing}}
        vn = maybe_prefix(vn, _model_prefix(model))
        throw(
            ArgumentError(
                "LHS variable `$vn` contains `$(binding.value)`; make it latent with `decondition`.",
            ),
        )
    end
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

# Keep the array owner through PartialArray's copies and element-type changes.
struct ModelBindingArray{T,N,A<:AbstractArray{T,N},V<:AbstractArray} <: AbstractArray{T,N}
    data::A
    template::V
end
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
Base.copy(value::ModelBindingArray) = ModelBindingArray(copy(value.data), value.template)
function Base.similar(value::ModelBindingArray, ::Type{T}) where {T}
    return ModelBindingArray(similar(value.data, T), value.template)
end
function Base.similar(value::ModelBindingArray, ::Type{T}, dims::Dims) where {T}
    return ModelBindingArray(similar(value.data, T, dims), value.template)
end

function _expand_model_binding(previous::ModelValue{R,<:AbstractArray}) where {R}
    value = previous.value
    T = isconcretetype(eltype(value)) ? ModelValue{R,eltype(value)} : ModelValue{R}
    data = similar(value, T)
    mask = similar(value, Bool)
    for i in eachindex(value)
        mask[i] = isassigned(value, i)
        if mask[i]
            data[i] = ModelValue{R}(value[i])
        end
    end
    # Arrays whose `similar` preserves their container already carry the owner type.
    # Leave those visible to array-specific binding protocols (e.g. ComponentArrays).
    if Core.Compiler.return_type(similar, Tuple{typeof(value)}) !== typeof(value)
        data = ModelBindingArray(data, value)
    end
    return VarNamedTuples.PartialArray(data, mask)
end
function _expand_model_binding(previous::ModelValue{R,<:Base.Pairs}) where {R}
    return _expand_model_binding(ModelValue{R}(NamedTuple(previous.value)))
end
function _expand_model_binding(previous::ModelValue{R,<:Tuple}) where {R}
    return ModelValueTree(previous.value, map(ModelValue{R}, previous.value))
end
function _expand_model_binding(previous::ModelValue{R}) where {R}
    fields = _defined_model_properties(previous.value, Val(propertynames(previous.value)))
    return ModelValueTree(previous.value, _tag_model_values(R, VarNamedTuple(fields)))
end
_defined_model_properties(value, ::Val{()}) = NamedTuple()
function _defined_model_properties(value, ::Val{names}) where {names}
    name = first(names)
    rest = _defined_model_properties(value, Val(Base.tail(names)))
    return if hasfield(typeof(value), name) && !isdefined(value, name)
        rest
    else
        merge(NamedTuple{(name,)}((getproperty(value, name),)), rest)
    end
end
function VarNamedTuples._setindex_optic!!(
    previous::ModelValue{R,<:Union{AbstractArray,Tuple}},
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
function VarNamedTuples._getindex_optic(
    tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index, vn
)
    optic = AbstractPPL.concretize_top_level(optic, tree.template)
    return VarNamedTuples._getindex_optic(
        getindex(tree.values, optic.ix...; optic.kw...), optic.child, vn
    )
end

function VarNamedTuples._haskey_optic(tree::ModelValueTree, optic::AbstractPPL.Property)
    return VarNamedTuples._haskey_optic(tree.values, optic)
end
function VarNamedTuples._haskey_optic(
    tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index
)
    value = _model_tuple_getindex(tree, optic)
    return !(value isa NoModelBinding) && VarNamedTuples._haskey_optic(value, optic.child)
end
function VarNamedTuples._haskey_optic(tree::ModelValueTree, ::AbstractPPL.Iden)
    values = tree.values
    if values isa VarNamedTuple
        return all(propertynames(tree.template)) do name
            haskey(values.data, name) &&
                VarNamedTuples._haskey_optic(values.data[name], AbstractPPL.Iden())
        end
    end
    return all(
        value ->
            !(value isa NoModelBinding) &&
                VarNamedTuples._haskey_optic(value, AbstractPPL.Iden()),
        values,
    )
end

function VarNamedTuples._setindex_optic!!(
    tree::ModelValueTree, value, optic::AbstractPPL.Property, template, permissions
)
    template = template isa ModelValueTree ? template.values : tree.template
    values = VarNamedTuples._setindex_optic!!(
        copy(tree.values), value, optic, template, permissions
    )
    return ModelValueTree(tree.template, values)
end
function VarNamedTuples._setindex_optic!!(
    tree::ModelValueTree{<:Tuple}, value, optic::AbstractPPL.Index, template, permissions
)
    optic = AbstractPPL.concretize_top_level(optic, tree.template)
    length(optic.ix) == 1 && only(optic.ix) isa Integer && isempty(optic.kw) ||
        throw(ArgumentError("Tuple bindings require a single integer index"))
    i = only(optic.ix)
    previous = tree.values[i]
    previous = if previous isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
        copy(previous)
    else
        previous
    end
    child_template = template isa ModelValueTree ? template.values[i] : tree.template[i]
    updated = if previous isa NoModelBinding
        permissions isa VarNamedTuples.MustOverwrite &&
            throw(VarNamedTuples.MustOverwriteError(permissions))
        VarNamedTuples.make_leaf(value, optic.child, child_template)
    else
        VarNamedTuples._setindex_optic!!(
            previous, value, optic.child, child_template, permissions
        )
    end
    return ModelValueTree(tree.template, Base.setindex(tree.values, updated, i))
end

function VarNamedTuples._mapreduce_recursive(
    f, op, tree::ModelValueTree{T,<:VarNamedTuple}, vn, init
) where {T}
    return VarNamedTuples._mapreduce_recursive(f, op, tree.values, vn, init)
end
@generated function VarNamedTuples._mapreduce_recursive(
    f, op, tree::ModelValueTree{T,V}, vn, init
) where {T,V<:Tuple}
    exs = map(1:fieldcount(V)) do i
        quote
            if !(tree.values[$i] isa NoModelBinding)
                result = VarNamedTuples._mapreduce_recursive(
                    f,
                    op,
                    tree.values[$i],
                    AbstractPPL.append_optic(vn, AbstractPPL.Index(($i,), (;))),
                    result,
                )
            end
        end
    end
    return quote
        result = init
        $(exs...)
        result
    end
end
function VarNamedTuples._map_values_recursive!!(f, tree::ModelValueTree)
    values = if tree.values isa VarNamedTuple
        map_values!!(f, copy(tree.values))
    else
        map(tree.values) do value
            if value isa NoModelBinding
                value
            else
                VarNamedTuples._map_values_recursive!!(f, _copy_model_node(value))
            end
        end
    end
    return ModelValueTree(tree.template, values)
end
function VarNamedTuples._map_pairs_recursive!!(f, tree::ModelValueTree, vn)
    values = if tree.values isa VarNamedTuple
        VarNamedTuples._map_pairs_recursive!!(f, copy(tree.values), vn)
    else
        ntuple(length(tree.values)) do i
            value = tree.values[i]
            if value isa NoModelBinding
                value
            else
                VarNamedTuples._map_pairs_recursive!!(
                    f,
                    _copy_model_node(value),
                    AbstractPPL.append_optic(vn, AbstractPPL.Index((i,), (;))),
                )
            end
        end
    end
    return ModelValueTree(tree.template, values)
end

function _empty_model_tree(tree::ModelValueTree)
    values =
        tree.values isa Tuple ? map(_ -> NoModelBinding(), tree.values) : VarNamedTuple()
    return ModelValueTree(tree.template, values)
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
function VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Index, template::ModelValueTree{<:Tuple}
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
function _model_tuple_getindex(tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index)
    optic = AbstractPPL.concretize_top_level(optic, tree.template)
    isempty(optic.kw) && checkbounds(Bool, Base.OneTo(length(tree.values)), optic.ix...) ||
        return NoModelBinding()
    return getindex(tree.values, optic.ix...)
end
function _model_role_at(tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index, vn)
    value = _model_tuple_getindex(tree, optic)
    return value isa NoModelBinding ? nothing : _model_role_at(value, optic.child, vn)
end
function _model_argument_binding(tree::ModelValueTree, optic::AbstractPPL.Property)
    return _model_argument_binding(tree.values, optic)
end
function _model_argument_binding(tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index)
    optic = AbstractPPL.concretize_top_level(optic, tree.template)
    value = _model_tuple_getindex(tree, optic)
    if value isa Tuple
        value = ModelValueTree(getindex(tree.template, optic.ix...), value)
    end
    return value isa NoModelBinding ? nothing : _model_argument_binding(value, optic.child)
end
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

function _check_namedtuple_index(value, optic, prefix=AbstractPPL.Iden())
    optic isa AbstractPPL.Iden && return nothing
    template = value isa ModelValue ? value.value : value
    template = template isa ModelValueTree ? template.template : template
    if template isa NamedTuple && optic isa AbstractPPL.Index
        index = AbstractPPL.concretize_top_level(optic, template)
        if index.ix isa Tuple{Integer}
            field = get(keys(template), only(index.ix), nothing)
            suggestion = if field === nothing
                "a field name"
            else
                "`$(AbstractPPL.optic_to_varname(AbstractPPL.Property{field}(optic.child) ∘ prefix))`"
            end
            message = "Integer indexing into a NamedTuple at `$(AbstractPPL.optic_to_varname(optic ∘ prefix))` is unsupported; use $suggestion instead."
            throw(ArgumentError(message))
        end
    end
    optic.child isa AbstractPPL.Iden && return nothing
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(value, head)
    return _check_namedtuple_index(child, optic.child, head ∘ prefix)
end

@generated function _merge_model_values(
    previous::VarNamedTuple{P}, updates::VarNamedTuple{U}
) where {P,U}
    names = Tuple(union(P, U))
    fields = map(names) do name
        if name in P && name in U
            quote
                _check_model_binding(
                    previous.data.$name, updates.data.$name, $(VarName{name}())
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
_check_model_binding(previous, updates, vn; check_bounds=true) = nothing
function _check_model_binding(
    previous,
    updates::Union{VarNamedTuple,VarNamedTuples.PartialArray},
    vn;
    check_bounds=true,
)
    _fold_model_indices(nothing, updates) do _, update, optic, _
        _check_namedtuple_index(previous, optic, AbstractPPL.varname_to_optic(vn))
        if previous isa ModelValue
            compatible = if updates isa VarNamedTuples.PartialArray
                previous.value isa Union{AbstractArray,Tuple}
            else
                previous.value isa NamedTuple ||
                    all(name -> hasproperty(previous.value, name), keys(updates.data))
            end
            compatible || throw(
                ArgumentError(
                    "Cannot bind parts below `$vn` with value of type $(typeof(previous.value)). " *
                    "If `$vn` holds a submodel return value, condition or fix the child model before wrapping it with `to_submodel`. " *
                    "For other bindings, use `decondition(model, @varname($vn))` first.",
                ),
            )
        end
        child = _model_argument_binding(previous, optic)
        if check_bounds &&
            child === nothing &&
            (
                (
                    previous isa ModelValue && previous.value isa Union{AbstractArray,Tuple}
                ) ||
                (
                    previous isa VarNamedTuples.PartialArray &&
                    optic isa AbstractPPL.Index &&
                    !(previous.data isa VarNamedTuples.GrowableArray) &&
                    !checkbounds(Bool, previous.data, optic.ix...; optic.kw...)
                ) ||
                (
                    previous isa ModelValueTree{<:Tuple} &&
                    optic isa AbstractPPL.Index &&
                    !checkbounds(Bool, Base.OneTo(length(previous.values)), optic.ix...)
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
    values =
        value.values isa Tuple ? map(_copy_model_node, value.values) : copy(value.values)
    return ModelValueTree(value.template, values)
end
_merge_model_node(previous, updates) = updates
_merge_model_node(previous, ::NoModelBinding) = previous
function _merge_model_node(previous, updates::VarNamedTuple)
    previous isa NoModelBinding && return copy(updates)
    if previous isa Union{ModelValue{<:Any,<:NamedTuple},ModelValueTree{<:NamedTuple}}
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
        return ModelValueTree(
            previous.template, _merge_model_values(previous.values, updates)
        )
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
    if updates.values isa Tuple
        values = ntuple(length(updates.values)) do i
            child = _previous_model_child(previous, AbstractPPL.Index((i,), (;)))
            update = updates.values[i]
            return child isa NoModelBinding ? update : _merge_model_node(child, update)
        end
        return ModelValueTree(updates.template, values)
    end
    fields = _merge_model_fields(previous, updates, Val(propertynames(updates.template)))
    return ModelValueTree(updates.template, VarNamedTuple(fields))
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
        map(ModelValue{R}, value.value)
    else
        value
    end
end

_model_data(value) = value
_model_data(value::ModelValue) = value.value
_model_data(values::AbstractArray) = map(_model_data, values)
function _model_data(values::ModelBindingArray)
    return _restore_model_array(map(_model_data, values.data), values.template)
end
VarNamedTuples.unwrap_internal_array(values::ModelBindingArray) = _model_data(values)
function _restore_model_array(data, template)
    if axes(template) == axes(data) && !(data isa typeof(template))
        result = _writable_model_argument(_copy_model_argument(template))
        for i in eachindex(data)
            result = BangBang.setindex!!(result, data[i], i)
        end
        return result
    end
    return data
end
_model_data(values::VarNamedTuple) = map(_model_data, values.data)
function _model_data(values::VarNamedTuples.PartialArray)
    data = VarNamedTuples.unwrap_internal_array(values)
    return values.data isa ModelBindingArray ? data : _model_data(data)
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
function _retain_argument_leaves!(memo, value, seen)
    isbits(value) && return nothing
    value isa Union{Type,Symbol,AbstractString,Module} && return nothing
    value in seen && return nothing
    push!(seen, value)
    if value isa Number && ismutable(value)
        memo[value] = value
    else
        _retain_argument_children!(memo, value, seen)
    end
    return nothing
end
function _retain_argument_children!(memo, value, seen)
    for i in 1:fieldcount(typeof(value))
        isdefined(value, i) && _retain_argument_leaves!(memo, getfield(value, i), seen)
    end
    return nothing
end
function _retain_argument_children!(memo, value::AbstractArray, seen)
    isbitstype(eltype(value)) && return nothing
    for i in eachindex(value)
        isassigned(value, i) && _retain_argument_leaves!(memo, value[i], seen)
    end
    for i in 1:fieldcount(typeof(value))
        isdefined(value, i) && _retain_argument_leaves!(memo, getfield(value, i), seen)
    end
    return nothing
end
function _retain_argument_children!(memo, value::AbstractDict, seen)
    keys = Base.IdSet{Any}()
    for (key, child) in value
        # Keys are addresses also used by other arguments and the model body.
        memo[key] = key
        _argument_graph!(keys, key)
        _retain_argument_leaves!(memo, key, seen)
        _retain_argument_leaves!(memo, child, seen)
    end
    for key in keys
        memo[key] = key
    end
    return nothing
end
# Resolve the optional adapter from type metadata so ordinary preparation remains
# inferred even when ReverseDiff's type-changing facade adapter is loaded.
_argument_ad_storage(::Type) = false
function _argument_adapter_expr(T, seen)
    T <: Union{Number,Type,Symbol,AbstractString} && return false
    T in seen && return false
    push!(seen, T)
    children = if T isa Union
        Base.uniontypes(T)
    elseif !isconcretetype(T)
        return true
    elseif T <: Array
        (eltype(T),)
    else
        fieldtypes(T)
    end
    result = :(_argument_ad_storage($T))
    for child in children
        result = :($result || $(_argument_adapter_expr(child, seen)))
    end
    return result
end
@generated function _argument_may_need_adapter(::Type{T}) where {T}
    return _argument_adapter_expr(T, Set{Any}())
end
function _copy_model_argument(value)
    copied = deepcopy(ModelArgumentCopy(value))
    return if _argument_may_need_adapter(typeof(value)) && copied.adapt
        _adapt_copied_argument(copied.value)
    else
        copied.value
    end
end
_copy_model_argument(value::Union{Number,Type}) = value
# With numerical leaves preserved, copying a single dense numeric array is shallow.
_copy_model_argument(value::Array{<:Number}) = copy(value)
# ReverseDiff supplies writable storage for its immutable tracked-array facade.
_writable_model_argument(value) = value

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
function _argument_graph!(seen, value)
    isbits(value) && return nothing
    value isa Union{Number,Type,Symbol,AbstractString,Module} && return nothing
    value in seen && return nothing
    push!(seen, value)
    if value isa AbstractArray && !isbitstype(eltype(value))
        for i in eachindex(value)
            isassigned(value, i) && _argument_graph!(seen, value[i])
        end
    end
    # Include backing storage, especially parents of views and reshaped arrays.
    for i in 1:fieldcount(typeof(value))
        isdefined(value, i) && _argument_graph!(seen, getfield(value, i))
    end
    return nothing
end
function _argument_storage_policy!(latent, bound, storage)
    value, children = storage.value, storage.children
    push!(latent, value)
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
    return _argument_graph!(leaf.bound ? bound : latent, value)
end
function _argument_child_policy!(latent, bound, value, storage::ModelArgumentStorage)
    # A nested owner can replace the original child, including its size and type.
    value === storage.value || _argument_graph!(bound, value)
    return _argument_storage_policy!(latent, bound, storage)
end
_argument_storage(value, template) = ModelArgumentLeaf(true)
function _argument_storage(tree::ModelValueTree, template)
    return _argument_storage(tree.values, tree.template)
end
function _argument_storage(values::VarNamedTuple, template)
    template isa NoTemplate && return nothing
    children = map(keys(values.data)) do name
        hasproperty(template, name) || throw(
            ArgumentError(
                "Cannot override nonexistent property `$name` of $(typeof(template)). If it holds a submodel return value, condition or fix the child model before wrapping it with `to_submodel`.",
            ),
        )
        _argument_child_storage(values.data[name], template, AbstractPPL.Property{name}())
    end
    return ModelArgumentStorage(template, NamedTuple{keys(values.data)}(children))
end
function _argument_storage(values::Tuple, template::Tuple)
    children = map(values, template) do value, child
        if value isa NoModelBinding
            ModelArgumentLeaf(false)
        else
            _argument_storage(value, child)
        end
    end
    return ModelArgumentStorage(template, children)
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
    values.data isa ModelBindingArray && (template = values.data.template)
    if template isa AbstractArray &&
        !(values.data isa VarNamedTuples.GrowableArray) &&
        axes(template) != axes(values.data)
        resized = similar(template, axes(values.data))
        # A new extent still keeps stale argument values at surviving latent indices.
        for i in eachindex(template)
            if checkbounds(Bool, resized, i) &&
                !haskey(values, i) &&
                isassigned(template, i)
                resized[i] = template[i]
            end
        end
        template = resized
    end
    children = map(CartesianIndices(values.mask)) do i
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
    return ModelArgumentStorage(template, children)
end

_model_argument_value(value, template) = value
_model_argument_value(::Nothing, template) = _copy_model_argument(template)
_model_argument_value(value::ModelValue, template) = value.value
_model_argument_value(values::AbstractArray, template) = _model_data(values)
function _model_argument_value(
    values::Union{ModelValueTree,VarNamedTuple,VarNamedTuples.PartialArray}, template
)
    storage = _argument_storage(values, template)
    return _apply_model_bindings(values, _copy_model_argument(storage))
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
function _apply_model_bindings(values::Tuple, storage)
    return map(values, storage.value, storage.children) do value, original, child
        value isa NoModelBinding ? original : _apply_model_bindings(value, child)
    end
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
    result = _writable_model_argument(storage.value)
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
    return BangBang.setindex!!(result, value, indices...; kwargs...)
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
    values = if tree.values isa Tuple
        # Tuple entries are indexed bindings, not a complete array replacement.
        mask = [!(value isa NoModelBinding) for value in tree.values]
        last = findlast(mask)
        VarNamedTuples.PartialArray(
            VarNamedTuples.GrowableArray(collect(tree.values[1:last])),
            VarNamedTuples.GrowableArray(mask[1:last]),
        )
    else
        tree.values
    end
    return _plain_model_values(values)
end
function _plain_model_values(values::VarNamedTuples.PartialArray)
    return _fold_model_indices(empty(values), values) do selected, value, optic, template
        return VarNamedTuples._setindex_optic!!(
            selected, _plain_model_values(value), optic, template, VarNamedTuples.AllowAll()
        )
    end
end

# ----------------
# Model definition
# ----------------

struct PrefixTemplate{V<:VarName,T,I}
    prefix::V
    template::T
    inner::I
end
_apply_prefix_template(::Nothing, template) = template
function _apply_prefix_template(prefix::VarName, template)
    return SkipTemplate{optic_skip_length(AbstractPPL.getoptic(prefix)) + 1}(template)
end
function _apply_prefix_template(prefix::PrefixTemplate, template)
    return VarNamedTuples.nested_template(
        AbstractPPL.getoptic(prefix.prefix),
        prefix.template,
        _apply_prefix_template(prefix.inner, template),
    )
end
_compose_prefix_templates(::Nothing, inner) = inner
function _compose_prefix_templates(prefix::VarName, inner)
    return PrefixTemplate(prefix, NoTemplate(), inner)
end
function _compose_prefix_templates(prefix::PrefixTemplate, inner)
    return PrefixTemplate(
        prefix.prefix, prefix.template, _compose_prefix_templates(prefix.inner, inner)
    )
end

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")
function unsplat_symbol(s::Symbol)
    return is_splat_symbol(s) ? Symbol(chopprefix(string(s), "#splat#")) : s
end

# The existing argument metadata slot also carries macro-known LHS addresses.
struct ModelBindingMetadata{Arguments,LHS,Submodels,Types} end
_args_on_lhs(::ModelBindingMetadata{A}) where {A} = A
_args_on_lhs(names::Union{Tuple,Vector{Symbol}}) = Tuple(names)
_lhs_names(::ModelBindingMetadata{A,L}) where {A,L} = L
_lhs_names(::Tuple) = nothing
_may_have_submodels(::ModelBindingMetadata{A,L,S}) where {A,L,S} = !isempty(S)
_submodel_lhs_names(::ModelBindingMetadata{A,L,S}) where {A,L,S} = S
_may_have_submodels(::Tuple) = true

_declared_argument_type(::Tuple, name) = Any
function _declared_argument_type(::ModelBindingMetadata{A,L,S,T}, name) where {A,L,S,T}
    return fieldtype(T, name)
end

function _reconstruct_model end

"""
    Model{Threaded}(f, args::NamedTuple, defaults::NamedTuple, context=DefaultContext(); args_on_lhs=())

Store a model function, arguments, and context. Prefer [`@model`](@ref) for construction.
The names of arguments with LHS variables are stored as immutable type metadata.
Set `args_on_lhs` to the tuple of argument names that occur on the left-hand side of
`~`, for example `Model{false}(f, (; y=1.0), (;); args_on_lhs=(:y,))`.
These arguments record argument-supplied observations, just as with `@model`, and can
be bound with [`condition`](@ref) or [`fix`](@ref). Use [`decondition`](@ref) to remove
argument-supplied observations. Without `args_on_lhs`, direct construction records
no argument-supplied observations and its arguments cannot be bound.
Whole `missing` or `nothing` arguments throw `ArgumentError` when a tilde reads their
argument-supplied observations, even if the body replaces them. Use [`decondition`](@ref)
to make their LHS variables latent.
Partial bindings into whole `missing`/`nothing` arguments, even after deconditioning, throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
At a submodel tilde, an argument LHS variable receives the submodel return value. The
argument supplies only its value before the tilde runs; its argument-supplied observation
is ignored at that tilde, so it needs no deconditioning. See [Binding rules](@ref).

Handwritten evaluators are responsible for the argument preparation and binding-aware tilde
protocol generated by `@model`. In particular, prepare argument LHS variables with
`prepare_model_argument` before the body, resolve their roles with `_get_argument_role`,
and validate observed or fixed values with `_check_tilde_value`. Calling `tilde_observe!!`
directly does not perform binding lookup or the whole-placeholder check. The constructor
records observations; it does not wrap a handwritten evaluator to enforce this protocol.
These helpers are internal; prefer reusing an evaluator generated by `@model`.

!!! note "Why keep `Model.f`?"
    Local models can capture variables from the enclosing function:

    ```julia
    function run_analysis(offset)
        @model function demo()
            x ~ Normal(offset, 1)
            return x
        end
        return demo()
    end
    ```

    `run_analysis(1)` and `run_analysis(2)` have the same evaluator type `F` but capture different
    offsets. The field `f` holds these values; the type alone cannot.

    Defining `(model::Model)(...)` or extending a global `get_evaluator` inside `run_analysis`
    requires a global method definition, which Julia rejects there. Using `eval` installs
    the method, but world age prevents the running `run_analysis` from calling it directly.
    We'd need a bridge such as `invokelatest` and separate storage for captures. Keeping `f`
    avoids both.
"""
struct Model{
    F,
    argnames,
    defaultnames,
    Targs,
    Tdefaults,
    C<:AbstractContext,
    Values<:Union{
        VarNamedTuple,ModelBindingLayers,LocalModelValues,UnprefixedArgumentValues
    },
    Threaded,
    ArgsOnLHS,
} <: AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    context::C
    values::Values
    function Model{Threaded}(
        f::F,
        args::NamedTuple{A,Ta},
        defaults::NamedTuple{D,Td},
        context::C,
        values::V;
        args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
    ) where {F,A,Ta,D,Td,C,V,Threaded}
        mapreduce(
            pair -> pair.second isa ModelValue, &, _model_values(values); init=true
        ) || throw(ArgumentError("Model values must carry a condition or fix role"))
        argument_names = _args_on_lhs(args_on_lhs)
        metadata = args_on_lhs isa ModelBindingMetadata ? args_on_lhs : argument_names
        for name in argument_names
            name in (_argument_names(args)..., _argument_names(defaults)...) || throw(
                ArgumentError(
                    "`$name` in `args_on_lhs` is not an argument or default name"
                ),
            )
        end
        return new{F,A,D,Ta,Td,C,V,Threaded,metadata}(f, args, defaults, context, values)
    end
    # Internal reconstruction reuses already-validated bindings.
    function DynamicPPL._reconstruct_model(
        model::Model{F,A,D,Ta,Td}, context::C, values::V, ::Val{Threaded}
    ) where {F,A,D,Ta,Td,C,V,Threaded}
        return new{F,A,D,Ta,Td,C,V,Threaded,_binding_metadata(model)}(
            model.f, model.args, model.defaults, context, values
        )
    end
end

function _binding_metadata(
    ::Model{F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
) where {F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
    return ArgsOnLHS
end

_args_on_lhs(model::Model) = _args_on_lhs(_binding_metadata(model))

Base.@constprop :aggressive function Model{Threaded}(
    f,
    args::NamedTuple,
    defaults::NamedTuple,
    context::AbstractContext=DefaultContext();
    args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
) where {Threaded}
    values = _argument_defaults(merge(args, defaults), Val(_args_on_lhs(args_on_lhs)))
    return Model{Threaded}(f, args, defaults, context, values; args_on_lhs)
end

"""
    Model{Threaded}(f, args::NamedTuple; kwargs...)

Create a model with evaluation function `f` and arguments `args`.

Arguments are ordinary inputs; no argument-supplied observations or metadata about
argument LHS variables are recorded, and these arguments cannot be bound.
Use [`@model`](@ref) or the constructor accepting `defaults` and `args_on_lhs` to
construct a model with argument-supplied observations.

Keyword arguments `kwargs` are stored in the model's `defaults` field.
"""
function Model{Threaded}(f, args::NamedTuple; kwargs...) where {Threaded}
    return Model{Threaded}(f, args, NamedTuple(kwargs))
end

"""
    requires_threadsafe(model::Model)

Return whether `model` has been marked as needing threadsafe evaluation (using
`setthreadsafe`).
"""
requires_threadsafe(::Model{F,A,D,Ta,Td,C,V,Threaded}) where {F,A,D,Ta,Td,C,V,Threaded} =
    Threaded
function _reconstruct_model(model::Model; context=model.context, values=model.values)
    return _reconstruct_model(model, context, values, Val(requires_threadsafe(model)))
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
    contextualize(model::Model, context::AbstractContext)

Return a model with its context replaced by `context`.
"""
function contextualize(model::Model, context::AbstractContext)
    if model.values isa UnprefixedArgumentValues && (
        last(extract_prefixes(context)) != _model_prefix(model) ||
        _prefix_template(context) !== _model_prefix_template(model)
    )
        model = _materialize_argument_values(model)
    end
    return _reconstruct_model(model; context)
end
"""Return a model with its leaf context replaced by `context`."""
function setleafcontext(model::Model, context::AbstractContext)
    return contextualize(model, setleafcontext(model.context, context))
end
_model_prefix(model::Model) = last(extract_prefixes(model.context))
_prefix_template(::AbstractContext) = nothing
_prefix_template(context::AbstractParentContext) = _prefix_template(childcontext(context))
_prefix_template(context::PrefixContext) = context.template
_model_prefix_template(model::Model) = _prefix_template(model.context)

"""
    setthreadsafe(model::Model, threadsafe::Bool)

Returns a new `Model` with its threadsafe flag set to `threadsafe`.

Threadsafe evaluation ensures correctness when executing model statements that mutate the
internal `VarInfo` object in parallel. For example, this is needed if tilde-statements are
nested inside `Threads.@threads` or similar constructs.

It is not needed for generic multithreaded operations that don't involve VarInfo. For
example, calculating a log-likelihood term in parallel and then calling `@addlogprob!`
outside of the parallel region is safe without needing to set `threadsafe=true`.

It is also not needed for multithreaded sampling with AbstractMCMC's `MCMCThreads()`.

Setting `threadsafe` to `true` increases the overhead in evaluating the model. Please see
[the Turing.jl docs](https://turinglang.org/docs/usage/threadsafe-evaluation/) for more
details.
"""
function setthreadsafe(model::Model, threadsafe::Bool)
    return if requires_threadsafe(model) == threadsafe
        model
    else
        _reconstruct_model(model, model.context, model.values, Val(threadsafe))
    end
end

"""
    model | (x = 1.0, ...)

Return a `Model` which now treats variables on the right-hand side as observations.

See [`condition`](@ref) for more information and examples.
"""
Base.:|(model::Model, values::Union{NamedTuple,AbstractDict,Pair,Tuple,VarNamedTuple}) =
    _bind_ordered_inputs(Condition, model, _binding_inputs(values))

function _check_binding_addresses(model, values)
    metadata = _binding_metadata(model)
    names = _lhs_names(metadata)
    names === nothing && return nothing
    prefix = _model_prefix(model)
    if !(model.values isa LocalModelValues) && prefix !== nothing
        for name in keys(values.data)
            name === AbstractPPL.getsym(prefix) || throw(
                ArgumentError(
                    "Cannot bind `$name`: it is outside this model's prefix `$prefix`."
                ),
            )
        end
    end
    _may_have_submodels(metadata) && return nothing
    local_values = if model.values isa LocalModelValues || _model_prefix(model) === nothing
        values
    else
        _submodel_values(values, _model_prefix(model))
    end
    for name in keys(local_values.data)
        name in names || throw(
            ArgumentError(
                "Cannot bind `$name`: it is not an LHS top symbol of this model. Use an LHS address, or bind a child model through its submodel namespace.",
            ),
        )
    end
    return nothing
end

@generated function _check_argument_bindings(model::Model, values)
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
                if binding isa ModelValue
                    T = _declared_argument_type(_binding_metadata(model), stored_name)
                    binding.value isa T || throw(
                        ArgumentError(
                            "Bound value at `$vn` in model `$(nameof(model))` must be an instance of declared argument type $T; supplied $(typeof(binding.value)).",
                        ),
                    )
                end
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
                    argument isa Union{Nothing,Missing} && throw(
                        ArgumentError(
                            "Cannot make a partial binding of argument `$vn` with whole `$argument` storage; supply a concrete argument (e.g. `$(nameof(model))(zeros(n))`) or a whole binding instead.",
                        ),
                    )
                    binding = _prepare_argument_fields(argument, binding, vn)
                    values = templated_setindex!!(
                        values, binding, vn, values.data[AbstractPPL.getsym(vn)]
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
        _check_namedtuple_index(
            ModelValue{Condition}(template), optic, AbstractPPL.varname_to_optic(vn)
        )
        if template isa Union{AbstractArray,Tuple} && optic isa AbstractPPL.Index
            indices = AbstractPPL.concretize_top_level(optic, template)
            bounds = template isa Tuple ? Base.OneTo(length(template)) : template
            checkbounds(Bool, bounds, indices.ix...; indices.kw...) || throw(
                ArgumentError(
                    "Cannot bind `$(AbstractPPL.append_optic(vn, optic))`: index is outside argument `$vn`",
                ),
            )
        end
        if template isa NamedTuple &&
            optic isa AbstractPPL.Property &&
            !VarNamedTuples._haskey_optic(template, optic)
            throw(
                ArgumentError(
                    "Cannot override nonexistent field `$(AbstractPPL.append_optic(vn, optic))` of argument `$vn`. If it holds a submodel return value, condition or fix the child model before wrapping it with `to_submodel`.",
                ),
            )
        end
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
    elseif template isa Tuple && !multi
        fieldtype(typeof(template), only(optic.ix))
    else
        Any
    end
    converted = multi ? map(v -> convert(T, v), value) : convert(T, value)
    isequal(converted, value) || throw(
        ArgumentError(
            "Cannot exactly represent partial binding at `$vn` in storage element type $T",
        ),
    )
    if optic isa AbstractPPL.Property && ismutabletype(typeof(template))
        name = _binding_property_name(optic)
        if hasfield(typeof(template), name) && isconst(typeof(template), name)
            throw(
                ArgumentError(
                    "Cannot bind const property `$name` of argument type $(typeof(template)); bind the whole value instead.",
                ),
            )
        end
    end
    if optic isa AbstractPPL.Property && !ismutabletype(typeof(template))
        name = _binding_property_name(optic)
        _argument_reconstructible(template, NamedTuple{(name,)}((converted,))) || throw(
            ArgumentError(
                "Cannot rebuild argument of type $(typeof(template)) when binding property `$name`; provide a ConstructionBase.setproperties method or bind the whole value.",
            ),
        )
    end
    return ModelValue{R}(converted)
end
_check_binding_template_bounds(template, ::AbstractPPL.Iden, vn) = nothing
function _check_binding_template_bounds(
    template, optic::AbstractPPL.Property{S}, vn
) where {S}
    optic.child isa AbstractPPL.Iden && return nothing
    child = VarNamedTuples.SharedGetProperty{S}()(template)
    return _check_binding_template_bounds(child, optic.child, vn)
end
function _check_binding_template_bounds(template, optic::AbstractPPL.Index, vn)
    array = if template isa VarNamedTuples.PartialArray
        template.data
    else
        VarNamedTuples.template_array(template)
    end
    coptic = AbstractPPL.concretize_top_level(optic, array)
    inbounds = if array isa AbstractArray
        checkbounds(Bool, array, coptic.ix...; coptic.kw...)
    elseif array isa Union{NoTemplate,VarNamedTuples.SkipTemplate,Missing}
        dims = VarNamedTuples.get_maximum_size_from_indices(coptic.ix...; coptic.kw...)
        all(>=(0), dims) || throw(ArgumentError("invalid Array dimensions"))
        Base.checkbounds_indices(Bool, map(Base.OneTo, dims), coptic.ix)
    else
        true
    end
    inbounds || throw(
        ArgumentError(
            "Cannot bind `$vn`: index is outside the storage at `$(AbstractPPL.getsym(vn))`",
        ),
    )
    if !(coptic.child isa AbstractPPL.Iden)
        child = VarNamedTuples.index_template(template, coptic)
        _check_binding_template_bounds(child, coptic.child, vn)
    end
    return nothing
end

"""
    condition(model::Model; values...)
    condition(model::Model, values..., [schema])

Return a `Model` which treats the LHS variables bound by `values` as observations: they replace
sampling and contribute to the likelihood.

See also: [`decondition`](@ref), [`conditioned`](@ref)

Fixed bindings shadow observations. Within each layer, later bindings replace earlier
ones where they overlap. Subvariables of one LHS variable must have the same role.
A local LHS variable reads its binding at its tilde. For an argument LHS variable,
every binding replaces the argument before the body runs; observations read its current
value in the body, while fixed LHS variables reset to their bound value at the tilde.
Observe raw data under a separate name if the body transforms the argument first.

Use NamedTuples/keywords for whole top-level values, `VarName` pairs for any address
(`:x => v` abbreviates `@varname(x) => v`), or a [`VarNamedTuple`](@ref) produced by
DynamicPPL. Positional inputs and tuples apply left to right. Every `AbstractDict` throws
`ArgumentError`. One positional binding schema, e.g. `@of(z = of(Array, 3))`, supplies
storage for partially bound local LHS variables without binding values (`using AbstractPPL: of, @of`).
Existing owners in the observation layer take precedence over schemas; conflicts throw.
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
Partial bindings take a shallow snapshot when made: the owner's container is copied one
level deep, but nested mutable values remain shared and must not be mutated either.
`missing`/`nothing` throw where a tilde reads them, naming the LHS variable. Unread parts
may contain either; whole argument placeholders throw even if the body replaces them.
Use [`decondition`](@ref) to make observations latent; see [Missing data](@ref).
Partial bindings into whole `missing`/`nothing` arguments, even after deconditioning, throw
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
are applied left to right. Other inputs, including every `AbstractDict`, throw `ArgumentError`.

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

Use `VarName` pairs with a binding schema as above. A schema supplies storage without
observing or fixing any values. It is an AbstractPPL `OfNamedTuple` type, equivalently
written `of((m=of(Array, 2),))`, and may appear anywhere among the positional inputs,
at most once per call. Keywords remain binding data; `|` does not take a schema.
An existing owner within the layer being edited takes precedence; conflicting storage
throws `ArgumentError`. Schema entries for arguments, unrelated names, or names the call
does not bind also throw `ArgumentError`. Resolve symbolic sizes before binding.
Use whole bindings for custom arrays and structs that `of` cannot describe.
Values DynamicPPL produces, such as `rand(model)` and `conditioned(model)`, are
[`VarNamedTuple`](@ref)s and can be passed straight back for round trips.

## Nested models

`condition` also supports the use of nested models through the use of [`to_submodel`](@ref).

At a submodel tilde, an argument LHS variable receives the submodel return value. The
argument supplies only its value before the tilde runs; its argument-supplied observation
is ignored at that tilde. Explicit bindings
at or below that address are rejected during evaluation, including named-tuple submodel namespaces.
Condition or fix the child model before wrapping it with `to_submodel` instead.

```jldoctest condition
julia> @model demo_inner() = m ~ Normal()
demo_inner (generic function with 2 methods)

julia> @model function demo_outer()
           # By default, `to_submodel` prefixes the LHS variables using the left-hand side of `~`.
           inner ~ to_submodel(demo_inner())
           return inner
       end
demo_outer (generic function with 2 methods)

julia> model = demo_outer();

julia> model() ≠ 1.0
true

julia> # To condition the LHS variable inside `demo_inner` we need to refer to it as `inner.m`.
       conditioned_model = model | (@varname(inner.m) => 1.0, );

julia> conditioned_model()
1.0

julia> # Binding `inner` supplies a submodel namespace. For example, this will work:
       conditioned_model2 = model | (inner = (m = 1.0,), );

julia> conditioned_model2()
1.0

julia> # Conditioning a submodel's return value is not supported.
       conditioned_model_fail = model | (inner = "something else", );

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

function _prepare_local_binding_types(::Type{R}, model, values) where {R}
    metadata = _binding_metadata(model)
    _lhs_names(metadata) === nothing && return values
    prefix = _model_prefix(model)
    local_values = if model.values isa LocalModelValues || prefix === nothing
        values
    else
        _submodel_values(values, prefix)
    end
    local_previous = _submodel_values(model, nothing)
    for name in keys(local_values.data)
        name in _args_on_lhs(model) && continue
        name in _lhs_names(metadata) || continue
        name in _submodel_lhs_names(metadata) && continue
        previous = get(local_previous.data, name, nothing)
        previous === nothing && continue
        update = local_values.data[name]
        vn = VarName{name}()
        if update isa ModelValue
            owner = _binding_owner(R, previous)
            owner isa NoTemplate ||
                update.value isa typeof(owner) ||
                throw(
                    ArgumentError(
                        "Bound value at `$vn` must be an instance of template type $(typeof(owner)); supplied $(typeof(update.value)).",
                    ),
                )
        else
            owner = _binding_owner(R, previous)
            owner isa NoTemplate && continue
            update = _prepare_argument_fields(owner, update, vn)
            local_values = VarNamedTuple(
                merge(local_values.data, NamedTuple{(name,)}((update,)))
            )
        end
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

# Flatten ordered input groups without interpreting keyword values as schemas.
function _binding_inputs(values::Tuple)
    return mapreduce(_binding_inputs, (a, b) -> (a..., b...), values; init=())
end
_binding_inputs(value) = (value,)
_binding_inputs((name, value)::Pair{Symbol}) = (VarName{name}() => value,)

function _bind_ordered_inputs(::Type{R}, model, values) where {R}
    return foldl((m, v) -> _bind_model(R, m, v), values; init=model)
end

function _bind_inputs(::Type{R}, model::Model, inputs::Tuple) where {R}
    inputs = _binding_inputs(inputs)
    schemas = filter(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, inputs)
    length(schemas) <= 1 ||
        throw(ArgumentError("At most one binding schema is allowed per call."))
    values = filter(x -> !(x isa Type{<:AbstractPPL.OfNamedTuple}), inputs)
    isempty(schemas) && return _bind_ordered_inputs(R, model, values)
    schema = VarNamedTuples.materialize_template(only(schemas))
    for name in keys(schema)
        name in map(unsplat_symbol, keys(merge(model.args, model.defaults))) && throw(
            ArgumentError(
                "Binding schema entry `$name` names a model argument; use its existing storage.",
            ),
        )
        names = _lhs_names(_binding_metadata(model))
        (names !== nothing && name in names) || throw(
            ArgumentError("Binding schema entry `$name` is not a local LHS top symbol.")
        )
        root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        any(v -> _input_binds_root(v, root), values) ||
            throw(ArgumentError("Binding schema entry `$name` is not bound by this call."))
    end
    for value in values
        model = _bind_schema_input(R, model, value, schema)
    end
    return model
end
_input_binds_root(value::Pair{<:VarName}, root) = subsumes(root, first(value))
_input_binds_root(value::NamedTuple, root) = haskey(value, AbstractPPL.getsym(root))
_input_binds_root(value::VarNamedTuple, root) = any(vn -> subsumes(root, vn), keys(value))
_input_binds_root(value, root) = false

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

function _schema_has_address(model, ::NamedTuple{names}, vn::VarName{sym}) where {names,sym}
    if model.values isa LocalModelValues || _model_prefix(model) === nothing
        return sym in names
    end
    return any(names) do name
        root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        subsumes(root, vn) || subsumes(vn, root)
    end
end

function _bind_schema_input(::Type{R}, model, input, schema) where {R}
    layer_model = _binding_layer_model(R, model)
    layer = _model_values(layer_model.values)
    templates = VarNamedTuple()
    for (name, storage) in pairs(schema)
        root = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        previous = _model_argument_binding(layer, AbstractPPL.varname_to_optic(root))
        if previous !== nothing
            owner = if previous isa Union{VarNamedTuples.PartialArray,VarNamedTuple}
                _select_model_node(R, previous)
            else
                _schema_storage(previous)
            end
            _same_schema_storage(owner, storage) || throw(
                ArgumentError(
                    "Binding schema storage for `$name` conflicts with its existing owner.",
                ),
            )
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
    prepared = copy(plain)
    for (vn, value) in entries
        uses_schema = _schema_has_address(model, schema, vn)
        if !uses_schema
            is_pair && return _bind_model(R, model, input; preparation_model=layer_model)
            continue
        end
        template = _binding_template(model, templates, vn)
        optic = AbstractPPL.getoptic(vn)
        _check_binding_template_bounds(template, optic, vn)
        value = _convert_binding_template(value, template, optic, vn)
        prepared = templated_setindex!!(prepared, value, vn, template)
    end
    input = prepared
    return _bind_model(R, model, input; preparation_model=layer_model)
end

function _convert_binding_template(value, template, ::AbstractPPL.Iden, vn)
    template isa Union{NoTemplate,VarNamedTuples.SkipTemplate} && return value
    value isa typeof(template) || throw(
        ArgumentError(
            "Bound value at `$vn` must be an instance of template type $(typeof(template)); supplied $(typeof(value)).",
        ),
    )
    _same_schema_storage(value, template) || throw(
        ArgumentError("Bound value at `$vn` conflicts with its binding schema storage.")
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
    # Descend to the local root before applying whole-value schema checks.
    if optic.child isa AbstractPPL.Iden && !(template isa VarNamedTuple)
        return _convert_partial_argument_binding(
            ModelValue{Condition}(value), template, head, vn
        ).value
    end
    child = _binding_child_template(template, head)
    return _convert_binding_template(value, child, optic.child, vn)
end

function _bind_model(::Type{R}, model::Model, values...; preparation_model=model) where {R}
    model = _materialize_argument_values(model)
    preparation_model = _binding_layer_model(
        R, _materialize_argument_values(preparation_model)
    )
    values = _tag_model_values(R, _make_condfix_values(preparation_model, values...))
    values = _check_argument_bindings(preparation_model, values)
    _check_binding_addresses(model, values)
    values = _prepare_local_binding_types(R, preparation_model, values)
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
    return _reconstruct_model(model; values)
end
function AbstractPPL.condition(model::Model; values...)
    return condition(model, NamedTuple(values))
end

"""
    _make_condfix_values(model, values...)

Convert normalised binding values to a `VarNamedTuple`.
Input ordering, keyword arguments and schemas are handled by the binding entry points.
"""
function _make_condfix_values(model, values...)
    throw(
        ArgumentError(
            "Bindings require a VarNamedTuple, NamedTuple, or VarName pairs (or an ordered tuple of these); use keywords for whole values or @varname(x) => value for an address. AbstractDict inputs are not supported.",
        ),
    )
end
_make_condfix_values(model, values::NamedTuple) = VarNamedTuple(values)
_make_condfix_values(model, values::VarNamedTuple) = values

_binding_storage(value) = value
_binding_storage(::Union{Nothing,Missing,Number}) = NoTemplate()
_binding_storage(value::ModelValue) = _binding_storage(value.value)
_binding_storage(value::ModelValueTree) = _binding_storage(_model_data(value))
_binding_storage(value::NamedTuple) = map(_binding_storage, value)
_binding_storage(value::Tuple) = map(_binding_storage, collect(value))
_binding_storage(value::VarNamedTuple) = VarNamedTuple(map(_binding_storage, value.data))
function _binding_storage(value::VarNamedTuples.PartialArray)
    # Inferred, growable storage has no owner to constrain subsequent indices.
    value.data isa VarNamedTuples.GrowableArray && return NoTemplate()
    return VarNamedTuples._map_values_recursive!!(_binding_storage, copy(value))
end

_binding_storage(values::VarNamedTuple, ::Tuple{}) = VarNamedTuple()
function _binding_storage(values::VarNamedTuple, pairs::Tuple{Pair,Vararg{Pair}})
    templates = _binding_storage(values, Base.tail(pairs))
    name = AbstractPPL.getsym(first(first(pairs)))
    if haskey(templates.data, name) || !haskey(values.data, name)
        return templates
    end
    storage = _binding_storage(values.data[name])
    return VarNamedTuple(merge(templates.data, NamedTuple{(name,)}((storage,))))
end

function _make_condfix_values(model, values::Pair{<:VarName}...)
    # Existing local owners also supply storage, including axes and nested fields.
    # Only traverse roots addressed by these pairs; unrelated partial storage can be large.
    templates = _binding_storage(_model_values(model.values), values)
    for (stored_name, argument) in pairs(merge(model.args, model.defaults))
        name = unsplat_symbol(stored_name)
        vn = _model_value_varname(model.values, VarName{name}(), _model_prefix(model))
        previous = _model_argument_binding(
            _model_values(model.values), AbstractPPL.varname_to_optic(vn)
        )
        template =
            previous === nothing ? argument : prepare_model_argument(previous, argument)
        template isa AbstractArray || continue
        templates = templated_setindex!!(
            templates,
            template,
            vn,
            _binding_template(model, _model_values(model.values), vn),
        )
    end
    result = VarNamedTuple()
    for (name, value) in values
        vn = name
        for stored_name in keys(model.defaults)
            is_splat_symbol(stored_name) || continue
            argument = unsplat_symbol(stored_name)
            root = _model_value_varname(
                model.values, VarName{argument}(), _model_prefix(model)
            )
            if root != vn && subsumes(root, vn)
                throw(
                    ArgumentError(
                        "Entries of keyword-splat argument `$argument` cannot be bound; replace the whole argument with `condition` or `fix` instead.",
                    ),
                )
            end
        end
        _check_namedtuple_index(
            _model_values(model.values), AbstractPPL.varname_to_optic(vn)
        )
        template = _binding_template(model, templates, vn)
        _check_namedtuple_index(
            ModelValue{Condition}(template),
            AbstractPPL.getoptic(vn),
            AbstractPPL.Property{AbstractPPL.getsym(vn)}(),
        )
        _check_binding_template_bounds(template, AbstractPPL.getoptic(vn), vn)
        result = templated_setindex!!(result, value, vn, template)
    end
    return result
end
function _binding_template(model, templates::VarNamedTuple, vn::VarName)
    template = get(templates.data, AbstractPPL.getsym(vn), NoTemplate())
    prefix = _model_prefix(model)
    if template isa NoTemplate &&
        prefix !== nothing &&
        AbstractPPL.getsym(vn) === AbstractPPL.getsym(prefix)
        return _apply_prefix_template(_model_prefix_template(model), NoTemplate())
    end
    return template
end

"""
    decondition(model::Model)
    decondition(model::Model, names...)

Remove this model's conditioned bindings at `names...`, or all conditioned bindings
if no names are supplied.

Unlike [`unfix`](@ref), `decondition(m, :x)` removes explicit and argument-supplied
observations, even beneath a fixed binding. `x` becomes latent unless still fixed.
A latent argument retains its old value before its tilde; its sampled value replaces
that value at the tilde and is used by subsequent body statements.

A name matches when it equals, contains, or is contained in a stored binding's address.
NamedTuple integer indices are rejected: use `x.a` instead of `x[1]`; Tuples keep integer indices.
Only the matching conditioned parts are removed. A name with no match throws
`ArgumentError`, including names with only fixed bindings. With no names, removing all
observations is always valid.

Only bindings stored on this model are removed. This cannot remove a child submodel's
argument-supplied observations: `decondition(outer_arg(), @varname(a.x))` throws when `a.x`
is supplied only by the child argument. Decondition the child before wrapping it with
`to_submodel` instead.

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
    model = _materialize_argument_values(model)
    observations = _observation_values(model.values)
    _check_removal_addresses(observations, syms...)
    _check_model_removal(Condition, observations, syms...)
    values = _remove_model_values(Condition, observations, syms...)
    fixed_values = _fixed_values(model.values)
    isempty(fixed_values) ||
        (values = ModelBindingLayers(values, fixed_values, _fixed_owners(model.values)))
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    return _reconstruct_model(model; values)
end

function _check_removal_addresses(values, names...)
    for name in names
        vn = name isa VarName ? name : VarName{name}()
        _check_shapeless_removal(
            values, AbstractPPL.varname_to_optic(vn), AbstractPPL.Iden(), vn
        )
        _check_namedtuple_index(values, AbstractPPL.varname_to_optic(vn))
    end
    return nothing
end

function _check_shapeless_removal(binding, optic, prefix, vn)
    optic isa AbstractPPL.Iden && return nothing
    if binding isa ModelValue{<:Any,<:Union{Missing,Nothing}}
        root = AbstractPPL.optic_to_varname(prefix)
        remove = binding isa ModelValue{Fix} ? "unfix" : "decondition"
        throw(
            ArgumentError(
                "Cannot remove part `$vn`: the whole value `$(binding.value)` at `$root` supplies no template. Use `$remove(model, @varname($root))` to remove the whole binding.",
            ),
        )
    end
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(binding, head)
    return _check_shapeless_removal(child, optic.child, head ∘ prefix, vn)
end

function _check_model_removal(::Type{R}, values, args...) where {R}
    for arg in args
        vn = arg isa VarName ? arg : VarName{arg}()
        _check_namedtuple_index(values, AbstractPPL.varname_to_optic(vn))
        binding = _model_argument_binding(values, AbstractPPL.varname_to_optic(vn))
        VarNamedTuples._mapreduce_recursive(
            pair -> _matches_model_role(R, pair.second), |, binding, vn, false
        ) && continue
        mapreduce(
            pair -> subsumes(vn, pair.first) && _matches_model_role(R, pair.second),
            |,
            values;
            init=false,
        ) && continue
        role = R === Condition ? "conditioned" : "fixed"
        message = "Cannot remove `$vn`: no $role binding is stored at this address."
        if VarNamedTuples._mapreduce_recursive(
            pair -> pair.second isa ModelValue, |, binding, vn, false
        )
            other = R === Condition ? "fixed" : "conditioned"
            message *= " The stored binding is $other."
        elseif R === Condition && !(AbstractPPL.getoptic(vn) isa AbstractPPL.Iden)
            message *= " If this is a child model's argument, decondition the child model before wrapping it with `to_submodel`."
        end
        throw(ArgumentError(message))
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

function _remove_model_values(
    ::Type{R}, values::VarNamedTuple, args::Union{Symbol,VarName}...
) where {R}
    for arg in args
        vn = arg isa VarName ? arg : VarName{arg}()
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
            values = _remove_model_binding(R, values, AbstractPPL.varname_to_optic(vn))
        end
    end
    return _prune_model_bindings(values)
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
    return ModelValueTree(tree.template, _remove_model_binding(R, tree.values, optic))
end
function _remove_model_binding(
    ::Type{R},
    values::Union{VarNamedTuple,ModelValueTree{<:NamedTuple}},
    optic::AbstractPPL.Index{Tuple{Symbol},NamedTuple{(),Tuple{}}},
) where {R}
    return _remove_model_binding(
        R, values, AbstractPPL.Property{only(optic.ix)}(optic.child)
    )
end

function _remove_model_binding(
    ::Type{R}, tree::ModelValueTree{<:Tuple}, optic::AbstractPPL.Index
) where {R}
    optic = AbstractPPL.concretize_top_level(optic, tree.template)
    checkbounds(Bool, Base.OneTo(length(tree.values)), optic.ix...) || return tree
    indices = getindex(ntuple(identity, length(tree.values)), optic.ix...)
    if indices isa Integer
        child = _remove_model_binding(R, tree.values[indices], optic.child)
        return ModelValueTree(tree.template, Base.setindex(tree.values, child, indices))
    end
    selected = ModelValueTree(
        getindex(tree.template, optic.ix...), getindex(tree.values, optic.ix...)
    )
    removed = _remove_model_binding(R, selected, optic.child)
    values = tree.values
    for (i, child) in zip(indices, removed.values)
        values = Base.setindex(values, child, i)
    end
    return ModelValueTree(tree.template, values)
end
function _remove_model_binding(
    ::Type{R}, values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index
) where {R}
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    checkbounds(Bool, values.data, optic.ix...; optic.kw...) || return values
    selected = if VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
        VarNamedTuples._subset_partialarray(values, optic.ix...; optic.kw...)
    elseif haskey(values, optic.ix...; optic.kw...)
        getindex(values, optic.ix...; optic.kw...)
    else
        return values
    end
    child = _remove_model_binding(R, selected, optic.child)
    return VarNamedTuples._setindex_optic!!(
        copy(values),
        child,
        AbstractPPL.Index(optic.ix, optic.kw),
        values,
        VarNamedTuples.AllowAll(),
    )
end

"""
    conditioned(model::Model)

Return this model's conditioned values as plain values, independent of binding history.

This may include argument-supplied observations that evaluation ignores when their argument LHS variables
receive submodel return values.

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
    fix(model::Model, values..., [schema])

Return a `Model` which treats the LHS variables bound by `values` as constants: they replace
sampling and contribute no log probability. Fixed argument LHS variables reset to their bound
value when their tilde statement runs, even if the body has computed a different value.

Fixed bindings shadow observations. [`unfix`](@ref) uncovers the observation below,
or leaves the LHS variable latent if none remains; [`decondition`](@ref) removes
observations even beneath a fixed binding.

Inputs, conversion errors, aliasing and argument preparation follow [`condition`](@ref).
Partial bindings into whole `missing`/`nothing` arguments, even after deconditioning, throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
Partial bindings copy the owner's container one level deep; nested mutable values remain
shared and must not be mutated.
For example, `fix(model, @varname(z[2]) => 1.0, @of(z = of(Array, 3)))` supplies local
storage (`using AbstractPPL: of, @of`). Owners in the fixed layer take precedence over
the schema. Runtime bindings under ForwardDiff/ReverseDiff need storage compatible with
AD values, e.g. `@of(z = of(Array, typeof(m), n))`, or a whole value.

Fixed values must cover their LHS variables with a static size and shape. Changing a
fixed argument's size or shape in the body throws `ArgumentError` naming the LHS variable.
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
    unfix(model::Model, names...)

Remove this model's fixed bindings at `names...`, or all fixed bindings if no names
are supplied. NamedTuple integer indices are rejected: use `x.a` instead of `x[1]`; Tuples keep integer indices.
Matching follows [`decondition`](@ref). A name with no stored fixed match throws `ArgumentError`,
including a name supplied only by a child submodel or only conditioned on this model.

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
    model = _materialize_argument_values(model)
    fixed_values = _fixed_values(model.values)
    _check_removal_addresses(fixed_values, syms...)
    _check_model_removal(Fix, fixed_values, syms...)
    fixed_values = _remove_model_values(Fix, fixed_values, syms...)
    observations = _observation_values(model.values)
    values = if isempty(fixed_values)
        observations
    else
        owners = filter(_fixed_owners(model.values)) do owner
            any(vn -> subsumes(owner, vn) || subsumes(vn, owner), keys(fixed_values))
        end
        ModelBindingLayers(observations, fixed_values, owners)
    end
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
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

For argument LHS variables that receive submodel return values, `conditioned` lists argument-supplied observations
that evaluation ignores, while `fixed` lists explicit bindings that evaluation rejects.

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

function _prefix_values(values::ModelBindingLayers, vn::VarName, template)
    return ModelBindingLayers(
        _prefix_values(values.observations, vn, template),
        _prefix_values(values.fixed, vn, template),
        map(owner -> maybe_prefix(owner, vn), values.owners),
    )
end
function _prefix_values(values::VarNamedTuple, vn::VarName, template)
    isempty(values) && return values
    return templated_setindex!!(VarNamedTuple(), values, vn, template)
end

# Prefix templates can cross submodel boundaries without reading parent return values.
function _concretize_prefix(vn::VarName{S}, template) where {S}
    return VarName{S}(_concretize_prefix(AbstractPPL.getoptic(vn), template))
end
_concretize_prefix(optic::AbstractPPL.Iden, template) = optic
function _concretize_prefix(optic::AbstractPPL.Property{S}, template) where {S}
    AbstractPPL.is_dynamic(optic) || return optic
    child = _concretize_prefix(optic.child, VarNamedTuples.SharedGetProperty{S}()(template))
    return AbstractPPL.Property{S}(child)
end
function _concretize_prefix(optic::AbstractPPL.Index, template)
    AbstractPPL.is_dynamic(optic) || return optic
    optic = AbstractPPL.concretize_top_level(optic, VarNamedTuples.template_array(template))
    AbstractPPL.is_dynamic(optic.child) || return optic
    child = _concretize_prefix(optic.child, VarNamedTuples.index_template(template, optic))
    return AbstractPPL.Index(optic.ix, optic.kw, child)
end

maybe_prefix(vn::VarName, ::Nothing) = vn
maybe_prefix(::Nothing, ::Nothing) = nothing
maybe_prefix(::Nothing, prefix::VarName) = prefix
maybe_prefix(vn::VarName, prefix::VarName) = AbstractPPL.prefix(vn, prefix)

"""
    prefix(model::Model, x::VarName; template=NoTemplate())
    prefix(model::Model, x::Val{sym})
    prefix(model::Model, x::Any)

Return `model` but with all random variables prefixed by `x`, where `x` is either:
- a `VarName` (e.g. `@varname(a)`),
- a `Val{sym}` (e.g. `Val(:a)`), or
- for any other type, `x` is converted to a Symbol and then to a `VarName`. Note that
  this will introduce runtime overheads so is not recommended unless absolutely
  necessary.

For an indexed prefix, `template` supplies the enclosing container's shape and resolves
`begin` and `end` indices.

# Examples

```jldoctest
julia> using DynamicPPL: prefix

julia> @model demo() = x ~ Dirac(1)
demo (generic function with 2 methods)

julia> rand(prefix(demo(), @varname(my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1

julia> rand(prefix(demo(), Val(:my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1
```
"""
function prefix(model::Model, x::VarName; template=NoTemplate())
    template = VarNamedTuples.materialize_template(template)
    x = _concretize_prefix(x, template)
    model = _materialize_argument_values(model)
    values =
        if model.values isa VarNamedTuple &&
            _model_prefix(model) === nothing &&
            !isempty(model.values) &&
            mapreduce(
                pair -> pair.second isa ModelValue{ArgumentCondition},
                &,
                model.values;
                init=true,
            )
            UnprefixedArgumentValues(model.values)
        else
            _prefix_values(model.values, x, template)
        end
    return _prefix_model(model, x, template, values)
end
function _prefix_model(model::Model, x::VarName, template, values)
    model_prefix = maybe_prefix(_model_prefix(model), x)
    prefix_template =
        if template isa NoTemplate && _model_prefix_template(model) === nothing
            nothing
        else
            inner = if _model_prefix_template(model) === nothing
                _model_prefix(model)
            else
                _model_prefix_template(model)
            end
            PrefixTemplate(x, template, inner)
        end
    context = PrefixContext(
        model_prefix, first(extract_prefixes(model.context)), prefix_template
    )
    return _reconstruct_model(model; context, values)
end
function prefix(model::Model, ::Val{sym}) where {sym}
    return prefix(model, VarName{sym}())
end
function prefix(model::Model, x)
    return prefix(model, VarName{Symbol(x)}())
end

function _prefix_varname_and_template(vn::VarName, template::Any, model::Model)
    return _prefix_varname_and_template(
        vn, template, _model_prefix(model), _model_prefix_template(model)
    )
end
function _prefix_varname_and_template(vn::VarName, template, prefix, prefix_template)
    prefix === nothing && return vn, template
    pt = prefix_template === nothing ? prefix : prefix_template
    return AbstractPPL.prefix(vn, prefix), _apply_prefix_template(pt, template)
end

function tilde_assume!!(
    model::Model,
    context::AbstractContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return tilde_assume!!(context, right, vn, template, vi)
end

function _check_tilde_value(value, vn, role::Union{Condition,Fix})
    remove = role isa Fix ? "unfix" : "decondition"
    absent = if _contains_missing(value)
        "missing"
    elseif _contains_nothing(value)
        "nothing"
    else
        nothing
    end
    absent === nothing || throw(
        ArgumentError(
            "LHS variable `$vn` contains `$absent`; make it latent with `$remove`."
        ),
    )
    return value
end

"""
    tilde_observe!!(prefix, prefix_template, right::Distribution, left, vn, template, vi)

Accumulate an observation and return `(left, vi)` with the updated varinfo.

`left` is supplied by the model's conditioned values or by a literal expression. `vn` is
the variable name before prefixing, or `nothing` for a literal. `template` describes the
top-level variable's storage; literals use `NoTemplate()`.

Apply `prefix` (a `VarName` or `nothing`) and its storage template `prefix_template`,
then delegate to [`accumulate_observe!!`](@ref). The compiler passes this metadata directly
so observations do not box the model. Every observation calls this function, independently
of the evaluation context. Fixed LHS variables bypass it and do not contribute to the log probability.
"""
function tilde_observe!!(
    prefix, prefix_template, right::Distribution, left, vn, template, vi
)
    vn, template = if vn === nothing
        vn, NoTemplate()
    else
        _prefix_varname_and_template(vn, template, prefix, prefix_template)
    end
    left = _check_tilde_value(left, vn, Condition())
    vi = accumulate_observe!!(vi, right, left, vn, template)
    return left, vi
end

"""
    store_coloneq_value!!(model::Model, vn::VarName, right, template, vi)

Store a tracked assignment's value in the raw-value accumulator and return the updated `vi`.

Apply the model's prefix to `vn` and its storage `template`. The evaluator calls this
function only when tracked-value extraction is enabled; no context hook is involved.
"""
function store_coloneq_value!!(
    model::Model, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return map_accumulator!!(
        acc -> store_colon_eq!!(acc, vn, right, template), vi, Val(RAW_VALUE_ACCNAME)
    )
end

"""
    (model::Model)([rng, varinfo])

Sample from the prior of the `model` with random number generator `rng`.

Returns the model's return value.

Note that calling this with an existing `varinfo` object will mutate it.
"""
(model::Model)() = model(Random.default_rng(), VarInfo())
function (model::Model)(varinfo::AbstractVarInfo)
    return model(Random.default_rng(), varinfo)
end
# ^ Weird Documenter.jl bug means that we have to write the two above separately
# as it can only detect the `function`-less syntax.
function (model::Model)(rng::Random.AbstractRNG, varinfo::AbstractVarInfo=VarInfo(()))
    return first(init!!(rng, model, varinfo, InitFromPrior(), UnlinkAll()))
end

"""
    init!!(
        [rng::Random.AbstractRNG,]
        model::Model,
        varinfo::AbstractVarInfo,
        init_strategy::AbstractInitStrategy,
        [transform_strategy::AbstractTransformStrategy=UnlinkAll(),]
    )

Evaluate `model` with the given initialisation and transform strategies, resetting and
filling the accumulators in `varinfo`. Parameter values are recorded only if a value
accumulator is present.

`transform_strategy` controls the output representation and defaults to `UnlinkAll()`.

Returns a tuple of the model's return value, plus the updated `varinfo` object.
"""
function init!!(
    rng::Random.AbstractRNG,
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    ctx = InitContext(rng, init_strategy, transform_strategy)
    model = DynamicPPL.setleafcontext(model, ctx)
    return DynamicPPL.evaluate_nowarn!!(model, vi)
end
function init!!(
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    return init!!(Random.default_rng(), model, vi, init_strategy, transform_strategy)
end

"""
    evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`, wrapping it in a `ThreadSafeVarInfo` if the
model is marked as needing threadsafe evaluation.

!!! warning
    The semantics of this method are complicated. We **strongly** recommend that users do
    *not* use this method unless absolutely necessary. In the future this method will be
    deprecated and removed. As far as possible (and it should **always** be possible --
    please open an issue if you do not know how to adapt your code!) you should use the
    five-argument `init!!([rng,] model, ::VarInfo, init_strategy,
    transform_strategy)` method, which has more explicit semantics and allows you to have
    more control over each part of the evaluation process.

The exact semantics depend on the `model`'s context. Fundamentally, this method executes the
model evaluation function (i.e., the function used to define the model) using the given
`varinfo` as an argument. At each tilde-statement, `tilde_assume!!` or `tilde_observe!!` is
called, whose behaviour depends on the model's context.

Broadly speaking, if the leaf context is an `InitContext`, then this function:

- uses the initialisation strategy inside the `InitContext`;
- uses the transform strategy inside the `InitContext`;
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- overwrites the values in `varinfo` with the new values obtained from the initialisation strategy.

If the leaf context is a `DefaultContext`, then this function:

- uses the values inside the `varinfo` as the initialisation strategy;
- derives a transform strategy from the `varinfo`'s stored variables (if a linked variable is
  stored, then the transform strategy will treat that variable as linked; likewise for
  unlinked)
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- records the values of executed LHS variables in the reset value accumulator, omitting LHS variables
  that are no longer executed.

The long-term plan for this method is to:

- Replace `DefaultContext` with `InitContext` by splitting up the functionality of `DefaultContext`
  into its constituent components
- Remove the `VarInfo` argument, and instead use only an `AccumulatorTuple`
- Separate the initialisation and transform strategies into separate arguments, instead of storing
  them inside the model's context.
"""
function AbstractPPL.evaluate!!(model::Model, varinfo::AbstractVarInfo)
    @warn (
        "Calling `evaluate!!(model, varinfo)` directly is not recommended and will be" *
        " deprecated in the future. Please switch to using `init!!([rng,] model," *
        " ::VarInfo, init_strategy, transform_strategy)` instead, which" *
        " has more explicit semantics and allows you to have more control over each" *
        " part of the evaluation process. Please see the DynamicPPL documentation" *
        " for more details: https://turinglang.org/DynamicPPL.jl/stable/evaluation"
    ) maxlog = 5
    return DynamicPPL.evaluate_nowarn!!(model, varinfo)
end

"""
    evaluate_nowarn!!(model::Model, varinfo)

This is the same as `evaluate!!(model, varinfo)` but without the deprecation warning.

!!! warning
    This is meant for internal use in DynamicPPL.jl only! If you rely on this method in your
    code, please note that it may break at any time.
"""
function evaluate_nowarn!!(model::Model, varinfo::AbstractVarInfo)
    if leafcontext(model.context) isa DefaultContext
        values, strategy = if hasacc(varinfo, Val(VECTORVAL_ACCNAME))
            copy(get_vector_values(varinfo)),
            infer_transform_strategy_from_values(get_vector_values(varinfo))
        else
            VarNamedTuple(), UnlinkAll()
        end
        ctx = InitContext(InitFromParams(values, nothing), strategy)
        model = setleafcontext(model, ctx)
    end
    return if requires_threadsafe(model)
        # Use of float_type_with_fallback(eltype(x)) is necessary to deal with cases where x is
        # a gradient type of some AD backend.
        param_eltype = DynamicPPL.get_param_eltype(varinfo, model.context)
        wrapper = ThreadSafeVarInfo(varinfo, param_eltype)
        result, wrapper_new = _evaluate!!(model, wrapper)
        # TODO(penelopeysm): If seems that if you pass a TSVI to this method, it
        # will return the underlying VI, which is a bit counterintuitive (because
        # calling TSVI(::TSVI) returns the original TSVI, instead of wrapping it
        # again).
        accs = map(getaccs(wrapper_new)) do acc
            if acc isa TSVNTAccumulator
                VNTAccumulator{accumulator_name(acc)}(acc.f, acc.values)
            else
                acc
            end
        end
        return result, setaccs!!(wrapper_new.varinfo, accs)
    else
        _evaluate!!(model, resetaccs!!(varinfo))
    end
end

"""
    _evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`.

This function does not wrap the varinfo in a `ThreadSafeVarInfo`. It also does not
reset the log probability of the `varinfo` before running.
"""
function _evaluate!!(model::Model, varinfo::AbstractVarInfo)
    args, kwargs = make_evaluate_args_and_kwargs(model, varinfo)
    return model.f(args...; kwargs...)
end

"""
    make_evaluate_args_and_kwargs(model, varinfo)

Return the arguments and keyword arguments to be passed to the evaluator of the model, i.e. `model.f`e.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames,defaultnames}, varinfo::AbstractVarInfo
) where {_F,argnames,defaultnames}
    unwrap_args = [
        if is_splat_symbol(var)
            :(
                $convert_model_argument(
                    $get_param_eltype(varinfo, model.context), model.args.$var
                )...
            )
        else
            :($convert_model_argument(
                $get_param_eltype(varinfo, model.context), model.args.$var
            ))
        end for var in argnames
    ]
    unwrap_kwargs = [
        is_splat_symbol(var) ? :(model.defaults.$var...) : :($var = model.defaults.$var) for
        var in defaultnames
    ]
    return quote
        args = (model, varinfo, $(unwrap_args...))
        kwargs = (; $(unwrap_kwargs...))
        return args, kwargs
    end
end

"""
    get_param_eltype(varinfo::AbstractVarInfo, context::AbstractContext)

Get the element type of the parameters being used to evaluate a model, using a `varinfo`
under the given `context`. For example, when evaluating a model with ForwardDiff AD, this
should return `ForwardDiff.Dual`.

For `InitContext`, query its initialisation strategy. For other leaf contexts, infer
this type from recorded vectorised values, or return `Union{}` if no value accumulator
is present. Parent contexts delegate to their child context.

See the docstring of `get_param_eltype(strategy::AbstractInitStrategy)` for the
strategy interface.
"""
function get_param_eltype(vi::AbstractVarInfo, ctx::AbstractParentContext)
    return get_param_eltype(vi, DynamicPPL.childcontext(ctx))
end
function get_param_eltype(vi::AbstractVarInfo, ::AbstractContext)
    hasacc(vi, Val(VECTORVAL_ACCNAME)) || return Union{}
    return get_param_eltype(InitFromParams(get_vector_values(vi), nothing))
end
function get_param_eltype(::AbstractVarInfo, ctx::InitContext)
    return get_param_eltype(ctx.strategy)
end

@generated function _argument_names(::NamedTuple{names}) where {names}
    return QuoteNode(map(unsplat_symbol, names))
end

"""
    getargnames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getargnames(model::Model{_F,argnames}) where {argnames,_F} = argnames

"""
    nameof(model::Model)

Get the name of the `model` as `Symbol`.
"""
Base.nameof(model::Model) = Symbol(model.f)
Base.nameof(model::Model{<:Function}) = nameof(model.f)

"""
    rand([rng=Random.default_rng()], model::Model)

Sample a `VarNamedTuple` of raw values from the prior of `model`.
"""
function Base.rand(rng::Random.AbstractRNG, model::Model)
    vi = VarInfo((RawValueAccumulator(false),))
    vi = last(init!!(rng, model, vi, InitFromPrior(), UnlinkAll()))
    return get_raw_values(vi)
end
Base.rand(model::Model) = rand(Random.default_rng(), model)

"""
    logjoint(model::Model, params)
    logjoint(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables `params` for the probabilistic `model`, or the
log joint of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logjoint` does not depend on whether `VarInfo` has been linked or not.

See also [`logprior`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       logjoint(demo([1.0]), (m = 100.0, ))
-9902.33787706641

julia> # Using a `OrderedDict`.
       logjoint(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-9902.33787706641

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0) + logpdf(Normal(), 100.0)
-9902.33787706641
```
"""
function logjoint(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getlogjoint(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logjoint(model::Model, varinfo::AbstractVarInfo)
    return logjoint(model, get_values(varinfo))
end

"""
    logprior(model::Model, params)
    logprior(model::Model, varinfo::AbstractVarInfo)

Return the log prior probability of variables `params` for the probabilistic `model`, or the
log prior of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logprior` does not depend on whether `VarInfo` has been linked or not.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       logprior(demo([1.0]), (m = 100.0, ))
-5000.918938533205

julia> # Using a `OrderedDict`.
       logprior(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-5000.918938533205

julia> # Truth.
       logpdf(Normal(), 100.0)
-5000.918938533205
```
"""
function logprior(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogPriorAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getlogprior(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logprior(model::Model, varinfo::AbstractVarInfo)
    return logprior(model, get_values(varinfo))
end

"""
    loglikelihood(model::Model, params)
    loglikelihood(model::Model, varinfo::AbstractVarInfo)

Return the log likelihood of variables `params` for the probabilistic `model`, or the log
likelihood of the data in `varinfo` if provided.

See also [`logjoint`](@ref) and [`logprior`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       loglikelihood(demo([1.0]), (m = 100.0, ))
-4901.418938533205

julia> # Using a `OrderedDict`.
       loglikelihood(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-4901.418938533205

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0)
-4901.418938533205
"""
function Distributions.loglikelihood(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogLikelihoodAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getloglikelihood(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
    return loglikelihood(model, get_values(varinfo))
end

# Implemented & documented in DynamicPPLMCMCChainsExt
function predict end

"""
    returned(model::Model, parameters...)

Initialise a `model` using the given `parameters` and return the model's return value. The
parameters must be provided in a format that can be wrapped in an `InitFromParams`, i.e.,
`InitFromParams(parameters..., nothing)` must be a valid `AbstractInitStrategy` (where
`nothing` is the fallback strategy to use if parameters are not provided).

As far as DynamicPPL is concerned, `parameters` can be either a singular `NamedTuple` or an
`AbstractDict{<:VarName}`; however this method is left flexible to allow for other packages
that wish to extend `InitFromParams`.

# Example
```jldoctest
julia> using DynamicPPL, Distributions

julia> @model function demo()
           m ~ Normal()
           return (mp1 = m + 1,)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> returned(model, (; m = 1.0))
(mp1 = 2.0,)

julia> returned(model, Dict{VarName,Float64}(@varname(m) => 2.0))
(mp1 = 3.0,)
```
"""
function returned(model::Model, parameters...)
    # Note: we can't use `fix(model, parameters)` because
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1097
    return first(
        init!!(
            model,
            DynamicPPL.VarInfo(DynamicPPL.AccumulatorTuple()),
            # Use `nothing` as the fallback to ensure that any missing parameters cause an
            # error
            InitFromParams(parameters..., nothing),
            UnlinkAll(),
        ),
    )
end

function AbstractPPL.evaluate!!(model::Model, context::AbstractContext, vi::AbstractVarInfo)
    return evaluate_nowarn!!(setleafcontext(model, context), vi)
end
