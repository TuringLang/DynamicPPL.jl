maybe_prefix(vn, ::Nothing) = vn
maybe_prefix(vn, prefix::VarName) = AbstractPPL.prefix(vn, prefix)

struct ArgumentCondition end

struct ModelValue{R<:Union{Condition,ArgumentCondition,Fix},T}
    value::T
    function ModelValue{R}(value::T) where {R<:Union{Condition,ArgumentCondition,Fix},T}
        (R === Condition || R === ArgumentCondition || R === Fix) ||
            throw(ArgumentError("A model value must have one concrete role"))
        return new{R,T}(value)
    end
end

struct NoModelBinding end

# Partial bindings retain the NamedTuple from which their fields came.
# S records whether that whole owner reaches child namespaces.
struct ModelValueTree{T,V<:VarNamedTuple,S}
    template::T
    values::V
    observed::Bool
    function ModelValueTree(
        template::T, values::V, ::Val{S}=Val(false), observed=false
    ) where {T,V<:VarNamedTuple,S}
        all(name -> hasproperty(template, name), keys(values.data)) ||
            throw(ArgumentError("Field bindings must belong to their template"))
        return new{T,V,S}(template, values, observed)
    end
end

# Copies and partial edits retain the whole owner's recursive scope.
function ModelValueTree(tree::ModelValueTree{T,V,S}, values::VarNamedTuple) where {T,V,S}
    return ModelValueTree(tree.template, values, Val(S), tree.observed)
end

_valid_model_binding(value) = value isa ModelValue
_model_value_like(::ModelValue{R}, x) where {R} = ModelValue{R}(x)
_inherits_binding(::ModelValue{R}) where {R} = R !== ArgumentCondition
_inherits_shape(::Any) = false
_inherits_shape(value::ModelValue) = _inherits_binding(value)
_inherits_shape(::ModelValueTree{T,V,S}) where {T,V,S} = S

# Encoded for `any`: `true` (a `missing` placeholder) beats `missing` (a
# `nothing` placeholder), which beats `false`.
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

_copy_model_node(value) = value
_copy_model_node(value::Union{VarNamedTuple,VarNamedTuples.PartialArray}) = copy(value)
function _copy_model_node(value::ModelValueTree)
    values = copy(value.values)
    return ModelValueTree(value, values)
end
function _merge_model_node(previous, updates, vn)
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
function _resize_model_binding(previous::VarNamedTuples.PartialArray, axes)
    data = similar(previous.data, eltype(previous), axes)
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
# Preserve surviving tuple entries and their roles when the replacement supplies array
# storage.
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
function _model_data(tree::ModelValueTree)
    names = propertynames(tree.template)
    return NamedTuple{names}(map(name -> _model_data(tree.values.data[name]), names))
end
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
    return all(values.mask) && all(_has_complete_model_data, values.data)
end
# Do not use `hasvalue`: it rejects partial containers needed for argument preparation.
_model_argument_binding(values, ::AbstractPPL.AbstractOptic) = nothing
_model_argument_binding(values, ::AbstractPPL.Iden) = values
function _model_argument_binding(
    values::VarNamedTuple, optic::AbstractPPL.Property{S}
) where {S}
    return if haskey(values.data, S)
        _model_argument_binding(values.data[S], optic.child)
    else
        nothing
    end
end
function _model_argument_binding(
    values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index
)
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    checkbounds(Bool, values, optic.ix...; optic.kw...) || return nothing
    selected = if VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
        subset = VarNamedTuples._subset_partialarray(values, optic.ix...; optic.kw...)
        if all(subset.mask) &&
            !isempty(subset.data) &&
            VarNamedTuples._can_get_arraylikeblock(subset.data)
            first(subset.data).block
        else
            subset
        end
    elseif haskey(values, optic.ix...; optic.kw...)
        getindex(values, optic.ix...; optic.kw...)
    else
        return nothing
    end
    return _model_argument_binding(selected, optic.child)
end
function _model_argument_binding(values::ModelValue, optic::AbstractPPL.AbstractOptic)
    return if VarNamedTuples._haskey_optic(values, optic)
        VarNamedTuples._getindex_optic(values, optic, @varname(_))
    else
        nothing
    end
end
_model_argument_binding(values::ModelValue, ::AbstractPPL.Iden) = values

function _select_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return _map_model_nodes(v -> _select_model_node(R, v), values)
end
@generated function _map_model_nodes(
    f::F, values::VarNamedTuple{names}, wrap::W=VarNamedTuple
) where {names,F,W}
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
    return :(wrap(merge((;), $(fields...))))
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
    if !isconcretetype(T)
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

# Keep whole supplied values intact, but expose incomplete bindings as ordinary partial
# values.
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

function _bound_shape_error(vn, role, message)
    label = role === Fix ? "Fixed" : "Observed"
    throw(ArgumentError("$label LHS variable `$vn` requires $message"))
end

struct ResolvedBindings{
    V<:VarNamedTuple,
    S<:VarNamedTuple,
    O<:VarNamedTuple,
    F<:VarNamedTuple,
    A<:VarNamedTuple,
    K<:AbstractVector{<:VarName},
}
    values::V
    storage::S
    observed_templates::O
    fixed_templates::F
    arguments::A
    roots::K
    function ResolvedBindings(
        values::V, storage::S, observed::O, fixed::F, arguments::A, roots::K
    ) where {
        V<:VarNamedTuple,
        S<:VarNamedTuple,
        O<:VarNamedTuple,
        F<:VarNamedTuple,
        A<:VarNamedTuple,
        K<:AbstractVector{<:VarName},
    }
        _check_resolved_nodes(values)
        return new{V,S,O,F,A,K}(values, storage, observed, fixed, arguments, roots)
    end
end
function ResolvedBindings(; arguments=VarNamedTuple(), roots=())
    return ResolvedBindings(
        VarNamedTuple(),
        deepcopy(arguments),
        VarNamedTuple(),
        VarNamedTuple(),
        arguments,
        vcat(VarName[roots...], collect(keys(arguments))),
    )
end
_check_resolved_nodes(::ModelValue) = nothing
_check_resolved_nodes(::NoModelBinding) = nothing
_check_resolved_nodes(tree::ModelValueTree) = _check_resolved_nodes(tree.values)
function _check_resolved_nodes(values::VarNamedTuple)
    foreach(_check_resolved_nodes, values.data)
    return nothing
end
function _check_resolved_nodes(values::VarNamedTuples.PartialArray)
    values.data isa VarNamedTuples.GrowableArray &&
        throw(ArgumentError("Binding indices require concrete storage"))
    for i in eachindex(values.mask)
        values.mask[i] && _check_resolved_nodes(values.data[i])
    end
    return nothing
end
function _check_resolved_nodes(value::VarNamedTuples.ArrayLikeBlock)
    return _check_resolved_nodes(value.block)
end
function _check_resolved_nodes(value)
    throw(ArgumentError("Invalid resolved binding node $(typeof(value))"))
end

function _binding_inputs(values::Tuple)
    return mapreduce(_binding_inputs, (a, b) -> (a..., b...), values; init=())
end
_binding_inputs(value) = (value,)
_binding_inputs((name, value)::Pair{Symbol}) = (VarName{name}() => value,)

_binding_storage(value::ModelValue) = value.value
_binding_storage(value::ModelValueTree) = value.template
_binding_storage(value::VarNamedTuples.PartialArray) = value.data
_binding_storage(value) = value
function _binding_child_template(template, optic::AbstractPPL.Index)
    return VarNamedTuples.maybe_index_template(template, optic)
end
function _binding_child_template(template, ::AbstractPPL.Property{S}) where {S}
    return VarNamedTuples.SharedGetProperty{S}()(template)
end
_binding_property_name(::AbstractPPL.Property{S}) where {S} = S

function _same_storage_shape(a, b, parents=())
    if a === missing ||
        b === missing ||
        a isa NoTemplate ||
        b isa NoTemplate ||
        a === nothing ||
        b === nothing
        return true
    end
    return _same_concrete_shape(a, b, parents)
end
_binding_fields(::Union{Number,AbstractString,Symbol,Type,Nothing,Missing}) = nothing
_binding_fields(value::NamedTuple) = keys(value)
function _binding_fields(value)
    T = typeof(value)
    return parentmodule(T) in (Base, Core) || fieldcount(T) == 0 ? nothing : fieldnames(T)
end
function _same_concrete_shape(a, b, parents)
    a isa VarNamedTuple && (a = a.data)
    b isa VarNamedTuple && (b = b.data)
    a isa Base.Pairs && (a = values(a))
    b isa Base.Pairs && (b = values(b))
    array = a isa Union{AbstractArray,Tuple}
    array == (b isa Union{AbstractArray,Tuple}) || return false
    if array
        axes(a) == axes(b) || return false
        eltype(a) <: Number && eltype(b) <: Number && return true
        names = eachindex(a)
    else
        names = _binding_fields(a)
        names == _binding_fields(b) || return false
        names === nothing && return true
    end
    i = findfirst(p -> first(p) === a, parents)
    j = findfirst(p -> last(p) === b, parents)
    (i !== nothing || j !== nothing) && return i == j
    return all(names) do name
        if array
            (!(a isa Tuple) && !isassigned(a, name)) ||
                (!(b isa Tuple) && !isassigned(b, name)) ||
                _same_storage_shape(a[name], b[name], (parents..., (a, b)))
        else
            !isdefined(a, name) ||
                !isdefined(b, name) ||
                _same_storage_shape(
                    getfield(a, name), getfield(b, name), (parents..., (a, b))
                )
        end
    end
end

function _check_cross_roles(previous, updates, vn=nothing)
    previous isa Union{Nothing,NoModelBinding} && return nothing
    updates isa Union{Nothing,NoModelBinding} && return nothing
    updates isa ModelValueTree && (updates = updates.values)
    if updates isa ModelValue
        if previous isa ModelValue
            typeof(_model_role(previous)) === typeof(_model_role(updates)) || throw(
                ArgumentError(
                    "Cannot bind `$vn`: a position belongs to the other role; remove it first",
                ),
            )
            return nothing
        end
        return _check_cross_roles(updates, previous, vn)
    end
    return _fold_model_indices(nothing, updates) do _, update, optic, _
        address = if vn === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(vn, optic)
        end
        _check_cross_roles(_model_argument_binding(previous, optic), update, address)
    end
end

function _resolved_selections(storage, ::AbstractPPL.Iden, vn)
    return (vn,)
end
function _resolved_selections(storage, optic::AbstractPPL.Property{S}, vn) where {S}
    storage = _binding_storage(storage)
    if _partial_binding_selector(storage, optic)
        indices = _binding_index_storage(storage)
        selected = VarNamedTuples._getindex_optic(indices, optic, vn)
        positions = selected isa Integer ? (selected,) : selected
        return map(positions) do i
            AbstractPPL.append_optic(
                vn, AbstractPPL.Index(Tuple(CartesianIndices(axes(storage))[i]), (;))
            )
        end
    end
    child = _binding_child_template(storage, AbstractPPL.Property{S}())
    return _resolved_selections(
        child, optic.child, AbstractPPL.append_optic(vn, AbstractPPL.Property{S}())
    )
end
function _resolved_selections(storage, optic::AbstractPPL.Index, vn)
    storage = _binding_storage(storage)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    selected = getindex(CartesianIndices(axes(storage)), optic.ix...; optic.kw...)
    child = optic.child
    while selected isa AbstractArray && child isa AbstractPPL.Index
        head = AbstractPPL.concretize_top_level(child, selected)
        selected = getindex(selected, head.ix...; head.kw...)
        child = child.child
    end
    positions = selected isa CartesianIndex ? (selected,) : selected
    return mapreduce(vcat, positions; init=VarName[]) do i
        name = AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(i), (;)))
        value = isassigned(storage, i) ? storage[i] : NoTemplate()
        collect(_resolved_selections(value, child, name))
    end
end
function _binding_index_storage(storage)
    data = copyto!(similar(storage, Int), LinearIndices(storage))
    return VarNamedTuples.PartialArray(data, fill!(similar(storage, Bool), true))
end
function _binding_ancestors(vn)
    ancestors = VarName[]
    current, optic = VarName{AbstractPPL.getsym(vn)}(), AbstractPPL.getoptic(vn)
    while !(optic isa AbstractPPL.Iden)
        push!(ancestors, current)
        current = AbstractPPL.append_optic(current, AbstractPPL.ohead(optic))
        optic = optic.child
    end
    return ancestors
end
function _check_resolved_overlap!((seen, ancestors), selections)
    for vn in selections
        parents = _binding_ancestors(vn)
        (vn in seen || vn in ancestors || any(parent -> parent in seen, parents)) && throw(
            ArgumentError("Binding address `$vn` overlaps another position in this call"),
        )
        push!(seen, vn)
        union!(ancestors, parents)
    end
    return nothing
end

function _resolved_entries(input::NamedTuple)
    return map(_resolved_entries(VarNamedTuple(input))) do (vn, value, supplied)
        root = input[AbstractPPL.getsym(vn)]
        storage = if root isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            supplied
        else
            NoTemplate()
        end
        (vn, value, storage)
    end
end
_resolved_entries(input::Pair{<:VarName}) = ((first(input), last(input), NoTemplate()),)
function _resolved_entries(input::VarNamedTuple)
    entries = Tuple[]
    _resolved_vnt_entries!(entries, input, nothing, NoTemplate())
    return entries
end
function _resolved_vnt_entries!(entries, input, prefix, storage)
    return _fold_model_indices(entries, input) do entries, value, optic, _
        vn = if prefix === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(prefix, optic)
        end
        supplied = prefix === nothing ? value : storage
        if value isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
            _resolved_vnt_entries!(entries, value, vn, supplied)
        else
            push!(entries, (vn, value, supplied))
        end
        entries
    end
end
function _resolved_entries(input)
    throw(
        ArgumentError(
            "Bindings require a NamedTuple, VarNamedTuple, or Symbol/VarName pairs; dictionaries are unsupported",
        ),
    )
end

function _expand_namespace_entries!(entries, vn, value, supplied, roots)
    if value isa NamedTuple &&
        !_known_binding_root(vn, roots) &&
        any(root -> subsumes(vn, root), roots)
        for name in keys(value)
            child = AbstractPPL.append_optic(vn, AbstractPPL.Property{name}())
            _expand_namespace_entries!(entries, child, value[name], supplied, roots)
        end
    else
        push!(entries, (vn, value, supplied))
    end
    return entries
end

function _resolved_root_storage(stored, schema, supplied, vn)
    name = AbstractPPL.getsym(vn)
    previous = get(stored.data, name, NoTemplate())
    supplied = supplied isa VarNamedTuples.PartialArray ? supplied.data : supplied
    candidate = get(schema, name, supplied)
    return candidate isa NoTemplate ? previous : _fill_binding_storage(candidate, previous)
end

_fill_binding_storage(candidate, previous) = candidate
_fill_binding_storage(::NoTemplate, previous) = previous
_fill_binding_storage(candidate::Number, previous::VarNamedTuple) = previous
function _fill_binding_storage(candidate::AbstractArray, previous::AbstractArray)
    result = candidate
    for i in CartesianIndices(candidate)
        checkbounds(Bool, previous, i) && isassigned(previous, i) || continue
        child = if isassigned(candidate, i)
            _fill_binding_storage(candidate[i], previous[i])
        else
            previous[i]
        end
        isassigned(candidate, i) && child === candidate[i] && continue
        result === candidate && (result = copy(candidate))
        result = _setindex!!(result, child, Tuple(i)...)
    end
    return result
end
function _fill_binding_storage(
    candidate::NamedTuple, previous::Union{NamedTuple,VarNamedTuple}
)
    previous = previous isa VarNamedTuple ? previous.data : previous
    return NamedTuple{keys(candidate)}(
        map(keys(candidate)) do name
            if haskey(previous, name)
                _fill_binding_storage(candidate[name], previous[name])
            else
                candidate[name]
            end
        end,
    )
end

_known_binding_root(vn, roots) = any(root -> subsumes(root, vn), roots)

function _check_mixed_binding(previous, value, ::AbstractPPL.Iden, vn, roots)
    value isa NamedTuple || return nothing
    _known_binding_root(vn, roots) && return nothing
    if value isa NamedTuple && previous isa Union{VarNamedTuple,ModelValueTree}
        throw(
            ArgumentError(
                "Mixing whole and field bindings at `$vn` requires root-structured storage"
            ),
        )
    elseif value isa NamedTuple && previous isa ModelValue{<:Any,<:NamedTuple}
        keys(value) == keys(previous.value) || throw(
            ArgumentError("Changing the fields of `$vn` requires root-structured storage"),
        )
    end
    return nothing
end
function _check_mixed_binding(previous, value, optic, vn, roots)
    if previous isa ModelValue{<:Any,<:NamedTuple} && !_known_binding_root(vn, roots)
        throw(
            ArgumentError(
                "Mixing whole and field bindings at `$vn` requires root-structured storage"
            ),
        )
    end
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(previous, head)
    child === nothing && return nothing
    return _check_mixed_binding(
        child, value, optic.child, AbstractPPL.append_optic(vn, head), roots
    )
end

_path_storage(storage, ::AbstractPPL.Iden, vn, roots) = storage
function _path_storage(storage, optic::AbstractPPL.Property{S}, vn, roots) where {S}
    storage = _binding_storage(storage)
    if _partial_binding_selector(storage, optic)
        _partial_binding_array(storage) || _unsupported_partial_binding(storage, vn, vn)
        return storage
    end
    if storage isa NoTemplate
        storage = VarNamedTuple()
    elseif !(storage isa Union{NamedTuple,VarNamedTuple})
        _known_binding_root(vn, roots) && _unsupported_partial_binding(storage, vn, vn)
        storage isa Number || _unsupported_partial_binding(storage, vn, vn)
        storage = VarNamedTuple()
    end
    child = _binding_child_template(storage, AbstractPPL.Property{S}())
    child isa NoTemplate &&
        storage isa NamedTuple &&
        throw(ArgumentError("No field `$S` in storage for `$vn`"))
    next = AbstractPPL.append_optic(vn, AbstractPPL.Property{S}())
    prepared = _path_storage(child, optic.child, next, roots)
    prepared === child && return storage
    return if storage isa VarNamedTuple
        VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((prepared,))))
    else
        merge(storage, NamedTuple{(S,)}((prepared,)))
    end
end
function _path_storage(storage, optic::AbstractPPL.Index, vn, roots)
    storage = _binding_storage(storage)
    storage isa AbstractArray && !(storage isa VarNamedTuples.GrowableArray) || throw(
        ArgumentError(
            "Cannot bind `$vn`: indexed edits require concrete storage at binding time"
        ),
    )
    _partial_binding_array(storage) || _unsupported_partial_binding(storage, vn, vn)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    checkbounds(Bool, storage, optic.ix...; optic.kw...) || throw(
        ArgumentError(
            "Cannot bind `$vn`: indices $(optic.ix) are outside storage axes $(axes(storage))",
        ),
    )
    optic.child isa AbstractPPL.Iden && return storage
    child = if VarNamedTuples._is_multiindex(storage, optic.ix...; optic.kw...)
        getindex(storage, optic.ix...; optic.kw...)
    else
        _binding_child_template(storage, optic)
    end
    next = AbstractPPL.append_optic(vn, AbstractPPL.ohead(optic))
    prepared = _path_storage(child, optic.child, next, roots)
    prepared === child && return storage
    return _setindex!!(copy(storage), prepared, optic.ix...; optic.kw...)
end

function _retain_binding_storage(storage, value, ::AbstractPPL.Iden, vn)
    storage isa VarNamedTuple &&
        throw(ArgumentError("Cannot create binding root `$vn` across existing field roots"))
    _same_storage_shape(storage, value) || throw(
        ArgumentError("Binding shape at `$vn` cannot change; clear the whole root first"),
    )
    return value
end
function _retain_binding_storage(
    storage, value, optic::AbstractPPL.Property{S}, vn
) where {S}
    if _partial_binding_selector(storage, optic)
        indices = VarNamedTuples._getindex_optic(_binding_index_storage(storage), optic, vn)
        return _retain_binding_storage(
            storage, value, AbstractPPL.Index((indices,), (;)), vn
        )
    end
    child = _binding_child_template(storage, AbstractPPL.Property{S}())
    child = _retain_binding_storage(
        child, value, optic.child, AbstractPPL.append_optic(vn, AbstractPPL.Property{S}())
    )
    return if storage isa VarNamedTuple
        VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((child,))))
    else
        merge(storage, NamedTuple{(S,)}((child,)))
    end
end
function _retain_binding_storage(storage, value, optic::AbstractPPL.Index, vn)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    if optic.child isa AbstractPPL.Iden &&
        VarNamedTuples._is_multiindex(storage, optic.ix...; optic.kw...)
        selected = getindex(CartesianIndices(axes(storage)), optic.ix...; optic.kw...)
        for (index, replacement) in zip(selected, value)
            isassigned(storage, index) || continue
            _same_storage_shape(storage[index], replacement) ||
                throw(ArgumentError("Binding shape at `$vn` cannot change"))
        end
        return _setindex!!(storage, value, optic.ix...; optic.kw...)
    end
    child = _binding_child_template(storage, optic)
    child = _retain_binding_storage(
        child, value, optic.child, AbstractPPL.append_optic(vn, AbstractPPL.ohead(optic))
    )
    return _setindex!!(storage, child, optic.ix...; optic.kw...)
end

"""
    _bind_resolved(table, role, inputs...; roots=(), kwargs...)

Construct a resolved role table without evaluating a model.
Place a positional `@of` template last, after all positional value inputs;
otherwise throw `ArgumentError`. Keyword bindings may accompany a final template.
"""
function _bind_resolved(
    table::ResolvedBindings, ::Type{R}, inputs...; roots=(), kwargs...
) where {R<:Union{Condition,Fix,ArgumentCondition}}
    inputs = _binding_inputs(inputs)
    position = findfirst(x -> x isa Type{<:AbstractPPL.OfNamedTuple}, inputs)
    position === nothing ||
        position == length(inputs) ||
        throw(
            ArgumentError(
                "A positional binding template must be the last argument to condition/fix"
            ),
        )
    schema = position === nothing ? (;) : zero(last(inputs))
    inputs = position === nothing ? inputs : Base.front(inputs)
    inputs = isempty(kwargs) ? inputs : (inputs..., (; kwargs...))
    schema = Base.structdiff(schema, NamedTuple{keys(table.arguments.data)})
    templates = R === Fix ? table.fixed_templates : table.observed_templates
    current_schema = merge(templates.data, schema)
    templates = VarNamedTuple(current_schema)
    batches = map(_resolved_entries, inputs)
    stored, values = deepcopy(table.storage), table.values
    updates = VarNamedTuple()
    prepared = Dict{Symbol,Any}()
    occupied = Dict{Symbol,Bool}()
    seen = (Set{VarName}(), Set{VarName}())
    known = union(table.roots, roots)
    for input in inputs
        input isa VarNamedTuple || continue
        append!(known, keys(input))
    end
    for input in inputs
        fields = if input isa VarNamedTuple
            input.data
        elseif input isa NamedTuple
            input
        else
            (;)
        end
        for (name, value) in pairs(fields)
            value isa VarNamedTuples.PartialArray && isempty(value) || continue
            storage = _resolved_root_storage(stored, current_schema, value, VarName{name}())
            storage isa VarNamedTuples.GrowableArray &&
                throw(ArgumentError("Binding indices require concrete storage"))
            stored = setindex!!(stored, deepcopy(storage), VarName{name}())
        end
    end
    unique!(known)
    entries = Tuple[]
    for batch in batches, (vn, value, supplied) in batch
        _expand_namespace_entries!(entries, vn, value, supplied, known)
    end
    for (vn, value, supplied) in entries
        _classify_placeholder(value) === _NoPlaceholder ||
            throw(ArgumentError("Explicit binding `$vn` contains missing or nothing"))
        name, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
        root = VarName{name}()
        previous = get(values.data, name, nothing)
        _check_mixed_binding(previous, value, optic, root, known)
        _check_mixed_binding(get(updates.data, name, nothing), value, optic, root, known)
        storage = get!(prepared, name) do
            deepcopy(_resolved_root_storage(stored, current_schema, supplied, vn))
        end
        active = get!(occupied, name) do
            _contains_model_binding(previous)
        end
        if active && haskey(stored.data, name)
            _same_storage_shape(stored.data[name], storage) || throw(
                ArgumentError(
                    "Binding storage at `$root` cannot change shape while bindings remain",
                ),
            )
        end
        storage = _path_storage(storage, optic, root, known)
        _check_resolved_overlap!(seen, _resolved_selections(storage, optic, root))
        _check_binding_extents(value, storage, optic, vn)
        value = _convert_resolved_value(
            value, storage, optic, vn, haskey(current_schema, name)
        )
        retained = get(stored.data, name, NoTemplate())
        if !active &&
            !(retained isa VarNamedTuple) &&
            (
                haskey(schema, name) ||
                optic isa AbstractPPL.Iden &&
                !haskey(current_schema, name) &&
                !haskey(
                    (R === Fix ? table.observed_templates : table.fixed_templates).data,
                    name,
                )
            )
            retained = if optic isa AbstractPPL.Iden && !haskey(current_schema, name)
                NoTemplate()
            else
                storage
            end
        else
            retained = _path_storage(
                retained isa NoTemplate ? storage : retained, optic, root, known
            )
        end
        retained = _retain_binding_storage(retained, value, optic, root)
        updates = templated_setindex!!(updates, ModelValue{R}(value), vn, storage)
        prepared[name] = storage
        stored = setindex!!(stored, retained, root)
    end
    _check_cross_roles(values, updates)
    values = _merge_model_values(values, updates)
    return ResolvedBindings(
        values,
        stored,
        R === Fix ? table.observed_templates : templates,
        R === Fix ? templates : table.fixed_templates,
        table.arguments,
        known,
    )
end

function _convert_resolved_value(value, storage, ::AbstractPPL.Iden, vn, templated)
    if templated && !(storage isa NoTemplate)
        value isa typeof(storage) || throw(
            ArgumentError(
                "Bound value at `$vn` must have template type $(typeof(storage))"
            ),
        )
        _same_storage_shape(value, storage) ||
            throw(ArgumentError("Bound value at `$vn` has incompatible template shape"))
    end
    return value
end
function _convert_resolved_value(value, storage, optic, vn, templated)
    storage isa NoTemplate && return value
    head = AbstractPPL.ohead(optic)
    head isa AbstractPPL.Index && (head = AbstractPPL.concretize_top_level(head, storage))
    child = _binding_child_template(storage, head)
    if optic.child isa AbstractPPL.Iden &&
        !(storage isa VarNamedTuple) &&
        !(child isa NoTemplate)
        return _convert_partial_argument_binding(
            ModelValue{Condition}(value), storage, head, vn
        ).value
    end
    return _convert_resolved_value(value, child, optic.child, vn, templated)
end

function _check_field_removal(value, ::AbstractPPL.Iden, vn, roots)
    return nothing
end
function _check_field_removal(value, optic, vn, roots)
    if value isa ModelValue{<:Any,<:NamedTuple} && !any(root -> subsumes(root, vn), roots)
        throw(
            ArgumentError("Removing a field below `$vn` requires root-structured storage")
        )
    end
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(value, head)
    child === nothing && return nothing
    next = if vn === nothing
        AbstractPPL.optic_to_varname(head)
    else
        AbstractPPL.append_optic(vn, head)
    end
    return _check_field_removal(child, optic.child, next, roots)
end

function _contains_model_binding(value::Union{VarNamedTuple,VarNamedTuples.PartialArray})
    return _fold_model_indices(false, value) do found, child, _, _
        found || _contains_model_binding(child)
    end
end
_contains_model_binding(value::ModelValueTree) = _contains_model_binding(value.values)
_contains_model_binding(::ModelValue) = true
_contains_model_binding(::Union{Nothing,NoModelBinding}) = false

function _unbind_resolved(
    table::ResolvedBindings, ::Type{R}, names...
) where {R<:Union{Condition,Fix}}
    names = map(name -> name isa Symbol ? VarName{name}() : name, names)
    all(name -> name isa VarName, names) ||
        throw(ArgumentError("Removal addresses must be Symbols or VarNames"))
    if isempty(names)
        return ResolvedBindings(
            _remove_model_values(R, table.values),
            table.storage,
            table.observed_templates,
            table.fixed_templates,
            table.arguments,
            table.roots,
        )
    end
    values, storage = table.values, table.storage
    for vn in names
        name, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
        root_storage = get(storage.data, name, NoTemplate())
        _path_storage(root_storage, optic, VarName{name}(), table.roots)
        _check_field_removal(values, AbstractPPL.varname_to_optic(vn), nothing, table.roots)
        values = _prune_model_bindings(
            _remove_model_binding(R, values, AbstractPPL.varname_to_optic(vn))
        )
        if optic isa AbstractPPL.Iden &&
            !_contains_model_binding(get(values.data, name, nothing)) &&
            !haskey(table.observed_templates.data, name) &&
            !haskey(table.fixed_templates.data, name)
            storage = VarNamedTuple(Base.structdiff(storage.data, NamedTuple{(name,)}))
            if haskey(table.arguments.data, name)
                storage = setindex!!(storage, table.arguments[vn], vn)
            end
        end
    end
    return ResolvedBindings(
        values,
        storage,
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        table.roots,
    )
end

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
    converted = if multi
        map(v -> convert(T, v), value)
    else
        convert(T, value)
    end
    isequal(converted, value) || throw(
        ArgumentError(
            "Cannot exactly represent partial binding at `$vn` in storage element type $T",
        ),
    )
    return _model_value_like(binding, converted)
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
        # Slice assignment ignores singleton dimensions and permits a flattened trailing
        # extent.
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
