struct Condition end
struct Fix end

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
    return isempty(selected.data) ? NoModelBinding() : selected
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
    return if found || isempty(values) && !(values.mask isa ModelBindingArray)
        VarNamedTuples.PartialArray(data, mask)
    else
        NoModelBinding()
    end
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
    if VarNamedTuples._haskey_optic(tree, AbstractPPL.Iden())
        names = propertynames(tree.template)
        return NamedTuple{names}(
            map(name -> _plain_model_values(tree.values.data[name]), names)
        )
    end
    return _plain_model_values(tree.values)
end
function _plain_model_values(values::VarNamedTuples.PartialArray)
    VarNamedTuples._haskey_optic(values, AbstractPPL.Iden()) &&
        return VarNamedTuples.unwrap_internal_array(values)
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
    ::Type{R}, tree::ModelValueTree, optic::AbstractPPL.Iden
) where {R}
    values = _prune_model_bindings(_remove_model_binding(R, tree.values, optic))
    return isempty(values.data) ? NoModelBinding() : ModelValueTree(tree, values)
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
    K<:Union{Tuple{Vararg{VarName}},AbstractVector{<:VarName}},
}
    values::V
    storage::S
    observed_templates::O
    fixed_templates::F
    arguments::A
    roots::K
    @inline function ResolvedBindings(
        values::V, storage::S, observed::O, fixed::F, arguments::A, roots::K
    ) where {
        V<:VarNamedTuple,
        S<:VarNamedTuple,
        O<:VarNamedTuple,
        F<:VarNamedTuple,
        A<:VarNamedTuple,
        K<:Union{Tuple{Vararg{VarName}},AbstractVector{<:VarName}},
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

_binding_state(value, argument) = (typeof(value), value)
_binding_state(value::ModelValue{R}, argument) where {R} = (R, value.value)
_binding_state(::NoModelBinding, argument) = nothing
function _binding_state(tree::ModelValueTree, argument)
    return (
        ModelValueTree,
        tree.template,
        _inherits_shape(tree),
        tree.observed,
        _binding_state(tree.values, tree.template),
    )
end
function _binding_state(values::VarNamedTuples.PartialArray, argument)
    if !_inherits_shape(values.mask) &&
        argument isa AbstractArray &&
        axes(values) == axes(argument) &&
        !any(values.mask)
        return nothing
    end
    owner = if values.mask isa ModelBindingArray
        template = values.mask.template
        storage =
            argument isa AbstractArray ? (typeof(template), axes(template)) : template
        (storage, values.mask.observed)
    else
        nothing
    end
    child_argument = if argument isa AbstractArray && axes(values) == axes(argument)
        argument
    else
        NoTemplate()
    end
    entries = map(CartesianIndices(values.mask)) do i
        values.mask[i] || return nothing
        value = values.data[i]
        child = if value isa VarNamedTuples.ArrayLikeBlock
            child_argument
        elseif child_argument isa AbstractArray &&
            checkbounds(Bool, child_argument, i) &&
            isassigned(child_argument, i)
            child_argument[i]
        else
            NoTemplate()
        end
        _binding_state(value, child)
    end
    return (VarNamedTuples.PartialArray, axes(values), owner, entries)
end
function _binding_state(value::VarNamedTuples.ArrayLikeBlock, argument)
    child =
        if argument isa AbstractArray &&
            checkbounds(Bool, argument, value.ix...; value.kw...)
            Base.maybeview(argument, value.ix...; value.kw...)
        else
            NoTemplate()
        end
    return (
        VarNamedTuples.ArrayLikeBlock,
        value.ix,
        value.kw,
        _binding_state(value.block, child),
    )
end
@generated function _binding_state(values::VarNamedTuple{names}, argument) where {names}
    ordered = Tuple(sort(collect(names)))
    return :(_binding_field_state(values.data, argument, Val{$ordered}()))
end
_binding_field_state(values, argument, ::Val{()}) = (;)
function _binding_field_state(values, argument, ::Val{names}) where {names}
    name = first(names)
    original = argument isa VarNamedTuple ? argument.data : argument
    child = original isa NamedTuple ? get(original, name, NoTemplate()) : NoTemplate()
    state = _binding_state(values[name], child)
    rest = _binding_field_state(values, argument, Val(Base.tail(names)))
    return state === nothing ? rest : merge(NamedTuple{(name,)}((state,)), rest)
end
function _binding_state(table::ResolvedBindings)
    # Binding nodes retain evaluation geometry; storage also holds editing-only shapes.
    return (
        _binding_state(table.values, table.arguments),
        _binding_state(table.observed_templates, NoTemplate()),
        _binding_state(table.fixed_templates, NoTemplate()),
    )
end
function Base.:(==)(a::ResolvedBindings, b::ResolvedBindings)
    return isequal(a, b)
end
function Base.isequal(a::ResolvedBindings, b::ResolvedBindings)
    return isequal(_binding_state(a), _binding_state(b))
end
function Base.hash(table::ResolvedBindings, h::UInt)
    return hash(_binding_state(table), hash(:ResolvedBindings, h))
end

function _missing_binding_storage(address=nothing; operation="bind")
    target = address === nothing ? "" : " `$address`"
    throw(
        ArgumentError(
            "Cannot $operation$target without storage: indexed edits require concrete storage at binding time; supply an @of template, a whole value, or a VarNamedTuple with storage",
        ),
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
    values.data isa VarNamedTuples.GrowableArray && _missing_binding_storage()
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
_binding_inputs(::NamedTuple{()}) = ()
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
    left = if a isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
        a.data
    elseif a isa Base.Pairs
        values(a)
    else
        a
    end
    right = if b isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
        b.data
    elseif b isa Base.Pairs
        values(b)
    else
        b
    end
    array = left isa Union{AbstractArray,Tuple}
    array == (right isa Union{AbstractArray,Tuple}) || return false
    if array
        axes(left) == axes(right) || return false
        eltype(left) <: Number && eltype(right) <: Number && return true
        names = eachindex(left)
    else
        names = _binding_fields(left)
        names == _binding_fields(right) || return false
        names === nothing && return true
    end
    i = findfirst(p -> first(p) === left, parents)
    j = findfirst(p -> last(p) === right, parents)
    (i !== nothing || j !== nothing) && return i == j
    return all(names) do name
        if array
            (!(left isa Tuple) && !isassigned(left, name)) ||
                (!(right isa Tuple) && !isassigned(right, name)) ||
                _same_storage_shape(left[name], right[name], (parents..., (left, right)))
        else
            !isdefined(left, name) ||
                !isdefined(right, name) ||
                _same_storage_shape(
                    getfield(left, name), getfield(right, name), (parents..., (left, right))
                )
        end
    end
end

function _check_cross_roles(previous, updates, vn=nothing; disjoint=false)
    previous isa Union{Nothing,NoModelBinding} && return nothing
    updates isa Union{Nothing,NoModelBinding} && return nothing
    updates isa ModelValueTree && (updates = updates.values)
    if updates isa ModelValue
        if previous isa ModelValue
            same_role = typeof(_model_role(previous)) === typeof(_model_role(updates))
            if disjoint || !same_role
                origin = same_role ? "another model" : "the other role"
                throw(
                    ArgumentError(
                        "Cannot bind `$vn`: a position is already bound by $origin; remove it with `decondition`/`unfix` first.",
                    ),
                )
            end
            return nothing
        end
        return _check_cross_roles(updates, previous, vn; disjoint)
    end
    return _fold_model_indices(nothing, updates) do _, update, optic, _
        address = if vn === nothing
            AbstractPPL.optic_to_varname(optic)
        else
            AbstractPPL.append_optic(vn, optic)
        end
        _check_cross_roles(
            _model_argument_binding(previous, optic), update, address; disjoint
        )
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
            ArgumentError(
                "Binding address `$vn` overlaps another position in this call; inputs must be disjoint",
            ),
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
            "Bindings require a NamedTuple, VarNamedTuple, or Symbol/VarName pairs; AbstractDict inputs are not supported",
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

function _check_existing_binding_template(binding::ModelValue, template, vn)
    _convert_resolved_value(binding.value, template, AbstractPPL.Iden(), vn, true)
    return nothing
end
_check_existing_binding_template(::Union{Nothing,NoModelBinding}, template, vn) = nothing
function _check_existing_binding_template(bindings, template, vn)
    bindings isa ModelValueTree && (bindings = bindings.values)
    _fold_model_indices(nothing, bindings) do _, binding, optic, _
        storage = _storage_at(template, optic)
        storage === nothing && return nothing
        _check_existing_binding_template(
            binding, storage, AbstractPPL.append_optic(vn, optic)
        )
    end
    return nothing
end

function _resolved_root_storage(stored, schema, supplied, vn)
    name = AbstractPPL.getsym(vn)
    previous = get(stored.data, name, NoTemplate())
    supplied = supplied isa VarNamedTuples.PartialArray ? supplied.data : supplied
    supplied isa VarNamedTuples.GrowableArray && (supplied = previous)
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
        result = _set_binding_storage(result, child, Tuple(i)...)
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

function _path_storage(storage, ::AbstractPPL.Iden, vn, roots; operation="bind", address=vn)
    return storage
end
function _path_storage(
    storage,
    optic::AbstractPPL.Property{S},
    vn,
    roots;
    operation="bind",
    address=AbstractPPL.append_optic(vn, optic),
) where {S}
    storage = _binding_storage(storage)
    if _partial_binding_selector(storage, optic)
        _partial_binding_array(storage) ||
            _unsupported_partial_binding(storage, address, vn; operation)
        return storage
    end
    if storage isa NoTemplate
        storage = VarNamedTuple()
    elseif !(storage isa Union{NamedTuple,VarNamedTuple})
        _known_binding_root(vn, roots) &&
            _unsupported_partial_binding(storage, address, vn; operation)
        storage isa Number || _unsupported_partial_binding(storage, address, vn; operation)
        storage = VarNamedTuple()
    end
    child = _binding_child_template(storage, AbstractPPL.Property{S}())
    child isa NoTemplate &&
        storage isa NamedTuple &&
        throw(
            ArgumentError(
                "Cannot $operation `$address`: nonexistent field `$S` in storage at `$vn`"
            ),
        )
    next = AbstractPPL.append_optic(vn, AbstractPPL.Property{S}())
    prepared = _path_storage(child, optic.child, next, roots; operation, address)
    prepared === child && return storage
    return if storage isa VarNamedTuple
        VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((prepared,))))
    else
        merge(storage, NamedTuple{(S,)}((prepared,)))
    end
end
function _path_storage(
    storage,
    optic::AbstractPPL.Index,
    vn,
    roots;
    operation="bind",
    address=AbstractPPL.append_optic(vn, optic),
)
    storage = _binding_storage(storage)
    storage isa Tuple && _unsupported_partial_binding(storage, address, vn; operation)
    storage isa AbstractArray && !(storage isa VarNamedTuples.GrowableArray) ||
        _missing_binding_storage(address; operation)
    _partial_binding_array(storage) ||
        _unsupported_partial_binding(storage, address, vn; operation)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    checkbounds(Bool, storage, optic.ix...; optic.kw...) || throw(
        ArgumentError(
            "Cannot $operation `$address`: index is outside the storage at `$vn`"
        ),
    )
    optic.child isa AbstractPPL.Iden && return storage
    child = if VarNamedTuples._is_multiindex(storage, optic.ix...; optic.kw...)
        getindex(storage, optic.ix...; optic.kw...)
    else
        _binding_child_template(storage, optic)
    end
    next = AbstractPPL.append_optic(vn, AbstractPPL.ohead(optic))
    prepared = _path_storage(child, optic.child, next, roots; operation, address)
    prepared === child && return storage
    return _set_binding_storage(copy(storage), prepared, optic.ix...; optic.kw...)
end

function _set_binding_storage(storage, value, indices...; kwargs...)
    return _setindex!!(storage, value, indices...; kwargs...)
end
function _set_binding_storage(
    storage::AbstractArray{<:Union{Number,VarNamedTuple{()}}},
    value::VarNamedTuple,
    indices...;
    kwargs...,
)
    # Numeric namespace templates provide axes; their entries are not variable storage.
    return _setindex!!(similar(storage, typeof(value)), value, indices...; kwargs...)
end

function _check_whole_binding_storage(storage, vn)
    if storage isa VarNamedTuple && !isempty(storage.data)
        child = AbstractPPL.prefix(first(keys(storage)), vn)
        throw(
            ArgumentError(
                "Cannot create binding root `$vn` across existing root `$child`; rebuild the model to change its roots",
            ),
        )
    elseif storage isa AbstractArray{<:VarNamedTuple}
        throw(
            ArgumentError(
                "Cannot create binding root `$vn`: an existing binding root already occupies this address; rebuild the model to change its roots",
            ),
        )
    end
    return nothing
end

function _retain_binding_storage(
    storage, value, ::AbstractPPL.Iden, vn, binding, independent; copied=nothing
)
    _check_whole_binding_storage(storage, vn)
    (independent && !_has_binding_storage(binding)) ||
        _same_storage_shape(storage, value) ||
        throw(
            ArgumentError(
                "Binding shape at `$vn` cannot change; use decondition and unfix on the whole root first",
            ),
        )
    return value
end
function _retain_binding_storage(
    storage, value, optic::AbstractPPL.Property{S}, vn, binding, independent; copied=nothing
) where {S}
    if _partial_binding_selector(storage, optic)
        indices = VarNamedTuples._getindex_optic(_binding_index_storage(storage), optic, vn)
        return _retain_binding_storage(
            storage,
            value,
            AbstractPPL.Index((indices,), (;)),
            vn,
            binding,
            independent;
            copied,
        )
    end
    child = _binding_child_template(storage, AbstractPPL.Property{S}())
    child = _retain_binding_storage(
        child,
        value,
        optic.child,
        AbstractPPL.append_optic(vn, AbstractPPL.Property{S}()),
        _model_argument_binding(binding, AbstractPPL.Property{S}()),
        storage isa VarNamedTuple;
        copied,
    )
    return if storage isa VarNamedTuple
        VarNamedTuple(merge(storage.data, NamedTuple{(S,)}((child,))))
    else
        merge(storage, NamedTuple{(S,)}((child,)))
    end
end
function _retain_binding_storage(
    storage, value, optic::AbstractPPL.Index, vn, binding, independent; copied=nothing
)
    optic = AbstractPPL.concretize_top_level(optic, storage)
    if copied === nothing || !haskey(copied, storage)
        storage = copy(storage)
        copied === nothing || (copied[storage] = nothing)
    end
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
        child,
        value,
        optic.child,
        AbstractPPL.append_optic(vn, AbstractPPL.ohead(optic)),
        _model_argument_binding(binding, AbstractPPL.ohead(optic)),
        false;
        copied,
    )
    return _set_binding_storage(storage, child, optic.ix...; optic.kw...)
end

function _bind_structured_resolved(table, ::Type{R}, input::VarNamedTuple) where {R}
    supplied = _resolved_input_storage(input)
    known = table.roots
    for name in keys(input.data)
        _known_binding_root(VarName{name}(), known) && continue
        known = union(known, keys(VarNamedTuple(NamedTuple{(name,)}((input.data[name],)))))
    end
    updates = if isempty(table.values.data)
        _check_bound_placeholders(R, input)
        _tag_model_values(R, input)
    else
        map_pairs!!(copy(input)) do pair
            vn, value = pair
            _check_bound_placeholder(R, vn, value)
            name, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
            previous = get(table.values.data, name, nothing)
            _check_mixed_binding(previous, value, optic, VarName{name}(), known)
            converted = if optic isa AbstractPPL.Iden
                value
            else
                _convert_binding_edit(value, previous, optic, vn, R)
            end
            ModelValue{R}(converted)
        end
    end
    _check_cross_roles(table.values, updates)
    stored = NamedTuple{keys(supplied.data)}(
        map(keys(supplied.data)) do name
            storage = supplied.data[name]
            previous = get(table.storage.data, name, NoTemplate())
            if _has_binding_storage(get(table.values.data, name, nothing))
                _same_storage_shape(previous, storage) || throw(
                    ArgumentError(
                        "Binding shape at `$(VarName{name}())` cannot change; use decondition and unfix on the whole root first",
                    ),
                )
            end
            _fill_binding_storage(storage, previous)
        end,
    )
    return ResolvedBindings(
        isempty(table.values.data) ? updates : _merge_model_values(table.values, updates),
        VarNamedTuple(merge(table.storage.data, stored)),
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        known,
    )
end
_resolved_input_storage(value) = value
function _resolved_input_storage(value::VarNamedTuple)
    return VarNamedTuple(map(_resolved_input_storage, value.data))
end
function _resolved_input_storage(value::VarNamedTuples.PartialArray)
    value.data isa VarNamedTuples.GrowableArray && _missing_binding_storage()
    return value.data
end

function _check_bound_placeholders(::Type{R}, values::VarNamedTuple) where {R}
    _classify_placeholder(values) === _NoPlaceholder && return nothing
    for (vn, value) in pairs(values)
        _check_bound_placeholder(R, vn, value)
    end
    return nothing
end
function _check_bound_placeholder(::Type{R}, vn, value) where {R}
    _classify_placeholder(value) === _NoPlaceholder && return nothing
    remove = R === Fix ? "unfix" : "decondition"
    throw(
        ArgumentError(
            "Cannot bind `$vn` to a value containing `missing` or `nothing`; bindings cannot hold placeholders. To make it latent, leave it unbound or use `$remove(model, @varname($vn))`.",
        ),
    )
end

_whole_binding_input(input) = false
_whole_binding_value(value) = !(value isa Union{VarNamedTuple,VarNamedTuples.PartialArray})
function _whole_binding_input(input::Pair{<:VarName{<:Any,<:AbstractPPL.Iden}})
    return _whole_binding_value(last(input))
end
_whole_binding_input(input::NamedTuple) = all(_whole_binding_value, values(input))
_whole_binding_pairs(::Tuple{}) = ()
function _whole_binding_pairs(inputs::Tuple)
    entries = _whole_binding_pairs(first(inputs))
    return (entries..., _whole_binding_pairs(Base.tail(inputs))...)
end
_whole_binding_pairs(input::Pair) = (input,)
@generated function _whole_binding_pairs(input::NamedTuple{names}) where {names}
    pairs = [:($(VarName{name}()) => input.$name) for name in names]
    return Expr(:tuple, pairs...)
end
function _bind_whole_resolved(table, ::Type{R}, entries) where {R}
    names = map(pair -> AbstractPPL.getsym(first(pair)), entries)
    allunique(names) ||
        throw(ArgumentError("Binding addresses `$names` overlap in this call"))
    updates, storage = _whole_binding_updates(table, R, entries)
    _check_cross_roles(table.values, updates)
    return ResolvedBindings(
        _merge_model_values(table.values, updates),
        VarNamedTuple(merge(table.storage.data, storage.data)),
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        table.roots,
    )
end
function _whole_binding_updates(table, ::Type{R}, ::Tuple{}) where {R}
    return (VarNamedTuple(), VarNamedTuple())
end
function _whole_binding_updates(table, ::Type{R}, entries::Tuple) where {R}
    vn, value = first(entries)
    name = AbstractPPL.getsym(vn)
    _check_bound_placeholder(R, vn, value)
    previous = get(table.values.data, name, nothing)
    _check_mixed_binding(previous, value, AbstractPPL.Iden(), vn, table.roots)
    storage = get(table.storage.data, name, NoTemplate())
    _check_cross_roles(previous, ModelValue{R}(value), vn)
    _check_whole_binding_storage(storage, vn)
    constraint = get(
        (R === Fix ? table.fixed_templates : table.observed_templates).data,
        name,
        NoTemplate(),
    )
    constraint isa NoTemplate ||
        _convert_resolved_value(value, constraint, AbstractPPL.Iden(), vn, true)
    other = get(
        (R === Fix ? table.observed_templates : table.fixed_templates).data,
        name,
        NoTemplate(),
    )
    _same_storage_shape(other, value) || throw(
        ArgumentError(
            "Bound value at `$vn` has incompatible template shape; supply a new template with the value",
        ),
    )
    if _has_binding_storage(previous)
        _same_storage_shape(storage, value) || throw(
            ArgumentError(
                "Binding shape at `$vn` cannot change; use decondition and unfix on the whole root first",
            ),
        )
    end
    updates, stored = _whole_binding_updates(table, R, Base.tail(entries))
    return (
        VarNamedTuple(merge(NamedTuple{(name,)}((ModelValue{R}(value),)), updates.data)),
        VarNamedTuple(merge(NamedTuple{(name,)}((value,)), stored.data)),
    )
end

_field_binding_path(::AbstractPPL.Iden) = true
_field_binding_path(optic::AbstractPPL.Property) = _field_binding_path(optic.child)
_field_binding_path(optic) = false
_nested_binding(value, ::AbstractPPL.Iden) = value
function _nested_binding(value, optic::AbstractPPL.Property{S}) where {S}
    return VarNamedTuple(NamedTuple{(S,)}((_nested_binding(value, optic.child),)))
end
function _bind_field_resolved(table, ::Type{R}, pair) where {R}
    vn, value = pair
    _check_bound_placeholder(R, vn, value)
    optic = AbstractPPL.varname_to_optic(vn)
    values = _nested_binding(ModelValue{R}(value), optic)
    storage = _nested_binding(value, optic)
    return ResolvedBindings(
        values,
        storage,
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        table.roots,
    )
end

@inline function _bind_template_resolved(
    table, ::Type{R}, vn, value, template, prefix
) where {R}
    _check_bound_placeholder(R, vn, value)
    name = AbstractPPL.getsym(vn)
    root = VarName{name}()
    address = maybe_prefix(vn, prefix)
    optic = _collapse_binding_selection(
        template, AbstractPPL.getoptic(vn), address; location=maybe_prefix(root, prefix)
    )
    _path_storage(template, optic, root, table.roots; address)
    _check_resolved_overlap!(
        (Set{VarName}(), Set{VarName}()),
        _resolved_selections(template, optic, maybe_prefix(root, prefix)),
    )
    _check_binding_extents(value, template, optic, address)
    value = _convert_resolved_value(value, template, optic, address, true)
    stored = _retain_binding_storage(template, value, optic, root, nothing, true)
    values = templated_setindex!!(
        VarNamedTuple(), ModelValue{R}(value), VarName{name}(optic), template
    )
    templates = VarNamedTuple(NamedTuple{(name,)}((template,)))
    return ResolvedBindings(
        values,
        VarNamedTuple(NamedTuple{(name,)}((stored,))),
        R === Fix ? table.observed_templates : templates,
        R === Fix ? templates : table.fixed_templates,
        table.arguments,
        table.roots,
    )
end

"""
    _bind_resolved(table, role, inputs...; roots=(), kwargs...)

Construct a resolved role table without evaluating a model.
Place a positional `@of` template last, after all positional value inputs;
otherwise throw `ArgumentError`. Keyword bindings may accompany a final template.
"""
function _bind_resolved(
    table::ResolvedBindings, ::Type{R}, inputs...; roots=(), error_prefix=nothing, kwargs...
) where {R<:Union{Condition,Fix,ArgumentCondition}}
    inputs = _binding_inputs(inputs)
    if isempty(roots) && isempty(kwargs) && all(_whole_binding_input, inputs)
        return _bind_whole_resolved(table, R, _whole_binding_pairs(inputs))
    end
    if isempty(roots) &&
        isempty(kwargs) &&
        length(inputs) == 1 &&
        first(inputs) isa Pair{<:VarName} &&
        _whole_binding_value(last(first(inputs))) &&
        _field_binding_path(AbstractPPL.getoptic(first(first(inputs)))) &&
        isempty(table.values.data) &&
        isempty(table.storage.data) &&
        isempty(table.observed_templates.data) &&
        isempty(table.fixed_templates.data)
        return _bind_field_resolved(table, R, first(inputs))
    end
    if isempty(roots) &&
        isempty(kwargs) &&
        length(inputs) == 1 &&
        first(inputs) isa Union{NamedTuple,VarNamedTuple} &&
        !_has_growable_binding(first(inputs)) &&
        isempty(table.observed_templates.data) &&
        isempty(table.fixed_templates.data)
        input = first(inputs)
        input = input isa NamedTuple ? VarNamedTuple(input) : input
        return _bind_structured_resolved(table, R, input)
    end
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
    if length(inputs) == 1 &&
        first(inputs) isa Pair{<:VarName} &&
        isempty(table.values.data) &&
        isempty(table.storage.data) &&
        isempty(table.observed_templates.data) &&
        isempty(table.fixed_templates.data)
        vn, value = first(inputs)
        root = VarName{AbstractPPL.getsym(vn)}()
        template = get(schema, AbstractPPL.getsym(vn), NoTemplate())
        optic = AbstractPPL.getoptic(vn)
        if template isa Array &&
            optic isa AbstractPPL.Index &&
            optic.child isa AbstractPPL.Iden &&
            _known_binding_root(root, table.roots)
            return _bind_template_resolved(table, R, vn, value, template, error_prefix)
        end
    end
    schema_names = VarName[
        vn for input in inputs for (vn, _, _) in _resolved_entries(input)
    ]
    schema =
        _used_binding_schema(
            VarNamedTuple(schema), schema_names, union(table.roots, roots)
        ).data
    for name in keys(schema)
        _check_existing_binding_template(
            get(table.values.data, name, nothing), schema[name], VarName{name}()
        )
    end
    templates = R === Fix ? table.fixed_templates : table.observed_templates
    current_schema = _fill_binding_storage(VarNamedTuple(schema), templates).data
    templates = VarNamedTuple(current_schema)
    batches = map(_resolved_entries, inputs)
    stored, values = table.storage, table.values
    updates = VarNamedTuple()
    prepared = Dict{Symbol,Any}()
    occupied = Dict{Symbol,Bool}()
    copied = IdDict{Any,Nothing}()
    supplied_roots = Set{Symbol}()
    seen = (Set{VarName}(), Set{VarName}())
    known = union(VarName[table.roots...], roots)
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
            storage isa VarNamedTuples.GrowableArray && _missing_binding_storage()
            stored = setindex!!(stored, storage, VarName{name}())
        end
    end
    unique!(known)
    entries = Tuple[]
    for batch in batches, (vn, value, supplied) in batch
        _expand_namespace_entries!(entries, vn, value, supplied, known)
    end
    for (vn, value, supplied) in entries
        _check_bound_placeholder(R, vn, value)
        name, optic = AbstractPPL.getsym(vn), AbstractPPL.getoptic(vn)
        root = VarName{name}()
        supplied isa NoTemplate || push!(supplied_roots, name)
        previous = get(values.data, name, nothing)
        storage = get!(prepared, name) do
            _resolved_root_storage(stored, current_schema, supplied, vn)
        end
        active = get!(occupied, name) do
            _has_binding_storage(previous)
        end
        if active && haskey(stored.data, name)
            _same_storage_shape(stored.data[name], storage) || throw(
                ArgumentError(
                    "Binding storage at `$root` cannot change shape from $(stored.data[name] isa AbstractArray ? size(stored.data[name]) : typeof(stored.data[name])) to $(storage isa AbstractArray ? size(storage) : typeof(storage)); use decondition and unfix on the whole root first",
                ),
            )
        end
        if !(optic isa AbstractPPL.Iden) && storage isa Union{Nothing,Missing}
            throw(
                ArgumentError(
                    "Cannot make a partial binding of argument `$root` with whole `$storage` storage; supply a concrete argument or a whole binding instead.",
                ),
            )
        end
        optic = _collapse_binding_selection(
            storage,
            optic,
            maybe_prefix(vn, error_prefix);
            location=maybe_prefix(root, error_prefix),
        )
        vn = VarName{name}(optic)
        if !(storage isa NoTemplate)
            _check_namedtuple_index(
                ModelValue{Condition}(storage),
                optic,
                AbstractPPL.Property{name}();
                check_container=_known_binding_root(root, known),
            )
        end
        _check_mixed_binding(previous, value, optic, root, known)
        _check_mixed_binding(get(updates.data, name, nothing), value, optic, root, known)
        storage = _path_storage(
            storage, optic, root, known; address=maybe_prefix(vn, error_prefix)
        )
        _check_resolved_overlap!(
            seen, _resolved_selections(storage, optic, maybe_prefix(root, error_prefix))
        )
        _check_binding_extents(value, storage, optic, vn)
        length(entries) == 1 && _check_cross_roles(
            _model_argument_binding(previous, optic),
            ModelValue{R}(value),
            maybe_prefix(vn, error_prefix),
        )
        constraint = _storage_at(templates, AbstractPPL.varname_to_optic(vn))
        other = _storage_at(
            R === Fix ? table.observed_templates : table.fixed_templates,
            AbstractPPL.varname_to_optic(vn),
        )
        _same_storage_shape(other, value) || throw(
            ArgumentError(
                "Bound value at `$vn` has incompatible template shape; supply a new template with the value",
            ),
        )
        value = if constraint !== nothing
            _convert_resolved_value(value, storage, optic, vn, true)
        elseif optic isa AbstractPPL.Iden
            value
        elseif !_has_binding_role(R, previous) && haskey(table.arguments.data, name)
            _check_namedtuple_index(
                ModelValue{Condition}(table.arguments.data[name]),
                optic,
                AbstractPPL.Property{name}();
                check_container=true,
            )
            _convert_resolved_value(value, table.arguments.data[name], optic, vn, false)
        elseif name in supplied_roots && !_has_binding_role(R, previous)
            _convert_resolved_value(value, storage, optic, vn, false)
        else
            _convert_binding_edit(value, previous, optic, vn, R)
        end
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
            if retained isa VarNamedTuple && storage isa VarNamedTuple
                retained = _fill_binding_storage(storage, retained)
            end
            retained = _path_storage(
                retained isa NoTemplate ? storage : retained, optic, root, known
            )
        end
        retained = _retain_binding_storage(
            retained, value, optic, root, previous, true; copied
        )
        updates = templated_setindex!!(updates, ModelValue{R}(value), vn, storage)
        prepared[name] = retained
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

_has_growable_binding(value) = false
_has_growable_binding(value::NamedTuple) = any(_has_growable_binding, values(value))
_has_growable_binding(value::VarNamedTuple) = _has_growable_binding(value.data)
function _has_growable_binding(value::VarNamedTuples.PartialArray)
    value.data isa VarNamedTuples.GrowableArray && return true
    return _fold_model_indices(false, value) do found, child, _, _
        found || _has_growable_binding(child)
    end
end

function _convert_binding_edit(value, previous, optic, vn, R)
    previous === nothing && return value
    if previous isa ModelValue && _matches_model_role(R, previous)
        return if optic isa AbstractPPL.Iden
            value
        else
            _convert_resolved_value(value, previous.value, optic, vn, false)
        end
    end
    optic isa AbstractPPL.Iden && return value
    if _model_argument_binding(previous, optic) === nothing &&
        _has_binding_role(R, previous)
        template = if previous isa ModelValueTree
            previous.template
        elseif previous isa VarNamedTuples.PartialArray &&
            previous.mask isa ModelBindingArray
            previous.mask.template
        else
            NoTemplate()
        end
        template isa NoTemplate ||
            return _convert_resolved_value(value, template, optic, vn, false)
    end
    head = AbstractPPL.ohead(optic)
    child = _model_argument_binding(previous, head)
    if (
            previous isa VarNamedTuples.PartialArray ||
            previous isa ModelValueTree{<:NamedTuple}
        ) &&
        child isa ModelValue &&
        _matches_model_role(R, child) &&
        optic.child isa AbstractPPL.Iden
        converted = convert(typeof(child.value), value)
        isequal(converted, value) || throw(
            ArgumentError(
                "Cannot exactly represent partial binding at `$vn` in storage element type $(typeof(child.value))",
            ),
        )
        return converted
    elseif previous isa ModelValueTree && _has_binding_role(R, child)
        return _convert_resolved_value(value, previous.template, optic, vn, false)
    end
    return _convert_binding_edit(value, child, optic.child, vn, R)
end

function _convert_resolved_value(value, storage, ::AbstractPPL.Iden, vn, templated)
    if templated && !(storage isa NoTemplate)
        value isa typeof(storage) || throw(
            ArgumentError(
                "Bound value at `$vn` must have template type $(typeof(storage))"
            ),
        )
        _same_storage_shape(value, storage) || throw(
            ArgumentError(
                "Bound value at `$vn` has incompatible template shape; supply a new template with the value",
            ),
        )
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
@inline function _check_field_removal(value, optic, vn, roots)
    if value isa ModelValue &&
        value.value isa NamedTuple &&
        !any(root -> subsumes(root, vn), roots)
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

_has_binding_storage(::Union{Nothing,NoModelBinding}) = false
_has_binding_storage(::Union{ModelValue,ModelValueTree,VarNamedTuples.PartialArray}) = true
_has_binding_storage(values::VarNamedTuple) = _contains_model_binding(values)

function _contains_model_binding(value::Union{VarNamedTuple,VarNamedTuples.PartialArray})
    return _fold_model_indices(false, value) do found, child, _, _
        found || _contains_model_binding(child)
    end
end
_contains_model_binding(value::ModelValueTree) = _contains_model_binding(value.values)
_contains_model_binding(::ModelValue) = true
_contains_model_binding(::Union{Nothing,NoModelBinding}) = false

@inline function _restore_argument_storage(table, values, storage, ::Val{name}) where {name}
    if haskey(table.arguments.data, name) &&
        !_contains_model_binding(get(values.data, name, nothing))
        return setindex!!(storage, table.arguments.data[name], VarName{name}())
    end
    return storage
end

@generated function _restore_argument_storage(
    table, values, storage::VarNamedTuple{names}
) where {names}
    arguments = fieldtype(table, :arguments).parameters[1]
    fields = map(names) do name
        name in arguments || return :(storage.data.$name)
        return :(
            if _contains_model_binding(get(values.data, $(QuoteNode(name)), nothing))
                storage.data.$name
            else
                table.arguments.data.$name
            end
        )
    end
    return :(VarNamedTuple(NamedTuple{$names}(($(fields...),))))
end

function _unbind_resolved(
    table::ResolvedBindings, ::Type{R}, names...; error_prefix=nothing
) where {R<:Union{Condition,Fix}}
    names = map(name -> name isa Symbol ? VarName{name}() : name, names)
    all(name -> name isa VarName, names) ||
        throw(ArgumentError("Removal addresses must be Symbols or VarNames"))
    if isempty(names)
        values = _remove_model_values(R, table.values)
        storage = _restore_argument_storage(table, values, table.storage)
        return ResolvedBindings(
            values,
            storage,
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
        optic = _collapse_binding_selection(
            root_storage,
            optic,
            maybe_prefix(vn, error_prefix);
            operation="remove",
            location=maybe_prefix(VarName{name}(), error_prefix),
        )
        vn = VarName{name}(optic)
        root_storage isa NoTemplate || _check_namedtuple_index(
            ModelValue{Condition}(root_storage),
            optic,
            AbstractPPL.Property{name}();
            operation="remove",
            check_container=_known_binding_root(VarName{name}(), table.roots),
        )
        _path_storage(
            root_storage,
            optic,
            VarName{name}(),
            table.roots;
            operation="remove",
            address=maybe_prefix(vn, error_prefix),
        )
        _check_field_removal(values, AbstractPPL.varname_to_optic(vn), nothing, table.roots)
        values = _prune_model_bindings(
            _remove_model_binding(R, values, AbstractPPL.varname_to_optic(vn))
        )
        if optic isa AbstractPPL.Iden
            storage = _restore_argument_storage(table, values, storage, Val(name))
        end
        if optic isa AbstractPPL.Iden &&
            root_storage isa VarNamedTuple &&
            !_contains_model_binding(get(values.data, name, nothing)) &&
            !haskey(table.observed_templates.data, name) &&
            !haskey(table.fixed_templates.data, name)
            storage = VarNamedTuple(Base.structdiff(storage.data, NamedTuple{(name,)}))
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

function _convert_binding_elements(::Type{T}, values) where {T}
    return map(value -> convert(T, value), values)
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
        _convert_binding_elements(T, value)
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

using SparseArrays

struct ModelArgumentCopy{T}
    value::T
end
function Base.deepcopy_internal(value::ModelArgumentCopy, memo::IdDict)
    _retain_argument_leaves!(memo, value.value, Base.IdSet{Any}())
    _retain_bound_argument_storage!(memo, value.value)
    return ModelArgumentCopy(Base.deepcopy_internal(value.value, memo))
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
function _copy_model_argument(value)
    return deepcopy(ModelArgumentCopy(value)).value
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
# One bits type keeps dense arrays of leaf plans cheap for Julia to deepcopy.
struct ModelArgumentLeaf
    bound::Bool
end
# Copy templates and arguments in one graph so overlapping templates share an identity table.
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

# Visit owning arrays' elements but wrappers' fields; visiting wrapper elements too
# would count their parent's storage twice.
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
function _check_shared_latent_storage(
    table::ResolvedBindings, arguments::NamedTuple, prefix=nothing
)
    _argument_may_alias(typeof(arguments)) || return nothing
    lhs = keys(table.arguments.data)
    latent, repeated = Base.IdSet{Any}(), Any[]
    roots, bound = Tuple{Any,Any,Any,Union{Bool,Nothing}}[], Any[]
    for (name, value) in pairs(arguments)
        _argument_may_alias(typeof(value)) || continue
        vn = VarName{name}()
        binding = if name in lhs
            _model_argument_binding(table.values, AbstractPPL.varname_to_optic(vn))
        else
            nothing
        end
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
        push!(roots, (vn, value, binding, role))
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
    selected = map(
        name -> maybe_prefix(name, prefix), _shared_argument_names(roots, shared)
    )
    throw(
        ArgumentError(
            "Storage reachable from a latent argument is shared by $(join(map(name -> "`$name`", selected), " and ")). A latent draw may replace storage and leave another reference stale, even if that reference is only read; pass independent storage, for example `(a=v, b=copy(v))`.",
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

    # ConstructionBase's fallback calls constructorof(T) on properties in field order.
    # Check actual types because declared fields may be abstract.
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

function _collapse_binding_selection(
    storage,
    optic::AbstractPPL.Iden,
    vn;
    operation="bind",
    location=VarName{AbstractPPL.getsym(vn)}(),
)
    return optic
end
function _collapse_binding_selection(
    storage,
    optic::AbstractPPL.Property{S},
    vn;
    operation="bind",
    location=VarName{AbstractPPL.getsym(vn)}(),
) where {S}
    child = if storage isa Union{VarNamedTuple,NoTemplate} || hasproperty(storage, S)
        _binding_child_template(storage, AbstractPPL.Property{S}())
    else
        NoTemplate()
    end
    return AbstractPPL.Property{S}(
        _collapse_binding_selection(
            child,
            optic.child,
            vn;
            operation,
            location=AbstractPPL.append_optic(location, AbstractPPL.Property{S}()),
        ),
    )
end
function _collapse_binding_selection(
    storage,
    optic::AbstractPPL.Index,
    vn;
    operation="bind",
    location=VarName{AbstractPPL.getsym(vn)}(),
)
    storage = _binding_storage(storage)
    storage isa AbstractArray || return optic
    head = AbstractPPL.concretize_top_level(optic, storage)
    checkbounds(Bool, storage, head.ix...; head.kw...) || throw(
        ArgumentError(
            "Cannot $operation `$vn`: index is outside the storage at `$location`"
        ),
    )
    selected_location = AbstractPPL.append_optic(location, AbstractPPL.ohead(head))
    child = head.child
    if child isa AbstractPPL.Index &&
        VarNamedTuples._is_multiindex(storage, head.ix...; head.kw...)
        selected = getindex(CartesianIndices(axes(storage)), head.ix...; head.kw...)
        while selected isa AbstractArray && child isa AbstractPPL.Index
            next = AbstractPPL.concretize_top_level(child, selected)
            checkbounds(Bool, selected, next.ix...; next.kw...) || throw(
                ArgumentError(
                    "Cannot $operation `$vn`: index is outside the storage at `$selected_location`",
                ),
            )
            selected = getindex(selected, next.ix...; next.kw...)
            selected_location = AbstractPPL.append_optic(
                selected_location, AbstractPPL.ohead(next)
            )
            child = next.child
        end
        head = if selected isa CartesianIndex
            AbstractPPL.Index(Tuple(selected), (;), child)
        else
            AbstractPPL.Index((selected,), (;), child)
        end
    end
    child isa AbstractPPL.Iden && return head
    _check_assigned_binding_storage(storage, head, vn)
    selected_storage = _binding_child_template(storage, AbstractPPL.ohead(head))
    return AbstractPPL.Index(
        head.ix,
        head.kw,
        _collapse_binding_selection(
            selected_storage, child, vn; operation, location=selected_location
        ),
    )
end

function _used_binding_schema(schema, names, roots, prefix=nothing)
    isempty(names) && return VarNamedTuple()
    prefix !== nothing &&
        (_known_binding_root(prefix, roots) || any(==(prefix), names)) &&
        return schema
    schema isa Union{NamedTuple,VarNamedTuple} || return schema
    fields = schema isa VarNamedTuple ? schema.data : schema
    selected = VarNamedTuple()
    for name in keys(fields)
        vn = if prefix === nothing
            VarName{name}()
        else
            AbstractPPL.append_optic(prefix, AbstractPPL.Property{name}())
        end
        descendants = filter(n -> subsumes(vn, n), names)
        isempty(descendants) && continue
        value = _used_binding_schema(fields[name], descendants, roots, vn)
        selected = setindex!!(selected, value, VarName{name}())
    end
    return selected
end
function _fill_binding_storage(candidate::VarNamedTuple, previous::VarNamedTuple)
    result = previous
    for name in keys(candidate.data)
        child = _fill_binding_storage(
            candidate.data[name], get(previous.data, name, NoTemplate())
        )
        result = setindex!!(result, child, VarName{name}())
    end
    return result
end
function _same_concrete_shape(a::VarNamedTuple, b::VarNamedTuple, parents)
    return all(keys(a.data)) do name
        !haskey(b.data, name) || _same_storage_shape(a.data[name], b.data[name], parents)
    end
end

function _same_concrete_shape(a::NamedTuple, b::VarNamedTuple, parents)
    return all(keys(b.data)) do name
        haskey(a, name) && _same_storage_shape(a[name], b.data[name], parents)
    end
end
function _same_concrete_shape(a::VarNamedTuple, b::NamedTuple, parents)
    return _same_concrete_shape(b, a, parents)
end
