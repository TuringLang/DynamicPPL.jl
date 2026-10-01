# ------------
# Model values
# ------------

struct Condition end
struct ArgumentCondition end
struct Fix end

_contains_missing(::Any) = false
_contains_missing(::Missing) = true
_contains_missing(value::Base.Pairs) = _contains_missing(values(value))
_contains_missing(value::TransformedValue) = _contains_missing(get_internal_value(value))
_contains_missing(::AbstractArray{<:Number}) = false
function _contains_missing(values::AbstractArray)
    return any(
        i -> isassigned(values, i) && _contains_missing(values[i]), eachindex(values)
    )
end
function _contains_missing(values::Union{Tuple,NamedTuple})
    return any(_contains_missing, values)
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

# Evaluation-local bindings avoid expanding each child into its parent's storage shape.
struct LocalModelValues{V<:VarNamedTuple}
    values::V
end
# Default arguments need no enclosing namespace storage during evaluation.
struct UnprefixedArgumentValues{V<:VarNamedTuple}
    values::V
end
_model_values(values::UnprefixedArgumentValues) = values.values
_model_value_varname(::UnprefixedArgumentValues, vn, prefix) = vn

_model_values(values::VarNamedTuple) = values
_model_values(values::LocalModelValues) = values.values
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
function VarNamedTuples._haskey_optic(value::ModelValue, optic::AbstractPPL.AbstractOptic)
    return VarNamedTuples._haskey_optic(value.value, optic)
end
VarNamedTuples._haskey_optic(::ModelValue, ::AbstractPPL.Iden) = true
function VarNamedTuples._haskey_optic(
    value::ModelValue{R,<:Tuple}, optic::AbstractPPL.Index
) where {R<:Union{Condition,ArgumentCondition,Fix}}
    optic = AbstractPPL.concretize_top_level(optic, value.value)
    isempty(optic.kw) && checkbounds(Bool, Base.OneTo(length(value.value)), optic.ix...) ||
        return false
    return VarNamedTuples._haskey_optic(getindex(value.value, optic.ix...), optic.child)
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
    _defer_argument_binding(model, argument) ||
        _check_fixed_shape(binding, local_value, AbstractPPL.getoptic(vn), vn)
    return _get_model_data(model, vn)
end

function _check_fixed_shape(binding, local_value, optic, vn)
    value = binding isa ModelValue ? binding.value : binding
    array = value isa VarNamedTuples.PartialArray ? value.data : value
    tuple = value isa ModelValueTree ? value.template : value
    if tuple isa Tuple
        local_value isa Tuple && length(tuple) == length(local_value) || _fixed_shape_error(
            vn, "a static size and shape; the model body changed its argument's length."
        )
    elseif (array isa AbstractArray || local_value isa AbstractArray) && !(
        value isa VarNamedTuples.PartialArray && array isa VarNamedTuples.GrowableArray
    )
        array isa AbstractArray &&
            local_value isa AbstractArray &&
            axes(array) == axes(local_value) || _fixed_shape_error(
            vn, "a static size and shape; the model body changed its argument's shape."
        )
    end
    return _check_fixed_shape_child(binding, local_value, optic, vn)
end
# Keep error construction out of the hot path so the shape checks can inline.
@noinline function _fixed_shape_error(vn, message)
    return throw(ArgumentError("Fixed LHS variable `$vn` requires $message"))
end
_check_fixed_shape_child(binding, local_value, ::AbstractPPL.Iden, vn) = nothing
function _check_fixed_shape_child(
    binding, local_value, optic::AbstractPPL.Property{S}, vn
) where {S}
    child = _model_argument_binding(binding, AbstractPPL.Property{S}())
    child !== nothing && hasproperty(local_value, S) ||
        _fixed_shape_error(vn, "coverage with a static size and shape.")
    return _check_fixed_shape(child, getproperty(local_value, S), optic.child, vn)
end
function _check_fixed_shape_child(binding, local_value, optic::AbstractPPL.Index, vn)
    optic = AbstractPPL.concretize_top_level(optic, local_value)
    child = _model_argument_binding(binding, AbstractPPL.Index(optic.ix, optic.kw))
    child === nothing && _fixed_shape_error(vn, "coverage with a static size and shape.")
    selected = if VarNamedTuples._is_multiindex(local_value, optic.ix...; optic.kw...)
        Base.maybeview(local_value, optic.ix...; optic.kw...)
    else
        getindex(local_value, optic.ix...; optic.kw...)
    end
    return _check_fixed_shape(child, selected, optic.child, vn)
end

function _tag_model_values(::Type{R}, values::VarNamedTuple) where {R}
    tagged = map_pairs!!(pair -> _tag_model_value(R, pair.second, pair.first), copy(values))
    return R === ArgumentCondition ? _prune_model_bindings(tagged) : tagged
end

_tag_model_value(::Type{R}, value, vn) where {R} = ModelValue{R}(value)
# Only a whole `nothing` argument is a latent placeholder. A `nothing` inside a container
# stays bound data, so a tilde that reads it fails with a `MethodError` in `logpdf`.
function _tag_model_value(
    ::Type{ArgumentCondition}, ::Nothing, ::VarName{S,AbstractPPL.Iden}
) where {S}
    return NoModelBinding()
end

function _expand_model_binding(previous::ModelValue{R,<:AbstractArray}) where {R}
    data = map(ModelValue{R}, previous.value)
    return VarNamedTuples.PartialArray(data, fill!(similar(data, Bool), true))
end
function _expand_model_binding(previous::ModelValue{R,<:Base.Pairs}) where {R}
    return _expand_model_binding(ModelValue{R}(NamedTuple(previous.value)))
end
function _expand_model_binding(previous::ModelValue{R,<:Tuple}) where {R}
    return ModelValueTree(previous.value, map(ModelValue{R}, previous.value))
end
function _expand_model_binding(previous::ModelValue{R}) where {R}
    names = propertynames(previous.value)
    fields = NamedTuple{names}(map(name -> getproperty(previous.value, name), names))
    return ModelValueTree(previous.value, _tag_model_values(R, VarNamedTuple(fields)))
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
_check_model_binding(previous, updates, vn) = nothing
function _check_model_binding(
    previous, updates::Union{VarNamedTuple,VarNamedTuples.PartialArray}, vn
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
        if child === nothing && (
            (previous isa ModelValue && previous.value isa Union{AbstractArray,Tuple}) ||
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
        child === nothing ||
            _check_model_binding(child, update, AbstractPPL.append_optic(vn, optic))
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
_model_data(values::VarNamedTuple) = map(_model_data, values.data)
function _model_data(values::VarNamedTuples.PartialArray)
    return _model_data(VarNamedTuples.unwrap_internal_array(values))
end
function _model_data(tree::ModelValueTree)
    return if tree.values isa Tuple
        map(tree.values, tree.template) do value, template
            if value isa NoModelBinding
                deepcopy(template)
            else
                _model_argument_value(value, template)
            end
        end
    else
        _model_argument_value(tree.values, tree.template)
    end
end
function VarNamedTuples.unwrap_internal_array(tree::ModelValueTree)
    VarNamedTuples._haskey_optic(tree, AbstractPPL.Iden()) ||
        throw(ArgumentError("Cannot extract a partially supplied model value"))
    return _model_data(tree)
end

_model_argument_value(value, template) = value
_model_argument_value(::Nothing, template) = deepcopy(template)
_model_argument_value(value::ModelValue, template) = value.value
_model_argument_value(tree::ModelValueTree, template) = _model_data(tree)
_model_argument_value(values::AbstractArray, template) = _model_data(values)

_has_complete_model_data(::Any) = true
_has_complete_model_data(::NoModelBinding) = false
# A namespace does not replace the submodel return value.
_has_complete_model_data(::VarNamedTuple) = false
_has_complete_model_data(::VarNamedTuples.ArrayLikeBlock) = false
function _has_complete_model_data(values::VarNamedTuples.PartialArray)
    # Growable arrays describe supplied indices, not the extent of the argument.
    return !(values.data isa VarNamedTuples.GrowableArray) &&
           all(values.mask) &&
           all(_has_complete_model_data, values.data)
end

function _defer_argument_binding(binding, value)
    return value === nothing && binding isa Union{VarNamedTuple,VarNamedTuples.PartialArray}
end

function _model_argument_value(values::VarNamedTuples.PartialArray, template)
    return if _has_complete_model_data(values)
        _model_data(values)
    else
        result = template isa Tuple ? template : copy(template)
        for i in eachindex(template)
            if !haskey(values, i) && (template isa Tuple || isassigned(template, i))
                result = BangBang.setindex!!(result, deepcopy(template[i]), i)
            end
        end
        _fold_model_indices(_set_model_argument, result, values)
    end
end
function _set_model_argument(result, value, optic, template)
    child_template = VarNamedTuples.maybe_index_template(result, optic)
    return Accessors.set(
        result,
        AbstractPPL.with_mutation(optic),
        _model_argument_value(value, child_template),
    )
end
function _model_argument_value(values::VarNamedTuple, template)
    template isa NoTemplate && return _model_data(values)
    for name in keys(values.data)
        hasproperty(template, name) || throw(
            ArgumentError(
                "Cannot override nonexistent property `$name` of $(typeof(template)). If it holds a submodel return value, condition or fix the child model before wrapping it with `to_submodel`.",
            ),
        )
    end
    fields = _model_argument_fields(values, ConstructionBase.getproperties(template))
    return ConstructionBase.setproperties(template, fields)
end
@generated function _model_argument_fields(
    values::VarNamedTuple{names}, template::NamedTuple{fields}
) where {names,fields}
    updates = map(fields) do name
        if name in names
            :(_model_argument_value(values.data.$name, template.$name))
        else
            :(deepcopy(template.$name))
        end
    end
    return :(NamedTuple{$fields}(($(updates...),)))
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
An argument equal to `nothing` supplies no argument-supplied observation; its LHS
variables are latent unless explicitly bound.
At a submodel tilde, an argument LHS variable receives the submodel return value. The
argument supplies only its value before the tilde runs; its argument-supplied observation
is ignored at that tilde, so it needs no deconditioning. See [Binding rules](@ref).
"""
struct Model{
    F,
    argnames,
    defaultnames,
    Targs,
    Tdefaults,
    C<:AbstractContext,
    Values<:Union{VarNamedTuple,LocalModelValues,UnprefixedArgumentValues},
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
        args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol}}=(),
    ) where {F,A,Ta,D,Td,C,V,Threaded}
        mapreduce(
            pair -> pair.second isa ModelValue, &, _model_values(values); init=true
        ) || throw(ArgumentError("Model values must carry a condition or fix role"))
        argument_names = Tuple(args_on_lhs)
        return new{F,A,D,Ta,Td,C,V,Threaded,argument_names}(
            f, args, defaults, context, values
        )
    end
    # Internal reconstruction reuses already-validated bindings.
    function DynamicPPL._reconstruct_model(
        model::Model{F,A,D,Ta,Td}, context::C, values::V, ::Val{Threaded}
    ) where {F,A,D,Ta,Td,C,V,Threaded}
        return new{F,A,D,Ta,Td,C,V,Threaded,_args_on_lhs(model)}(
            model.f, model.args, model.defaults, context, values
        )
    end
end

function _args_on_lhs(
    ::Model{F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
) where {F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
    return ArgsOnLHS
end

Base.@constprop :aggressive function Model{Threaded}(
    f,
    args::NamedTuple,
    defaults::NamedTuple,
    context::AbstractContext=DefaultContext();
    args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol}}=(),
) where {Threaded}
    values = _argument_defaults(merge(args, defaults), Val(Tuple(args_on_lhs)))
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
    condition(model, values)

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
                    binding = _prepare_argument_fields(
                        prepare_model_argument(previous, argument), binding, vn
                    )
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
            child = VarNamedTuples._getindex_optic(template, optic, vn)
            address = AbstractPPL.append_optic(vn, optic)
            binding = _prepare_argument_fields(child, binding, address)
            if template isa AbstractArray && binding isa ModelValue
                binding = _convert_partial_argument_binding(
                    binding, template, optic, address
                )
            end
        end
        return VarNamedTuples._setindex_optic!!(
            result, binding, optic, storage, VarNamedTuples.AllowAll()
        )
    end
end

function _convert_partial_argument_binding(
    binding::ModelValue{R}, template, optic, vn
) where {R}
    value = binding.value
    converted = try
        if VarNamedTuples._is_multiindex(template, optic.ix...; optic.kw...)
            map(v -> convert(eltype(template), v), value)
        else
            convert(eltype(template), value)
        end
    catch err
        err isa InterruptException && rethrow()
        throw(
            ArgumentError(
                "Cannot represent partial binding at `$vn` in argument element type $(eltype(template))",
            ),
        )
    end
    isequal(converted, value) || throw(
        ArgumentError(
            "Cannot exactly represent partial binding at `$vn` in argument element type $(eltype(template))",
        ),
    )
    return ModelValue{R}(converted)
end

"""
    condition(model::Model; values...)
    condition(model::Model, values::NamedTuple)

Return a `Model` which treats the LHS variables bound by `values` as observations: they replace
sampling and contribute to the likelihood.

See also: [`decondition`](@ref), [`conditioned`](@ref)

Later bindings replace earlier ones where they overlap: a whole binding replaces the entire
value; a partial binding changes part of an already-bound value and preserves the rest.
Subvariables of one LHS variable cannot have different roles. A value containing `missing`
is rejected when a tilde statement observes or fixes it, naming the LHS variable. Parts of
an argument or binding that no tilde statement reads may contain `missing`. It no longer
marks an LHS variable as latent: omit its explicit binding or use [`decondition`](@ref) to
make an argument LHS variable latent; see [Missing data](@ref).
Bindings unused by executed LHS variables are ignored.

Binding an argument with no LHS variables throws `ArgumentError` at this call; construct
the model with a new argument value instead. For submodel names, precedence, and errors
at evaluation time, see [`to_submodel`](@ref) and [Binding rules](@ref).

Whole bindings use the supplied object without copying. A partial binding snapshots the
remaining parts when it splits a whole binding; later changes to the supplied container's
entries are not reflected in those parts. The model body must not mutate bound values,
directly or through an alias such as a `view`. This also applies to [`fix`](@ref).

Binding a whole argument replaces its value, shape, and dispatch type parameters
from the start of the model body. Observed LHS variables use the value computed by the body;
fixed LHS variables reset to their bound value at the tilde statement. Partial updates preserve the
remaining stored values and their array templates. Values in partial bindings are converted to the argument array's element
type; values that cannot be represented exactly throw `ArgumentError`.
Arguments with unobserved entries retain their original storage
template; the corresponding tilde statements fill those entries during evaluation.
Defaults derived from an argument are evaluated at model construction; binding that
argument does not recompute them.
`model.args` and `model.defaults` retain construction values when bindings change;
bindings live in the model's binding table, so use [`conditioned`](@ref) and
[`fixed`](@ref) to inspect effective observations and fixed values.

!!! note
    Partial bindings on an array argument (for example, `@varname(x[1])`) rebuild the
    argument on every evaluation, costing O(length(x)). For large arrays or hot loops,
    prefer replacing the whole argument, for example `condition(model; x=newx)` with
    `newx` already containing the override, or construct the model with the updated argument.
    Under reverse-mode AD such as Mooncake, partial bindings can be far more expensive,
    so bind whole arrays when gradients are needed.

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
provide a `NamedTuple`, `AbstractDict{<:VarName}`, or a `VarNamedTuple`; internally these are
all converted to a `VarNamedTuple`.

For example, here we use a `Dict`:

```jldoctest condition
julia> conditioned_model_dict = condition(model, Dict(@varname(x) => 100.0));

julia> m, x = conditioned_model_dict(); (m != 1.0 && x == 100.0)
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

julia> observations = @vnt begin
           @template m=zeros(2)
           m[2] := 1.0
       end;

julia> conditioned_model = condition(model, observations);

julia> # `m[1]` is sampled while `m[2]` is observed.
       m = conditioned_model(); (m[1] != 1.0 && m[2] == 1.0)
true
```

Intuitively one might also expect to be able to write `model | (m[2] = 1.0, )`. You cannot
do this with a `NamedTuple` because the `VarName` `m[2]` cannot be represented as a `Symbol`
(i.e., `Symbol("m[2]")` is not the same as `@varname(m[2])`).

```jldoctest condition
julia> # (×) `m[2]` is not set to 1.0.
       m = condition(model, var"m[2]" = 1.0)(); m[2] == 1.0
false
```

But you _can_ do this if you use a `Dict` or a `VarNamedTuple` as the underlying storage
instead:

```jldoctest condition
julia> vnt = @vnt begin
           @template m = zeros(2)
           m[2] := 1.0
       end
VarNamedTuple
└─ m => PartialArray size=(2,) data::Vector{Float64}
        └─ (2,) => 1.0

julia> m = condition(model, vnt)(); (m[1] != 1.0 && m[2] == 1.0)
true
```

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
AbstractPPL.condition(model::Model, values...) = _bind_model(Condition, model, values...)

function _bind_model(::Type{R}, model::Model, values...) where {R}
    model = _materialize_argument_values(model)
    values = _tag_model_values(R, _make_condfix_values(model, values...))
    values = _check_argument_bindings(model, values)
    values = _merge_model_values(_model_values(model.values), values)
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    return _reconstruct_model(model; values)
end
function AbstractPPL.condition(model::Model; values...)
    return condition(model, NamedTuple(values))
end
function AbstractPPL.condition(model::Model, first::Pair, second::Pair, rest::Pair...)
    return condition(condition(model, first), second, rest...)
end
function AbstractPPL.condition(model::Model, values::Tuple{Vararg{Pair}})
    return condition(model, values...)
end
function AbstractPPL.condition(model::Model, values::AbstractDict{<:VarName})
    return condition(model, pairs(values)...)
end

"""
    _make_condfix_values(vals...)

Convert different types of input to a `VarNamedTuple` of values, suitable for storage in a
`Model`.

This handles all the cases where `vals` is either already a `NamedTuple` or `AbstractDict`
(e.g. `model | (x=1, y=2)`), as well as if they are splatted (e.g. `condition(model, x=1,
y=2)`).
"""
_make_condfix_values(model, values::NamedTuple) = VarNamedTuple(values)
_make_condfix_values(model, values::VarNamedTuple) = values
function _make_condfix_values(model, values::Pair{<:Union{VarName,Symbol}}...)
    templates = VarNamedTuple()
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
            get(_model_values(model.values).data, AbstractPPL.getsym(vn), NoTemplate()),
        )
    end
    result = VarNamedTuple()
    for (name, value) in values
        vn = name isa Symbol ? VarName{name}() : name
        _check_namedtuple_index(
            _model_values(model.values), AbstractPPL.varname_to_optic(vn)
        )
        result = try
            templated_setindex!!(
                result,
                value,
                vn,
                get(templates.data, AbstractPPL.getsym(vn), NoTemplate()),
            )
        catch err
            err isa BoundsError || rethrow()
            throw(
                ArgumentError(
                    "Cannot bind `$vn`: index is outside the argument template at `$(AbstractPPL.getsym(vn))`",
                ),
            )
        end
    end
    return result
end

"""
    decondition(model::Model)
    decondition(model::Model, names...)

Remove this model's conditioned bindings at `names...`, or all conditioned bindings
if no names are supplied.

Unlike [`unfix`](@ref), `decondition(m, :x)` removes explicit and argument-supplied
observations, making `x` latent. After deconditioning, an LHS variable's
sampled value replaces its local argument value and is used by subsequent model statements.

A name matches when it equals, contains, or is contained in a stored binding's address.
NamedTuple integer indices are rejected: use `x.a` instead of `x[1]`; Tuples keep integer indices.
Only the matching conditioned parts are removed. A name with no match throws
`ArgumentError`, including names with only fixed bindings. With no names, removing all
observations is always valid.

Only bindings stored on this model are removed. This cannot remove a child submodel's
argument-supplied observations: `decondition(outer_arg(), @varname(a.x))` throws when `a.x`
is supplied only by the child argument. Decondition the child before wrapping it with
`to_submodel` instead.

This is essentially the inverse of [`condition`](@ref).

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
    _check_model_removal(Condition, _model_values(model.values), syms...)
    values = _remove_model_values(Condition, _model_values(model.values), syms...)
    values = model.values isa LocalModelValues ? LocalModelValues(values) : values
    return _reconstruct_model(model; values)
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
    fix(model::Model, values::NamedTuple)

Return a `Model` which treats the LHS variables bound by `values` as constants: they replace
sampling and contribute no log probability. Fixed argument LHS variables reset to their bound
value when their tilde statement runs, even if the body has computed a different value.

Whole bindings use the supplied object without copying. A partial binding snapshots the
remaining parts when it splits a whole binding; later changes to the supplied container's
entries are not reflected in those parts. The model body must not mutate bound values,
directly or through a `view`. `missing` rejection has the scope and remedies documented
in [`condition`](@ref). Replacement, unused bindings, and
argument restrictions follow [`condition`](@ref). See [Binding rules](@ref) for the shared rules, including the
cost of partial bindings on array arguments, and [`to_submodel`](@ref) for submodels.

Fixed values must cover every LHS variable they bind with a static size and shape. If the model
body changes a fixed argument's size or shape, evaluation throws `ArgumentError` naming
the LHS variable. This restriction does not apply to [`condition`](@ref).

Removing a fixed binding with [`unfix`](@ref) restores the argument-supplied observation, if any,
without restoring an earlier explicit conditioned binding.

See also: [`unfix`](@ref), [`fixed`](@ref)

!!! warning "Fixing applies to whole LHS variables"
    A multivariate draw (e.g. `x ~ MvNormal(...)`) is a single LHS variable, so a subset of its
    subvariables cannot be fixed independently; only fixing the whole LHS variable is supported.
    Partly bound LHS variables throw `ArgumentError` during evaluation. Declare separate LHS variables in a loop (`x[i] ~ ...`) if you need to fix them
    individually.

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
fix(model::Model, values...) = _bind_model(Fix, model, values...)
function fix(model::Model; values...)
    return fix(model, NamedTuple(values))
end
function fix(model::Model, first::Pair, second::Pair, rest::Pair...)
    return fix(fix(model, first), second, rest...)
end
function fix(model::Model, values::Tuple{Vararg{Pair}})
    return fix(model, values...)
end
function fix(model::Model, values::AbstractDict{<:VarName})
    return fix(model, pairs(values)...)
end

"""
    unfix(model::Model)
    unfix(model::Model, names...)

Remove this model's fixed bindings at `names...`, or all fixed bindings if no names
are supplied. NamedTuple integer indices are rejected: use `x.a` instead of `x[1]`; Tuples keep integer indices.
Matching follows [`decondition`](@ref). A name with no stored fixed match throws `ArgumentError`,
including a name supplied only by a child submodel or only conditioned on this model.

Unlike [`decondition`](@ref), removal restores the argument-supplied observation, if any,
otherwise making the LHS variable latent; it never restores an earlier explicit conditioned binding.
Argument-supplied observations are rebuilt from the model's arguments, even if previously removed
with `decondition`. For `@model f(x) = x ~ Normal()`, both
`unfix(fix(f(1.0); x=5.0), :x)` and
`unfix(fix(decondition(f(1.0), :x); x=5.0), :x)` observe `x = 1.0` again.

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
    _check_model_removal(Fix, _model_values(model.values), syms...)
    values = _remove_model_values(Fix, _model_values(model.values), syms...)
    removed = _removed_fixed_bindings(_model_values(model.values), values)
    defaults = _argument_defaults(
        merge(model.args, model.defaults), Val(_args_on_lhs(model))
    )
    if !(model.values isa LocalModelValues) && _model_prefix(model) !== nothing
        defaults = _prefix_values(
            defaults,
            _model_prefix(model),
            _apply_prefix_template(_model_prefix_template(model), NoTemplate()),
        )
    end
    values = mapfoldl(
        identity,
        function (restored, pair)
            vn, binding = pair
            return if binding isa ModelValue{Fix} && haskey(defaults, vn)
                templated_setindex!!(
                    restored, defaults[vn], vn, defaults.data[AbstractPPL.getsym(vn)]
                )
            else
                restored
            end
        end,
        removed;
        init=values,
    )
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
            _tag_model_value(ArgumentCondition, arguments.$stored_name, $(VarName{name}())),
        )))
    end
    return :(_prune_model_bindings(VarNamedTuple(merge((;), $(fields...)))))
end

_removed_fixed_bindings(previous, remaining) = NoModelBinding()
function _removed_fixed_bindings(previous::ModelValue{Fix}, remaining)
    remaining isa ModelValue{Fix} && return NoModelBinding()
    remaining === nothing && return previous
    return _removed_fixed_bindings(_expand_model_binding(previous), remaining)
end
function _removed_fixed_bindings(
    previous::Union{VarNamedTuple,VarNamedTuples.PartialArray}, remaining
)
    remaining === nothing && return previous
    return _fold_model_indices(empty(previous), previous) do removed, value, optic, template
        child = _removed_fixed_bindings(value, _model_argument_binding(remaining, optic))
        return if child isa NoModelBinding
            removed
        else
            VarNamedTuples._setindex_optic!!(
                removed, child, optic, template, VarNamedTuples.AllowAll()
            )
        end
    end
end
function _removed_fixed_bindings(previous::ModelValueTree, remaining)
    remaining === nothing && return previous
    values = if previous.values isa Tuple
        ntuple(length(previous.values)) do i
            _removed_fixed_bindings(
                previous.values[i],
                _model_argument_binding(remaining, AbstractPPL.Index((i,), (;))),
            )
        end
    else
        _removed_fixed_bindings(
            previous.values, remaining isa ModelValueTree ? remaining.values : remaining
        )
    end
    return ModelValueTree(previous.template, values)
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
    x = _concretize_prefix(x, template)
    model = _materialize_argument_values(model)
    values =
        if _model_prefix(model) === nothing &&
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
    _contains_missing(value) && throw(
        ArgumentError(
            "LHS variable `$vn` contains `missing`; make it latent with `$remove`."
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
