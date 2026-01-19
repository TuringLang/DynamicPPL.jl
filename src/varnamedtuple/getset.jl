# We define our own getindex, setindex!!, and haskey functions, which we use to
# get/set/check values in VarNamedTuple and PartialArray. We do this because we want to be
# able to override their behaviour for some types exported from elsewhere without type
# piracy. This is needed because
# 1. We would want to index into things with lenses (from AbstractPPL.jl) using getindex and
# setindex!!, but AbstractPPL does not define these methods.
# 2. We would want `haskey` to fall back onto `checkbounds` when called on Base.Arrays.

const IndexWithoutChild = AbstractPPL.Index{<:Tuple,<:NamedTuple,AbstractPPL.Iden}

"""
    DynamicPPL._getindex_optic(collection, optic::AbstractPPL.Optic)
    DynamicPPL._getindex_optic(collection, vn::VarName)

Access the value in `collection` at the location specified by the given `optic`. If a `VarName`
is provided, it is first converted to an optic using `AbstractPPL.varname_to_optic`.

Here, `collection` can be either a `VarNamedTuple` or a `PartialArray`, or a leaf value stored
within one of these.

This is semantically similar to `AbstractPPL.getvalue` but is specialised for `VarNamedTuple`
and `PartialArray`, and skips a number of checks that are unnecessary here.

Note that it is only valid to index into a `VarNamedTuple` with a `Property` optic, and a
`PartialArray` with an `Index` optic. Other combinations are not valid. When we have reached
the leaf of the VNT i.e. a value, we could still handle pure `Index` optics if the value is
an `AbstractArray`, but otherwise the only valid optic is `Iden`.
"""
function _getindex_optic(vnt::VarNamedTuple, vn::VarName)
    return _getindex_optic(vnt, AbstractPPL.varname_to_optic(vn))
end
@inline _getindex_optic(@nospecialize(x::Any), ::AbstractPPL.Iden) = x
function _getindex_optic(vnt::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return _getindex_optic(getindex(vnt.data, S), optic.child)
end
function _getindex_optic(pa::PartialArray, optic::AbstractPPL.Index)
    coptic = AbstractPPL.concretize_top_level(optic, pa.data)
    child_value =
        if _is_multiindex(pa, coptic.ix...; coptic.kw...) &&
            coptic.child isa AbstractPPL.Index
            # getindex can error if we try to access an index that is masked. However, it 
            # may be that the child index subsets into only valid indices (i.e. unmasked
            # parts). For example, consider the case where `pa[1]` is valid but `pa[2]` is
            # masked. Then `pa[1:2][1]` is valid, even though `pa[1:2]` would error.
            _subset_partialarray(pa, coptic.ix...; coptic.kw...)
        else
            getindex(pa, coptic.ix...; coptic.kw...)
        end
    return _getindex_optic(child_value, optic.child)
end
function _getindex_optic(arr::AbstractArray, optic::IndexWithoutChild)
    coptic = AbstractPPL.concretize_top_level(optic, arr)
    return Base.getindex(arr, coptic.ix...; coptic.kw...)
end

function _haskey_optic(vnt::VarNamedTuple, name::VarName)
    return _haskey_optic(vnt, AbstractPPL.varname_to_optic(name))
end
@inline _haskey_optic(@nospecialize(::Any), ::AbstractPPL.Iden) = true
@inline _haskey_optic(::VarNamedTuple, ::AbstractPPL.Index) = false
function _haskey_optic(vnt::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return Base.haskey(vnt.data, S) && _haskey_optic(getindex(vnt.data, S), optic.child)
end
function _haskey_optic(pa::PartialArray, optic::AbstractPPL.Index)
    return Base.haskey(pa, optic.ix...; optic.kw...) &&
           _haskey_optic(Base.getindex(pa, optic.ix...; optic.kw...), optic.child)
end
function _haskey_optic(arr::AbstractArray, optic::IndexWithoutChild)
    # Note that this call to `checkbounds` can error, although it is technically out of our
    # hands: it depends on how the provider of the AbstractArray has implemented
    # checkbounds. For example, DimArray can error here:
    # https://github.com/rafaqz/DimensionalData.jl/issues/1156. But that is not our job to fix
    # -- it should be done upstream -- hence we just forward the indices.
    return checkbounds(Bool, arr, optic.ix...; optic.kw...)
end

"""
    _setindex_optic!!(collection, value, optic, template, top_level; allow_new=Val(true))

Like `setindex!!`, but special-cased for `VarNamedTuple` and `PartialArray` to recurse
into nested structures.

The `top_level` argument indicates whether `optic` refers to a top-level symbol (e.g. a
`Property{:x}` referring to a top-level value of `x`). In such a case, the `template`
argument refers to the structure of that top-level symbol (i.e., `x`). If not, then the
`template` argument refers to the structure that should be indexed into with that optic.

The `allow_new` keyword argument is a performance optimisation: If it is set to
`Val(false)`, the function can assume that the key being set already exists in `collection`.
This allows skipping some code paths, which may have a minor benefit at runtime, but more
importantly, allows for better constant propagation and type stability at compile time.

`allow_new` being set to `Val(false)` does _not_ guarantee that no new keys will be added.
It only gives the implementation of `_setindex!!` the permission to assume that the key
already exists. Setting it to `Val(false)` should be done only when the caller is sure that
the key already exists, anything else is a bug in the caller.

Most methods of _setindex!! ignore the `allow_new` keyword argument, as they have no use for
it. See the method for setting values in a `VarNamedTuple` with a `ComposedFunction` for
when it is useful.
"""
@inline function _setindex_optic!!(
    @nospecialize(::Any),
    value,
    ::AbstractPPL.Iden,
    @nospecialize(::Any),
    ::Bool,
    ;
    allow_new=Val(true),
)
    return value
end
function _setindex_optic!!(
    arr::AbstractArray,
    value,
    optic::IndexWithoutChild,
    template,
    is_top_level;
    allow_new=Val(true),
)
    return BangBang.setindex!!(arr, value, optic.ix...; optic.kw...)
end

function throw_setindex_allow_new_error()
    return error(
        "Attempted to set a value at a key that does not exist, but" *
        " `allow_new=Val(false)` was specified. If you did not attempt" *
        " to call this function yourself, this likely indicates a bug in" *
        " DynamicPPL. Please file an issue at" *
        " https://github.com/TuringLang/DynamicPPL.jl/issues.",
    )
end

struct SharedGetProperty{S} end
(::SharedGetProperty{S})(x) where {S} = getproperty(x, S)
(::SharedGetProperty)(::NoTemplate) = NoTemplate()
(::SharedGetProperty)(t::SkipTemplate{N}) where {N} = decrease_skip(t)
function (::SharedGetProperty{S})(x::VarNamedTuple) where {S}
    if Base.haskey(x.data, S)
        x.data[S]
    else
        NoTemplate()
    end
end

function _setindex_optic!!(
    pa::PartialArray,
    value,
    optic::AbstractPPL.Index,
    template,
    is_top_level;
    allow_new=Val(true),
)
    need_merge = false
    coptic = AbstractPPL.concretize_top_level(optic, pa.data)
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        child_template = if is_top_level
            template
        elseif template === NoTemplate()
            NoTemplate()
        elseif template isa SkipTemplate
            decrease_skip(template)
        elseif _is_multiindex(template, coptic.ix...; coptic.kw...)
            Base.getindex(template, coptic.ix...; coptic.kw...)
        elseif isassigned(template, coptic.ix...; coptic.kw...)
            Base.getindex(template, coptic.ix...; coptic.kw...)
        else
            NoTemplate()
        end
        if Base.haskey(pa, coptic.ix...; coptic.kw...)
            # The PartialArray already contains an unmasked value at this index. We need to
            # set that value in place there.
            _setindex_optic!!(
                Base.getindex(pa, coptic.ix...; coptic.kw...),
                value,
                coptic.child,
                child_template,
                false;
                allow_new=allow_new,
            )
        elseif allow_new isa Val{true}
            if any(view(pa.mask, coptic.ix...; coptic.kw...))
                # NOTE: This is a VERY subtle case, which can happen when you are setting
                # multiple indices at once, but some of them were masked. When they are
                # masked, it will cause haskey to return false, so we can't go into the
                # previous branch. However, if we naively call make_leaf to create the 
                # sub-value and then setindex it inside the PartialArray, it will overwrite
                # ALL the indices with the new leaf value, which will overwrite any
                # previously active values! To avoid this, we will set a flag to indicate
                # that we can create a new leaf, but at the end of the function we need to
                # merge the new leaf into the existing PartialArray instead of overwriting
                # it.
                # This situation can happen e.g. with
                #     using DynamicPPL
                #     vnt = VarNamedTuple()
                #     x = zeros(2)
                #     vnt = DynamicPPL.templated_setindex!!(vnt, 1.0, @varname(x[1:2][1]), x)
                #     vnt = DynamicPPL.templated_setindex!!(vnt, 2.0, @varname(x[1:2][2]), x)
                need_merge = true
            end
            # No new data but we are allowed to create it.
            make_leaf(value, coptic.child, child_template)
        else
            throw_setindex_allow_new_error()
        end
    end
    return if need_merge
        new_pa = BangBang.setindex!!(copy(pa), sub_value, coptic.ix...; coptic.kw...)
        _merge_norecurse(pa, new_pa)
    else
        BangBang.setindex!!(pa, sub_value, coptic.ix...; coptic.kw...)
    end
end

function _setindex_optic!!(
    vnt::VarNamedTuple,
    value,
    optic::AbstractPPL.Property{S},
    template,
    is_top_level;
    allow_new=Val(true),
) where {S}
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        child_template = is_top_level ? template : SharedGetProperty{S}()(template)
        if Base.haskey(vnt.data, S)
            # Data already exists; we need to recurse into it
            _setindex_optic!!(
                vnt.data[S],
                value,
                optic.child,
                child_template,
                false;
                allow_new=allow_new,
            )
        elseif allow_new isa Val{true}
            # No new data but we are allowed to create it.
            make_leaf(value, optic.child, child_template)
        else
            throw_setindex_allow_new_error()
        end
    end
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((sub_value,))))
end

@generated function _is_multiindex_static(::T) where {T<:Tuple}
    for x in T.parameters
        if x <: AbstractVector{<:Int} || x <: Colon
            return :(return true)
        end
    end
    return :(return false)
end

"""
    make_leaf(value, optic, template)

Make a new leaf node for a VarNamedTuple.

This is the function that sets any `optic` that is a `Property` to be stored as a
`VarNamedTuple`, any `Index` to be stored as a `PartialArray`, and other `Iden` optics to be
stored as raw values. It is the link that joins `VarNamedTuple` and `PartialArray` together.
"""
@inline make_leaf(@nospecialize(value::Any), ::AbstractPPL.Iden, ::Any) = value
function make_leaf(value, optic::AbstractPPL.Property{S}, template) where {S}
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        child_template = SharedGetProperty{S}()(template)
        make_leaf(value, optic.child, child_template)
    end
    return VarNamedTuple(NamedTuple{(S,)}((sub_value,)))
end
function make_leaf(value, optic::AbstractPPL.Index, template)
    coptic = AbstractPPL.concretize_top_level(optic, template)
    is_multiindex = if template isa NoTemplate || template isa SkipTemplate
        _is_multiindex_static(coptic.ix)
    else
        _is_multiindex(template, coptic.ix...; coptic.kw...)
    end
    sub_value = if coptic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        # Note: don't need to check isassigned here because if the child template would be
        # an undefined reference, then it is meaningless for coptic.child to be anything but
        # Iden, which would have been caught above. Similar logic is used in
        # AbstractPPL.concretize.
        child_template = if template === NoTemplate()
            NoTemplate()
        elseif template isa SkipTemplate
            decrease_skip(template)
        else
            getindex(template, coptic.ix...; coptic.kw...)
        end
        make_leaf(value, coptic.child, child_template)
    end
    # We need to be careful here, as depending on the indices, `sub_value` might either
    # represent a single element or a slice of elements. When reconstructing the data
    # to use for the PartialArray creation, we need to make sure to not accidentally
    # create an array of arrays.
    correct_template_eltype =
        if is_multiindex &&
            sub_value isa AbstractArray &&
            (
                template isa NoTemplate ||
                template isa SkipTemplate ||
                template isa AbstractArray
            )
            # In this branch, we know that sub_value represents a slice of elements. Since
            # it's an AbstractArray, we can safely get its element type.
            eltype(sub_value)
        elseif is_multiindex
            # This is the case where sub_value represents a slice of elements, but it's
            # not an AbstractArray. This can happen if sub_value is something that needs
            # to be set as an ArrayLikeBlock. In order to get the right element type for
            # the template (to avoid type instability), we need to pre-construct the
            # expected type that would be used here if we were to set it directly.
            #
            # Note that we don't perform any check here that vnt_size lines up with the
            # size of the indices. That's checked for us inside PartialArray code.
            ArrayLikeBlock{
                typeof(sub_value),
                typeof(coptic.ix),
                typeof(coptic.kw),
                typeof(vnt_size(value)),
            }
        else
            # Single-element indexing.
            typeof(sub_value)
        end
    pa_data = if template isa NoTemplate || template isa SkipTemplate
        # If no template was provided, we have to make a GrowableArray.
        template_sz = get_implied_size_from_indices(coptic.ix...; coptic.kw...)
        GrowableArray(Array{correct_template_eltype}(undef, template_sz))
    elseif is_multiindex && sub_value isa PartialArray
        # In this case, sub_value is actually just a subset of the data that we are going to
        # create! This can happen with varnames like x[1:2][1].
        # Note that this is different from varnames like x[1][1]. In that case, x should be
        # a PartialArray that itself stores PartialArrays.
        similar(sub_value.data, size(template))
    elseif !(eltype(template) <: correct_template_eltype)
        # If coptic.child was a Property lens, then sub_value will always be normalised into
        # a VNT (since that's what the inner make_leaf returns). However, `template` may
        # contain things that are not VNTs, but could be either e.g. NamedTuples or just
        # generic structs. In this case, it can be type unstable to create a PartialArray
        # from the template and then setindex!! a VNT into it. So, we create a new template
        # with the appropriate type here before creating the PartialArray.
        similar(template, correct_template_eltype)
    else
        similar(template)
    end
    pa_mask = similar(pa_data, Bool)
    fill!(pa_mask, false)
    pa = PartialArray(pa_data, pa_mask)
    return BangBang.setindex!!(pa, sub_value, coptic.ix...; coptic.kw...)
end
function make_leaf(value, optic::AbstractPPL.Index, template::PartialArray)
    return make_leaf(value, optic, template.data)
end
