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
@inline _getindex_optic(x::Any, o::AbstractPPL.AbstractOptic) = o(x)
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
@inline _haskey_optic(x::Any, o::AbstractPPL.AbstractOptic) = AbstractPPL.canview(o, x)
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
    _setindex_optic!!(collection, value, optic, template; allow_new=Val(true))

Like `setindex!!`, but special-cased for `VarNamedTuple` and `PartialArray` to recurse
into nested structures.

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
    ;
    allow_new=Val(true),
)
    return value
end
function _setindex_optic!!(
    arr::AbstractArray, value, optic::IndexWithoutChild, template; allow_new=Val(true)
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
# TODO(penelopeysm): The check on hasproperty can be type unstable! This avoids erroring
# when an incorrect template is passed (e.g. trying to set `x.a.b` on something that doesn't
# have a field `a`) -- but is it worth it?
(::SharedGetProperty{S})(x) where {S} = hasproperty(x, S) ? getproperty(x, S) : NoTemplate()
(::SharedGetProperty)(::NoTemplate) = NoTemplate()
(::SharedGetProperty)(t::SkipTemplate{N}) where {N} = decrease_skip(t)
# Unsure if this needs to be a generated function, but it's not too complex.
@generated function (::SharedGetProperty{S})(x::VarNamedTuple{names}) where {S,names}
    return if S in names
        :(x.data.$S)
    else
        :(NoTemplate())
    end
end

function _setindex_optic!!(
    pa::PartialArray, value, optic::AbstractPPL.Index, template; allow_new=Val(true)
)
    need_merge = false
    coptic = AbstractPPL.concretize_top_level(optic, pa.data)
    # We might be attempting to set a slice into a PartialArray{GrowableArray} that
    # doesn't yet have enough indices for that slice. Expand it if so.
    pa = grow_to_indices!!(pa, coptic.ix...; coptic.kw...)

    is_multiindex = if template isa AbstractArray || template isa PartialArray
        _is_multiindex(template, coptic.ix...; coptic.kw...)
    else
        isempty(coptic.kw) || throw_kw_error()
        _is_multiindex_static(coptic.ix)
    end

    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        child_template = if template === NoTemplate()
            NoTemplate()
        elseif template isa SkipTemplate
            decrease_skip(template)
        elseif is_multiindex
            Base.getindex(template, coptic.ix...; coptic.kw...)
        elseif isassigned(template, coptic.ix...; coptic.kw...)
            # Single-index, but we should check that there is actual data there before
            # calling getindex, as otherwise it will error
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
                child_template;
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
                # MERGE the new leaf into the existing PartialArray instead of overwriting
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

    # If sub_value is a GrowableArray, we need to make sure it is grown to the right size to
    # fit into the indices specified by `coptic`. This is the same logic as in
    # `make_leaf_multiindex`. Again, if there is no GrowableArray underpinning sub_value
    # then grow_to_indices!! is a no-op so won't hurt.
    grown_sub_value = if is_multiindex && sub_value isa PartialArray
        grow_to_indices!!(sub_value, coptic.ix...; coptic.kw...)
    else
        sub_value
    end

    return if need_merge
        new_pa = BangBang.setindex!!(copy(pa), grown_sub_value, coptic.ix...; coptic.kw...)
        _merge_norecurse(pa, new_pa)
    else
        BangBang.setindex!!(pa, grown_sub_value, coptic.ix...; coptic.kw...)
    end
end

function _setindex_optic!!(
    vnt::VarNamedTuple{names},
    value,
    optic::AbstractPPL.Property{S},
    template;
    allow_new=Val(true),
) where {names,S}
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    else
        child_template = SharedGetProperty{S}()(template)
        if S in names
            # Data already exists; we need to recurse into it
            _setindex_optic!!(
                vnt.data[S], value, optic.child, child_template; allow_new=allow_new
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

A lot of the complexity in the VarNamedTuple code comes from this function. For this reason,
it is heavily documented, not only with comments in the code, but also in the built
DynamicPPL documentation. If you are modifying this function, please read it!

https://turinglang.org/DynamicPPL.jl/stable/vnt/implementation/
"""
@inline make_leaf(@nospecialize(value::Any), ::AbstractPPL.Iden, ::Any) = value
function make_leaf(value, optic::AbstractPPL.Property{S}, template) where {S}
    sub_value = if optic.child isa AbstractPPL.Iden
        value
    else
        child_template = SharedGetProperty{S}()(template)
        make_leaf(value, optic.child, child_template)
    end
    return VarNamedTuple(NamedTuple{(S,)}((sub_value,)))
end

# This function handles Index optics. Since Index optics can represent either single-element
# indexing or multi-element indexing (e.g., slices, arrays of indices), and their
# implementation can be somewhat different, we dispatch to separate functions for each case.
function make_leaf(value, optic::AbstractPPL.Index, template::PartialArray)
    # If the template is a PA, use its data as the template.
    return make_leaf(value, optic, template.data)
end
function make_leaf(value, optic::AbstractPPL.Index, template)
    # First we need to resolve any dynamic indices, since _is_multiindex doesn't work with
    # them. This also helpfully catches errors if there is a dynamic index and a suitable
    # template is not provided (e.g., if someone tries to set `x[end]` without a template).
    coptic = AbstractPPL.concretize_top_level(optic, template)
    is_multiindex = if template isa AbstractArray || template isa PartialArray
        _is_multiindex(template, coptic.ix...; coptic.kw...)
    else
        # This handles the case where no template is provided, or a nonsense template is
        # provided.
        isempty(coptic.kw) || throw_kw_error()
        # This will error if there are things like colons.
        _is_multiindex_static(coptic.ix)
    end
    return if is_multiindex
        make_leaf_multiindex(value, coptic, template)
    else
        make_leaf_singleindex(value, coptic, template)
    end
end

function make_sub_value(value, coptic::AbstractPPL.Index, template)
    return if coptic.child isa AbstractPPL.Iden
        value
    else
        child_template = if template isa NoTemplate
            NoTemplate()
        elseif template isa SkipTemplate
            decrease_skip(template)
        elseif template isa AbstractArray
            # Note template can't be a PartialArray as that would have been unwrapped in
            # another make_leaf method.
            getindex(template, coptic.ix...; coptic.kw...)
        else
            NoTemplate()
        end
        make_leaf(value, coptic.child, child_template)
    end
end

# This is the easy case, and is directly analogous to the Property optic case above.
# The optic is called `coptic` to indicate that it has already been concretised
# by the caller.
function make_leaf_singleindex(value, coptic::AbstractPPL.Index, template)
    sub_value = make_sub_value(value, coptic, template)
    # `sub_value` will become a single element inside the PartialArray that we create. To
    # ensure type stability, we want to make sure that the PartialArray is created with the
    # correct eltype to begin with, otherwise setindex!! may become type unstable.
    pa_eltype = typeof(sub_value)
    pa_data = if template isa AbstractArray
        if pa_eltype <: eltype(template)
            # Reuse the eltype of template: it might be more abstract than sub_value_type,
            # but that's fine since it will mimic the semantics of what the user passed in.
            similar(template)
        else
            # Use the eltype of sub_value.
            similar(template, pa_eltype)
        end
    else
        # If no template was provided, or an incorrectly typed template was provided, we
        # have to make a GrowableArray. No need to use `kw` since make_leaf already errors
        # if there are uninterpretable keyword indices.
        template_sz = get_maximum_size_from_indices(coptic.ix...)
        _warn_growable_array_creation(template_sz)
        GrowableArray(Array{pa_eltype}(undef, template_sz))
    end
    pa_mask = similar(pa_data, Bool)
    fill!(pa_mask, false)
    pa = PartialArray(pa_data, pa_mask)
    return BangBang.setindex!!(pa, sub_value, coptic.ix...; coptic.kw...)
end

# This is more complex.
function make_leaf_multiindex(value, coptic::AbstractPPL.Index, template)
    sub_value = make_sub_value(value, coptic, template)

    # Firstly, we need to make sure that sub_value has *exactly* the right size to fit into
    # the indices specified by `coptic`. This might not always be the case. Consider
    # _[2:3][1] without a template -- the inner `make_leaf` call will create a GrowableArray
    # of size 1 (because that's the minimum size it infers from the inner indices), but we
    # actually need a slice of length 2. (If there's a template, we don't have this problem,
    # because the inner make_leaf will use the template to set the correct size of 2.) Note
    # that `grow_to_indices!!` is a no-op for any PartialArray that doesn't contain a
    # GrowableArray.
    grown_sub_value = if sub_value isa PartialArray
        grow_to_indices!!(sub_value, coptic.ix...; coptic.kw...)
    else
        sub_value
    end

    # `sub_value` is not just one element, but is actually a slice inside the PartialArray
    # that we create. To add to that, it might not just be a slice, but it might be an
    # ArrayLikeBlock that is set at multiple indices.
    pa_eltype = if sub_value isa AbstractArray || sub_value isa PartialArray
        eltype(sub_value)
    else
        ArrayLikeBlock{
            typeof(sub_value),
            typeof(coptic.ix),
            typeof(coptic.kw),
            typeof(vnt_size(value)),
        }
    end

    # The rest is the same as the single-index case.
    pa_data = if template isa AbstractArray
        if pa_eltype <: eltype(template)
            similar(template)
        else
            similar(template, pa_eltype)
        end
    else
        # No template, or incorrectly typed template
        template_sz = get_maximum_size_from_indices(coptic.ix...)
        _warn_growable_array_creation(template_sz)
        GrowableArray(Array{pa_eltype}(undef, template_sz))
    end
    pa_mask = similar(pa_data, Bool)
    fill!(pa_mask, false)
    pa = PartialArray(pa_data, pa_mask)
    return BangBang.setindex!!(pa, grown_sub_value, coptic.ix...; coptic.kw...)
end
