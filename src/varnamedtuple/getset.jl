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
    return _getindex_optic(Base.getindex(pa, optic.ix...; optic.kw...), optic.child)
end
function _getindex_optic(arr::AbstractArray, optic::IndexWithoutChild)
    return Base.getindex(arr, optic.ix...; optic.kw...)
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
    _setindex_optic!!(collection, value, key; allow_new=Val(true))

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
function _setindex_optic!!(vnt::VarNamedTuple, value, name::VarName; allow_new=Val(true))
    return _setindex_optic!!(
        vnt, value, AbstractPPL.varname_to_optic(name); allow_new=allow_new
    )
end
@inline function _setindex_optic!!(
    @nospecialize(::Any), value, ::AbstractPPL.Iden; allow_new=Val(true)
)
    return value
end
function _setindex_optic!!(
    arr::AbstractArray, value, optic::IndexWithoutChild; allow_new=Val(true)
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

function _setindex_optic!!(
    pa::PartialArray, value, optic::AbstractPPL.Index; allow_new=Val(true)
)
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    elseif Base.haskey(pa, optic.ix...; optic.kw...)
        # Data already exists; we need to recurse into it
        _setindex_optic!!(
            Base.getindex(pa, optic.ix...; optic.kw...),
            value,
            optic.child;
            allow_new=allow_new,
        )
    elseif allow_new isa Val{true}
        # No new data but we are allowed to create it.
        make_leaf(value, optic.child)
    else
        throw_setindex_allow_new_error()
    end
    return BangBang.setindex!!(pa, sub_value, optic.ix...; optic.kw...)
end

function _setindex_optic!!(
    vnt::VarNamedTuple, value, optic::AbstractPPL.Property{S}; allow_new=Val(true)
) where {S}
    sub_value = if optic.child isa AbstractPPL.Iden
        # Skip recursion
        value
    elseif Base.haskey(vnt.data, S)
        # Data already exists; we need to recurse into it
        _setindex_optic!!(vnt.data[S], value, optic.child; allow_new=allow_new)
    elseif allow_new isa Val{true}
        # No new data but we are allowed to create it.
        make_leaf(value, optic.child)
    else
        # If this branch is ever reached, then someone has used allow_new=Val(false)
        # incorrectly.
        error("""
            _setindex_optic was called with allow_new=Val(false) but the key does not exist.
            This indicates a bug in DynamicPPL: Please file an issue on GitHub.""")
    end
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((sub_value,))))
end

"""
    make_leaf(value, optic)

Make a new leaf node for a VarNamedTuple.

This is the function that sets any `optic` that is a `Property` to be stored as a
`VarNamedTuple`, any `Index` to be stored as a `PartialArray`, and other `Iden` optics to be
stored as raw values. It is the link that joins `VarNamedTuple` and `PartialArray` together.
"""
@inline make_leaf(@nospecialize(value::Any), ::AbstractPPL.Iden) = value
function make_leaf(value, optic::AbstractPPL.Property{S}) where {S}
    sub_value = make_leaf(value, optic.child)
    return VarNamedTuple(NamedTuple{(S,)}((sub_value,)))
end
function make_leaf(value, optic::AbstractPPL.Index)
    isempty(optic.kw) || error_kw_indices()
    sub_value = make_leaf(value, optic.child)
    inds = optic.ix
    num_inds = length(inds)
    # The element type of the PartialArray depends on whether we are setting a single value
    # or a range of values.
    et = if !_is_multiindex(inds)
        typeof(sub_value)
    elseif _needs_arraylikeblock(sub_value, inds...)
        ArrayLikeBlock{typeof(sub_value),typeof(inds)}
    else
        eltype(sub_value)
    end
    pa = PartialArray{et,num_inds}()
    return BangBang.setindex!!(pa, sub_value, optic.ix...; optic.kw...)
end
