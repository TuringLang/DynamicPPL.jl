# We define our own getindex, setindex!!, and haskey functions, which we use to
# get/set/check values in VarNamedTuple and PartialArray. We do this because we want to be
# able to override their behaviour for some types exported from elsewhere without type
# piracy. This is needed because
# 1. We would want to index into things with lenses (from AbstractPPL.jl) using getindex and
# setindex!!, but AbstractPPL does not define these methods.
# 2. We would want `haskey` to fall back onto `checkbounds` when called on Base.Arrays.
#
# The difference between _getindex_optic and _getindex is that the former takes an optic
# as the second argument, whereas the latter takes indices. The latter is therefore closer
# in spirit to Base.getindex, whereas the former is more like AbstractPPL.getvalue.
function _getindex_optic end
function _getindex end

function _haskey end

# In many places, we don't yet know how to cleverly handle keyword indices. Sometimes we can
# forward them to `Base.getindex`, but often we can't, unless we adopt a shadow-array
# mechanism. See https://github.com/TuringLang/DynamicPPL.jl/issues/1194.
function _error_if_kw_indices(i::AbstractPPL.Index)
    if !isempty(i.kw)
        throw(
            ArgumentError("Keyword indices in VarName are not yet supported in DynamicPPL.")
        )
    end
    return nothing
end

# When we have reached the bottom of the VNT i.e. we are only left with an AbstractArray
# rather than some flavour of PartialArray, we don't know how to further handle any child
# optics -- hence the type bound on `optic`.
const IndexWithoutChild = AbstractPPL.Index{<:Tuple,<:NamedTuple,AbstractPPL.Iden}
function _getindex_optic(arr::AbstractArray, optic::IndexWithoutChild)
    return getindex(arr, optic.ix...; optic.kw...)
end
_haskey(arr::AbstractArray, optic::IndexWithoutChild) = _haskey(arr, optic.ix; optic.kw...)
function _haskey(arr::AbstractArray, ix...; kw...)
    # Note that this call to `checkbounds` can error, although it is technically out of our
    # hands: it depends on how the provider of the AbstractArray has implemented
    # checkbounds. For example, DimArray can error here:
    # https://github.com/rafaqz/DimensionalData.jl/issues/1156. But that is not our job to fix
    # -- it should be done upstream -- hence we just forward the indices.
    return checkbounds(Bool, arr, ix...; kw...)
end
function _setindex_optic!!(
    arr::AbstractArray, value, optic::IndexWithoutChild; allow_new=Val(true)
)
    # See comment in _getindex_optic above as to the type bound on `optic`.
    return setindex!!(arr, value, optic.ix...; optic.kw...)
end

"""
    _setindex!!(collection, value, key; allow_new=Val(true))

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
function _setindex!! end
function _setindex_optic!! end

function _getindex_optic(vnt::VarNamedTuple, vn::VarName)
    return _getindex_optic(vnt, AbstractPPL.varname_to_optic(vn))
end
function _getindex_optic(vnt::VarNamedTuple, optic::AbstractPPL.Property{S}) where {S}
    return _getindex_optic(getindex(vnt.data, S), optic.child)
end

function _haskey(vnt::VarNamedTuple, name::VarName)
    return _haskey(vnt, AbstractPPL.varname_to_optic(name))
end

function _setindex_optic!!(vnt::VarNamedTuple, value, name::VarName; allow_new=Val(true))
    return _setindex_optic!!(
        vnt, value, AbstractPPL.varname_to_optic(name); allow_new=allow_new
    )
end

function _setindex_optic!!(
    vnt::VarNamedTuple, value, optic::AbstractPPL.Property{S}; allow_new=Val(true)
) where {S}
    sub_value = if haskey(vnt.data, S)
        # Data already exists; we need to recurse into it
        _setindex_optic!!(vnt.data[S], value, optic.child; allow_new=allow_new)
    elseif allow_new isa Val{true}
        # No new data but we are allowed to create it.
        make_leaf(value, optic.outer)
    else
        # If this branch is ever reached, then someone has used allow_new=Val(false)
        # incorrectly.
        error("""
            _setindex_optic was called with allow_new=Val(false) but the key does not exist.
            This indicates a bug in DynamicPPL: Please file an issue on GitHub.""")
    end
    return VarNamedTuple(merge(vnt.data, NamedTuple{(S,)}((sub_value,))))
end

function _haskey(vnt::Union{PartialArray,VarNamedTuple}, optic::ComposedFunction)
    return _haskey(vnt, optic.inner) &&
           _haskey(_getindex_optic(vnt, optic.inner), optic.outer)
end

# The entry points for getting, setting, and checking, using the familiar functions.
Base.haskey(vnt::VarNamedTuple, vn::VarName) = _haskey(vnt, vn)

# PartialArrays are an implementation detail of VarNamedTuple, and should never be the
# return value of getindex. Thus, we automatically convert them to dense arrays if needed.
# TODO(mhauru) The below doesn't handle nested PartialArrays. Is that a problem?
_dense_array_if_needed(pa::PartialArray) = _dense_array(pa)
_dense_array_if_needed(x) = x
function Base.getindex(vnt::VarNamedTuple, vn::VarName)
    return _dense_array_if_needed(_getindex_optic(vnt, vn))
end

BangBang.setindex!!(vnt::VarNamedTuple, value, vn::VarName) = _setindex!!(vnt, value, vn)

Base.haskey(pa::PartialArray, key) = _haskey(pa, key)
Base.getindex(pa::PartialArray, inds...) = _getindex(pa, inds...)
BangBang.setindex!!(pa::PartialArray, value, inds...) = _setindex!!(pa, value, inds...)

"""
    make_leaf(value, optic)

Make a new leaf node for a VarNamedTuple.

This is the function that sets any `optic` that is a `Property` to be stored as a
`VarNamedTuple`, any `Index` to be stored as a `PartialArray`, and other `Iden` optics to be
stored as raw values. It is the link that joins `VarNamedTuple` and `PartialArray` together.
"""
make_leaf(value, ::AbstractPPL.Iden) = value
function make_leaf(value, optic::AbstractPPL.Property{S}) where {S}
    sub = VarNamedTuple(NamedTuple{(S,)}((value,)))
    return make_leaf(sub, optic.child)
end
function make_leaf(value, optic::AbstractPPL.Index)
    _error_if_kw_indices(optic)
    inds = optic.ix
    num_inds = length(inds)
    # The element type of the PartialArray depends on whether we are setting a single value
    # or a range of values.
    et = if !_is_multiindex(inds)
        typeof(value)
    elseif _needs_arraylikeblock(value, inds...)
        ArrayLikeBlock{typeof(value),typeof(inds)}
    else
        eltype(value)
    end
    pa = PartialArray{et,num_inds}()
    sub = _setindex_optic!!(pa, value, optic)
    return make_leaf(sub, optic.child)
end
