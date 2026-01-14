"""
    VarInfo{T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo

The default implementation of `AbstractVarInfo`, storing variable values and accumulators.

`VarInfo` is quite a thin wrapper around a `VarNamedTuple` storing the variable values,
and a tuple of accumulators. The only really noteworthy thing about it is that it stores
the values of variables vectorised as instances of `TransformedValue`. That is, it stores
each value as a vector and a transformation to be applied to that vector to get the actual
value. It also stores whether the transformation is such that it guarantees all real vectors
to be valid internal representations of the variable (i.e., whether the variable has been
linked), as well as the size of the actual post-transformation value. These are all fields
of [`TransformedValue`](@ref).

Note that `setindex!!` and `getindex` on `VarInfo` take and return values in the support of
the original distribution. To get access to the internal vectorised values, use
[`getindex_internal`](@ref), [`setindex_internal!!`](@ref), and [`unflatten!!`](@ref).

There's also a `VarInfo`-specific function [`setindex_with_dist!!`](@ref), which sets a
variable's value with a transformation based on the statistical distribution this value is
a sample for.

For more details on the internal storage, see documentation of [`TransformedValue`](@ref) and
[`VarNamedTuple`](@ref).

# Fields
$(TYPEDFIELDS)

"""
struct VarInfo{T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo
    values::T
    accs::Accs
end

# TODO(mhauru) The policy of vectorising all values was set when the old VarInfo type was
# using a Vector as the internal storage in all cases. We should revisit this, and allow
# values to be stored "raw", since VarNamedTuple supports it.

# TODO(mhauru) Related to the above, I think we should reconsider whether we should store
# transformations at all. We rarely use them, since they may be dynamic in a model.
# tilde_assume!! rather gets the transformation from the current distribution encountered
# during model execution. However, this would change the interface quite a lot, so I want to
# finish implementing VarInfo using VNT (mostly) respecting the old interface first.

"""
    TransformedValue{ValType,TransformType,SizeType}

A struct for storing a variable's value in its internal (vectorised) form.

# Fields
$(TYPEDFIELDS)
"""
struct TransformedValue{ValType,TransformType,SizeType}
    "The internal (vectorised) value."
    val::ValType
    """Boolean indicating whether the variable is linked, i.e. the transformation maps all
    real vectors to valid values."""
    linked::Bool
    """The transformation from internal (vectorised) to actual value. In other words, the
    actual value of the variable being stored is `transform(val)`."""
    transform::TransformType
    """The size of the actual value after transformation. This is needed when a
    `TransformedValue` is stored as a block in an array."""
    size::SizeType
end

VarNamedTuples.vnt_size(tv::TransformedValue) = tv.size

VarInfo() = VarInfo(VarNamedTuple(), default_accumulators())

function VarInfo(values::Union{NamedTuple,AbstractDict})
    vi = VarInfo()
    for (k, v) in pairs(values)
        vn = k isa Symbol ? VarName{k}() : k
        vi = setindex!!(vi, v, vn)
    end
    return vi
end

function VarInfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return VarInfo(Random.default_rng(), model, init_strategy)
end

function VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return last(init!!(rng, model, VarInfo(), init_strategy))
end

getaccs(vi::VarInfo) = vi.accs
setaccs!!(vi::VarInfo, accs::AccumulatorTuple) = VarInfo(vi.values, accs)

transformation(::VarInfo) = DynamicTransformation()

Base.copy(vi::VarInfo) = VarInfo(copy(vi.values), copy(getaccs(vi)))
Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.values, vn)
Base.length(vi::VarInfo) = length(vi.values)
Base.keys(vi::VarInfo) = keys(vi.values)
Base.values(vi::VarInfo) = mapreduce(p -> p.second.val, push!, vi.values; init=Any[])

function Base.getindex(vi::VarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.transform(tv.val)
end

function Base.getindex(vi::VarInfo, vns::AbstractVector{<:VarName})
    return [getindex(vi, vn) for vn in vns]
end

Base.isempty(vi::VarInfo) = isempty(vi.values)
Base.empty(vi::VarInfo) = VarInfo(empty(vi.values), map(reset, vi.accs))
BangBang.empty!!(vi::VarInfo) = VarInfo(empty!!(vi.values), map(reset, vi.accs))

"""
    setindex_internal!!(vi::VarInfo, val, vn::VarName)

Set the internal (vectorised) value of variable `vn` in `vi` to `val`.

This does not change the transformation or linked status of the variable.
"""
function setindex_internal!!(vi::VarInfo, val, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(val, old_tv.linked, old_tv.transform, old_tv.size)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo(new_values, vi.accs)
end

# TODO(mhauru) It shouldn't really be VarInfo's business to know about `dist`. However,
# we need `dist` to determine the linking transformation (or even just the vectorisation
# transformation in the case of ProductNamedTupleDistribions), and if we leave the work
# of doing the transformation to the caller (tilde_assume!!), it'll be done even when e.g.
# using OnlyAccsVarInfo. Hence having this function. It should eventually hopefully be
# removed once VAIMAcc is the only way to get values out of an evaluation.
"""
    setindex_with_dist!!(vi::VarInfo, val, dist::Distribution, vn::VarName)

Set the value of `vn` in `vi` to `val`, applying a transformation based on `dist`.

`val` is taken to be the actual value of the variable, and is transformed into the internal
(vectorised) representation using a transformation based on `dist`. If the variable is
currently linked in `vi`, or doesn't exist in `vi` but all other variables in `vi` are
linked, the linking transformation is used; otherwise, the standard vector transformation is
used.

Returns the modified `vi` together with the log absolute determinant of the Jacobian of the
transformation applied.
"""
function setindex_with_dist!!(vi::VarInfo, val, dist::Distribution, vn::VarName)
    link = haskey(vi, vn) ? is_transformed(vi, vn) : is_transformed(vi)
    transform = if link
        from_linked_vec_transform(dist)
    else
        from_vec_transform(dist)
    end
    transformed_val, logjac = with_logabsdet_jacobian(inverse(transform), val)
    # All values for which `size` is not defined are assumed to be scalars.
    val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    tv = TransformedValue(transformed_val, link, transform, val_size)
    vi = VarInfo(setindex!!(vi.values, tv, vn), vi.accs)
    return vi, logjac
end

# TODO(mhauru) The below is somewhat unsafe or incomplete: For instance, from_vec_transform
# isn't defined for NamedTuples. However, this is needed in some places where values for
# in a VarInfo are set outside the context of a `tilde_assume!!` and no distribution is
# available. Hopefully we'll get rid of this eventually.
"""
    setindex!!(vi::VarInfo, val, vn::VarName)

Set the value of `vn` in `vi` to `val`.

The transformation for `vn` is reset to be the standard vector transformation for values of
the type of `val` and linking status is set to false.
"""
function BangBang.setindex!!(vi::VarInfo, val, vn::VarName)
    transform = from_vec_transform(val)
    transformed_val = inverse(transform)(val)
    tv = TransformedValue(transformed_val, false, transform, size(val))
    return VarInfo(setindex!!(vi.values, tv, vn), vi.accs)
end

"""
    set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)

Set the linked status of variable `vn` in `vi` to `linked`.

This does not change the value or transformation of the variable.
"""
function set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(old_tv.val, linked, old_tv.transform, old_tv.size)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo(new_values, vi.accs)
end

# VarInfo does not care whether the transformation was Static or Dynamic, it just tracks
# whether one was applied at all.
function set_transformed!!(vi::VarInfo, ::AbstractTransformation, vn::VarName)
    return set_transformed!!(vi, true, vn)
end

set_transformed!!(vi::VarInfo, ::AbstractTransformation) = set_transformed!!(vi, true)

function set_transformed!!(vi::VarInfo, ::NoTransformation, vn::VarName)
    return set_transformed!!(vi, false, vn)
end

set_transformed!!(vi::VarInfo, ::NoTransformation) = set_transformed!!(vi, false)

function set_transformed!!(vi::VarInfo, linked::Bool)
    new_values = map_values!!(vi.values) do tv
        TransformedValue(tv.val, linked, tv.transform, tv.size)
    end
    return VarInfo(new_values, vi.accs)
end

"""
    getindex_internal(vi::VarInfo, vn::VarName)

Get the internal (vectorised) value of variable `vn` in `vi`.
"""
getindex_internal(vi::VarInfo, vn::VarName) = getindex(vi.values, vn).val
# TODO(mhauru) The below should be removed together with unflatten!!.
getindex_internal(vi::VarInfo, ::Colon) = values_as(vi, Vector)

is_transformed(vi::VarInfo, vn::VarName) = getindex(vi.values, vn).linked

function from_internal_transform(::VarInfo, ::VarName, dist::Distribution)
    return from_vec_transform(dist)
end

function from_linked_internal_transform(::VarInfo, ::VarName, dist::Distribution)
    return from_linked_vec_transform(dist)
end

function from_internal_transform(vi::VarInfo, vn::VarName)
    return getindex(vi.values, vn).transform
end

function from_linked_internal_transform(vi::VarInfo, vn::VarName)
    if !is_transformed(vi, vn)
        error("Variable $vn is not linked; cannot get linked transformation.")
    end
    return getindex(vi.values, vn).transform
end

"""
    _link_or_invlink!!(vi::VarInfo, vns, model::Model, ::Val{link}) where {link isa Bool}

The internal function that implements both link!! and invlink!!.

The last argument controls whether linking (true) or invlinking (false) is performed. If
`vns` is `nothing`, all variables in `vi` are transformed; otherwise, only the variables
in `vns` are transformed. Existing variables already in the desired state are left
unchanged.
"""
function _link_or_invlink!!(vi::VarInfo, vns, model::Model, ::Val{link}) where {link}
    @assert link isa Bool
    # Note that extract_priors causes a model execution. In the past with the Metadata-based
    # VarInfo we rather derived the transformations from the distributions stored in the
    # VarInfo itself. However, that is not fail-safe with dynamic models, and would require
    # storing the distributions in TransformedValue (which we could start doing). Instead we
    # use extract_priors to get the current, correct transformations. This logic is very
    # similar to what DynamicTransformation used to do, and we might replace this with a
    # context that transforms each variable in turn during the execution.
    dists = extract_priors(model, vi)
    cumulative_logjac = zero(LogProbType)
    new_values = map_pairs!!(vi.values) do pair
        vn, tv = pair
        if vns !== nothing && !any(x -> subsumes(x, vn), vns)
            # Not one of the target variables.
            return tv
        end
        if tv.linked == link
            # Already in the desired state.
            return tv
        end
        dist = getindex(dists, vn)
        vec_transform = from_vec_transform(dist)
        link_transform = from_linked_vec_transform(dist)
        current_transform, new_transform = if link
            (vec_transform, link_transform)
        else
            (link_transform, vec_transform)
        end
        val_untransformed, logjac1 = with_logabsdet_jacobian(current_transform, tv.val)
        val_new, logjac2 = with_logabsdet_jacobian(
            inverse(new_transform), val_untransformed
        )
        new_tv = TransformedValue(val_new, link, new_transform, tv.size)
        cumulative_logjac += logjac1 + logjac2
        return new_tv
    end
    vi = VarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function link!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return _link_or_invlink!!(vi, vns, model, Val(true))
end
function link!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return _link_or_invlink!!(vi, nothing, model, Val(true))
end
function invlink!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return _link_or_invlink!!(vi, vns, model, Val(false))
end
function invlink!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return _link_or_invlink!!(vi, nothing, model, Val(false))
end

function link!!(t::StaticTransformation{<:Bijectors.Transform}, vi::VarInfo, ::Model)
    # TODO(mhauru) This assumes that the user has defined the bijector using the same
    # variable ordering as what `vi[:]` and `unflatten!!(vi, x)` use. This is a bad user
    # interface.
    b = inverse(t.bijector)
    x = vi[:]
    y, logjac = with_logabsdet_jacobian(b, x)
    # TODO(mhauru) This doesn't set the transforms of `vi`. With the old Metadata that meant
    # that getindex(vi, vn) would apply the default link transform of the distribution. With
    # the new VarNamedTuple-based VarInfo it means that getindex(vi, vn) won't apply any
    # link transform. Neither is correct, rather the transform should be the inverse of b.
    vi = unflatten!!(vi, y)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, logjac)
    end
    return set_transformed!!(vi, t)
end

function invlink!!(t::StaticTransformation{<:Bijectors.Transform}, vi::VarInfo, ::Model)
    b = t.bijector
    y = vi[:]
    x, inv_logjac = with_logabsdet_jacobian(b, y)

    # Mildly confusing: we need to _add_ the logjac of the inverse transform,
    # because we are trying to remove the logjac of the forward transform
    # that was previously accumulated when linking.
    vi = unflatten!!(vi, x)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, inv_logjac)
    end
    return set_transformed!!(vi, NoTransformation())
end

# TODO(mhauru) I don't think this should return the internal values, but that's the current
# convention.
function values_as(vi::VarInfo, ::Type{Vector})
    return mapfoldl(pair -> tovec(pair.second.val), vcat, vi.values; init=Union{}[])
end

function values_as(vi::VarInfo, ::Type{T}) where {T<:AbstractDict}
    return mapfoldl(identity, function (cumulant, pair)
        vn, tv = pair
        val = tv.transform(tv.val)
        return setindex!!(cumulant, val, vn)
    end, vi.values; init=T())
end

# TODO(mhauru) I really dislike this sort of conversion to Symbols, but it's the current
# interface provided by rand(::Model). We should change that to return a VarNamedTuple
# instead, and then this method (and any other values_as methods for NamedTuple) could be
# removed.
function values_as(vi::VarInfo, ::Type{NamedTuple})
    return mapfoldl(
        identity,
        function (cumulant, pair)
            vn, tv = pair
            val = tv.transform(tv.val)
            return setindex!!(cumulant, val, Symbol(vn))
        end,
        vi.values;
        init=NamedTuple(),
    )
end

"""
    VectorChunkIterator{T<:AbstractVector}

A tiny struct for getting chunks of a vector sequentially.

The only function provided is `get_next_chunk!`, which takes a length and returns
a view into the next chunk of that length, updating the internal index.
"""
mutable struct VectorChunkIterator{T<:AbstractVector}
    vec::T
    index::Int
end

function get_next_chunk!(vci::VectorChunkIterator, len::Int)
    i = vci.index
    chunk = @view vci.vec[i:(i + len - 1)]
    vci.index += len
    return chunk
end

function unflatten!!(vi::VarInfo, vec::AbstractVector)
    # You may wonder, why have a whole struct for this, rather than just an index variable
    # that the mapping function would close over. I wonder too. But for some reason type
    # inference fails on such an index variable, turning it into a Core.Box.
    vci = VectorChunkIterator(vec, 1)
    new_values = map_values!!(vi.values) do tv
        old_val = tv.val
        if !(old_val isa AbstractVector)
            error(
                "Can't unflatten a VarInfo for which existing values are not vectors:" *
                " Got value of type $(typeof(old_val)).",
            )
        end
        len = length(old_val)
        new_val = get_next_chunk!(vci, len)
        return TransformedValue(new_val, tv.linked, tv.transform, tv.size)
    end
    return VarInfo(new_values, vi.accs)
end

"""
    subset(varinfo::VarInfo, vns)

Create a new `VarInfo` containing only the variables in `vns`.

`vns` can be almost any collection of `VarName`s, e.g. a `Set`, `Vector`, or `Tuple`.
"""
function subset(varinfo::VarInfo, vns)
    new_values = subset(varinfo.values, vns)
    return VarInfo(new_values, map(copy, getaccs(varinfo)))
end

"""
    merge(varinfo_left::VarInfo, varinfo_right::VarInfo)

Merge two `VarInfo`s into a new `VarInfo` containing all variables from both.

The accumulators are taken exclusively from `varinfo_right`.

If a variable exists in both `varinfo_left` and `varinfo_right`, the value from
`varinfo_right` is used.
"""
function Base.merge(varinfo_left::VarInfo, varinfo_right::VarInfo)
    new_values = merge(varinfo_left.values, varinfo_right.values)
    new_accs = map(copy, getaccs(varinfo_right))
    return VarInfo(new_values, new_accs)
end
