struct VNTVarInfo{T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo
    values::T
    accs::Accs
end

# TODO(mhauru) Make this renaming permanent.
const VarInfo = VNTVarInfo

struct TransformedValue{ValType,TransformType,SizeType}
    val::ValType
    linked::Bool
    transform::TransformType
    size::SizeType
end

VarNamedTuples.vnt_size(tv::TransformedValue) = tv.size

VNTVarInfo() = VNTVarInfo(VarNamedTuple(), default_accumulators())

function VNTVarInfo(values::Union{NamedTuple,AbstractDict})
    vi = VarInfo()
    for (k, v) in pairs(values)
        vn = k isa Symbol ? VarName{k}() : k
        vi = setindex!!(vi, v, vn)
    end
    return vi
end

function VNTVarInfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return VNTVarInfo(Random.default_rng(), model, init_strategy)
end

function VNTVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return last(init!!(rng, model, VNTVarInfo(), init_strategy))
end

getaccs(vi::VNTVarInfo) = vi.accs
setaccs!!(vi::VNTVarInfo, accs::AccumulatorTuple) = VNTVarInfo(vi.values, accs)

transformation(::VNTVarInfo) = DynamicTransformation()

Base.copy(vi::VNTVarInfo) = VNTVarInfo(copy(vi.values), copy(getaccs(vi)))

Base.haskey(vi::VNTVarInfo, vn::VarName) = haskey(vi.values, vn)

Base.length(vi::VNTVarInfo) = length(vi.values)

function Base.getindex(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.transform(tv.val)
end

function Base.getindex(vi::VNTVarInfo, vns::Vector{<:VarName})
    return [getindex(vi, vn) for vn in vns]
end

function Base.getindex(vi::VNTVarInfo, vn::VarName, dist::Distribution)
    val = getindex_internal(vi, vn)
    return from_maybe_linked_internal(vi, vn, dist, val)
end

Base.isempty(vi::VNTVarInfo) = isempty(vi.values)
Base.empty(vi::VNTVarInfo) = VNTVarInfo(empty(vi.values), map(reset, vi.accs))
BangBang.empty!!(vi::VNTVarInfo) = VNTVarInfo(empty!!(vi.values), map(reset, vi.accs))

function setindex_internal!!(vi::VNTVarInfo, val, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(val, old_tv.linked, old_tv.transform, old_tv.size)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VNTVarInfo(new_values, vi.accs)
end

# TODO(mhauru) It shouldn't really be VarInfo's business to know about `dist`. However,
# we need `dist` to determine the linking transformation (or even just the vectorisation
# transformation, in the case of ProductNamedTupleDistribions), and if we leave the work
# of doing the transformation to the caller, it'll be done even when e.g. using
# OnlyAccsVarInfo. Hence having this function. It should eventually hopefully be removed
# once VAIMAcc is the only way to get values out of an evaluation.
function setindex_with_dist!!(vi::VNTVarInfo, val, dist::Distribution, vn::VarName)
    # Determine whether to insert a transformed value into `vi`.
    # If the VarInfo alrady had a value for this variable, we will
    # keep the same linked status as in the original VarInfo. If not, we
    # check the rest of the VarInfo to see if other variables are linked.
    # is_transformed(vi) returns true if vi is nonempty and all variables in vi
    # are linked.
    insert_transformed_value = haskey(vi, vn) ? is_transformed(vi, vn) : is_transformed(vi)
    # TODO(mhauru) We should move away from having all values vectorised by default.
    # That messes with our use of unflatten though, so will require some thought.
    transform = if insert_transformed_value
        from_linked_vec_transform(dist)
    else
        from_vec_transform(dist)
    end
    transformed_val, logjac = with_logabsdet_jacobian(inverse(transform), val)
    val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    tv = TransformedValue(transformed_val, insert_transformed_value, transform, val_size)
    vi = VNTVarInfo(setindex!!(vi.values, tv, vn), vi.accs)
    return vi, logjac
end

function BangBang.setindex!!(vi::VNTVarInfo, val, vn::VarName)
    transform = from_vec_transform(val)
    transformed_val = inverse(transform)(val)
    tv = TransformedValue(transformed_val, false, transform, size(val))
    return VNTVarInfo(setindex!!(vi.values, tv, vn), vi.accs)
end

Base.keys(vi::VNTVarInfo) = keys(vi.values)
Base.values(vi::VNTVarInfo) = mapreduce(p -> p.second.val, push!, vi.values; init=Any[])

function set_transformed!!(vi::VNTVarInfo, linked::Bool, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(old_tv.val, linked, old_tv.transform, old_tv.size)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VNTVarInfo(new_values, vi.accs)
end

# VNTVarInfo does not care whether the transformation was Static or Dynamic, it just tracks
# whether one was applied at all.
function set_transformed!!(vi::VNTVarInfo, ::AbstractTransformation, vn::VarName)
    return set_transformed!!(vi, true, vn)
end

set_transformed!!(vi::VNTVarInfo, ::AbstractTransformation) = set_transformed!!(vi, true)

function set_transformed!!(vi::VNTVarInfo, ::NoTransformation, vn::VarName)
    return set_transformed!!(vi, false, vn)
end

set_transformed!!(vi::VNTVarInfo, ::NoTransformation) = set_transformed!!(vi, false)

function set_transformed!!(vi::VNTVarInfo, linked::Bool)
    new_values = map_values!!(vi.values) do tv
        TransformedValue(tv.val, linked, tv.transform, tv.size)
    end
    return VNTVarInfo(new_values, vi.accs)
end

function getindex_internal(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.val
end

# TODO(mhauru) This is mimicing old behaviour, but is now wrong: The internal
# representation does not have to be a Vector.
getindex_internal(vi::VNTVarInfo, ::Colon) = values_as(vi, Vector)

function is_transformed(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.linked
end

# TODO(mhauru) Other VarInfos have something like this. Do we need it? Or should we use the
# below version?
function from_internal_transform(::VNTVarInfo, ::VarName, dist::Distribution)
    return from_vec_transform(dist)
end

# function from_internal_transform(vi::VNTVarInfo, vn::VarName, ::Distribution)
#     return getindex(vi.values, vn).transform
# end

function from_linked_internal_transform(::VNTVarInfo, ::VarName, dist::Distribution)
    return from_linked_vec_transform(dist)
end

function from_linked_internal_transform(vi::VNTVarInfo, vn::VarName)
    return getindex(vi.values, vn).transform
end

function link!!(::DynamicTransformation, vi::VNTVarInfo, vns, model::Model)
    dists = extract_priors(model, vi)
    cumulative_logjac = zero(LogProbType)
    new_values = map_pairs!!(vi.values) do pair
        vn, tv = pair
        if vns !== nothing && !any(x -> subsumes(x, vn), vns)
            # Not one of the target variables.
            return tv
        end
        dist = getindex(dists, vn)
        vec_transform = from_vec_transform(dist)
        link_transform = from_linked_vec_transform(dist)
        val_untransformed, logjac1 = with_logabsdet_jacobian(vec_transform, tv.val)
        val_new, logjac2 = with_logabsdet_jacobian(
            inverse(link_transform), val_untransformed
        )
        new_tv = TransformedValue(val_new, true, link_transform, tv.size)
        cumulative_logjac += logjac1 + logjac2
        return new_tv
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function link!!(t::DynamicTransformation, vi::VNTVarInfo, model::Model)
    return link!!(t, vi, nothing, model)
end

function invlink!!(::DynamicTransformation, vi::VNTVarInfo, vns, model::Model)
    dists = extract_priors(model, vi)
    cumulative_logjac = zero(LogProbType)
    new_values = map_pairs!!(vi.values) do pair
        vn, tv = pair
        if vns !== nothing && !any(x -> subsumes(x, vn), vns)
            # Not one of the target variables.
            return tv
        end
        current_val = tv.val
        dist = getindex(dists, vn)
        vec_transform = from_vec_transform(dist)
        link_transform = from_linked_vec_transform(dist)
        val_untransformed, logjac1 = with_logabsdet_jacobian(link_transform, current_val)
        val_new, logjac2 = with_logabsdet_jacobian(
            inverse(vec_transform), val_untransformed
        )
        new_tv = TransformedValue(val_new, false, vec_transform, tv.size)
        cumulative_logjac += logjac1 + logjac2
        return new_tv
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function invlink!!(t::DynamicTransformation, vi::VNTVarInfo, model::Model)
    return invlink!!(t, vi, nothing, model)
end

function link!!(t::DynamicTransformation, vi::ThreadSafeVarInfo{<:VNTVarInfo}, model::Model)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, model)
end

function link!!(
    t::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VNTVarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, vns, model)
end

function invlink!!(
    t::DynamicTransformation, vi::ThreadSafeVarInfo{<:VNTVarInfo}, model::Model
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.invlink!!(t, vi.varinfo, model)
end

function invlink!!(
    ::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VNTVarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.invlink!!(vi.varinfo, vns, model)
end

# TODO(mhauru) I don't think this should return the internal values, but that's the current
# convention.
function values_as(vi::VNTVarInfo, ::Type{Vector})
    return mapfoldl(pair -> tovec(pair.second.val), vcat, vi.values; init=Union{}[])
end

function values_as(vi::VNTVarInfo, ::Type{T}) where {T<:AbstractDict}
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
function values_as(vi::VNTVarInfo, ::Type{NamedTuple})
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

# TODO(mhauru) These two are now redundant, just conforming to the old interface
# temporarily.
function untyped_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return VNTVarInfo(rng, model, init_strategy)
end

function typed_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return VNTVarInfo(rng, model, init_strategy)
end

typed_varinfo(vi::VNTVarInfo) = vi

function typed_varinfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return typed_varinfo(Random.default_rng(), model, init_strategy)
end

function untyped_varinfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return untyped_varinfo(Random.default_rng(), model, init_strategy)
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

function unflatten!!(vi::VNTVarInfo, vec::AbstractVector)
    # You may wonder, why have a whole struct for this, rather than just an index variable
    # that the mapping function would close over. I wonder too. But for some reason type
    # inference fails on such an index variable, turning it into a Core.Box.
    vci = VectorChunkIterator(vec, 1)
    new_values = map_values!!(vi.values) do tv
        old_val = tv.val
        if !(old_val isa AbstractVector)
            error(
                "Can not unflatten a VarInfo for which existing values are not vectors:" *
                " Got value of type $(typeof(old_val)).",
            )
        end
        len = length(old_val)
        new_val = get_next_chunk!(vci, len)
        return TransformedValue(new_val, tv.linked, tv.transform, tv.size)
    end
    return VNTVarInfo(new_values, vi.accs)
end

function subset(varinfo::VNTVarInfo, vns)
    new_values = subset(varinfo.values, vns)
    return VNTVarInfo(new_values, map(copy, getaccs(varinfo)))
end

function Base.merge(varinfo_left::VNTVarInfo, varinfo_right::VNTVarInfo)
    new_values = merge(varinfo_left.values, varinfo_right.values)
    new_accs = map(copy, getaccs(varinfo_right))
    return VNTVarInfo(new_values, new_accs)
end
