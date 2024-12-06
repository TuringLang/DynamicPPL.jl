"""
    ThreadSafeVarInfo

A `ThreadSafeVarInfo` object wraps an [`AbstractVarInfo`](@ref) object and an
array of log probabilities for thread-safe execution of a probabilistic model.
"""
struct ThreadSafeVarInfo{V<:AbstractVarInfo,L} <: AbstractVarInfo
    varinfo::V
    logps::L
end
function ThreadSafeVarInfo(vi::AbstractVarInfo)
    return ThreadSafeVarInfo(vi, [Ref(zero(getlogp(vi))) for _ in 1:Threads.nthreads()])
end
ThreadSafeVarInfo(vi::ThreadSafeVarInfo) = vi

const ThreadSafeVarInfoWithRef{V<:AbstractVarInfo} = ThreadSafeVarInfo{
    V,<:AbstractArray{<:Ref}
}

transformation(vi::ThreadSafeVarInfo) = transformation(vi.varinfo)

# Instead of updating the log probability of the underlying variables we
# just update the array of log probabilities.
function acclogp!!(vi::ThreadSafeVarInfo, logp)
    vi.logps[Threads.threadid()] += logp
    return vi
end
function acclogp!!(vi::ThreadSafeVarInfoWithRef, logp)
    vi.logps[Threads.threadid()][] += logp
    return vi
end

# The current log probability of the variables has to be computed from
# both the wrapped variables and the thread-specific log probabilities.
getlogp(vi::ThreadSafeVarInfo) = getlogp(vi.varinfo) + sum(vi.logps)
getlogp(vi::ThreadSafeVarInfoWithRef) = getlogp(vi.varinfo) + sum(getindex, vi.logps)

# TODO: Make remaining methods thread-safe.
function resetlogp!!(vi::ThreadSafeVarInfo)
    return ThreadSafeVarInfo(resetlogp!!(vi.varinfo), zero(vi.logps))
end
function resetlogp!!(vi::ThreadSafeVarInfoWithRef)
    for x in vi.logps
        x[] = zero(x[])
    end
    return ThreadSafeVarInfo(resetlogp!!(vi.varinfo), vi.logps)
end
function setlogp!!(vi::ThreadSafeVarInfo, logp)
    return ThreadSafeVarInfo(setlogp!!(vi.varinfo, logp), zero(vi.logps))
end
function setlogp!!(vi::ThreadSafeVarInfoWithRef, logp)
    for x in vi.logps
        x[] = zero(x[])
    end
    return ThreadSafeVarInfo(setlogp!!(vi.varinfo, logp), vi.logps)
end

has_varnamedvector(vi::DynamicPPL.ThreadSafeVarInfo) = has_varnamedvector(vi.varinfo)

function BangBang.push!!(
    vi::ThreadSafeVarInfo, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)
    return Accessors.@set vi.varinfo = push!!(vi.varinfo, vn, r, dist, gidset)
end

get_num_produce(vi::ThreadSafeVarInfo) = get_num_produce(vi.varinfo)
increment_num_produce!(vi::ThreadSafeVarInfo) = increment_num_produce!(vi.varinfo)
reset_num_produce!(vi::ThreadSafeVarInfo) = reset_num_produce!(vi.varinfo)
set_num_produce!(vi::ThreadSafeVarInfo, n::Int) = set_num_produce!(vi.varinfo, n)

syms(vi::ThreadSafeVarInfo) = syms(vi.varinfo)

function setgid!(vi::ThreadSafeVarInfo, gid::Selector, vn::VarName)
    return setgid!(vi.varinfo, gid, vn)
end
setorder!(vi::ThreadSafeVarInfo, vn::VarName, index::Int) = setorder!(vi.varinfo, vn, index)
setval!(vi::ThreadSafeVarInfo, val, vn::VarName) = setval!(vi.varinfo, val, vn)

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)

link!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = link!(vi.varinfo, spl)
invlink!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = invlink!(vi.varinfo, spl)
islinked(vi::ThreadSafeVarInfo, spl::AbstractSampler) = islinked(vi.varinfo, spl)

function link!!(
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return Accessors.@set vi.varinfo = link!!(t, vi.varinfo, spl, model)
end

function invlink!!(
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return Accessors.@set vi.varinfo = invlink!!(t, vi.varinfo, spl, model)
end

function link(
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return Accessors.@set vi.varinfo = link(t, vi.varinfo, spl, model)
end

function invlink(
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return Accessors.@set vi.varinfo = invlink(t, vi.varinfo, spl, model)
end

# Need to define explicitly for `DynamicTransformation` to avoid method ambiguity.
# NOTE: We also can't just defer to the wrapped varinfo, because we need to ensure
# consistency between `vi.logps` field and `getlogp(vi.varinfo)`, which accumulates
# to define `getlogp(vi)`.
function link!!(
    t::DynamicTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(last(evaluate!!(model, vi, DynamicTransformationContext{false}())), t)
end

function invlink!!(
    ::DynamicTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(
        last(evaluate!!(model, vi, DynamicTransformationContext{true}())),
        NoTransformation(),
    )
end

function link(
    t::DynamicTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return link!!(t, deepcopy(vi), spl, model)
end

function invlink(
    t::DynamicTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return invlink!!(t, deepcopy(vi), spl, model)
end

function maybe_invlink_before_eval!!(
    vi::ThreadSafeVarInfo, context::AbstractContext, model::Model
)
    # Defer to the wrapped `AbstractVarInfo` object.
    # NOTE: When computing `getlogp` for `ThreadSafeVarInfo` we do include the `getlogp(vi.varinfo)`
    # hence the log-absdet-jacobian term will correctly be included in the `getlogp(vi)`.
    return Accessors.@set vi.varinfo = maybe_invlink_before_eval!!(
        vi.varinfo, context, model
    )
end

# `getindex`
getindex(vi::ThreadSafeVarInfo, ::Colon) = getindex(vi.varinfo, Colon())
getindex(vi::ThreadSafeVarInfo, vn::VarName) = getindex(vi.varinfo, vn)
getindex(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}) = getindex(vi.varinfo, vns)
function getindex(vi::ThreadSafeVarInfo, vn::VarName, dist::Distribution)
    return getindex(vi.varinfo, vn, dist)
end
function getindex(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}, dist::Distribution)
    return getindex(vi.varinfo, vns, dist)
end
getindex(vi::ThreadSafeVarInfo, spl::AbstractSampler) = getindex(vi.varinfo, spl)

function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::AbstractSampler)
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::SampleFromPrior)
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::SampleFromUniform)
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end

function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vn::VarName)
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vn)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vns::AbstractVector{<:VarName})
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vns)
end

vector_length(vi::ThreadSafeVarInfo) = vector_length(vi.varinfo)
vector_getrange(vi::ThreadSafeVarInfo, vn::VarName) = vector_getrange(vi.varinfo, vn)
function vector_getranges(vi::ThreadSafeVarInfo, vns::Vector{<:VarName})
    return vector_getranges(vi.varinfo, vns)
end

function set_retained_vns_del_by_spl!(vi::ThreadSafeVarInfo, spl::Sampler)
    return set_retained_vns_del_by_spl!(vi.varinfo, spl)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function BangBang.empty!!(vi::ThreadSafeVarInfo)
    return resetlogp!!(Accessors.@set(vi.varinfo = empty!!(vi.varinfo)))
end

values_as(vi::ThreadSafeVarInfo) = values_as(vi.varinfo)
values_as(vi::ThreadSafeVarInfo, ::Type{T}) where {T} = values_as(vi.varinfo, T)

function unset_flag!(
    vi::ThreadSafeVarInfo, vn::VarName, flag::String, ignoreable::Bool=false
)
    return unset_flag!(vi.varinfo, vn, flag, ignoreable)
end
function is_flagged(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return is_flagged(vi.varinfo, vn, flag)
end

# Transformations.
function settrans!!(vi::ThreadSafeVarInfo, trans::Bool, vn::VarName)
    return Accessors.@set vi.varinfo = settrans!!(vi.varinfo, trans, vn)
end
function settrans!!(vi::ThreadSafeVarInfo, spl::AbstractSampler, dist::Distribution)
    return Accessors.@set vi.varinfo = settrans!!(vi.varinfo, spl, dist)
end

istrans(vi::ThreadSafeVarInfo, vn::VarName) = istrans(vi.varinfo, vn)
istrans(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}) = istrans(vi.varinfo, vns)

getindex_internal(vi::ThreadSafeVarInfo, vn::VarName) = getindex_internal(vi.varinfo, vn)

function unflatten(vi::ThreadSafeVarInfo, x::AbstractVector)
    return Accessors.@set vi.varinfo = unflatten(vi.varinfo, x)
end
function unflatten(vi::ThreadSafeVarInfo, spl::AbstractSampler, x::AbstractVector)
    return Accessors.@set vi.varinfo = unflatten(vi.varinfo, spl, x)
end

function subset(varinfo::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return Accessors.@set varinfo.varinfo = subset(varinfo.varinfo, vns)
end

function Base.merge(varinfo_left::ThreadSafeVarInfo, varinfo_right::ThreadSafeVarInfo)
    return Accessors.@set varinfo_left.varinfo = merge(
        varinfo_left.varinfo, varinfo_right.varinfo
    )
end

function invlink_with_logpdf(vi::ThreadSafeVarInfo, vn::VarName, dist, y)
    return invlink_with_logpdf(vi.varinfo, vn, dist, y)
end

function from_internal_transform(varinfo::ThreadSafeVarInfo, vn::VarName)
    return from_internal_transform(varinfo.varinfo, vn)
end
function from_internal_transform(varinfo::ThreadSafeVarInfo, vn::VarName, dist)
    return from_internal_transform(varinfo.varinfo, vn, dist)
end

function from_linked_internal_transform(varinfo::ThreadSafeVarInfo, vn::VarName)
    return from_linked_internal_transform(varinfo.varinfo, vn)
end
function from_linked_internal_transform(varinfo::ThreadSafeVarInfo, vn::VarName, dist)
    return from_linked_internal_transform(varinfo.varinfo, vn, dist)
end
