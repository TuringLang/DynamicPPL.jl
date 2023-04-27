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

function BangBang.push!!(
    vi::ThreadSafeVarInfo, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)
    return Setfield.@set vi.varinfo = push!!(vi.varinfo, vn, r, dist, gidset)
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
    return link!!(t, vi.varinfo, spl, model)
end

function invlink!!(
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return invlink!!(t, vi.varinfo, spl, model)
end

function maybe_invlink_before_eval!!(
    vi::ThreadSafeVarInfo, context::AbstractContext, model::Model
)
    # Defer to the wrapped `AbstractVarInfo` object.
    # NOTE: When computing `getlogp` for `ThreadSafeVarInfo` we do include the `getlogp(vi.varinfo)`
    # hence the log-absdet-jacobian term will correctly be included in the `getlogp(vi)`.
    return Setfield.@set vi.varinfo = maybe_invlink_before_eval!!(
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

getindex_raw(vi::ThreadSafeVarInfo, ::Colon) = getindex_raw(vi.varinfo, Colon())
getindex_raw(vi::ThreadSafeVarInfo, vn::VarName) = getindex_raw(vi.varinfo, vn)
function getindex_raw(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return getindex_raw(vi.varinfo, vns)
end
function getindex_raw(vi::ThreadSafeVarInfo, vn::VarName, dist::Distribution)
    return getindex_raw(vi.varinfo, vn, dist)
end
function getindex_raw(
    vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}, dist::Distribution
)
    return getindex_raw(vi.varinfo, vns, dist)
end
getindex_raw(vi::ThreadSafeVarInfo, spl::AbstractSampler) = getindex_raw(vi.varinfo, spl)

function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::AbstractSampler)
    return Setfield.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::SampleFromPrior)
    return Setfield.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, val, spl::SampleFromUniform)
    return Setfield.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, val, spl)
end

function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vn::VarName)
    return Setfield.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vn)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vns::AbstractVector{<:VarName})
    return Setfield.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vns)
end

function set_retained_vns_del_by_spl!(vi::ThreadSafeVarInfo, spl::Sampler)
    return set_retained_vns_del_by_spl!(vi.varinfo, spl)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function BangBang.empty!!(vi::ThreadSafeVarInfo)
    return resetlogp!!(Setfield.@set!(vi.varinfo = empty!!(vi.varinfo)))
end

values_as(vi::ThreadSafeVarInfo, ::Type{T}) where {T} = values_as(vi.varinfo, T)

function unset_flag!(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return unset_flag!(vi.varinfo, vn, flag)
end
function is_flagged(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return is_flagged(vi.varinfo, vn, flag)
end

tonamedtuple(vi::ThreadSafeVarInfo) = tonamedtuple(vi.varinfo)

# Transformations.
function settrans!!(vi::ThreadSafeVarInfo, trans::Bool, vn::VarName)
    return Setfield.@set vi.varinfo = settrans!!(vi.varinfo, trans, vn)
end
function settrans!!(vi::ThreadSafeVarInfo, spl::AbstractSampler, dist::Distribution)
    return Setfield.@set vi.varinfo = settrans!!(vi.varinfo, spl, dist)
end

istrans(vi::ThreadSafeVarInfo, vn::VarName) = istrans(vi.varinfo, vn)
istrans(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}) = istrans(vi.varinfo, vns)

getval(vi::ThreadSafeVarInfo, vn::VarName) = getval(vi.varinfo, vn)
