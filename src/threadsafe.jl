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
    return ThreadSafeVarInfo(vi, [zero(getlogp(vi)) for _ in 1:Threads.nthreads()])
end
ThreadSafeVarInfo(vi::ThreadSafeVarInfo) = vi

# Instead of updating the log probability of the underlying variables we
# just update the array of log probabilities.
function acclogp!(vi::ThreadSafeVarInfo, logp::Real)
    vi.logps[Threads.threadid()] += logp
    return getlogp(vi)
end

# The current log probability of the variables has to be computed from
# both the wrapped variables and the thread-specific log probabilities.
getlogp(vi::ThreadSafeVarInfo) = getlogp(vi.varinfo) + sum(vi.logps)

# TODO: Make remaining methods thread-safe.

function resetlogp!(vi::ThreadSafeVarInfo)
    resetlogp!(vi.varinfo)
    z = zero(getlogp(vi))
    fill!(vi.logps, z)
    z
end
function setlogp!(vi::ThreadSafeVarInfo, logp::Real)
    if length(vi.logp) == 0
        push!(vi.logp, logp)
    else
        vi.logp[1] = logp
    end
    vi.lastidx[] = 1
    return logp
end

get_num_produce(vi::ThreadSafeVarInfo) = get_num_produce(vi.varinfo)
increment_num_produce!(vi::ThreadSafeVarInfo) = increment_num_produce!(vi.varinfo)
reset_num_produce!(vi::ThreadSafeVarInfo) = reset_num_produce!(vi.varinfo)
set_num_produce!(vi::ThreadSafeVarInfo, n::Int) = set_num_produce!(vi.varinfo, n)

getall(vi::ThreadSafeVarInfo) = getall(vi.varinfo)
setall!(vi::ThreadSafeVarInfo, val) = setall!(vi.varinfo, val)

syms(vi::ThreadSafeVarInfo) = syms(vi.varinfo)

getmetadata(vi::ThreadSafeVarInfo, vn::VarName) = getmetadata(vi.varinfo, vn)
getidx(vi::ThreadSafeVarInfo, vn::VarName) = getidx(vi.varinfo, vn)
getrange(vi::ThreadSafeVarInfo, vn::VarName) = getrange(vi.varinfo, vn)
getdist(vi::ThreadSafeVarInfo, vn::VarName) = getdist(vi.varinfo, vn)
getval(vi::ThreadSafeVarInfo, vn::VarName) = getval(vi.varinfo, vn)

function setgid!(vi::ThreadSafeVarInfo, gid::Selector, vn::VarName)
    setgid!(vi.varinfo, gid, vn)
end
setval!(vi::ThreadSafeVarInfo, val, vn::VarName) = setval!(vi.varinfo, val, vn)

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)

_getranges(vi::ThreadSafeVarInfo, idcs::NamedTuple) = _getranges(vi.varinfo, idcs)
_getidcs(vi::ThreadSafeVarInfo, spl::SampleFromPrior) = _getidcs(vi.varinfo, spl)
_getidcs(vi::ThreadSafeVarInfo, s::Selector, space) = _getidcs(vi.varinfo, s, space)
_getvns(vi::ThreadSafeVarInfo, spl::SampleFromPrior) = _getvns(vi.varinfo, spl)
_getvns(vi::ThreadSafeVarInfo, s::Selector, space) = _getvns(vi.varinfo, s, space)

link!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = link!(vi.varinfo, spl)
invlink!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = invlink!(vi.varinfo, spl)
islinked(vi::ThreadSafeVarInfo, spl::AbstractSampler) = islinked(vi.varinfo, spl)

getindex(vi::ThreadSafeVarInfo, spl::Sampler) = getindex(vi.varinfo, spl)
setindex!(vi::ThreadSafeVarInfo, val, spl::Sampler) = setindex!(vi.varinfo, val, spl)

function set_retained_vns_del_by_spl!(vi::ThreadSafeVarInfo, spl::Sampler)
    return set_retained_vns_del_by_spl!(vi.varinfo, spl)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function empty!(vi::ThreadSafeVarInfo)
    empty!(vi.varinfo)
    fill!(vi.logps, zero(getlogp(vi)))
    return vi
end

function push!(
    vi::ThreadSafeVarInfo,
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector}
)
    push!(vi.varinfo, vn, r, dist, gidset)
end
function push_assert(vi::ThreadSafeVarInfo, vn::VarName, dist, gidset)
    return push_assert(vi.varinfo, vn, dist, gidset)
end

function unset_flag!(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return unset_flag!(vi.varinfo, vn, flag)
end
function is_flagged(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return is_flagged(vi.varinfo, vn, flag)
end

tonamedtuple(vi::ThreadSafeVarInfo) = tonamedtuple(vi.varinfo)
