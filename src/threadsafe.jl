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

# Instead of updating the log probability of the underlying variables we
# just update the array of log probabilities.
function acclogp!(vi::ThreadSafeVarInfo, logp)
    vi.logps[Threads.threadid()][] += logp
    return vi
end

# The current log probability of the variables has to be computed from
# both the wrapped variables and the thread-specific log probabilities.
getlogp(vi::ThreadSafeVarInfo) = getlogp(vi.varinfo) + sum(getindex, vi.logps)

# TODO: Make remaining methods thread-safe.

function resetlogp!(vi::ThreadSafeVarInfo)
    for x in vi.logps
        x[] = zero(x[])
    end
    return resetlogp!(vi.varinfo)
end
function setlogp!(vi::ThreadSafeVarInfo, logp)
    for x in vi.logps
        x[] = zero(x[])
    end
    return setlogp!(vi.varinfo, logp)
end

Bijectors.link(vi::ThreadSafeVarInfo) = ThreadSafeVarInfo(link(vi.varinfo), vi.logps)
Bijectors.invlink(vi::ThreadSafeVarInfo) = ThreadSafeVarInfo(invlink(vi.varinfo), vi.logps)
initlink(vi::ThreadSafeVarInfo) = ThreadSafeVarInfo(initlink(vi.varinfo), vi.logps)

getrange(vi::ThreadSafeVarInfo, vn::VarName) = getrange(vi.varinfo, vn)
get_num_produce(vi::ThreadSafeVarInfo) = get_num_produce(vi.varinfo)
increment_num_produce!(vi::ThreadSafeVarInfo) = increment_num_produce!(vi.varinfo)
reset_num_produce!(vi::ThreadSafeVarInfo) = reset_num_produce!(vi.varinfo)
set_num_produce!(vi::ThreadSafeVarInfo, n::Int) = set_num_produce!(vi.varinfo, n)

syms(vi::ThreadSafeVarInfo) = syms(vi.varinfo)

function setgid!(vi::ThreadSafeVarInfo, gid::Selector, vn::VarName)
    setgid!(vi.varinfo, gid, vn)
end
setorder!(vi::ThreadSafeVarInfo, vn::VarName, index::Int) = setorder!(vi.varinfo, vn, index)

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)
getmode(vi::ThreadSafeVarInfo) = getmode(vi.varinfo)
issynced(vi::ThreadSafeVarInfo) = issynced(vi.varinfo)
function setsynced!(vi::ThreadSafeVarInfo, b::Bool)
    setsynced!(vi.varinfo, b)
    return vi
end
getmetadata(vi::ThreadSafeVarInfo, vn::VarName) = getmetadata(vi.varinfo, vn)

init_dist_link!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = init_dist_link!(vi.varinfo, spl)
init_dist_invlink!(vi::ThreadSafeVarInfo, spl::AbstractSampler) = init_dist_invlink!(vi.varinfo, spl)
islinked(vi::ThreadSafeVarInfo, spl::AbstractSampler) = islinked(vi.varinfo, spl)
getinitdist(vi::ThreadSafeVarInfo, vn::VarName) = getinitdist(vi.varinfo, vn)
has_fixed_support(vi::ThreadSafeVarInfo) = has_fixed_support(vi.varinfo)
set_fixed_support!(vi::ThreadSafeVarInfo, b::Bool) = set_fixed_support!(vi.varinfo, b)

getindex(vi::ThreadSafeVarInfo, spl::AbstractSampler) = getindex(vi.varinfo, spl)
getindex(vi::ThreadSafeVarInfo, spl::SampleFromPrior) = getindex(vi.varinfo, spl)
getindex(vi::ThreadSafeVarInfo, spl::SampleFromUniform) = getindex(vi.varinfo, spl)

function setindex!(vi::ThreadSafeVarInfo, val, spl::AbstractSampler)
    setindex!(vi.varinfo, val, spl)
end
function setindex!(vi::ThreadSafeVarInfo, val, spl::SampleFromPrior)
    setindex!(vi.varinfo, val, spl)
end
function setindex!(vi::ThreadSafeVarInfo, val, spl::SampleFromUniform)
    setindex!(vi.varinfo, val, spl)
end

function set_retained_vns_del_by_spl!(vi::ThreadSafeVarInfo, spl::Sampler)
    return set_retained_vns_del_by_spl!(vi.varinfo, spl)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function empty!(vi::ThreadSafeVarInfo)
    empty!(vi.varinfo)
    fill!(vi.logps, zero(getlogp(vi)))
    return vi
end
function empty!(vi::ThreadSafeVarInfo, spl::AbstractSampler)
    empty!(vi.varinfo, spl)
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

function unset_flag!(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return unset_flag!(vi.varinfo, vn, flag)
end
function is_flagged(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return is_flagged(vi.varinfo, vn, flag)
end

tonamedtuple(vi::ThreadSafeVarInfo) = tonamedtuple(vi.varinfo)
