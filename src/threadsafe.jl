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
function acclogp!!(vi::ThreadSafeVarInfo, logp)
    vi.logps[Threads.threadid()][] += logp
    return vi
end

# The current log probability of the variables has to be computed from
# both the wrapped variables and the thread-specific log probabilities.
getlogp(vi::ThreadSafeVarInfo) = getlogp(vi.varinfo) + sum(getindex, vi.logps)

# TODO: Make remaining methods thread-safe.

function resetlogp!!(vi::ThreadSafeVarInfo)
    for x in vi.logps
        x[] = zero(x[])
    end
    return resetlogp!!(vi.varinfo)
end
function setlogp!!(vi::ThreadSafeVarInfo, logp)
    for x in vi.logps
        x[] = zero(x[])
    end
    return setlogp!!(vi.varinfo, logp)
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

getindex(vi::ThreadSafeVarInfo, spl::AbstractSampler) = getindex(vi.varinfo, spl)
getindex(vi::ThreadSafeVarInfo, spl::SampleFromPrior) = getindex(vi.varinfo, spl)
getindex(vi::ThreadSafeVarInfo, spl::SampleFromUniform) = getindex(vi.varinfo, spl)
getindex(vi::ThreadSafeVarInfo, vn::VarName) = getindex(vi.varinfo, vn)
getindex(vi::ThreadSafeVarInfo, vns::Vector{<:VarName}) = getindex(vi.varinfo, vns)

function setindex!(vi::ThreadSafeVarInfo, val, spl::AbstractSampler)
    return setindex!(vi.varinfo, val, spl)
end
function setindex!(vi::ThreadSafeVarInfo, val, spl::SampleFromPrior)
    return setindex!(vi.varinfo, val, spl)
end
function setindex!(vi::ThreadSafeVarInfo, val, spl::SampleFromUniform)
    return setindex!(vi.varinfo, val, spl)
end

function set_retained_vns_del_by_spl!(vi::ThreadSafeVarInfo, spl::Sampler)
    return set_retained_vns_del_by_spl!(vi.varinfo, spl)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function empty!!(vi::ThreadSafeVarInfo)
    empty!!(vi.varinfo)
    fill!(vi.logps, zero(getlogp(vi)))
    return vi
end

function push!!(
    vi::ThreadSafeVarInfo, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)
    return push!!(vi.varinfo, vn, r, dist, gidset)
end

function unset_flag!(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return unset_flag!(vi.varinfo, vn, flag)
end
function is_flagged(vi::ThreadSafeVarInfo, vn::VarName, flag::String)
    return is_flagged(vi.varinfo, vn, flag)
end

tonamedtuple(vi::ThreadSafeVarInfo) = tonamedtuple(vi.varinfo)
