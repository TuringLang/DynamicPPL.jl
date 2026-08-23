mutable struct TaskId end

mutable struct TaskAccumulators{L<:AccumulatorTuple}
    task_id::TaskId
    accs::L
end

const _TASK_ID_KEY = Ref{Nothing}(nothing)

# Task-local identities survive migration without retaining live `Task` objects.
function _current_task_id()
    return get!(task_local_storage(), _TASK_ID_KEY) do
        TaskId()
    end::TaskId
end

function _task_accs_cache(::Type{L}) where {L<:AccumulatorTuple}
    return Vector{Union{Nothing,TaskAccumulators{L}}}(nothing, Threads.maxthreadid())
end

"""
    ThreadSafeVarInfo

A `ThreadSafeVarInfo` object wraps an [`AbstractVarInfo`](@ref) object and stores one
set of accumulators per task for thread-safe execution of a probabilistic model.
"""
mutable struct ThreadSafeVarInfo{V<:AbstractVarInfo,L<:AccumulatorTuple} <: AbstractVarInfo
    varinfo::V
    accs_by_task::IdDict{TaskId,TaskAccumulators{L}}
    @atomic task_accs_cache::Vector{Union{Nothing,TaskAccumulators{L}}}
    accs_lock::ReentrantLock
end
function ThreadSafeVarInfo(vi::AbstractVarInfo)
    L = typeof(map(split, getaccs(vi)))
    accs_by_task = IdDict{TaskId,TaskAccumulators{L}}()
    task_accs_cache = _task_accs_cache(L)
    return ThreadSafeVarInfo(vi, accs_by_task, task_accs_cache, ReentrantLock())
end
ThreadSafeVarInfo(vi::ThreadSafeVarInfo) = vi

function _get_task_accs_cache(vi::ThreadSafeVarInfo{V,L}) where {V,L}
    cache = @atomic :acquire vi.task_accs_cache
    # A serialized cache may have been sized for a process with fewer threads.
    length(cache) == Threads.maxthreadid() && return cache
    return lock(vi.accs_lock) do
        current_cache = @atomic :acquire vi.task_accs_cache
        if length(current_cache) != Threads.maxthreadid()
            current_cache = _task_accs_cache(L)
            @atomic :release vi.task_accs_cache = current_cache
        end
        current_cache
    end
end

# Per-thread cache entries avoid registry locking after registration. Stale entries fail the
# task identity check, so migration only causes another registry lookup.
function _get_task_accs(vi::ThreadSafeVarInfo{V,L}) where {V,L}
    task_id = _current_task_id()
    task_accs_cache = _get_task_accs_cache(vi)
    task_accs = task_accs_cache[Threads.threadid()]
    if task_accs === nothing || task_accs.task_id !== task_id
        task_accs = lock(vi.accs_lock) do
            get!(vi.accs_by_task, task_id) do
                TaskAccumulators(task_id, map(split, getaccs(vi.varinfo))::L)
            end
        end
        task_accs_cache[Threads.threadid()] = task_accs
    end
    return task_accs
end

"""
    ThreadSafeVarInfo(varinfo::AbstractVarInfo, param_eltype::Type{T})

Construct a `ThreadSafeVarInfo` that promotes any accumulators in `varinfo` to their
versions for use in TSVI.

This method also resets the accumulators' contents.

# Extended help

The reason why this is needed in general is to ensure that the function call
`map_accumulator!!(tsvi::ThreadSafeVarInfo, ...)` does not fail. Suppose first that
`TaskAccumulators.accs` has a concrete `AccumulatorTuple` type containing a
`LogLikelihoodAccumulator(::Float64)`.

Now, consider a situation where we evaluate the gradient of the log-probability with
ForwardDiff. This would cause the wrapped log-likelihood to be promoted to
`ForwardDiff.Dual`. If there were only one accumulator, this would be fine. However, the
promoted accumulator cannot be stored in a field whose concrete type contains `Float64`.

This means that *before* model evaluation even begins, the eltype of *all* log-probability
accumulators must be promoted to `ForwardDiff.Dual`.

For log-probability accumulators, construction of the thread-safe versions therefore
requires knowledge of `param_eltype`, which is the type of the parameters about to be used
for model evaluation. See the docstring of `get_param_eltype` for more information about
this. For accumulators that wrap `VarNamedTuple`s, thread safety is accomplished by removing
the VNT type parameter from its type.
"""
function ThreadSafeVarInfo(varinfo::AbstractVarInfo, param_eltype::Type{T}) where {T}
    # The below line is finicky for type stability. For instance, assigning the eltype to
    # convert to into an intermediate variable makes this unstable (constant propagation
    # fails). Take care when editing.
    accs = map(DynamicPPL.getaccs(varinfo)) do acc
        DynamicPPL.promote_for_threadsafe_eval(acc, param_eltype)
    end
    varinfo = DynamicPPL.setaccs!!(varinfo, accs)
    return ThreadSafeVarInfo(resetaccs!!(varinfo))
end

function setacc!!(vi::ThreadSafeVarInfo, acc::AbstractAccumulator)
    inner_vi = setaccs!!(vi.varinfo, getaccs(vi))
    return ThreadSafeVarInfo(setacc!!(inner_vi, acc))
end

get_values(vi::ThreadSafeVarInfo) = get_values(vi.varinfo)

# This flag is accumulator configuration, not accumulated task state.
function is_extracting_colon_eq_values(vi::ThreadSafeVarInfo)
    return is_extracting_colon_eq_values(vi.varinfo)
end

function getacc(vi::ThreadSafeVarInfo, accname::Val)
    main_acc = getacc(vi.varinfo, accname)
    # Protect dictionary traversal from concurrent registration. Accumulator contents may
    # only be read after their tasks complete.
    other_accs = lock(vi.accs_lock) do
        map(values(vi.accs_by_task)) do task_accs
            getacc(task_accs.accs, accname)
        end
    end
    return foldl(combine, other_accs; init=main_acc)
end

function Base.copy(vi::ThreadSafeVarInfo)
    inner_vi = setaccs!!(vi.varinfo, getaccs(vi))
    return ThreadSafeVarInfo(copy(inner_vi))
end

hasacc(vi::ThreadSafeVarInfo, accname::Val) = hasacc(vi.varinfo, accname)
acckeys(vi::ThreadSafeVarInfo) = acckeys(vi.varinfo)

function getaccs(vi::ThreadSafeVarInfo)
    # This method is a bit finicky to maintain type stability. For instance, moving the
    # accname -> Val(accname) part in the main `map` call makes constant propagation fail
    # and this becomes unstable. Do check the effects if you make edits.
    accnames = acckeys(vi)
    accname_vals = map(Val, accnames)
    return AccumulatorTuple(map(anv -> getacc(vi, anv), accname_vals))
end

# Calls to map_accumulator(s)!! are task-specific by default. For any use of them that
# should not be task-specific a specific method has to be written.
function map_accumulator!!(func::Function, vi::ThreadSafeVarInfo, accname::Val)
    task_accs = _get_task_accs(vi)
    task_accs.accs = map_accumulator(func, task_accs.accs, accname)
    return vi
end

function map_accumulators!!(func::Function, vi::ThreadSafeVarInfo)
    task_accs = _get_task_accs(vi)
    task_accs.accs = map(func, task_accs.accs)
    return vi
end

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)

is_transformed(vi::ThreadSafeVarInfo) = is_transformed(vi.varinfo)

function link!!(vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = link!!(vi.varinfo, args...)
end

function invlink!!(vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = invlink!!(vi.varinfo, args...)
end
get_transform_strategy(vi::ThreadSafeVarInfo) = get_transform_strategy(vi.varinfo)

getindex(vi::ThreadSafeVarInfo, ::Colon) = getindex(vi.varinfo, Colon())

function setindex_with_dist!!(
    vi::ThreadSafeVarInfo, tval, dist::Distribution, vn::VarName, template
)
    vi_inner = setindex_with_dist!!(vi.varinfo, tval, dist, vn, template)
    return Accessors.@set(vi.varinfo = vi_inner)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function BangBang.empty!!(vi::ThreadSafeVarInfo)
    return resetaccs!!(Accessors.@set(vi.varinfo = empty!!(vi.varinfo)))
end

function resetaccs!!(vi::ThreadSafeVarInfo{V,L}) where {V,L}
    vi = Accessors.@set vi.varinfo = resetaccs!!(vi.varinfo)
    lock(vi.accs_lock) do
        empty!(vi.accs_by_task)
        @atomic :release vi.task_accs_cache = _task_accs_cache(L)
    end
    return vi
end

internal_values_as_vector(vi::ThreadSafeVarInfo) = internal_values_as_vector(vi.varinfo)

is_transformed(vi::ThreadSafeVarInfo, vn::VarName) = is_transformed(vi.varinfo, vn)
function is_transformed(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return is_transformed(vi.varinfo, vns)
end

getindex_internal(vi::ThreadSafeVarInfo, vn::VarName) = getindex_internal(vi.varinfo, vn)
function get_transformed_value(vi::ThreadSafeVarInfo, vn::VarName)
    return get_transformed_value(vi.varinfo, vn)
end

function unflatten!!(vi::ThreadSafeVarInfo, x::AbstractVector)
    return Accessors.@set vi.varinfo = unflatten!!(vi.varinfo, x)
end

function subset(varinfo::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return Accessors.@set varinfo.varinfo = subset(varinfo.varinfo, vns)
end

function Base.merge(varinfo_left::ThreadSafeVarInfo, varinfo_right::ThreadSafeVarInfo)
    return Accessors.@set varinfo_left.varinfo = merge(
        varinfo_left.varinfo, varinfo_right.varinfo
    )
end
