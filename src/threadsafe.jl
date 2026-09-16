mutable struct TaskId end

mutable struct TaskAccumulators{L<:AccumulatorTuple}
    task_id::TaskId
    accs::L
    # Once widened, this holds the current state and `accs` is no longer used.
    widened_accs::Union{Nothing,AccumulatorTuple}
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
    accs = map(acc -> promote_for_threadsafe_eval(acc, Any), getaccs(vi))
    vi = setaccs!!(vi, accs)
    return _threadsafe_varinfo(vi, typeof(map(split, getaccs(vi))))
end
function _threadsafe_varinfo(vi::AbstractVarInfo, ::Type{L}) where {L<:AccumulatorTuple}
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
                TaskAccumulators{L}(task_id, map(split, getaccs(vi.varinfo))::L, nothing)
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

This method resets the accumulators and promotes their initial values using the supplied
parameter type. Numeric types are converted to floating equivalents; empty or unknown
types leave the initial accumulator types unchanged.

Each task can replace its accumulator tuple with a wider type during evaluation, preserving
contributions from observations, fallback parameters, and AD tracers.

For accumulators that wrap `VarNamedTuple`s, thread safety also requires removing the VNT
type parameter; see [`promote_for_threadsafe_eval`](@ref).
"""
function ThreadSafeVarInfo(varinfo::AbstractVarInfo, ::Type{T}) where {T}
    # Capturing a runtime type loses accumulator inference on Julia 1.10; use static T.
    accs = map(DynamicPPL.getaccs(varinfo)) do acc
        DynamicPPL.promote_for_threadsafe_eval(
            acc,
            if T === Any || T === Union{}
                Any
            else
                float_type_with_fallback(T)
            end,
        )
    end
    varinfo = DynamicPPL.setaccs!!(varinfo, accs)
    varinfo = resetaccs!!(varinfo)
    return if T === Any || T === Union{}
        _threadsafe_varinfo(varinfo, AccumulatorTuple)
    else
        ThreadSafeVarInfo(varinfo)
    end
end

function setacc!!(vi::ThreadSafeVarInfo, acc::AbstractAccumulator)
    inner_vi = setaccs!!(vi.varinfo, getaccs(vi))
    return ThreadSafeVarInfo(setacc!!(inner_vi, acc))
end

# This flag is accumulator configuration, not accumulated task state.
function is_extracting_colon_eq_values(vi::ThreadSafeVarInfo)
    return is_extracting_colon_eq_values(vi.varinfo)
end

function _getaccs(f::F, vi::ThreadSafeVarInfo, accnames::Tuple) where {F}
    main_accs = map(name -> copy(getacc(vi.varinfo, name)), accnames)
    # Protect dictionary traversal from concurrent registration. Accumulator contents may
    # only be read after their tasks complete.
    task_accs = lock(vi.accs_lock) do
        collect(values(vi.accs_by_task))
    end
    # Keep the reduction and result construction inferred when no task has widened.
    if all(task -> task.widened_accs === nothing, task_accs)
        accs = foldl(task_accs; init=main_accs) do accs, task
            map(accs, accnames) do acc, name
                combine(acc, getacc(task.accs, name))
            end
        end
        return f(accs)
    else
        accs = foldl(task_accs; init=main_accs) do accs, task
            other = task.widened_accs === nothing ? task.accs : task.widened_accs
            map(accs, accnames) do acc, name
                combine(acc, getacc(other, name))
            end
        end
        return f(accs)
    end
end
getacc(vi::ThreadSafeVarInfo, accname::Val) = _getaccs(only, vi, (accname,))

function Base.copy(vi::ThreadSafeVarInfo)
    inner_vi = setaccs!!(vi.varinfo, getaccs(vi))
    return ThreadSafeVarInfo(copy(inner_vi))
end

hasacc(vi::ThreadSafeVarInfo, accname::Val) = hasacc(vi.varinfo, accname)
acckeys(vi::ThreadSafeVarInfo) = acckeys(vi.varinfo)

function getaccs(vi::ThreadSafeVarInfo)
    return _getaccs(vi, map(Val, acckeys(vi))) do accs
        AccumulatorTuple(accs)
    end
end

function _map_task_accs!!(func::F, task_accs::TaskAccumulators{L}) where {F,L}
    widened_accs = task_accs.widened_accs
    if widened_accs === nothing
        accs = func(task_accs.accs)
        if accs isa L
            task_accs.accs = accs
        else
            task_accs.widened_accs = accs
        end
    else
        task_accs.widened_accs = func(widened_accs)
    end
    return task_accs
end

# Calls to map_accumulator(s)!! are task-specific by default. For any use of them that
# should not be task-specific a specific method has to be written.
function map_accumulator!!(func::Function, vi::ThreadSafeVarInfo, accname::Val)
    _map_task_accs!!(_get_task_accs(vi)) do accs
        map_accumulator(func, accs, accname)
    end
    return vi
end

function map_accumulators!!(func::Function, vi::ThreadSafeVarInfo)
    _map_task_accs!!(_get_task_accs(vi)) do accs
        map(func, accs)
    end
    return vi
end

function setaccs!!(vi::ThreadSafeVarInfo, accs::AccumulatorTuple)
    return ThreadSafeVarInfo(setaccs!!(vi.varinfo, accs))
end

keys(vi::ThreadSafeVarInfo) = keys(get_vector_values(vi))
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(get_vector_values(vi), vn)
isempty(vi::ThreadSafeVarInfo) = isempty(get_vector_values(vi))
BangBang.empty!!(vi::ThreadSafeVarInfo) = resetaccs!!(vi)

function resetaccs!!(vi::ThreadSafeVarInfo{V,L}) where {V,L}
    vi = Accessors.@set vi.varinfo = resetaccs!!(vi.varinfo)
    lock(vi.accs_lock) do
        empty!(vi.accs_by_task)
        @atomic :release vi.task_accs_cache = _task_accs_cache(L)
    end
    return vi
end

function is_transformed(vi::ThreadSafeVarInfo, vn::VarName)
    return get_transform(get_transformed_value(vi, vn)) isa DynamicLink
end

function link!!(vi::ThreadSafeVarInfo, args...)
    return ThreadSafeVarInfo(link!!(setaccs!!(vi.varinfo, getaccs(vi)), args...))
end
function invlink!!(vi::ThreadSafeVarInfo, args...)
    return ThreadSafeVarInfo(invlink!!(setaccs!!(vi.varinfo, getaccs(vi)), args...))
end
function unflatten!!(vi::ThreadSafeVarInfo, x::AbstractVector)
    return ThreadSafeVarInfo(unflatten!!(setaccs!!(vi.varinfo, getaccs(vi)), x))
end
function subset(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return ThreadSafeVarInfo(subset(setaccs!!(vi.varinfo, getaccs(vi)), vns))
end
function Base.merge(left::ThreadSafeVarInfo, right::ThreadSafeVarInfo)
    return ThreadSafeVarInfo(
        merge(
            setaccs!!(left.varinfo, getaccs(left)), setaccs!!(right.varinfo, getaccs(right))
        ),
    )
end
