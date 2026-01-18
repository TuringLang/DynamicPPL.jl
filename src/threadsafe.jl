"""
    ThreadSafeVarInfo

A `ThreadSafeVarInfo` object wraps an [`AbstractVarInfo`](@ref) object and an
array of accumulators for thread-safe execution of a probabilistic model.
"""
struct ThreadSafeVarInfo{V<:AbstractVarInfo,L<:AccumulatorTuple} <: AbstractVarInfo
    varinfo::V
    accs_by_thread::Vector{L}
end
function ThreadSafeVarInfo(vi::AbstractVarInfo)
    # In ThreadSafeVarInfo we use threadid() to index into the array of logp
    # fields. This is not good practice --- see
    # https://github.com/TuringLang/DynamicPPL.jl/issues/924 for a full
    # explanation --- but it has worked okay so far.
    accs_by_thread = [map(split, getaccs(vi)) for _ in 1:Threads.maxthreadid()]
    return ThreadSafeVarInfo(vi, accs_by_thread)
end
ThreadSafeVarInfo(vi::ThreadSafeVarInfo) = vi

transformation(vi::ThreadSafeVarInfo) = transformation(vi.varinfo)

# Set the accumulator in question in vi.varinfo, and set the thread-specific
# accumulators of the same type to be empty.
function setacc!!(vi::ThreadSafeVarInfo, acc::AbstractAccumulator)
    inner_vi = setacc!!(vi.varinfo, acc)
    news_accs_by_thread = map(accs -> setacc!!(accs, split(acc)), vi.accs_by_thread)
    return ThreadSafeVarInfo(inner_vi, news_accs_by_thread)
end

# Get both the main accumulator and the thread-specific accumulators of the same type and
# combine them.
function getacc(vi::ThreadSafeVarInfo, accname::Val)
    main_acc = getacc(vi.varinfo, accname)
    other_accs = map(accs -> getacc(accs, accname), vi.accs_by_thread)
    return foldl(combine, other_accs; init=main_acc)
end

function Base.copy(vi::ThreadSafeVarInfo)
    return ThreadSafeVarInfo(copy(vi.varinfo), deepcopy(vi.accs_by_thread))
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

# Calls to map_accumulator(s)!! are thread-specific by default. For any use of them that
# should _not_ be thread-specific a specific method has to be written.
function map_accumulator!!(func::Function, vi::ThreadSafeVarInfo, accname::Val)
    tid = Threads.threadid()
    vi.accs_by_thread[tid] = map_accumulator(func, vi.accs_by_thread[tid], accname)
    return vi
end

function map_accumulators!!(func::Function, vi::ThreadSafeVarInfo)
    tid = Threads.threadid()
    vi.accs_by_thread[tid] = map(func, vi.accs_by_thread[tid])
    return vi
end

syms(vi::ThreadSafeVarInfo) = syms(vi.varinfo)

setval!(vi::ThreadSafeVarInfo, val, vn::VarName) = setval!(vi.varinfo, val, vn)

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)

is_transformed(vi::ThreadSafeVarInfo) = is_transformed(vi.varinfo)

function link!!(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = link!!(t, vi.varinfo, args...)
end

function invlink!!(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = invlink!!(t, vi.varinfo, args...)
end

function maybe_invlink_before_eval!!(vi::ThreadSafeVarInfo, model::Model)
    # Defer to the wrapped `AbstractVarInfo` object.
    # NOTE: When computing `getacc` for `ThreadSafeVarInfo` we do include the
    # `getacc(vi.varinfo)` hence the log-absdet-jacobian term will correctly be included in
    # the `getlogprior(vi)`.
    return Accessors.@set vi.varinfo = maybe_invlink_before_eval!!(vi.varinfo, model)
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

function setindex_with_dist!!(
    vi::ThreadSafeVarInfo, val, dist::Distribution, vn::VarName, template
)
    vi_inner, logjac = setindex_with_dist!!(vi.varinfo, val, dist, vn, template)
    return Accessors.@set(vi.varinfo = vi_inner), logjac
end

function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vn::VarName)
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vn)
end
function BangBang.setindex!!(vi::ThreadSafeVarInfo, vals, vns::AbstractVector{<:VarName})
    return Accessors.@set vi.varinfo = BangBang.setindex!!(vi.varinfo, vals, vns)
end

vector_length(vi::ThreadSafeVarInfo) = vector_length(vi.varinfo)

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function BangBang.empty!!(vi::ThreadSafeVarInfo)
    return resetaccs!!(Accessors.@set(vi.varinfo = empty!!(vi.varinfo)))
end

function resetaccs!!(vi::ThreadSafeVarInfo)
    vi = Accessors.@set vi.varinfo = resetaccs!!(vi.varinfo)
    for i in eachindex(vi.accs_by_thread)
        vi.accs_by_thread[i] = map(reset, vi.accs_by_thread[i])
    end
    return vi
end

values_as(vi::ThreadSafeVarInfo) = values_as(vi.varinfo)
values_as(vi::ThreadSafeVarInfo, ::Type{T}) where {T} = values_as(vi.varinfo, T)

function set_transformed!!(vi::ThreadSafeVarInfo, val::Bool, vn::VarName)
    return Accessors.@set vi.varinfo = set_transformed!!(vi.varinfo, val, vn)
end

is_transformed(vi::ThreadSafeVarInfo, vn::VarName) = is_transformed(vi.varinfo, vn)
function is_transformed(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName})
    return is_transformed(vi.varinfo, vns)
end

getindex_internal(vi::ThreadSafeVarInfo, vn::VarName) = getindex_internal(vi.varinfo, vn)

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
