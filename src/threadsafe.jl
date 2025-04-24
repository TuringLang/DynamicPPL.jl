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
    accs_by_thread = [
        AccumulatorTuple(map(split, vi.accs.nt)) for _ in 1:Threads.nthreads()
    ]
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

has_varnamedvector(vi::ThreadSafeVarInfo) = has_varnamedvector(vi.varinfo)

function BangBang.push!!(vi::ThreadSafeVarInfo, vn::VarName, r, dist::Distribution)
    return Accessors.@set vi.varinfo = push!!(vi.varinfo, vn, r, dist)
end

# TODO(mhauru) Why these short-circuits? Why not use the thread-specific ones?
get_num_produce(vi::ThreadSafeVarInfo) = get_num_produce(vi.varinfo)
function increment_num_produce!!(vi::ThreadSafeVarInfo)
    return ThreadSafeVarInfo(increment_num_produce!!(vi.varinfo), vi.accs_by_thread)
end
function reset_num_produce!!(vi::ThreadSafeVarInfo)
    return ThreadSafeVarInfo(reset_num_produce!!(vi.varinfo), vi.accs_by_thread)
end
function set_num_produce!!(vi::ThreadSafeVarInfo, n::Int)
    return ThreadSafeVarInfo(set_num_produce!!(vi.varinfo, n), vi.accs_by_thread)
end

syms(vi::ThreadSafeVarInfo) = syms(vi.varinfo)

setorder!(vi::ThreadSafeVarInfo, vn::VarName, index::Int) = setorder!(vi.varinfo, vn, index)
setval!(vi::ThreadSafeVarInfo, val, vn::VarName) = setval!(vi.varinfo, val, vn)

keys(vi::ThreadSafeVarInfo) = keys(vi.varinfo)
haskey(vi::ThreadSafeVarInfo, vn::VarName) = haskey(vi.varinfo, vn)

islinked(vi::ThreadSafeVarInfo) = islinked(vi.varinfo)

function link!!(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = link!!(t, vi.varinfo, args...)
end

function invlink!!(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = invlink!!(t, vi.varinfo, args...)
end

function link(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = link(t, vi.varinfo, args...)
end

function invlink(t::AbstractTransformation, vi::ThreadSafeVarInfo, args...)
    return Accessors.@set vi.varinfo = invlink(t, vi.varinfo, args...)
end

# Need to define explicitly for `DynamicTransformation` to avoid method ambiguity.
# NOTE: We also can't just defer to the wrapped varinfo, because we need to ensure
# consistency between `vi.accs_by_thread` field and `getacc(vi.varinfo)`, which accumulates
# to define `getacc(vi)`.
function link!!(t::DynamicTransformation, vi::ThreadSafeVarInfo, model::Model)
    return settrans!!(last(evaluate!!(model, vi, DynamicTransformationContext{false}())), t)
end

function invlink!!(::DynamicTransformation, vi::ThreadSafeVarInfo, model::Model)
    return settrans!!(
        last(evaluate!!(model, vi, DynamicTransformationContext{true}())),
        NoTransformation(),
    )
end

function link(t::DynamicTransformation, vi::ThreadSafeVarInfo, model::Model)
    return link!!(t, deepcopy(vi), model)
end

function invlink(t::DynamicTransformation, vi::ThreadSafeVarInfo, model::Model)
    return invlink!!(t, deepcopy(vi), model)
end

# These two StaticTransformation methods needed to resolve ambiguities
function link!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::ThreadSafeVarInfo, model::Model
)
    return Accessors.@set vi.varinfo = link!!(t, vi.varinfo, model)
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::ThreadSafeVarInfo, model::Model
)
    return Accessors.@set vi.varinfo = invlink!!(t, vi.varinfo, model)
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

function set_retained_vns_del!(vi::ThreadSafeVarInfo)
    return set_retained_vns_del!(vi.varinfo)
end

isempty(vi::ThreadSafeVarInfo) = isempty(vi.varinfo)
function BangBang.empty!!(vi::ThreadSafeVarInfo)
    return resetlogp!!(Accessors.@set(vi.varinfo = empty!!(vi.varinfo)))
end

function resetlogp!!(vi::ThreadSafeVarInfo)
    vi = Accessors.@set vi.varinfo = resetlogp!!(vi.varinfo)
    for i in eachindex(vi.accs_by_thread)
        vi.accs_by_thread[i] = map_accumulator(zero, vi.accs_by_thread[i], Val(:LogPrior))
        vi.accs_by_thread[i] = map_accumulator(
            zero, vi.accs_by_thread[i], Val(:LogLikelihood)
        )
    end
    return vi
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

function settrans!!(vi::ThreadSafeVarInfo, trans::Bool, vn::VarName)
    return Accessors.@set vi.varinfo = settrans!!(vi.varinfo, trans, vn)
end

istrans(vi::ThreadSafeVarInfo, vn::VarName) = istrans(vi.varinfo, vn)
istrans(vi::ThreadSafeVarInfo, vns::AbstractVector{<:VarName}) = istrans(vi.varinfo, vns)

getindex_internal(vi::ThreadSafeVarInfo, vn::VarName) = getindex_internal(vi.varinfo, vn)

function unflatten(vi::ThreadSafeVarInfo, x::AbstractVector)
    return Accessors.@set vi.varinfo = unflatten(vi.varinfo, x)
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
