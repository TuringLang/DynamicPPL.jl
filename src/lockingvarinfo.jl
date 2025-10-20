"""
    LockingVarInfo

A `LockingVarInfo` object wraps an [`AbstractVarInfo`](@ref) object and an
array of accumulators for thread-safe execution of a probabilistic model.
"""
struct LockingVarInfo{T<:AbstractVarInfo} <: AbstractVarInfo
    inner::Ref{T}
    lock::ReentrantLock
end
LockingVarInfo(vi::AbstractVarInfo) = LockingVarInfo(Ref(vi), ReentrantLock())

transformation(vi::LockingVarInfo) = transformation(vi.inner[])

function setacc!!(vi::LockingVarInfo, acc::AbstractAccumulator)
    @lock vi.lock begin
        vi.inner[] = setacc!!(vi.varinfo, acc)
    end
    return vi
end

function getacc(vi::LockingVarInfo, accname::Val)
    return getacc(vi.inner[], accname)
end

hasacc(vi::LockingVarInfo, accname::Val) = hasacc(vi.inner[], accname)
acckeys(vi::LockingVarInfo) = acckeys(vi.inner[])

function getaccs(vi::LockingVarInfo)
    return getaccs(vi.inner[])
end

function map_accumulator!!(func::Function, vi::LockingVarInfo, accname::Val)
    @lock vi.lock begin
        vi.inner[] = map_accumulator!!(func, vi.inner[], accname)
    end
    return vi
end

function map_accumulators!!(func::Function, vi::LockingVarInfo)
    @lock vi.lock begin
        vi.inner[] = map_accumulators!!(func, vi.inner[])
    end
    return vi
end

has_varnamedvector(vi::LockingVarInfo) = has_varnamedvector(vi.inner[])

function BangBang.push!!(vi::LockingVarInfo, vn::VarName, r, dist::Distribution)
    @lock vi.lock begin
        vi.inner[] = BangBang.push!!(vi.inner[], vn, r, dist)
    end
    return vi
end

syms(vi::LockingVarInfo) = syms(vi.inner[])

function setval!(vi::LockingVarInfo, val, vn::VarName)
    @lock vi.lock begin
        vi.inner[] = setval!(vi.inner[], val, vn)
    end
    return vi
end

keys(vi::LockingVarInfo) = keys(vi.inner[])
haskey(vi::LockingVarInfo, vn::VarName) = haskey(vi.inner[], vn)

is_transformed(vi::LockingVarInfo) = is_transformed(vi.inner[])

function link!!(t::AbstractTransformation, vi::LockingVarInfo, args...)
    @lock vi.lock begin
        vi.inner[] = link!!(t, vi.inner[], args...)
    end
end

function invlink!!(t::AbstractTransformation, vi::LockingVarInfo, args...)
    @lock vi.lock begin
        vi.inner[] = invlink!!(t, vi.inner[], args...)
    end
end

function link(t::AbstractTransformation, vi::LockingVarInfo, args...)
    return LockingVarInfo(link(t, vi.inner[], args...))
end

function invlink(t::AbstractTransformation, vi::LockingVarInfo, args...)
    return LockingVarInfo(invlink(t, vi.inner[], args...))
end

function link!!(t::DynamicTransformation, vi::LockingVarInfo, model::Model)
    @lock vi.lock begin
        vi.inner[] = link!!(t, vi.inner[], model)
    end
    return vi
end

function invlink!!(::DynamicTransformation, vi::LockingVarInfo, model::Model)
    @lock vi.lock begin
        vi.inner[] = invlink!!(t, vi.inner[], model)
    end
    return vi
end

function link(t::DynamicTransformation, vi::LockingVarInfo, model::Model)
    return LockingVarInfo(link(t, vi.inner[], model))
end

function invlink(t::DynamicTransformation, vi::LockingVarInfo, model::Model)
    return LockingVarInfo(invlink(t, vi.inner[], model))
end

# These two StaticTransformation methods needed to resolve ambiguities
function link!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::LockingVarInfo, model::Model
)
    @lock vi.lock begin
        vi.inner[] = link!!(t, vi.inner[], model)
    end
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::LockingVarInfo, model::Model
)
    @lock vi.lock begin
        vi.inner[] = invlink!!(t, vi.inner[], model)
    end
end

function maybe_invlink_before_eval!!(vi::LockingVarInfo, model::Model)
    @lock vi.lock begin
        vi.inner[] = maybe_invlink_before_eval!!(vi.inner[], model)
    end
    return vi
end

# `getindex`
getindex(vi::LockingVarInfo, ::Colon) = getindex(vi.inner[], :)
getindex(vi::LockingVarInfo, vn::VarName) = getindex(vi.inner[], vn)
getindex(vi::LockingVarInfo, vns::AbstractVector{<:VarName}) = getindex(vi.inner[], vns)
function getindex(vi::LockingVarInfo, vn::VarName, dist::Distribution)
    return getindex(vi.inner[], vn, dist)
end
function getindex(vi::LockingVarInfo, vns::AbstractVector{<:VarName}, dist::Distribution)
    return getindex(vi.inner[], vns, dist)
end

function BangBang.setindex!!(vi::LockingVarInfo, vals, vn::VarName)
    @lock vi.lock begin
        vi.inner[] = BangBang.setindex!!(vi.inner[], vals, vn)
    end
    return vi
end
function BangBang.setindex!!(vi::LockingVarInfo, vals, vns::AbstractVector{<:VarName})
    @lock vi.lock begin
        vi.inner[] = BangBang.setindex!!(vi.inner[], vals, vns)
    end
    return vi
end

vector_length(vi::LockingVarInfo) = vector_length(vi.inner[])
vector_getrange(vi::LockingVarInfo, vn::VarName) = vector_getrange(vi.inner[], vn)
function vector_getranges(vi::LockingVarInfo, vns::Vector{<:VarName})
    return vector_getranges(vi.inner[], vns)
end

isempty(vi::LockingVarInfo) = isempty(vi.inner[])
function BangBang.empty!!(vi::LockingVarInfo)
    @lock vi.lock begin
        vi.inner[] = BangBang.empty!!(vi.inner[])
    end
    return vi
end

function resetaccs!!(vi::LockingVarInfo)
    @lock vi.lock begin
        vi.inner[] = resetaccs!!(vi.inner[])
    end
    return vi
end

function setaccs!!(vi::LockingVarInfo, accs::NTuple{N,AbstractAccumulator}) where {N}
    @lock vi.lock begin
        vi.inner[] = setaccs!!(vi.inner[], accs)
    end
    return vi
end

function setacc!!(vi::LockingVarInfo, acc::AbstractAccumulator)
    @lock vi.lock begin
        vi.inner[] = setacc!!(vi.inner[], acc)
    end
    return vi
end

function setaccs!!(vi::LockingVarInfo, accs::AccumulatorTuple)
    @lock vi.lock begin
        vi.inner[] = setaccs!!(vi.inner[], accs)
    end
    return vi
end

values_as(vi::LockingVarInfo) = values_as(vi.inner[])
values_as(vi::LockingVarInfo, ::Type{T}) where {T} = values_as(vi.inner[], T)

function set_transformed!!(vi::LockingVarInfo, val::Bool, vn::VarName)
    @lock vi.lock begin
        vi.inner[] = set_transformed!!(vi.inner[], val, vn)
    end
    return vi
end

is_transformed(vi::LockingVarInfo, vn::VarName) = is_transformed(vi.inner[], vn)
function is_transformed(vi::LockingVarInfo, vns::AbstractVector{<:VarName})
    return is_transformed(vi.inner[], vns)
end

getindex_internal(vi::LockingVarInfo, vn::VarName) = getindex_internal(vi.inner[], vn)

function unflatten(vi::LockingVarInfo, x::AbstractVector)
    return LockingVarInfo(unflatten(vi.inner[], x))
end

function subset(varinfo::LockingVarInfo, vns::AbstractVector{<:VarName})
    return LockingVarInfo(subset(varinfo.inner[], vns))
end

function Base.merge(varinfo_left::LockingVarInfo, varinfo_right::LockingVarInfo)
    return LockingVarInfo(merge(varinfo_left.inner[], varinfo_right.inner[]))
end

function invlink_with_logpdf(vi::LockingVarInfo, vn::VarName, dist, y)
    return invlink_with_logpdf(vi.inner[], vn, dist, y)
end

function from_internal_transform(varinfo::LockingVarInfo, vn::VarName)
    return from_internal_transform(varinfo.inner[], vn)
end
function from_internal_transform(varinfo::LockingVarInfo, vn::VarName, dist)
    return from_internal_transform(varinfo.inner[], vn, dist)
end

function from_linked_internal_transform(varinfo::LockingVarInfo, vn::VarName)
    return from_linked_internal_transform(varinfo.inner[], vn)
end
function from_linked_internal_transform(varinfo::LockingVarInfo, vn::VarName, dist)
    return from_linked_internal_transform(varinfo.inner[], vn, dist)
end
