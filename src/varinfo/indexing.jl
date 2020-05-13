const VarView = Union{Int, UnitRange, Vector{Int}}

"""
    getval(vi::UntypedVarInfo, vview::Union{Int, UnitRange, Vector{Int}})

Return a view `vi.vals[vview]`.
"""
getval(vi::UntypedVarInfo, vview::VarView) = view(vi.metadata.vals, vview)

"""
    setval!(vi::UntypedVarInfo, val, vview::Union{Int, UnitRange, Vector{Int}})

Set the value of `vi.vals[vview]` to `val`.
"""
setval!(vi::UntypedVarInfo, val, vview::VarView) = vi.metadata.vals[vview] = val
function setval!(vi::UntypedVarInfo, val, vview::Vector{UnitRange})
    if length(vview) > 0
        vi.metadata.vals[[i for arr in vview for i in arr]] = val
    end
    return val
end

"""
    getval(vi::VarInfo, vn::VarName)

Return the value(s) of `vn`.

The values may or may not be transformed to Euclidean space.
"""
getval(vi::VarInfo, vn::VarName) = view(getmetadata(vi, vn).vals, getrange(vi, vn))

"""
    setval!(vi::VarInfo, val, vn::VarName)

Set the value(s) of `vn` in the metadata of `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
setval!(vi::VarInfo, val, vn::VarName) = getmetadata(vi, vn).vals[getrange(vi, vn)] = val

"""
    getval(vi::VarInfo, vns::Vector{<:VarName})

Return the value(s) of `vns`.

The values may or may not be transformed to Euclidean space.
"""
function getval(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return mapreduce(vn -> getval(vi, vn), vcat, vns)
end

"""
    getall(vi::VarInfo)

Return the values of all the variables in `vi`.

The values may or may not be transformed to Euclidean space.
"""
getall(vi::UntypedVarInfo) = vi.metadata.vals
getall(vi::TypedVarInfo) = vcat(_getall(vi.metadata)...)
@generated function _getall(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :(metadata.$f.vals))
    end
    return :($(exprs...),)
end

"""
    setall!(vi::VarInfo, val)

Set the values of all the variables in `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
setall!(vi::UntypedVarInfo, val) = vi.metadata.vals .= val
setall!(vi::TypedVarInfo, val) = _setall!(vi.metadata, val)
@generated function _setall!(metadata::NamedTuple{names}, val, start = 0) where {names}
    expr = Expr(:block)
    start = :(1)
    for f in names
        length = :(length(metadata.$f.vals))
        finish = :($start + $length - 1)
        push!(expr.args, :(metadata.$f.vals .= val[$start:$finish]))
        start = :($start + $length)
    end
    return expr
end

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
"""
    getindex(vi::VarInfo, vn::VarName)
    getindex(vi::VarInfo, vns::Vector{<:VarName})

Return the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s).

If the value(s) is (are) transformed to the Euclidean space, it is
(they are) transformed back.
"""
function getindex(vi::AbstractVarInfo, vn::VarName)
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vn)
    return istrans(vi, vn) ?
        Bijectors.invlink(dist, reconstruct(dist, getval(vi, vn))) :
        reconstruct(dist, getval(vi, vn))
end
function getindex(vi::AbstractVarInfo, vns::Vector{<:VarName})
    @assert haskey(vi, vns[1]) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vns[1])
    return istrans(vi, vns[1]) ?
        Bijectors.invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
        reconstruct(dist, getval(vi, vns), length(vns))
end

"""
    getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})

Return the current value(s) of the random variables sampled by `spl` in `vi`.

The value(s) may or may not be transformed to Euclidean space.
"""
getindex(vi::AbstractVarInfo, spl::SampleFromPrior) = copy(getall(vi))
getindex(vi::AbstractVarInfo, spl::SampleFromUniform) = copy(getall(vi))
getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, _getranges(vi, spl)))
function getindex(vi::TypedVarInfo, spl::Sampler)
    # Gets the ranges as a NamedTuple
    ranges = _getranges(vi, spl)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return vcat(_getindex(vi.metadata, ranges)...)
end
# Recursively builds a tuple of the `vals` of all the symbols
@generated function _getindex(metadata, ranges::NamedTuple{names}) where {names}
    expr = Expr(:tuple)
    for f in names
        push!(expr.args, :(metadata.$f.vals[ranges.$f]))
    end
    return expr
end

"""
    setindex!(vi::VarInfo, val, vn::VarName)

Set the current value(s) of the random variable `vn` in `vi` to `val`.

The value(s) may or may not be transformed to Euclidean space.
"""
setindex!(vi::AbstractVarInfo, val, vn::VarName) = setval!(vi, val, vn)

"""
    setindex!(vi::VarInfo, val, spl::Union{SampleFromPrior, Sampler})

Set the current value(s) of the random variables sampled by `spl` in `vi` to `val`.

The value(s) may or may not be transformed to Euclidean space.
"""
setindex!(vi::AbstractVarInfo, val, spl::SampleFromPrior) = setall!(vi, val)
setindex!(vi::UntypedVarInfo, val, spl::Sampler) = setval!(vi, val, _getranges(vi, spl))
function setindex!(vi::TypedVarInfo, val, spl::Sampler)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = _getranges(vi, spl)
    _setindex!(vi.metadata, val, ranges)
    return val
end
# Recursively writes the entries of `val` to the `vals` fields of all the symbols as if they were a contiguous vector.
@generated function _setindex!(metadata, val, ranges::NamedTuple{names}) where {names}
    expr = Expr(:block)
    offset = :(0)
    for f in names
        f_vals = :(metadata.$f.vals)
        f_range = :(ranges.$f)
        start = :($offset + 1)
        len = :(length($f_range))
        finish = :($offset + $len)
        push!(expr.args, :(@views $f_vals[$f_range] .= val[$start:$finish]))
        offset = :($offset + $len)
    end
    return expr
end

"""
    push!(vi::VarInfo, vn::VarName, r, dist::Distribution)

Push a new random variable `vn` with a sampled value `r` from a distribution `dist` to
the `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution)
    return push!(vi, vn, r, dist, Set{Selector}([]))
end

"""
    push!(vi::VarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler)

Push a new random variable `vn` with a sampled value `r` sampled with a sampler `spl`
from a distribution `dist` to `VarInfo` `vi`.

The sampler is passed here to invalidate its cache where defined.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, spl::Sampler)
    spl.info[:cache_updated] = CACHERESET
    return push!(vi, vn, r, dist, spl.selector)
end
function push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler)
    return push!(vi, vn, r, dist)
end

"""
    push!(vi::VarInfo, vn::VarName, r, dist::Distribution, gid::Selector)

Push a new random variable `vn` with a sampled value `r` sampled with a sampler of
selector `gid` from a distribution `dist` to `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Selector)
    return push!(vi, vn, r, dist, Set([gid]))
end
function push!(
            vi::VarInfo,
            vn::VarName,
            r,
            dist::Distribution,
            gidset::Set{Selector}
            )

    if vi isa UntypedVarInfo
        @assert ~(vn in keys(vi)) "[push!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gidset"
    elseif vi isa TypedVarInfo
        @assert ~(haskey(vi, vn)) "[push!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gidset"
    end

    val = vectorize(dist, r)

    meta = getmetadata(vi, vn)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals); n = length(val)
    push!(meta.ranges, l+1:l+n)
    append!(meta.vals, val)
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, get_num_produce(vi))
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)

    return vi
end
