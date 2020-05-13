## Vectorized value getters and setters ##

const VarView = Union{Int, UnitRange, Vector{Int}}

"""
    getval(vi::UntypedVarInfo, vview::Union{Int, UnitRange, Vector{Int}})

Return a view `vi.vals[vview]`.
"""
function getval(vi::UntypedVarInfo, vview::VarView)
    vals = getmode(vi) isa LinkMode ? vi.metadata.trans_vals : vi.metadata.vals
    return view(vals, vview)
end

"""
    setval!(vi::UntypedVarInfo, val, vview::Union{Int, UnitRange, Vector{Int}})

Set the value of `vi.vals[vview]` to `val`.
"""
function setval!(vi::UntypedVarInfo, val, vview::VarView)
    vals = getmode(vi) isa LinkMode ? vi.metadata.trans_vals : vi.metadata.vals
    return vals[vview] = val
end
function setval!(vi::UntypedVarInfo, val, vview::Vector{UnitRange})
    vals = getmode(vi) isa LinkMode ? vi.metadata.trans_vals : vi.metadata.vals
    if length(vview) > 0
        vals[[i for arr in vview for i in arr]] = val
    end
    return val
end

"""
    getval(vi::VarInfo, vn::VarName)

Return the value(s) of `vn`.

The values may or may not be transformed to Euclidean space.
"""
function getval(vi::AbstractVarInfo, vn::VarName)
    metadata = getmetadata(vi, vn)
    vals = getmode(vi) isa LinkMode ? metadata.trans_vals : metadata.vals
    return view(vals, getrange(vi, vn))
end

"""
    setval!(vi::VarInfo, val, vn::VarName)

Set the value(s) of `vn` in the metadata of `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
function setval!(vi::AbstractVarInfo, val, vn::VarName)
    metadata = getmetadata(vi, vn)
    vals = getmode(vi) isa LinkMode ? metadata.trans_vals : metadata.vals
    return vals[getrange(vi, vn)] = val
end

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
function getall(vi::UntypedVarInfo)
    return getmode(vi) isa LinkMode ? vi.metadata.trans_vals : vi.metadata.vals
end
function getall(vi::TypedVarInfo)
    return vcat(_getall(vi.metadata, Val(getmode(vi) isa LinkMode))...)
end
@generated function _getall(metadata::NamedTuple{names}, ::Val{linked}) where {names, linked}
    exprs = []
    for f in names
        if linked
            push!(exprs, :(metadata.$f.trans_vals))
        else
            push!(exprs, :(metadata.$f.vals))
        end
    end
    return :($(exprs...),)
end

"""
    setall!(vi::VarInfo, val)

Set the values of all the variables in `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
function setall!(vi::UntypedVarInfo, val)
    vals = getmode(vi) isa LinkMode ? vi.metadata.trans_vals : vi.metadata.vals
    return vals .= val
end
setall!(vi::TypedVarInfo, val) = _setall!(vi.metadata, val, Val(getmode(vi) isa LinkMode))
@generated function _setall!(metadata::NamedTuple{names}, val, ::Val{true}, start = 0) where {names}
    expr = Expr(:block)
    start = :(1)
    for f in names
        length = :(length(metadata.$f.trans_vals))
        finish = :($start + $length - 1)
        push!(expr.args, :(metadata.$f.trans_vals .= val[$start:$finish]))
        start = :($start + $length)
    end
    return expr
end
@generated function _setall!(metadata::NamedTuple{names}, val, ::Val{false}, start = 0) where {names}
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

## VarName getindex and setindex! ##

function zygote_setval!(vi, val, vn)
    return setval!(vi, val, vn)
end

"""
    getindex(vi::VarInfo, vn::VarName, dist::Distribution)
    getindex(vi::VarInfo, vns::Vector{<:VarName}, dist::Distribution)

Return the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s) `dist`.

If the value(s) is (are) transformed to the Euclidean space, it is
(they are) transformed back.
"""
function Base.getindex(
    vi::AbstractVarInfo,
    vn::VarName,
    dist::Distribution,
)
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    trans = istrans(vi, vn)
    if has_fixed_support(vi)
        set_fixed_support!(vi, bijector(dist) == bijector(getinitdist(vi, vn)))
    end
    if getmode(vi) isa LinkMode && trans
        trans_val = reconstruct(dist, getval(vi, vn))
        val = Bijectors.invlink(dist, trans_val)
        zygote_setval!(invlink(vi), value(vectorize(dist, val)), vn)
    elseif getmode(vi) isa InitLinkMode && trans
        val = reconstruct(dist, getval(vi, vn))
        trans_val = Bijectors.link(dist, val)
        zygote_setval!(link(vi), vectorize(dist, trans_val), vn)
    else
        val = reconstruct(dist, getval(vi, vn))
    end
    return val
end
function Base.getindex(
    vi::AbstractVarInfo,
    vn::VarName,
)
    @assert getmode(vi) isa StandardMode
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    return reconstruct(getinitdist(vi, vn), getval(vi, vn))
end
function Base.getindex(
    vi::AbstractVarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
)
    return mapreduce(hcat, vns) do vn
        vi[vn, dist]
    end
end
function Base.getindex(
    vi::AbstractVarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
)
    return broadcast(vns, dists) do vn, dist
        vi[vn, dist]
    end
end
function Base.getindex(
    vi::AbstractVarInfo,
    vns::Vector{<:VarName},
)
    return map(vns) do vn
        vi[vn]
    end
end

"""
    setindex!(vi::VarInfo, val, vn::VarName)

Set the current value(s) of the random variable `vn` in `vi` to `val`.

The value(s) may or may not be transformed to Euclidean space.
"""
function setindex!(vi::AbstractVarInfo, val, vn::VarName, dist::Distribution)
    @assert haskey(vi, vn) "[DynamicPPL] variable not found in VarInfo."
    trans = istrans(vi, vn)
    if getmode(vi) isa LinkMode && trans
        trans_val = Bijectors.link(dist, val)
        setval!(vi, vectorize(dist, trans_val), vn)
        setval!(invlink(vi), vectorize(dist, val), vn)
    elseif getmode(vi) isa InitLinkMode && trans
        trans_val = Bijectors.link(dist, val)
        setval!(vi, vectorize(dist, val), vn)
        setval!(link(vi), vectorize(dist, trans_val), vn)
    else
        setval!(vi, vectorize(dist, val), vn)
    end
    return vi
end
function setindex!(vi::AbstractVarInfo, val, vn::VarName)
    @assert getmode(vi) isa StandardMode
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    setval!(vi, vectorize(getinitdist(vi, vn), val), vn)
    return vi
end

## Sampler getindex and setindex! ##

"""
    getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})

Return the current value(s) of the random variables sampled by `spl` in `vi`.

The value(s) may or may not be transformed to Euclidean space.
"""
function getindex(vi::AbstractVarInfo, spl::Union{SampleFromPrior, SampleFromUniform})
    return copy(getall(vi))
end
function getindex(vi::UntypedVarInfo, spl::Sampler)
    return copy(getval(vi, getranges(vi, spl)))
end
function getindex(vi::TypedVarInfo, spl::Sampler)
    # Gets the ranges as a NamedTuple
    ranges = getranges(vi, spl)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return vcat(_getindex(vi.metadata, ranges, Val(getmode(vi) isa LinkMode))...)
end
# Recursively builds a tuple of the `vals` of all the symbols
@generated function _getindex(
    metadata,
    ranges::NamedTuple{names},
    ::Val{false},
) where {names}
    expr = Expr(:tuple)
    for f in names
        push!(expr.args, :(metadata.$f.vals[ranges.$f]))
    end
    return expr
end
@generated function _getindex(
    metadata,
    ranges::NamedTuple{names},
    ::Val{true},
) where {names}
    expr = Expr(:tuple)
    for f in names
        push!(expr.args, :(metadata.$f.trans_vals[ranges.$f]))
    end
    return expr
end

"""
    setindex!(vi::VarInfo, val, spl::Union{SampleFromPrior, Sampler})

Set the current value(s) of the random variables sampled by `spl` in `vi` to `val`.

The value(s) may or may not be transformed to Euclidean space.
"""
function setindex!(vi::AbstractVarInfo, val, spl::SampleFromPrior)
    setall!(vi, val)
    setsynced!(vi, false)
    return vi
end
function setindex!(vi::UntypedVarInfo, val, spl::Sampler)
    setval!(vi, val, getranges(vi, spl))
    setsynced!(vi, false)
    return vi
end
function setindex!(vi::TypedVarInfo, val, spl::Sampler)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = getranges(vi, spl)
    _setindex!(vi.metadata, val, ranges, Val(getmode(vi) isa LinkMode))
    setsynced!(vi, false)
    return vi
end
# Recursively writes the entries of `val` to the `vals` fields of all the symbols as if they were a contiguous vector.
@generated function _setindex!(
    metadata,
    val,
    ranges::NamedTuple{names},
    ::Val{linked},
) where {names, linked}
    expr = Expr(:block)
    offset = :(0)
    for f in names
        f_vals = linked ? :(metadata.$f.trans_vals) : :(metadata.$f.vals)
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
    val,
    dist::Distribution,
    gidset::Set{Selector},
)
    if vi isa UntypedVarInfo
        @assert ~(vn in keys(vi)) "[push!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gidset"
    elseif vi isa TypedVarInfo
        @assert ~(haskey(vi, vn)) "[push!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gidset"
    end

    meta = getmetadata(vi, vn)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)

    vectorized_val = vectorize(dist, val)
    l = length(meta.vals); n = length(vectorized_val)
    push!(meta.ranges, l+1:l+n)
    if getmode(vi) isa LinkMode || getmode(vi) isa InitLinkMode
        append!(meta.vals, vectorized_val)
        trans_val = Bijectors.link(dist, val)
        append!(meta.trans_vals, vectorize(dist, trans_val))
    else
        append!(meta.vals, vectorized_val)
        append!(meta.trans_vals, vectorized_val)
        setsynced!(vi, false)
    end
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, get_num_produce(vi))
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)

    return vi
end
