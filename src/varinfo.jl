####
#### Types for typed and untyped VarInfo
####

####################
# VarInfo metadata #
####################

"""
The `Metadata` struct stores some metadata about the parameters of the model. This helps
query certain information about a variable, such as its distribution, which samplers
sample this variable, its value and whether this value is transformed to real space or
not.

Let `md` be an instance of `Metadata`:
- `md.vns` is the vector of all `VarName` instances.
- `md.idcs` is the dictionary that maps each `VarName` instance to its index in
 `md.vns`, `md.ranges` `md.dists`, `md.orders` and `md.flags`.
- `md.vns[md.idcs[vn]] == vn`.
- `md.dists[md.idcs[vn]]` is the distribution of `vn`.
- `md.gids[md.idcs[vn]]` is the set of algorithms used to sample `vn`. This is used in
 the Gibbs sampling process.
- `md.orders[md.idcs[vn]]` is the number of `observe` statements before `vn` is sampled.
- `md.ranges[md.idcs[vn]]` is the index range of `vn` in `md.vals`.
- `md.vals[md.ranges[md.idcs[vn]]]` is the vector of values of corresponding to `vn`.
- `md.flags` is a dictionary of true/false flags. `md.flags[flag][md.idcs[vn]]` is the
 value of `flag` corresponding to `vn`.

To make `md::Metadata` type stable, all the `md.vns` must have the same symbol
and distribution type. However, one can have a Julia variable, say `x`, that is a
matrix or a hierarchical array sampled in partitions, e.g.
`x[1][:] ~ MvNormal(zeros(2), I); x[2][:] ~ MvNormal(ones(2), I)`, and is managed by
a single `md::Metadata` so long as all the distributions on the RHS of `~` are of the
same type. Type unstable `Metadata` will still work but will have inferior performance.
When sampling, the first iteration uses a type unstable `Metadata` for all the
variables then a specialized `Metadata` is used for each symbol along with a function
barrier to make the rest of the sampling type stable.
"""
struct Metadata{
    TIdcs<:Dict{<:VarName,Int},
    TDists<:AbstractVector{<:Distribution},
    TVN<:AbstractVector{<:VarName},
    TVal<:AbstractVector{<:Real},
    TGIds<:AbstractVector{Set{Selector}},
}
    # Mapping from the `VarName` to its integer index in `vns`, `ranges` and `dists`
    idcs::TIdcs # Dict{<:VarName,Int}

    # Vector of identifiers for the random variables, where `vns[idcs[vn]] == vn`
    vns::TVN # AbstractVector{<:VarName}

    # Vector of index ranges in `vals` corresponding to `vns`
    # Each `VarName` `vn` has a single index or a set of contiguous indices in `vals`
    ranges::Vector{UnitRange{Int}}

    # Vector of values of all the univariate, multivariate and matrix variables
    # The value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`
    vals::TVal # AbstractVector{<:Real}

    # Vector of distributions correpsonding to `vns`
    dists::TDists # AbstractVector{<:Distribution}

    # Vector of sampler ids corresponding to `vns`
    # Each random variable can be sampled using multiple samplers, e.g. in Gibbs, hence the `Set`
    gids::TGIds # AbstractVector{Set{Selector}}

    # Number of `observe` statements before each random variable is sampled
    orders::Vector{Int}

    # Each `flag` has a `BitVector` `flags[flag]`, where `flags[flag][i]` is the true/false flag value corresonding to `vns[i]`
    flags::Dict{String,BitVector}
end

###########
# VarInfo #
###########

"""
```
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
```

A light wrapper over one or more instances of `Metadata`. Let `vi` be an instance of
`VarInfo`. If `vi isa VarInfo{<:Metadata}`, then only one `Metadata` instance is used
for all the sybmols. `VarInfo{<:Metadata}` is aliased `UntypedVarInfo`. If
`vi isa VarInfo{<:NamedTuple}`, then `vi.metadata` is a `NamedTuple` that maps each
symbol used on the LHS of `~` in the model to its `Metadata` instance. The latter allows
for the type specialization of `vi` after the first sampling iteration when all the
symbols have been observed. `VarInfo{<:NamedTuple}` is aliased `TypedVarInfo`.

Note: It is the user's responsibility to ensure that each "symbol" is visited at least
once whenever the model is called, regardless of any stochastic branching. Each symbol
refers to a Julia variable and can be a hierarchical array of many random variables, e.g. `x[1] ~ ...` and `x[2] ~ ...` both have the same symbol `x`.
"""
struct VarInfo{Tmeta,Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
const UntypedVarInfo = VarInfo{<:Metadata}
const TypedVarInfo = VarInfo{<:NamedTuple}
const VarInfoOrThreadSafeVarInfo{Tmeta} = Union{
    VarInfo{Tmeta},ThreadSafeVarInfo{<:VarInfo{Tmeta}}
}

# NOTE: This is kind of weird, but it effectively preserves the "old"
# behavior where we're allowed to call `link!` on the same `VarInfo`
# multiple times.
transformation(vi::VarInfo) = DynamicTransformation()

function VarInfo(old_vi::VarInfo, spl, x::AbstractVector)
    md = newmetadata(old_vi.metadata, Val(getspace(spl)), x)
    return VarInfo(
        md, Base.RefValue{eltype(x)}(getlogp(old_vi)), Ref(get_num_produce(old_vi))
    )
end

function VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    varinfo = VarInfo()
    model(rng, varinfo, sampler, context)
    return TypedVarInfo(varinfo)
end
VarInfo(model::Model, args...) = VarInfo(Random.default_rng(), model, args...)

unflatten(vi::VarInfo, x::AbstractVector) = unflatten(vi, SampleFromPrior(), x)

# TODO: deprecate.
unflatten(vi::VarInfo, spl::AbstractSampler, x::AbstractVector) = VarInfo(vi, spl, x)

# without AbstractSampler
function VarInfo(rng::Random.AbstractRNG, model::Model, context::AbstractContext)
    return VarInfo(rng, model, SampleFromPrior(), context)
end

# TODO: Remove `space` argument when no longer needed. Ref: https://github.com/TuringLang/DynamicPPL.jl/issues/573
function newmetadata(metadata::Metadata, space, x)
    return Metadata(
        metadata.idcs,
        metadata.vns,
        metadata.ranges,
        x,
        metadata.dists,
        metadata.gids,
        metadata.orders,
        metadata.flags,
    )
end

@generated function newmetadata(
    metadata::NamedTuple{names}, ::Val{space}, x
) where {names,space}
    exprs = []
    offset = :(0)
    for f in names
        mdf = :(metadata.$f)
        if inspace(f, space) || length(space) == 0
            len = :(sum(length, $mdf.ranges))
            push!(
                exprs,
                :(
                    $f = Metadata(
                        $mdf.idcs,
                        $mdf.vns,
                        $mdf.ranges,
                        x[($offset + 1):($offset + $len)],
                        $mdf.dists,
                        $mdf.gids,
                        $mdf.orders,
                        $mdf.flags,
                    )
                ),
            )
            offset = :($offset + $len)
        else
            push!(exprs, :($f = $mdf))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

####
#### Internal functions
####

"""
    Metadata()

Construct an empty type unstable instance of `Metadata`.
"""
function Metadata()
    vals = Vector{Real}()
    flags = Dict{String,BitVector}()
    flags["del"] = BitVector()
    flags["trans"] = BitVector()

    return Metadata(
        Dict{VarName,Int}(),
        Vector{VarName}(),
        Vector{UnitRange{Int}}(),
        vals,
        Vector{Distribution}(),
        Vector{Set{Selector}}(),
        Vector{Int}(),
        flags,
    )
end

"""
    empty!(meta::Metadata)

Empty the fields of `meta`.

This is useful when using a sampling algorithm that assumes an empty `meta`, e.g. `SMC`.
"""
function empty!(meta::Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end

    return meta
end

# Removes the first element of a NamedTuple. The pairs in a NamedTuple are ordered, so this is well-defined.
if VERSION < v"1.1"
    _tail(nt::NamedTuple{names}) where {names} = NamedTuple{Base.tail(names)}(nt)
else
    _tail(nt::NamedTuple) = Base.tail(nt)
end

function subset(varinfo::UntypedVarInfo, vns::AbstractVector{<:VarName})
    metadata = subset(varinfo.metadata, vns)
    return VarInfo(metadata, varinfo.logp, varinfo.num_produce)
end

function subset(varinfo::TypedVarInfo, vns::AbstractVector{<:VarName{sym}}) where {sym}
    # If all the variables are using the same symbol, then we can just extract that field from the metadata.
    metadata = subset(getfield(varinfo.metadata, sym), vns)
    return VarInfo(NamedTuple{(sym,)}(tuple(metadata)), varinfo.logp, varinfo.num_produce)
end

function subset(varinfo::TypedVarInfo, vns::AbstractVector{<:VarName})
    syms = Tuple(unique(map(getsym, vns)))
    metadatas = map(syms) do sym
        subset(getfield(varinfo.metadata, sym), filter(==(sym) ∘ getsym, vns))
    end

    return VarInfo(NamedTuple{syms}(metadatas), varinfo.logp, varinfo.num_produce)
end

function subset(metadata::Metadata, vns_given::AbstractVector{<:VarName})
    # TODO: Should we error if `vns` contains a variable that is not in `metadata`?
    # For each `vn` in `vns`, get the variables subsumed by `vn`.
    vns = mapreduce(vcat, vns_given) do vn
        filter(Base.Fix1(subsumes, vn), metadata.vns)
    end
    indices_for_vns = map(Base.Fix1(getindex, metadata.idcs), vns)
    indices = Dict(vn => i for (i, vn) in enumerate(vns))
    # Construct new `vals` and `ranges`.
    vals_original = metadata.vals
    ranges_original = metadata.ranges
    # Allocate the new `vals`. and `ranges`.
    vals = similar(metadata.vals, sum(length, ranges_original[indices_for_vns]))
    ranges = similar(ranges_original)
    # The new range `r` for `vns[i]` is offset by `offset` and
    # has the same length as the original range `r_original`.
    # The new `indices` (from above) ensures ordering according to `vns`.
    # NOTE: This means that the order of the variables in `vns` defines the order
    # in the resulting `varinfo`! This can have performance implications, e.g.
    # if in the model we have something like
    #
    #     for i = 1:N
    #         x[i] ~ Normal()
    #     end
    #
    # and we then we do
    #
    #    subset(varinfo, [@varname(x[i]) for i in shuffle(keys(varinfo))])
    #
    # the resulting `varinfo` will have `vals` ordered differently from the
    # original `varinfo`, which can have performance implications.
    offset = 0
    for (idx, idx_original) in enumerate(indices_for_vns)
        r_original = ranges_original[idx_original]
        r = (offset + 1):(offset + length(r_original))
        vals[r] = vals_original[r_original]
        ranges[idx] = r
        offset = r[end]
    end

    flags = Dict(k => v[indices_for_vns] for (k, v) in metadata.flags)
    return Metadata(
        indices,
        vns,
        ranges,
        vals,
        metadata.dists[indices_for_vns],
        metadata.gids,
        metadata.orders[indices_for_vns],
        flags,
    )
end

function Base.merge(varinfo_left::VarInfo, varinfo_right::VarInfo)
    return _merge(varinfo_left, varinfo_right)
end

function _merge(varinfo_left::VarInfo, varinfo_right::VarInfo)
    metadata = merge_metadata(varinfo_left.metadata, varinfo_right.metadata)
    return VarInfo(
        metadata, Ref(getlogp(varinfo_right)), Ref(get_num_produce(varinfo_right))
    )
end

@generated function merge_metadata(
    metadata_left::NamedTuple{names_left}, metadata_right::NamedTuple{names_right}
) where {names_left,names_right}
    names = Expr(:tuple)
    vals = Expr(:tuple)
    # Loop over `names_left` first because we want to preserve the order of the variables.
    for sym in names_left
        push!(names.args, QuoteNode(sym))
        if sym in names_right
            push!(vals.args, :(merge_metadata(metadata_left.$sym, metadata_right.$sym)))
        else
            push!(vals.args, :(metadata_left.$sym))
        end
    end
    # Loop over remaining variables in `names_right`.
    names_right_only = filter(∉(names_left), names_right)
    for sym in names_right_only
        push!(names.args, QuoteNode(sym))
        push!(vals.args, :(metadata_right.$sym))
    end

    return :(NamedTuple{$names}($vals))
end

function merge_metadata(metadata_left::Metadata, metadata_right::Metadata)
    # Extract the varnames.
    vns_left = metadata_left.vns
    vns_right = metadata_right.vns
    vns_both = union(vns_left, vns_right)

    # Determine `eltype` of `vals`.
    T_left = eltype(metadata_left.vals)
    T_right = eltype(metadata_right.vals)
    T = promote_type(T_left, T_right)
    # TODO: Is this necessary?
    if !(T <: Real)
        T = Real
    end

    # Determine `eltype` of `dists`.
    D_left = eltype(metadata_left.dists)
    D_right = eltype(metadata_right.dists)
    D = promote_type(D_left, D_right)
    # TODO: Is this necessary?
    if !(D <: Distribution)
        D = Distribution
    end

    # Initialize required fields for `metadata`.
    vns = VarName[]
    idcs = Dict{VarName,Int}()
    ranges = Vector{UnitRange{Int}}()
    vals = T[]
    dists = D[]
    gids = metadata_right.gids  # NOTE: giving precedence to `metadata_right`
    orders = Int[]
    flags = Dict{String,BitVector}()
    # Initialize the `flags`.
    for k in union(keys(metadata_left.flags), keys(metadata_right.flags))
        flags[k] = BitVector()
    end

    # Range offset.
    offset = 0

    for (idx, vn) in enumerate(vns_both)
        # `idcs`
        idcs[vn] = idx
        # `vns`
        push!(vns, vn)
        if vn in vns_left && vn in vns_right
            # `vals`: only valid if they're the length.
            vals_left = getval(metadata_left, vn)
            vals_right = getval(metadata_right, vn)
            @assert length(vals_left) == length(vals_right)
            append!(vals, vals_right)
            # `ranges`
            r = (offset + 1):(offset + length(vals_left))
            push!(ranges, r)
            offset = r[end]
            # `dists`: only valid if they're the same.
            dist_right = getdist(metadata_right, vn)
            # Give precedence to `metadata_right`.
            push!(dists, dist_right)
            # `orders`: giving precedence to `metadata_right`
            push!(orders, getorder(metadata_right, vn))
            # `flags`
            for k in keys(flags)
                # Using `metadata_right`; should we?
                push!(flags[k], is_flagged(metadata_right, vn, k))
            end
        elseif vn in vns_left
            # Just extract the metadata from `metadata_left`.
            # `vals`
            vals_left = getval(metadata_left, vn)
            append!(vals, vals_left)
            # `ranges`
            r = (offset + 1):(offset + length(vals_left))
            push!(ranges, r)
            offset = r[end]
            # `dists`
            dist_left = getdist(metadata_left, vn)
            push!(dists, dist_left)
            # `orders`
            push!(orders, getorder(metadata_left, vn))
            # `flags`
            for k in keys(flags)
                push!(flags[k], is_flagged(metadata_left, vn, k))
            end
        else
            # Just extract the metadata from `metadata_right`.
            # `vals`
            vals_right = getval(metadata_right, vn)
            append!(vals, vals_right)
            # `ranges`
            r = (offset + 1):(offset + length(vals_right))
            push!(ranges, r)
            offset = r[end]
            # `dists`
            dist_right = getdist(metadata_right, vn)
            push!(dists, dist_right)
            # `orders`
            push!(orders, getorder(metadata_right, vn))
            # `flags`
            for k in keys(flags)
                push!(flags[k], is_flagged(metadata_right, vn, k))
            end
        end
    end

    return Metadata(idcs, vns, ranges, vals, dists, gids, orders, flags)
end

const VarView = Union{Int,UnitRange,Vector{Int}}

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
    getmetadata(vi::VarInfo, vn::VarName)

Return the metadata in `vi` that belongs to `vn`.
"""
getmetadata(vi::VarInfo, vn::VarName) = vi.metadata
getmetadata(vi::TypedVarInfo, vn::VarName) = getfield(vi.metadata, getsym(vn))

"""
    getidx(vi::VarInfo, vn::VarName)

Return the index of `vn` in the metadata of `vi` corresponding to `vn`.
"""
getidx(vi::VarInfo, vn::VarName) = getidx(getmetadata(vi, vn), vn)
getidx(md::Metadata, vn::VarName) = md.idcs[vn]

"""
    getrange(vi::VarInfo, vn::VarName)

Return the index range of `vn` in the metadata of `vi`.
"""
getrange(vi::VarInfo, vn::VarName) = getrange(getmetadata(vi, vn), vn)
getrange(md::Metadata, vn::VarName) = md.ranges[getidx(md, vn)]

"""
    setrange!(vi::VarInfo, vn::VarName, range)

Set the index range of `vn` in the metadata of `vi` to `range`.
"""
setrange!(vi::VarInfo, vn::VarName, range) = setrange!(getmetadata(vi, vn), vn, range)
setrange!(md::Metadata, vn::VarName, range) = md.ranges[getidx(md, vn)] = range

"""
    getranges(vi::VarInfo, vns::Vector{<:VarName})

Return the indices of `vns` in the metadata of `vi` corresponding to `vn`.
"""
function getranges(vi::VarInfo, vns::Vector{<:VarName})
    return mapreduce(vn -> getrange(vi, vn), vcat, vns; init=Int[])
end

"""
    getdist(vi::VarInfo, vn::VarName)

Return the distribution from which `vn` was sampled in `vi`.
"""
getdist(vi::VarInfo, vn::VarName) = getdist(getmetadata(vi, vn), vn)
getdist(md::Metadata, vn::VarName) = md.dists[getidx(md, vn)]

"""
    getval(vi::VarInfo, vn::VarName)

Return the value(s) of `vn`.

The values may or may not be transformed to Euclidean space.
"""
getval(vi::VarInfo, vn::VarName) = getval(getmetadata(vi, vn), vn)
getval(md::Metadata, vn::VarName) = view(md.vals, getrange(md, vn))

"""
    setval!(vi::VarInfo, val, vn::VarName)

Set the value(s) of `vn` in the metadata of `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
setval!(vi::VarInfo, val, vn::VarName) = setval!(getmetadata(vi, vn), val, vn)
function setval!(md::Metadata, val::AbstractVector, vn::VarName)
    return md.vals[getrange(md, vn)] = val
end
function setval!(md::Metadata, val, vn::VarName)
    return md.vals[getrange(md, vn)] = vectorize(getdist(md, vn), val)
end

"""
    getval(vi::VarInfo, vns::Vector{<:VarName})

Return the value(s) of `vns`.

The values may or may not be transformed to Euclidean space.
"""
getval(vi::VarInfo, vns::Vector{<:VarName}) = mapreduce(Base.Fix1(getval, vi), vcat, vns)

"""
    getall(vi::VarInfo)

Return the values of all the variables in `vi`.

The values may or may not be transformed to Euclidean space.
"""
getall(vi::UntypedVarInfo) = getall(vi.metadata)
# NOTE: `mapreduce` over `NamedTuple` results in worse type-inference.
# See for example https://github.com/JuliaLang/julia/pull/46381.
getall(vi::TypedVarInfo) = reduce(vcat, map(getall, vi.metadata))
function getall(md::Metadata)
    return mapreduce(Base.Fix1(getval, md), vcat, md.vns; init=similar(md.vals, 0))
end

"""
    setall!(vi::VarInfo, val)

Set the values of all the variables in `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
function setall!(vi::UntypedVarInfo, val)
    for r in vi.metadata.ranges
        vi.metadata.vals[r] .= val[r]
    end
end
setall!(vi::TypedVarInfo, val) = _setall!(vi.metadata, val)
@generated function _setall!(metadata::NamedTuple{names}, val) where {names}
    expr = Expr(:block)
    start = :(1)
    for f in names
        length = :(sum(length, metadata.$f.ranges))
        finish = :($start + $length - 1)
        push!(expr.args, :(copyto!(metadata.$f.vals, 1, val, $start, $length)))
        start = :($start + $length)
    end
    return expr
end

"""
    getgid(vi::VarInfo, vn::VarName)

Return the set of sampler selectors associated with `vn` in `vi`.
"""
getgid(vi::VarInfo, vn::VarName) = getmetadata(vi, vn).gids[getidx(vi, vn)]

function settrans!!(vi::VarInfo, trans::Bool, vn::VarName)
    if trans
        set_flag!(vi, vn, "trans")
    else
        unset_flag!(vi, vn, "trans")
    end

    return vi
end

function settrans!!(vi::VarInfo, trans::Bool)
    for vn in keys(vi)
        settrans!!(vi, trans, vn)
    end

    return vi
end

settrans!!(vi::VarInfo, trans::NoTransformation) = settrans!!(vi, false)
# HACK: This is necessary to make something like `link!!(transformation, vi, model)`
# work properly, which will transform the variables according to `transformation`
# and then call `settrans!!(vi, transformation)`. An alternative would be to add
# the `transformation` to the `VarInfo` object, but at the moment doesn't seem
# worth it as `VarInfo` has its own way of handling transformations.
settrans!!(vi::VarInfo, trans::AbstractTransformation) = settrans!!(vi, true)

"""
    syms(vi::VarInfo)

Returns a tuple of the unique symbols of random variables sampled in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(getsym, vi.metadata.vns)))  # get all symbols
syms(vi::TypedVarInfo) = keys(vi.metadata)

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
@inline function _getidcs(vi::UntypedVarInfo, ::SampleFromPrior)
    return filter(i -> isempty(vi.metadata.gids[i]), 1:length(vi.metadata.gids))
end
# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
@inline function _getidcs(vi::TypedVarInfo, ::SampleFromPrior)
    return _getidcs(vi.metadata)
end
@generated function _getidcs(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findinds(metadata.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get all indices of variables belonging to a given sampler
@inline function _getidcs(vi::VarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    # Checks if cache is valid, i.e. no new pushes were made, to return the cached idcs
    # Otherwise, it recomputes the idcs and caches it
    #if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
    #    spl.info[:idcs]
    #else
    #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
    idcs = _getidcs(vi, spl.selector, Val(getspace(spl)))
    #spl.info[:idcs] = idcs
    #end
    return idcs
end
@inline _getidcs(vi::UntypedVarInfo, s::Selector, space) = findinds(vi.metadata, s, space)
@inline _getidcs(vi::TypedVarInfo, s::Selector, space) = _getidcs(vi.metadata, s, space)
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@generated function _getidcs(
    metadata::NamedTuple{names}, s::Selector, ::Val{space}
) where {names,space}
    exprs = []
    # Iterate through each varname in metadata.
    for f in names
        # If the varname is in the sampler space
        # or the sample space is empty (all variables)
        # then return the indices for that variable.
        if inspace(f, space) || length(space) == 0
            push!(exprs, :($f = findinds(metadata.$f, s, Val($space))))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end
@inline function findinds(f_meta, s, ::Val{space}) where {space}
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter(
        (i) ->
            (s in f_meta.gids[i] || isempty(f_meta.gids[i])) &&
                (isempty(space) || inspace(f_meta.vns[i], space)),
        1:length(f_meta.gids),
    )
end
@inline function findinds(f_meta)
    # Get all the idcs of the vns
    return filter((i) -> isempty(f_meta.gids[i]), 1:length(f_meta.gids))
end

# Get all vns of variables belonging to spl
_getvns(vi::VarInfo, spl::Sampler) = _getvns(vi, spl.selector, Val(getspace(spl)))
function _getvns(vi::VarInfo, spl::Union{SampleFromPrior,SampleFromUniform})
    return _getvns(vi, Selector(), Val(()))
end
function _getvns(vi::UntypedVarInfo, s::Selector, space)
    return view(vi.metadata.vns, _getidcs(vi, s, space))
end
function _getvns(vi::TypedVarInfo, s::Selector, space)
    return _getvns(vi.metadata, _getidcs(vi, s, space))
end
# Get a NamedTuple for all the `vns` of indices `idcs`, one entry for each symbol
@generated function _getvns(metadata, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = metadata.$f.vns[idcs.$f]))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get the index (in vals) ranges of all the vns of variables belonging to spl
@inline function _getranges(vi::VarInfo, spl::Sampler)
    ## Uncomment the spl.info stuff when it is concretely typed, not Dict{Symbol, Any}
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    #if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
    #    spl.info[:ranges]
    #else
    #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
    ranges = _getranges(vi, spl.selector, Val(getspace(spl)))
    #spl.info[:ranges] = ranges
    return ranges
    #end
end
# Get the index (in vals) ranges of all the vns of variables belonging to selector `s` in `space`
@inline function _getranges(vi::VarInfo, s::Selector, space)
    return _getranges(vi, _getidcs(vi, s, space))
end
@inline function _getranges(vi::UntypedVarInfo, idcs::Vector{Int})
    return mapreduce(i -> vi.metadata.ranges[i], vcat, idcs; init=Int[])
end
@inline _getranges(vi::TypedVarInfo, idcs::NamedTuple) = _getranges(vi.metadata, idcs)

@generated function _getranges(metadata::NamedTuple, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findranges(metadata.$f.ranges, idcs.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

@inline function findranges(f_ranges, f_idcs)
    # Old implementation was using `mapreduce` but turned out
    # to be type-unstable.
    results = Int[]
    for i in f_idcs
        append!(results, f_ranges[i])
    end
    return results
end

"""
    set_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = true
end

####
#### APIs for typed and untyped VarInfo
####

# VarInfo

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{Float64}(0.0), Ref(0))

"""
    TypedVarInfo(vi::UntypedVarInfo)

This function finds all the unique `sym`s from the instances of `VarName{sym}` found in
`vi.metadata.vns`. It then extracts the metadata associated with each symbol from the
global `vi.metadata` field. Finally, a new `VarInfo` is created with a new `metadata` as
a `NamedTuple` mapping from symbols to type-stable `Metadata` instances, one for each
symbol.
"""
function TypedVarInfo(vi::UntypedVarInfo)
    meta = vi.metadata
    new_metas = Metadata[]
    # Symbols of all instances of `VarName{sym}` in `vi.vns`
    syms_tuple = Tuple(syms(vi))
    for s in syms_tuple
        # Find all indices in `vns` with symbol `s`
        inds = findall(vn -> getsym(vn) === s, meta.vns)
        n = length(inds)
        # New `vns`
        sym_vns = getindex.((meta.vns,), inds)
        # New idcs
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        # New dists
        sym_dists = getindex.((meta.dists,), inds)
        # New gids, can make a resizeable FillArray
        sym_gids = getindex.((meta.gids,), inds)
        @assert length(sym_gids) <= 1 || all(x -> x == sym_gids[1], @view sym_gids[2:end])
        # New orders
        sym_orders = getindex.((meta.orders,), inds)
        # New flags
        sym_flags = Dict(a => meta.flags[a][inds] for a in keys(meta.flags))

        # Extract new ranges and vals
        _ranges = getindex.((meta.ranges,), inds)
        # `copy.()` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(meta.vals[_ranges[i]]) for i in 1:n]
        sym_ranges = Vector{eltype(_ranges)}(undef, n)
        start = 0
        for i in 1:n
            sym_ranges[i] = (start + 1):(start + length(_vals[i]))
            start += length(_vals[i])
        end
        sym_vals = foldl(vcat, _vals)

        push!(
            new_metas,
            Metadata(
                sym_idcs,
                sym_vns,
                sym_ranges,
                sym_vals,
                sym_dists,
                sym_gids,
                sym_orders,
                sym_flags,
            ),
        )
    end
    logp = getlogp(vi)
    num_produce = get_num_produce(vi)
    nt = NamedTuple{syms_tuple}(Tuple(new_metas))
    return VarInfo(nt, Ref(logp), Ref(num_produce))
end
TypedVarInfo(vi::TypedVarInfo) = vi

function BangBang.empty!!(vi::VarInfo)
    _empty!(vi.metadata)
    resetlogp!!(vi)
    reset_num_produce!(vi)
    return vi
end
@inline _empty!(metadata::Metadata) = empty!(metadata)
@generated function _empty!(metadata::NamedTuple{names}) where {names}
    expr = Expr(:block)
    for f in names
        push!(expr.args, :(empty!(metadata.$f)))
    end
    return expr
end

# Functions defined only for UntypedVarInfo
Base.keys(vi::UntypedVarInfo) = keys(vi.metadata.idcs)

# HACK: Necessary to avoid returning `Any[]` which won't dispatch correctly
# on other methods in the codebase which requires `Vector{<:VarName}`.
Base.keys(vi::TypedVarInfo{<:NamedTuple{()}}) = VarName[]
@generated function Base.keys(vi::TypedVarInfo{<:NamedTuple{names}}) where {names}
    expr = Expr(:call)
    push!(expr.args, :vcat)

    for n in names
        push!(expr.args, :(vi.metadata.$n.vns))
    end

    return expr
end

"""
    setgid!(vi::VarInfo, gid::Selector, vn::VarName)

Add `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
function setgid!(vi::VarInfo, gid::Selector, vn::VarName)
    return push!(getmetadata(vi, vn).gids[getidx(vi, vn)], gid)
end

istrans(vi::VarInfo, vn::VarName) = is_flagged(vi, vn, "trans")

getlogp(vi::VarInfo) = vi.logp[]

function setlogp!!(vi::VarInfo, logp)
    vi.logp[] = logp
    return vi
end

function acclogp!!(vi::VarInfo, logp)
    vi.logp[] += logp
    return vi
end

"""
    get_num_produce(vi::VarInfo)

Return the `num_produce` of `vi`.
"""
get_num_produce(vi::VarInfo) = vi.num_produce[]

"""
    set_num_produce!(vi::VarInfo, n::Int)

Set the `num_produce` field of `vi` to `n`.
"""
set_num_produce!(vi::VarInfo, n::Int) = vi.num_produce[] = n

"""
    increment_num_produce!(vi::VarInfo)

Add 1 to `num_produce` in `vi`.
"""
increment_num_produce!(vi::VarInfo) = vi.num_produce[] += 1

"""
    reset_num_produce!(vi::VarInfo)

Reset the value of `num_produce` the log of the joint probability of the observed data
and parameters sampled in `vi` to 0.
"""
reset_num_produce!(vi::VarInfo) = set_num_produce!(vi, 0)

isempty(vi::UntypedVarInfo) = isempty(vi.metadata.idcs)
isempty(vi::TypedVarInfo) = _isempty(vi.metadata)
@generated function _isempty(metadata::NamedTuple{names}) where {names}
    expr = Expr(:&&, :true)
    for f in names
        push!(expr.args, :(isempty(metadata.$f.idcs)))
    end
    return expr
end

# X -> R for all variables associated with given sampler
function link!!(t::DynamicTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # Call `_link!` instead of `link!` to avoid deprecation warning.
    _link!(vi, spl)
    return vi
end

function link!!(
    t::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    spl::AbstractSampler,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Setfield.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, spl, model)
end

"""
    link!(vi::VarInfo, spl::Sampler)

Transform the values of the random variables sampled by `spl` in `vi` from the support
of their distributions to the Euclidean space and set their corresponding `"trans"`
flag values to `true`.
"""
function link!(vi::VarInfo, spl::AbstractSampler)
    Base.depwarn(
        "`link!(varinfo, sampler)` is deprecated, use `link!!(varinfo, sampler, model)` instead.",
        :link!,
    )
    return _link!(vi, spl)
end
function link!(vi::VarInfo, spl::AbstractSampler, spaceval::Val)
    Base.depwarn(
        "`link!(varinfo, sampler, spaceval)` is deprecated, use `link!!(varinfo, sampler, model)` instead.",
        :link!,
    )
    return _link!(vi, spl, spaceval)
end
function _link!(vi::UntypedVarInfo, spl::AbstractSampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = _getvns(vi, spl)
    if ~istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            _inner_transform!(vi, vn, dist, link_transform(dist))
            settrans!!(vi, true, vn)
        end
    else
        @warn("[DynamicPPL] attempt to link a linked vi")
    end
end
function _link!(vi::TypedVarInfo, spl::AbstractSampler)
    return _link!(vi, spl, Val(getspace(spl)))
end
function _link!(vi::TypedVarInfo, spl::AbstractSampler, spaceval::Val)
    vns = _getvns(vi, spl)
    return _link!(vi.metadata, vi, vns, spaceval)
end
@generated function _link!(
    metadata::NamedTuple{names}, vi, vns, ::Val{space}
) where {names,space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(
                expr.args,
                quote
                    f_vns = vi.metadata.$f.vns
                    if ~istrans(vi, f_vns[1])
                        # Iterate over all `f_vns` and transform
                        for vn in f_vns
                            dist = getdist(vi, vn)
                            _inner_transform!(vi, vn, dist, link_transform(dist))
                            settrans!!(vi, true, vn)
                        end
                    else
                        @warn("[DynamicPPL] attempt to link a linked vi")
                    end
                end,
            )
        end
    end
    return expr
end

# R -> X for all variables associated with given sampler
function invlink!!(::DynamicTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # Call `_invlink!` instead of `invlink!` to avoid deprecation warning.
    _invlink!(vi, spl)
    return vi
end

function invlink!!(
    ::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    spl::AbstractSampler,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Setfield.@set vi.varinfo = DynamicPPL.invlink!!(vi.varinfo, spl, model)
end

function maybe_invlink_before_eval!!(vi::VarInfo, context::AbstractContext, model::Model)
    # Because `VarInfo` does not contain any information about what the transformation
    # other than whether or not it has actually been transformed, the best we can do
    # is just assume that `default_transformation` is the correct one if `istrans(vi)`.
    t = istrans(vi) ? default_transformation(model, vi) : NoTransformation()
    return maybe_invlink_before_eval!!(t, vi, context, model)
end

"""
    invlink!(vi::VarInfo, spl::AbstractSampler)

Transform the values of the random variables sampled by `spl` in `vi` from the
Euclidean space back to the support of their distributions and sets their corresponding
`"trans"` flag values to `false`.
"""
function invlink!(vi::VarInfo, spl::AbstractSampler)
    Base.depwarn(
        "`invlink!(varinfo, sampler)` is deprecated, use `invlink!!(varinfo, sampler, model)` instead.",
        :invlink!,
    )
    return _invlink!(vi, spl)
end

function invlink!(vi::VarInfo, spl::AbstractSampler, spaceval::Val)
    Base.depwarn(
        "`invlink!(varinfo, sampler, spaceval)` is deprecated, use `invlink!!(varinfo, sampler, model)` instead.",
        :invlink!,
    )
    return _invlink!(vi, spl, spaceval)
end

function _invlink!(vi::UntypedVarInfo, spl::AbstractSampler)
    vns = _getvns(vi, spl)
    if istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            _inner_transform!(vi, vn, dist, invlink_transform(dist))
            settrans!!(vi, false, vn)
        end
    else
        @warn("[DynamicPPL] attempt to invlink an invlinked vi")
    end
end
function _invlink!(vi::TypedVarInfo, spl::AbstractSampler)
    return _invlink!(vi, spl, Val(getspace(spl)))
end
function _invlink!(vi::TypedVarInfo, spl::AbstractSampler, spaceval::Val)
    vns = _getvns(vi, spl)
    return _invlink!(vi.metadata, vi, vns, spaceval)
end
@generated function _invlink!(
    metadata::NamedTuple{names}, vi, vns, ::Val{space}
) where {names,space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(
                expr.args,
                quote
                    f_vns = vi.metadata.$f.vns
                    if istrans(vi, f_vns[1])
                        # Iterate over all `f_vns` and transform
                        for vn in f_vns
                            dist = getdist(vi, vn)
                            _inner_transform!(vi, vn, dist, invlink_transform(dist))
                            settrans!!(vi, false, vn)
                        end
                    else
                        @warn("[DynamicPPL] attempt to invlink an invlinked vi")
                    end
                end,
            )
        end
    end
    return expr
end

function _inner_transform!(vi::VarInfo, vn::VarName, dist, f)
    # TODO: Use inplace versions to avoid allocations
    y, logjac = with_logabsdet_jacobian_and_reconstruct(f, dist, getval(vi, vn))
    yvec = vectorize(dist, y)
    # Determine the new range.
    start = first(getrange(vi, vn))
    # NOTE: `length(yvec)` should never be longer than `getrange(vi, vn)`.
    setrange!(vi, vn, start:(start + length(yvec) - 1))
    # Set the new value.
    setval!(vi, yvec, vn)
    acclogp!!(vi, -logjac)
    return vi
end

# HACK: We need `SampleFromPrior` to result in ALL values which are in need
# of a transformation to be transformed. `_getvns` will by default return
# an empty iterable for `SampleFromPrior`, so we need to override it here.
# This is quite hacky, but seems safer than changing the behavior of `_getvns`.
_getvns_link(varinfo::VarInfo, spl::AbstractSampler) = _getvns(varinfo, spl)
_getvns_link(varinfo::UntypedVarInfo, spl::SampleFromPrior) = nothing
function _getvns_link(varinfo::TypedVarInfo, spl::SampleFromPrior)
    return map(Returns(nothing), varinfo.metadata)
end

function link(::DynamicTransformation, varinfo::VarInfo, spl::AbstractSampler, model::Model)
    return _link(varinfo, spl)
end
function link(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    spl::AbstractSampler,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Setfield.@set varinfo.varinfo = link(varinfo.varinfo, spl, model)
end

function _link(varinfo::UntypedVarInfo, spl::AbstractSampler)
    varinfo = deepcopy(varinfo)
    return VarInfo(
        _link_metadata!(varinfo, varinfo.metadata, _getvns_link(varinfo, spl)),
        Base.Ref(getlogp(varinfo)),
        Ref(get_num_produce(varinfo)),
    )
end

function _link(varinfo::TypedVarInfo, spl::AbstractSampler)
    varinfo = deepcopy(varinfo)
    md = _link_metadata_namedtuple!(
        varinfo, varinfo.metadata, _getvns_link(varinfo, spl), Val(getspace(spl))
    )
    return VarInfo(md, Base.Ref(getlogp(varinfo)), Ref(get_num_produce(varinfo)))
end

@generated function _link_metadata_namedtuple!(
    varinfo::VarInfo, metadata::NamedTuple{names}, vns::NamedTuple, ::Val{space}
) where {names,space}
    vals = Expr(:tuple)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(vals.args, :(_link_metadata!(varinfo, metadata.$f, vns.$f)))
        else
            push!(vals.args, :(metadata.$f))
        end
    end

    return :(NamedTuple{$names}($vals))
end
function _link_metadata!(varinfo::VarInfo, metadata::Metadata, target_vns)
    vns = metadata.vns

    # Construct the new transformed values, and keep track of their lengths.
    vals_new = map(vns) do vn
        # Return early if we're already in unconstrained space.
        # HACK: if `target_vns` is `nothing`, we ignore the `target_vns` check.
        if istrans(varinfo, vn) || (target_vns !== nothing && vn ∉ target_vns)
            return metadata.vals[getrange(metadata, vn)]
        end

        # Transform to constrained space.
        x = getval(varinfo, vn)
        dist = getdist(varinfo, vn)
        f = link_transform(dist)
        y, logjac = with_logabsdet_jacobian_and_reconstruct(f, dist, x)
        # Vectorize value.
        yvec = vectorize(dist, y)
        # Accumulate the log-abs-det jacobian correction.
        acclogp!!(varinfo, -logjac)
        # Mark as no longer transformed.
        settrans!!(varinfo, true, vn)
        # Return the vectorized transformed value.
        return yvec
    end

    # Determine new ranges.
    ranges_new = similar(metadata.ranges)
    offset = 0
    for (i, v) in enumerate(vals_new)
        r_start, r_end = offset + 1, length(v) + offset
        offset = r_end
        ranges_new[i] = r_start:r_end
    end

    # Now we just create a new metadata with the new `vals` and `ranges`.
    return Metadata(
        metadata.idcs,
        metadata.vns,
        ranges_new,
        reduce(vcat, vals_new),
        metadata.dists,
        metadata.gids,
        metadata.orders,
        metadata.flags,
    )
end

function invlink(
    ::DynamicTransformation, varinfo::VarInfo, spl::AbstractSampler, model::Model
)
    return _invlink(varinfo, spl)
end
function invlink(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    spl::AbstractSampler,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Setfield.@set varinfo.varinfo = invlink(varinfo.varinfo, spl, model)
end

function _invlink(varinfo::UntypedVarInfo, spl::AbstractSampler)
    varinfo = deepcopy(varinfo)
    return VarInfo(
        _invlink_metadata!(varinfo, varinfo.metadata, _getvns_link(varinfo, spl)),
        Base.Ref(getlogp(varinfo)),
        Ref(get_num_produce(varinfo)),
    )
end

function _invlink(varinfo::TypedVarInfo, spl::AbstractSampler)
    varinfo = deepcopy(varinfo)
    md = _invlink_metadata_namedtuple!(
        varinfo, varinfo.metadata, _getvns_link(varinfo, spl), Val(getspace(spl))
    )
    return VarInfo(md, Base.Ref(getlogp(varinfo)), Ref(get_num_produce(varinfo)))
end

@generated function _invlink_metadata_namedtuple!(
    varinfo::VarInfo, metadata::NamedTuple{names}, vns::NamedTuple, ::Val{space}
) where {names,space}
    vals = Expr(:tuple)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(vals.args, :(_invlink_metadata!(varinfo, metadata.$f, vns.$f)))
        else
            push!(vals.args, :(metadata.$f))
        end
    end

    return :(NamedTuple{$names}($vals))
end
function _invlink_metadata!(varinfo::VarInfo, metadata::Metadata, target_vns)
    vns = metadata.vns

    # Construct the new transformed values, and keep track of their lengths.
    vals_new = map(vns) do vn
        # Return early if we're already in constrained space OR if we're not
        # supposed to touch this `vn`, e.g. when `vn` does not belong to the current sampler. 
        # HACK: if `target_vns` is `nothing`, we ignore the `target_vns` check.
        if !istrans(varinfo, vn) || (target_vns !== nothing && vn ∉ target_vns)
            return metadata.vals[getrange(metadata, vn)]
        end

        # Transform to constrained space.
        y = getval(varinfo, vn)
        dist = getdist(varinfo, vn)
        f = invlink_transform(dist)
        x, logjac = with_logabsdet_jacobian_and_reconstruct(f, dist, y)
        # Vectorize value.
        xvec = vectorize(dist, x)
        # Accumulate the log-abs-det jacobian correction.
        acclogp!!(varinfo, -logjac)
        # Mark as no longer transformed.
        settrans!!(varinfo, false, vn)
        # Return the vectorized transformed value.
        return xvec
    end

    # Determine new ranges.
    ranges_new = similar(metadata.ranges)
    offset = 0
    for (i, v) in enumerate(vals_new)
        r_start, r_end = offset + 1, length(v) + offset
        offset = r_end
        ranges_new[i] = r_start:r_end
    end

    # Now we just create a new metadata with the new `vals` and `ranges`.
    return Metadata(
        metadata.idcs,
        metadata.vns,
        ranges_new,
        reduce(vcat, vals_new),
        metadata.dists,
        metadata.gids,
        metadata.orders,
        metadata.flags,
    )
end

"""
    islinked(vi::VarInfo, spl::Union{Sampler, SampleFromPrior})

Check whether `vi` is in the transformed space for a particular sampler `spl`.

Turing's Hamiltonian samplers use the `link` and `invlink` functions from 
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) to map a constrained variable
(for example, one bounded to the space `[0, 1]`) from its constrained space to the set of 
real numbers. `islinked` checks if the number is in the constrained space or the real space.
"""
function islinked(vi::UntypedVarInfo, spl::Union{Sampler,SampleFromPrior})
    vns = _getvns(vi, spl)
    return istrans(vi, vns[1])
end
function islinked(vi::TypedVarInfo, spl::Union{Sampler,SampleFromPrior})
    vns = _getvns(vi, spl)
    return _islinked(vi, vns)
end
@generated function _islinked(vi, vns::NamedTuple{names}) where {names}
    out = []
    for f in names
        push!(out, :(length(vns.$f) == 0 ? false : istrans(vi, vns.$f[1])))
    end
    return Expr(:||, false, out...)
end

function nested_setindex_maybe!(vi::UntypedVarInfo, val, vn::VarName)
    return _nested_setindex_maybe!(vi, getmetadata(vi, vn), val, vn)
end
function nested_setindex_maybe!(
    vi::VarInfo{<:NamedTuple{names}}, val, vn::VarName{sym}
) where {names,sym}
    return if sym in names
        _nested_setindex_maybe!(vi, getmetadata(vi, vn), val, vn)
    else
        nothing
    end
end
function _nested_setindex_maybe!(vi::VarInfo, md::Metadata, val, vn::VarName)
    # If `vn` is in `vns`, then we can just use the standard `setindex!`.
    vns = md.vns
    if vn in vns
        setindex!(vi, val, vn)
        return vn
    end

    # Otherwise, we need to check if either of the `vns` subsumes `vn`.
    i = findfirst(Base.Fix2(subsumes, vn), vns)
    i === nothing && return nothing

    vn_parent = vns[i]
    dist = getdist(md, vn_parent)
    val_parent = getindex(vi, vn_parent, dist)  # TODO: Ensure that we're working with a view here.
    # Split the varname into its tail lens.
    lens = remove_parent_lens(vn_parent, vn)
    # Update the value for the parent.
    val_parent_updated = set!!(val_parent, lens, val)
    setindex!(vi, val_parent_updated, vn_parent)
    return vn_parent
end

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
getindex(vi::VarInfo, vn::VarName) = getindex(vi, vn, getdist(vi, vn))
function getindex(vi::VarInfo, vn::VarName, dist::Distribution)
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    val = getval(vi, vn)
    return maybe_invlink_and_reconstruct(vi, vn, dist, val)
end
function getindex(vi::VarInfo, vns::Vector{<:VarName})
    # FIXME(torfjelde): Using `getdist(vi, first(vns))` won't be correct in cases
    # such as `x .~ [Normal(), Exponential()]`.
    # BUT we also can't fix this here because this will lead to "incorrect"
    # behavior if `vns` arose from something like `x .~ MvNormal(zeros(2), I)`,
    # where by "incorrect" we mean there exists pieces of code expecting this behavior.
    return getindex(vi, vns, getdist(vi, first(vns)))
end
function getindex(vi::VarInfo, vns::Vector{<:VarName}, dist::Distribution)
    @assert haskey(vi, vns[1]) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    vals_linked = mapreduce(vcat, vns) do vn
        getindex(vi, vn, dist)
    end
    return reconstruct(dist, vals_linked, length(vns))
end

getindex_raw(vi::VarInfo, vn::VarName) = getindex_raw(vi, vn, getdist(vi, vn))
function getindex_raw(vi::VarInfo, vn::VarName, dist::Distribution)
    return reconstruct(dist, getval(vi, vn))
end
function getindex_raw(vi::VarInfo, vns::Vector{<:VarName})
    return getindex_raw(vi, vns, getdist(vi, first(vns)))
end
function getindex_raw(vi::VarInfo, vns::Vector{<:VarName}, dist::Distribution)
    return reconstruct(dist, getval(vi, vns), length(vns))
end

"""
    getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})

Return the current value(s) of the random variables sampled by `spl` in `vi`.

The value(s) may or may not be transformed to Euclidean space.
"""
getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, _getranges(vi, spl)))
function getindex(vi::TypedVarInfo, spl::Sampler)
    # Gets the ranges as a NamedTuple
    ranges = _getranges(vi, spl)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return reduce(vcat, _getindex(vi.metadata, ranges))
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
setindex!(vi::VarInfo, val, vn::VarName) = (setval!(vi, val, vn); return vi)
function BangBang.setindex!!(vi::VarInfo, val, vn::VarName)
    setindex!(vi, val, vn)
    return vi
end

"""
    setindex!(vi::VarInfo, val, spl::Union{SampleFromPrior, Sampler})

Set the current value(s) of the random variables sampled by `spl` in `vi` to `val`.

The value(s) may or may not be transformed to Euclidean space.
"""
setindex!(vi::VarInfo, val, spl::SampleFromPrior) = setall!(vi, val)
setindex!(vi::UntypedVarInfo, val, spl::Sampler) = setval!(vi, val, _getranges(vi, spl))
function setindex!(vi::TypedVarInfo, val, spl::Sampler)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = _getranges(vi, spl)
    _setindex!(vi.metadata, val, ranges)
    return nothing
end

function BangBang.setindex!!(vi::VarInfo, val, spl::AbstractSampler)
    setindex!(vi, val, spl)
    return vi
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
        push!(expr.args, :(@views $f_vals[$f_range] .= val[($start):($finish)]))
        offset = :($offset + $len)
    end
    return expr
end

@inline function findvns(vi, f_vns)
    if length(f_vns) == 0
        throw("Unidentified error, please report this error in an issue.")
    end
    return map(vn -> vi[vn], f_vns)
end

"""
    haskey(vi::VarInfo, vn::VarName)

Check whether `vn` has been sampled in `vi`.
"""
haskey(vi::VarInfo, vn::VarName) = haskey(getmetadata(vi, vn).idcs, vn)
function haskey(vi::TypedVarInfo, vn::VarName)
    metadata = vi.metadata
    Tmeta = typeof(metadata)
    return getsym(vn) in fieldnames(Tmeta) && haskey(getmetadata(vi, vn).idcs, vn)
end

function Base.show(io::IO, ::MIME"text/plain", vi::UntypedVarInfo)
    vi_str = """
    /=======================================================================
    | VarInfo
    |-----------------------------------------------------------------------
    | Varnames  :   $(string(vi.metadata.vns))
    | Range     :   $(vi.metadata.ranges)
    | Vals      :   $(vi.metadata.vals)
    | GIDs      :   $(vi.metadata.gids)
    | Orders    :   $(vi.metadata.orders)
    | Logp      :   $(getlogp(vi))
    | #produce  :   $(get_num_produce(vi))
    | flags     :   $(vi.metadata.flags)
    \\=======================================================================
    """
    return print(io, vi_str)
end

const _MAX_VARS_SHOWN = 4

function _show_varnames(io::IO, vi)
    md = vi.metadata
    vns = md.vns

    vns_by_name = Dict{Symbol,Vector{VarName}}()
    for vn in vns
        group = get!(() -> Vector{VarName}(), vns_by_name, getsym(vn))
        push!(group, vn)
    end

    L = length(vns_by_name)
    if L == 0
        print(io, "0 variables, dimension 0")
    else
        (L == 1) ? print(io, "1 variable (") : print(io, L, " variables (")
        join(io, Iterators.take(keys(vns_by_name), _MAX_VARS_SHOWN), ", ")
        (L > _MAX_VARS_SHOWN) && print(io, ", ...")
        print(io, "), dimension ", length(md.vals))
    end
end

function Base.show(io::IO, vi::UntypedVarInfo)
    print(io, "VarInfo (")
    _show_varnames(io, vi)
    print(io, "; logp: ", round(getlogp(vi); digits=3))
    return print(io, ")")
end

function BangBang.push!!(
    vi::VarInfo, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)
    if vi isa UntypedVarInfo
        @assert ~(vn in keys(vi)) "[push!!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gidset"
    elseif vi isa TypedVarInfo
        @assert ~(haskey(vi, vn)) "[push!!] attempt to add an exisitng variable $(getsym(vn)) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gidset"
    end

    val = vectorize(dist, r)

    meta = getmetadata(vi, vn)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals)
    n = length(val)
    push!(meta.ranges, (l + 1):(l + n))
    append!(meta.vals, val)
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, get_num_produce(vi))
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)

    return vi
end

"""
    setorder!(vi::VarInfo, vn::VarName, index::Int)

Set the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe
statements run before sampling `vn`.
"""
function setorder!(vi::VarInfo, vn::VarName, index::Int)
    metadata = getmetadata(vi, vn)
    if metadata.orders[metadata.idcs[vn]] != index
        metadata.orders[metadata.idcs[vn]] = index
    end
    return vi
end

"""
    getorder(vi::VarInfo, vn::VarName)

Get the `order` of `vn` in `vi`, where `order` is the number of `observe` statements
run before sampling `vn`.
"""
getorder(vi::VarInfo, vn::VarName) = getorder(getmetadata(vi, vn), vn)
getorder(metadata::Metadata, vn::VarName) = metadata.orders[getidx(metadata, vn)]

#######################################
# Rand & replaying method for VarInfo #
#######################################

"""
    is_flagged(vi::VarInfo, vn::VarName, flag::String)

Check whether `vn` has a true value for `flag` in `vi`.
"""
function is_flagged(vi::VarInfo, vn::VarName, flag::String)
    return is_flagged(getmetadata(vi, vn), vn, flag)
end
function is_flagged(metadata::Metadata, vn::VarName, flag::String)
    return metadata.flags[flag][getidx(metadata, vn)]
end

"""
    unset_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `false` in `vi`.
"""
function unset_flag!(vi::VarInfo, vn::VarName, flag::String)
    getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = false
    return vi
end

"""
    set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)

Set the `"del"` flag of variables in `vi` with `order > vi.num_produce[]` to `true`.
"""
function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = _getidcs(vi, spl)
    if get_num_produce(vi) == 0
        for i in length(gidcs):-1:1
            vi.metadata.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in gidcs && vi.orders[i] > get_num_produce(vi)
                vi.metadata.flags["del"][i] = true
            end
        end
    end
    return nothing
end
function set_retained_vns_del_by_spl!(vi::TypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = _getidcs(vi, spl)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, get_num_produce(vi))
end
@generated function _set_retained_vns_del_by_spl!(
    metadata, gidcs::NamedTuple{names}, num_produce
) where {names}
    expr = Expr(:block)
    for f in names
        f_gidcs = :(gidcs.$f)
        f_orders = :(metadata.$f.orders)
        f_flags = :(metadata.$f.flags)
        push!(
            expr.args,
            quote
                # Set the flag for variables with symbol `f`
                if num_produce == 0
                    for i in length($f_gidcs):-1:1
                        $f_flags["del"][$f_gidcs[i]] = true
                    end
                else
                    for i in 1:length($f_orders)
                        if i in $f_gidcs && $f_orders[i] > num_produce
                            $f_flags["del"][i] = true
                        end
                    end
                end
            end,
        )
    end
    return expr
end

"""
    updategid!(vi::VarInfo, vn::VarName, spl::Sampler)

Set `vn`'s `gid` to `Set([spl.selector])`, if `vn` does not have a sampler selector linked
and `vn`'s symbol is in the space of `spl`.
"""
function updategid!(vi::VarInfoOrThreadSafeVarInfo, vn::VarName, spl::Sampler)
    if inspace(vn, getspace(spl))
        setgid!(vi, spl.selector, vn)
    end
end

# TODO: Maybe rename or something?
"""
    _apply!(kernel!, vi::VarInfo, values, keys)

Calls `kernel!(vi, vn, values, keys)` for every `vn` in `vi`.
"""
function _apply!(kernel!, vi::VarInfoOrThreadSafeVarInfo, values, keys)
    keys_strings = map(string, collectmaybe(keys))
    num_indices_seen = 0

    for vn in Base.keys(vi)
        indices_found = kernel!(vi, vn, values, keys_strings)
        if indices_found !== nothing
            num_indices_seen += length(indices_found)
        end
    end

    if length(keys) > num_indices_seen
        # Some keys have not been seen, i.e. attempted to set variables which
        # we were not able to locate in `vi`.
        # Find the ones we missed so we can warn the user.
        unused_keys = _find_missing_keys(vi, keys_strings)
        @warn "the following keys were not found in `vi`, and thus `kernel!` was not applied to these: $(unused_keys)"
    end

    return vi
end

function _apply!(kernel!, vi::TypedVarInfo, values, keys)
    return _typed_apply!(kernel!, vi, vi.metadata, values, collectmaybe(keys))
end

@generated function _typed_apply!(
    kernel!, vi::TypedVarInfo, metadata::NamedTuple{names}, values, keys
) where {names}
    updates = map(names) do n
        quote
            for vn in metadata.$n.vns
                indices_found = kernel!(vi, vn, values, keys_strings)
                if indices_found !== nothing
                    num_indices_seen += length(indices_found)
                end
            end
        end
    end

    return quote
        keys_strings = map(string, keys)
        num_indices_seen = 0

        $(updates...)

        if length(keys) > num_indices_seen
            # Some keys have not been seen, i.e. attempted to set variables which
            # we were not able to locate in `vi`.
            # Find the ones we missed so we can warn the user.
            unused_keys = _find_missing_keys(vi, keys_strings)
            @warn "the following keys were not found in `vi`, and thus `kernel!` was not applied to these: $(unused_keys)"
        end

        return vi
    end
end

function _find_missing_keys(vi::VarInfoOrThreadSafeVarInfo, keys)
    string_vns = map(string, collectmaybe(Base.keys(vi)))
    # If `key` isn't subsumed by any element of `string_vns`, it is not present in `vi`.
    missing_keys = filter(keys) do key
        !any(Base.Fix2(subsumes_string, key), string_vns)
    end

    return missing_keys
end

"""
    setval!(vi::VarInfo, x)
    setval!(vi::VarInfo, values, keys)
    setval!(vi::VarInfo, chains::AbstractChains, sample_idx::Int, chain_idx::Int)

Set the values in `vi` to the provided values and leave those which are not present in
`x` or `chains` unchanged.

## Notes
This is rather limited for two reasons:
1. It uses `subsumes_string(string(vn), map(string, keys))` under the hood,
   and therefore suffers from the same limitations as [`subsumes_string`](@ref).
2. It will set every `vn` present in `keys`. It will NOT however
   set every `k` present in `keys`. This means that if `vn == [m[1], m[2]]`,
   representing some variable `m`, calling `setval!(vi, (m = [1.0, 2.0]))` will
   be a no-op since it will try to find `m[1]` and `m[2]` in `keys((m = [1.0, 2.0]))`.

## Example
```jldoctest
julia> using DynamicPPL, Distributions, StableRNGs

julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1)
           end
       end;

julia> rng = StableRNG(42);

julia> m = demo([missing]);

julia> var_info = DynamicPPL.VarInfo(rng, m);

julia> var_info[@varname(m)]
-0.6702516921145671

julia> var_info[@varname(x[1])]
-0.22312984965118443

julia> DynamicPPL.setval!(var_info, (m = 100.0, )); # set `m` and and keep `x[1]`

julia> var_info[@varname(m)] # [✓] changed
100.0

julia> var_info[@varname(x[1])] # [✓] unchanged
-0.22312984965118443

julia> m(rng, var_info); # rerun model

julia> var_info[@varname(m)] # [✓] unchanged
100.0

julia> var_info[@varname(x[1])] # [✓] unchanged
-0.22312984965118443
```
"""
setval!(vi::VarInfo, x) = setval!(vi, values(x), keys(x))
setval!(vi::VarInfo, values, keys) = _apply!(_setval_kernel!, vi, values, keys)
function setval!(vi::VarInfo, chains::AbstractChains, sample_idx::Int, chain_idx::Int)
    return setval!(vi, chains.value[sample_idx, :, chain_idx], keys(chains))
end

function _setval_kernel!(vi::VarInfoOrThreadSafeVarInfo, vn::VarName, values, keys)
    indices = findall(Base.Fix1(subsumes_string, string(vn)), keys)
    if !isempty(indices)
        val = reduce(vcat, values[indices])
        setval!(vi, val, vn)
        settrans!!(vi, false, vn)
    end

    return indices
end

"""
    setval_and_resample!(vi::VarInfo, x)
    setval_and_resample!(vi::VarInfo, values, keys)
    setval_and_resample!(vi::VarInfo, chains::AbstractChains, sample_idx, chain_idx)

Set the values in `vi` to the provided values and those which are not present
in `x` or `chains` to *be* resampled.

Note that this does *not* resample the values not provided! It will call `setflag!(vi, vn, "del")`
for variables `vn` for which no values are provided, which means that the next time we call `model(vi)` these
variables will be resampled.

## Note
- This suffers from the same limitations as [`setval!`](@ref). See `setval!` for more info.

## Example
```jldoctest
julia> using DynamicPPL, Distributions, StableRNGs

julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1)
           end
       end;

julia> rng = StableRNG(42);

julia> m = demo([missing]);

julia> var_info = DynamicPPL.VarInfo(rng, m);

julia> var_info[@varname(m)]
-0.6702516921145671

julia> var_info[@varname(x[1])]
-0.22312984965118443

julia> DynamicPPL.setval_and_resample!(var_info, (m = 100.0, )); # set `m` and ready `x[1]` for resampling

julia> var_info[@varname(m)] # [✓] changed
100.0

julia> var_info[@varname(x[1])] # [✓] unchanged
-0.22312984965118443

julia> m(rng, var_info); # sample `x[1]` conditioned on `m = 100.0`

julia> var_info[@varname(m)] # [✓] unchanged
100.0

julia> var_info[@varname(x[1])] # [✓] changed
101.37363069798343
```

## See also
- [`setval!`](@ref)
"""
function setval_and_resample!(vi::VarInfoOrThreadSafeVarInfo, x)
    return setval_and_resample!(vi, values(x), keys(x))
end
function setval_and_resample!(vi::VarInfoOrThreadSafeVarInfo, values, keys)
    return _apply!(_setval_and_resample_kernel!, vi, values, keys)
end
function setval_and_resample!(
    vi::VarInfoOrThreadSafeVarInfo, chains::AbstractChains, sample_idx::Int, chain_idx::Int
)
    if supports_varname_indexing(chains)
        # First we need to set every variable to be resampled.
        for vn in keys(vi)
            set_flag!(vi, vn, "del")
        end
        # Then we set the variables in `varinfo` from `chain`.
        for vn in varnames(chains)
            vn_updated = nested_setindex_maybe!(
                vi, getindex_varname(chains, sample_idx, vn, chain_idx), vn
            )

            # Unset the `del` flag if we found something.
            if vn_updated !== nothing
                # NOTE: This will be triggered even if only a subset of a variable has been set!
                unset_flag!(vi, vn_updated, "del")
            end
        end
    else
        setval_and_resample!(vi, chains.value[sample_idx, :, chain_idx], keys(chains))
    end
end

function _setval_and_resample_kernel!(
    vi::VarInfoOrThreadSafeVarInfo, vn::VarName, values, keys
)
    indices = findall(Base.Fix1(subsumes_string, string(vn)), keys)
    if !isempty(indices)
        val = reduce(vcat, values[indices])
        setval!(vi, val, vn)
        settrans!!(vi, false, vn)
    else
        # Ensures that we'll resample the variable corresponding to `vn` if we run
        # the model on `vi` again.
        set_flag!(vi, vn, "del")
    end

    return indices
end

values_as(vi::VarInfo) = vi.metadata
values_as(vi::VarInfo, ::Type{Vector}) = copy(getall(vi))
function values_as(vi::UntypedVarInfo, ::Type{NamedTuple})
    iter = values_from_metadata(vi.metadata)
    return NamedTuple(map(p -> Symbol(p.first) => p.second, iter))
end
function values_as(vi::UntypedVarInfo, ::Type{D}) where {D<:AbstractDict}
    return ConstructionBase.constructorof(D)(values_from_metadata(vi.metadata))
end

function values_as(vi::VarInfo{<:NamedTuple{names}}, ::Type{NamedTuple}) where {names}
    iter = Iterators.flatten(values_from_metadata(getfield(vi.metadata, n)) for n in names)
    return NamedTuple(map(p -> Symbol(p.first) => p.second, iter))
end

function values_as(
    vi::VarInfo{<:NamedTuple{names}}, ::Type{D}
) where {names,D<:AbstractDict}
    iter = Iterators.flatten(values_from_metadata(getfield(vi.metadata, n)) for n in names)
    return ConstructionBase.constructorof(D)(iter)
end

function values_from_metadata(md::Metadata)
    return (
        vn => reconstruct(md.dists[md.idcs[vn]], md.vals[md.ranges[md.idcs[vn]]]) for
        vn in md.vns
    )
end
