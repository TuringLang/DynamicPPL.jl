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
const VectorVarInfo = VarInfo{<:VarNamedVector}
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
    md = replace_values(old_vi.metadata, Val(getspace(spl)), x)
    return VarInfo(
        md, Base.RefValue{eltype(x)}(getlogp(old_vi)), Ref(get_num_produce(old_vi))
    )
end

# No-op if we're already working with a `VarNamedVector`.
metadata_to_varnamedvector(vnv::VarNamedVector) = vnv
function metadata_to_varnamedvector(md::Metadata)
    idcs = copy(md.idcs)
    vns = copy(md.vns)
    ranges = copy(md.ranges)
    vals = copy(md.vals)
    is_unconstrained = map(Base.Fix1(istrans, md), md.vns)
    transforms = map(md.dists, is_unconstrained) do dist, trans
        if trans
            return from_linked_vec_transform(dist)
        else
            return from_vec_transform(dist)
        end
    end

    return VarNamedVector(
        OrderedDict{eltype(keys(idcs)),Int}(idcs),
        vns,
        ranges,
        vals,
        transforms,
        is_unconstrained,
    )
end

function VectorVarInfo(vi::UntypedVarInfo)
    md = metadata_to_varnamedvector(vi.metadata)
    lp = getlogp(vi)
    return VarInfo(md, Base.RefValue{eltype(lp)}(lp), Ref(get_num_produce(vi)))
end

function VectorVarInfo(vi::TypedVarInfo)
    md = map(metadata_to_varnamedvector, vi.metadata)
    lp = getlogp(vi)
    return VarInfo(md, Base.RefValue{eltype(lp)}(lp), Ref(get_num_produce(vi)))
end

function has_varnamedvector(vi::VarInfo)
    return vi.metadata isa VarNamedVector ||
           (vi isa TypedVarInfo && any(Base.Fix2(isa, VarNamedVector), values(vi.metadata)))
end

"""
    untyped_varinfo(model[, context, metadata])

Return an untyped varinfo object for the given `model` and `context`.

# Arguments
- `model::Model`: The model for which to create the varinfo object.
- `context::AbstractContext`: The context in which to evaluate the model. Default: `SamplingContext()`.
- `metadata::Union{Metadata,VarNamedVector}`: The metadata to use for the varinfo object.
    Default: `Metadata()`.
"""
function untyped_varinfo(
    model::Model,
    context::AbstractContext=SamplingContext(),
    metadata::Union{Metadata,VarNamedVector}=Metadata(),
)
    varinfo = VarInfo(metadata)
    return last(
        evaluate!!(model, varinfo, hassampler(context) ? context : SamplingContext(context))
    )
end

"""
    typed_varinfo(model[, context, metadata])

Return a typed varinfo object for the given `model`, `sampler` and `context`.

This simply calls [`DynamicPPL.untyped_varinfo`](@ref) and converts the resulting
varinfo object to a typed varinfo object.

See also: [`DynamicPPL.untyped_varinfo`](@ref)
"""
typed_varinfo(args...) = TypedVarInfo(untyped_varinfo(args...))

function VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
    metadata::Union{Metadata,VarNamedVector}=Metadata(),
)
    return typed_varinfo(model, SamplingContext(rng, sampler, context), metadata)
end
VarInfo(model::Model, args...) = VarInfo(Random.default_rng(), model, args...)

"""
    vector_length(varinfo::VarInfo)

Return the length of the vector representation of `varinfo`.
"""
vector_length(varinfo::VarInfo) = length(varinfo.metadata)
vector_length(varinfo::TypedVarInfo) = sum(length, varinfo.metadata)
vector_length(md::Metadata) = sum(length, md.ranges)

unflatten(vi::VarInfo, x::AbstractVector) = unflatten(vi, SampleFromPrior(), x)

# TODO: deprecate.
function unflatten(vi::VarInfo, spl::AbstractSampler, x::AbstractVector)
    md = unflatten(vi.metadata, spl, x)
    return VarInfo(md, Base.RefValue{eltype(x)}(getlogp(vi)), Ref(get_num_produce(vi)))
end

# The Val(getspace(spl)) is used to dispatch into the below generated function.
function unflatten(metadata::NamedTuple, spl::AbstractSampler, x::AbstractVector)
    return unflatten(metadata, Val(getspace(spl)), x)
end

@generated function unflatten(
    metadata::NamedTuple{names}, ::Val{space}, x
) where {names,space}
    exprs = []
    offset = :(0)
    for f in names
        mdf = :(metadata.$f)
        if inspace(f, space) || length(space) == 0
            len = :(sum(length, $mdf.ranges))
            push!(exprs, :($f = unflatten($mdf, x[($offset + 1):($offset + $len)])))
            offset = :($offset + $len)
        else
            push!(exprs, :($f = $mdf))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# For Metadata unflatten and replace_values are the same. For VarNamedVector they are not.
function unflatten(md::Metadata, x::AbstractVector)
    return replace_values(md, x)
end
function unflatten(md::Metadata, spl::AbstractSampler, x::AbstractVector)
    return replace_values(md, spl, x)
end

# without AbstractSampler
function VarInfo(rng::Random.AbstractRNG, model::Model, context::AbstractContext)
    return VarInfo(rng, model, SampleFromPrior(), context)
end

# TODO: Remove `space` argument when no longer needed. Ref: https://github.com/TuringLang/DynamicPPL.jl/issues/573
replace_values(metadata::Metadata, space, x) = replace_values(metadata, x)
function replace_values(metadata::Metadata, x)
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

@generated function replace_values(
    metadata::NamedTuple{names}, ::Val{space}, x
) where {names,space}
    exprs = []
    offset = :(0)
    for f in names
        mdf = :(metadata.$f)
        if inspace(f, space) || length(space) == 0
            len = :(sum(length, $mdf.ranges))
            push!(exprs, :($f = replace_values($mdf, x[($offset + 1):($offset + $len)])))
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
    return VarInfo(metadata, deepcopy(varinfo.logp), deepcopy(varinfo.num_produce))
end

function subset(varinfo::VectorVarInfo, vns::AbstractVector{<:VarName})
    metadata = subset(varinfo.metadata, vns)
    return VarInfo(metadata, deepcopy(varinfo.logp), deepcopy(varinfo.num_produce))
end

function subset(varinfo::TypedVarInfo, vns::AbstractVector{<:VarName{sym}}) where {sym}
    # If all the variables are using the same symbol, then we can just extract that field from the metadata.
    metadata = subset(getfield(varinfo.metadata, sym), vns)
    return VarInfo(
        NamedTuple{(sym,)}(tuple(metadata)),
        deepcopy(varinfo.logp),
        deepcopy(varinfo.num_produce),
    )
end

function subset(varinfo::TypedVarInfo, vns::AbstractVector{<:VarName})
    syms = Tuple(unique(map(getsym, vns)))
    metadatas = map(syms) do sym
        subset(getfield(varinfo.metadata, sym), filter(==(sym) ∘ getsym, vns))
    end

    return VarInfo(
        NamedTuple{syms}(metadatas), deepcopy(varinfo.logp), deepcopy(varinfo.num_produce)
    )
end

function subset(metadata::Metadata, vns_given::AbstractVector{VN}) where {VN<:VarName}
    # TODO: Should we error if `vns` contains a variable that is not in `metadata`?
    # For each `vn` in `vns`, get the variables subsumed by `vn`.
    vns = mapreduce(vcat, vns_given; init=VN[]) do vn
        filter(Base.Fix1(subsumes, vn), metadata.vns)
    end
    indices_for_vns = map(Base.Fix1(getindex, metadata.idcs), vns)
    indices = if isempty(vns)
        Dict{VarName,Int}()
    else
        Dict(vn => i for (i, vn) in enumerate(vns))
    end
    # Construct new `vals` and `ranges`.
    vals_original = metadata.vals
    ranges_original = metadata.ranges
    # Allocate the new `vals`. and `ranges`.
    vals = similar(metadata.vals, sum(length, ranges_original[indices_for_vns]; init=0))
    ranges = similar(ranges_original, length(vns))
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
        metadata.gids[indices_for_vns],
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

function merge_metadata(vnv_left::VarNamedVector, vnv_right::VarNamedVector)
    return merge(vnv_left, vnv_right)
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
    gids = Set{Selector}[]
    orders = Int[]
    flags = Dict{String,BitVector}()
    # Initialize the `flags`.
    for k in union(keys(metadata_left.flags), keys(metadata_right.flags))
        flags[k] = BitVector()
    end

    # Range offset.
    offset = 0

    for (idx, vn) in enumerate(vns_both)
        idcs[vn] = idx
        push!(vns, vn)
        metadata_for_vn = vn in vns_right ? metadata_right : metadata_left

        val = getindex_internal(metadata_for_vn, vn)
        append!(vals, val)
        r = (offset + 1):(offset + length(val))
        push!(ranges, r)
        offset = r[end]
        dist = getdist(metadata_for_vn, vn)
        push!(dists, dist)
        gid = metadata_for_vn.gids[getidx(metadata_for_vn, vn)]
        push!(gids, gid)
        push!(orders, getorder(metadata_for_vn, vn))
        for k in keys(flags)
            push!(flags[k], is_flagged(metadata_for_vn, vn, k))
        end
    end

    return Metadata(idcs, vns, ranges, vals, dists, gids, orders, flags)
end

const VarView = Union{Int,UnitRange,Vector{Int}}

"""
    setval!(vi::UntypedVarInfo, val, vview::Union{Int, UnitRange, Vector{Int}})

Set the value of `vi.vals[vview]` to `val`.
"""
setval!(vi::UntypedVarInfo, val, vview::VarView) = vi.metadata.vals[vview] = val

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
    return map(Base.Fix1(getrange, vi), vns)
end

"""
    vector_getrange(varinfo::VarInfo, varname::VarName)

Return the range corresponding to `varname` in the vector representation of `varinfo`.
"""
vector_getrange(vi::VarInfo, vn::VarName) = getrange(vi.metadata, vn)
function vector_getrange(vi::TypedVarInfo, vn::VarName)
    offset = 0
    for md in values(vi.metadata)
        # First, we need to check if `vn` is in `md`.
        # In this case, we can just return the corresponding range + offset.
        haskey(md, vn) && return getrange(md, vn) .+ offset
        # Otherwise, we need to get the cumulative length of the ranges in `md`
        # and add it to the offset.
        offset += sum(length, md.ranges)
    end
    # If we reach this point, `vn` is not in `vi.metadata`.
    throw(KeyError(vn))
end

"""
    vector_getranges(varinfo::VarInfo, varnames::Vector{<:VarName})

Return the range corresponding to `varname` in the vector representation of `varinfo`.
"""
function vector_getranges(varinfo::VarInfo, varname::Vector{<:VarName})
    return map(Base.Fix1(vector_getrange, varinfo), varname)
end
# Specialized version for `TypedVarInfo`.
function vector_getranges(varinfo::TypedVarInfo, vns::Vector{<:VarName})
    # TODO: Does it help if we _don't_ convert to a vector here?
    metadatas = collect(values(varinfo.metadata))
    # Extract the offsets.
    offsets = cumsum(map(vector_length, metadatas))
    # Extract the ranges from each metadata.
    ranges = Vector{UnitRange{Int}}(undef, length(vns))
    # Need to keep track of which ones we've seen.
    not_seen = fill(true, length(vns))
    for (i, metadata) in enumerate(metadatas)
        vns_metadata = filter(Base.Fix1(haskey, metadata), vns)
        # If none of the variables exist in the metadata, we return an empty array.
        isempty(vns_metadata) && continue
        # Otherwise, we extract the ranges.
        offset = i == 1 ? 0 : offsets[i - 1]
        for vn in vns_metadata
            r_vn = getrange(metadata, vn)
            # Get the index, so we return in the same order as `vns`.
            # NOTE: There might be duplicates in `vns`, so we need to handle that.
            indices = findall(==(vn), vns)
            for idx in indices
                not_seen[idx] = false
                ranges[idx] = r_vn .+ offset
            end
        end
    end
    # Raise key error if any of the variables were not found.
    if any(not_seen)
        inds = findall(not_seen)
        # Just use a `convert` to get the same type as the input; don't want to confuse by overly
        # specilizing the types in the error message.
        throw(KeyError(convert(typeof(vns), vns[inds])))
    end
    return ranges
end

"""
    getdist(vi::VarInfo, vn::VarName)

Return the distribution from which `vn` was sampled in `vi`.
"""
getdist(vi::VarInfo, vn::VarName) = getdist(getmetadata(vi, vn), vn)
getdist(md::Metadata, vn::VarName) = md.dists[getidx(md, vn)]
# TODO(mhauru) Remove this once the old Gibbs sampler stuff is gone.
function getdist(::VarNamedVector, ::VarName)
    throw(ErrorException("getdist does not exist for VarNamedVector"))
end

getindex_internal(vi::VarInfo, vn::VarName) = getindex_internal(getmetadata(vi, vn), vn)
# TODO(torfjelde): Use `view` instead of `getindex`. Requires addressing type-stability issues though,
# since then we might be returning a `SubArray` rather than an `Array`, which is typically
# what a bijector would result in, even if the input is a view (`SubArray`).
# TODO(torfjelde): An alternative is to implement `view` directly instead.
getindex_internal(md::Metadata, vn::VarName) = getindex(md.vals, getrange(md, vn))

function getindex_internal(vi::VarInfo, vns::Vector{<:VarName})
    return mapreduce(Base.Fix1(getindex_internal, vi), vcat, vns)
end

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
    return md.vals[getrange(md, vn)] = tovec(val)
end

"""
    getall(vi::VarInfo)

Return the values of all the variables in `vi`.

The values may or may not be transformed to Euclidean space.
"""
getall(vi::VarInfo) = getall(vi.metadata)
# NOTE: `mapreduce` over `NamedTuple` results in worse type-inference.
# See for example https://github.com/JuliaLang/julia/pull/46381.
getall(vi::TypedVarInfo) = reduce(vcat, map(getall, vi.metadata))
function getall(md::Metadata)
    return mapreduce(
        Base.Fix1(getindex_internal, md), vcat, md.vns; init=similar(md.vals, 0)
    )
end
getall(vnv::VarNamedVector) = getindex_internal(vnv, Colon())

"""
    setall!(vi::VarInfo, val)

Set the values of all the variables in `vi` to `val`.

The values may or may not be transformed to Euclidean space.
"""
setall!(vi::VarInfo, val) = _setall!(vi.metadata, val)

function _setall!(metadata::Metadata, val)
    for r in metadata.ranges
        metadata.vals[r] .= val[r]
    end
end
function _setall!(vnv::VarNamedVector, val)
    # TODO(mhauru) Do something more efficient here.
    for i in 1:length_internal(vnv)
        setindex_internal!(vnv, val[i], i)
    end
end
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
    settrans!!(getmetadata(vi, vn), trans, vn)
    return vi
end
function settrans!!(metadata::Metadata, trans::Bool, vn::VarName)
    if trans
        set_flag!(metadata, vn, "trans")
    else
        unset_flag!(metadata, vn, "trans")
    end

    return metadata
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

_getidcs(vi::UntypedVarInfo) = 1:length(vi.metadata.idcs)
_getidcs(vi::TypedVarInfo) = _getidcs(vi.metadata)

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
@inline function findinds(f_meta::Metadata, s, ::Val{space}) where {space}
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter(
        (i) ->
            (s in f_meta.gids[i] || isempty(f_meta.gids[i])) &&
                (isempty(space) || inspace(f_meta.vns[i], space)),
        1:length(f_meta.gids),
    )
end
@inline function findinds(f_meta::Metadata)
    # Get all the idcs of the vns
    return filter((i) -> isempty(f_meta.gids[i]), 1:length(f_meta.gids))
end

function findinds(vnv::VarNamedVector, ::Selector, ::Val{space}) where {space}
    # New Metadata objects are created with an empty list of gids, which is intrepreted as
    # all Selectors applying to all variables. We assume the same behavior for
    # VarNamedVector, and thus ignore the Selector argument.
    if space !== ()
        msg = "VarNamedVector does not support selecting variables based on samplers"
        throw(ErrorException(msg))
    else
        return findinds(vnv)
    end
end

function findinds(vnv::VarNamedVector)
    return 1:length(vnv.varnames)
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
        push!(exprs, :($f = Base.keys(metadata.$f)[idcs.$f]))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

"""
    all_varnames_namedtuple(vi::TypedVarInfo)

Return a `NamedTuple` of the variables in `vi` grouped by symbol.
"""
all_varnames_namedtuple(vi::TypedVarInfo) = all_varnames_namedtuple(vi.metadata)

@generated function all_varnames_namedtuple(md::NamedTuple{names}) where {names}
    expr = Expr(:tuple)
    for f in names
        push!(expr.args, :($f = keys(md.$f)))
    end
    return expr
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
@inline function _getranges(vi::VarInfo, idcs::Vector{Int})
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

# TODO(mhauru) These set_flag! methods return the VarInfo. They should probably be called
# set_flag!!.
"""
    set_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::VarInfo, vn::VarName, flag::String)
    set_flag!(getmetadata(vi, vn), vn, flag)
    return vi
end
function set_flag!(md::Metadata, vn::VarName, flag::String)
    return md.flags[flag][getidx(md, vn)] = true
end

function set_flag!(vnv::VarNamedVector, ::VarName, flag::String)
    if flag == "del"
        # The "del" flag is effectively always set for a VarNamedVector, so this is a no-op.
    else
        throw(ErrorException("Flag $flag not valid for VarNamedVector"))
    end
    return vnv
end

####
#### APIs for typed and untyped VarInfo
####

# VarInfo

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{Float64}(0.0), Ref(0))

function TypedVarInfo(vi::VectorVarInfo)
    new_metas = group_by_symbol(vi.metadata)
    logp = getlogp(vi)
    num_produce = get_num_produce(vi)
    nt = NamedTuple(new_metas)
    return VarInfo(nt, Ref(logp), Ref(num_produce))
end

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

_empty!(metadata) = empty!(metadata)
@generated function _empty!(metadata::NamedTuple{names}) where {names}
    expr = Expr(:block)
    for f in names
        push!(expr.args, :(empty!(metadata.$f)))
    end
    return expr
end

# `keys`
Base.keys(md::Metadata) = md.vns
Base.keys(vi::VarInfo) = Base.keys(vi.metadata)

# HACK: Necessary to avoid returning `Any[]` which won't dispatch correctly
# on other methods in the codebase which requires `Vector{<:VarName}`.
Base.keys(vi::TypedVarInfo{<:NamedTuple{()}}) = VarName[]
@generated function Base.keys(vi::TypedVarInfo{<:NamedTuple{names}}) where {names}
    expr = Expr(:call)
    push!(expr.args, :vcat)

    for n in names
        push!(expr.args, :(keys(vi.metadata.$n)))
    end

    return expr
end

# FIXME(torfjelde): Don't use `_getvns`.
Base.keys(vi::UntypedVarInfo, spl::AbstractSampler) = _getvns(vi, spl)
function Base.keys(vi::TypedVarInfo, spl::AbstractSampler)
    return mapreduce(values, vcat, _getvns(vi, spl))
end

"""
    setgid!(vi::VarInfo, gid::Selector, vn::VarName)

Add `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
setgid!(vi::VarInfo, gid::Selector, vn::VarName) = setgid!(getmetadata(vi, vn), gid, vn)

function setgid!(m::Metadata, gid::Selector, vn::VarName)
    return push!(m.gids[getidx(m, vn)], gid)
end

function setgid!(vnv::VarNamedVector, gid::Selector, vn::VarName)
    throw(ErrorException("Calling setgid! on a VarNamedVector isn't valid."))
end

istrans(vi::VarInfo, vn::VarName) = istrans(getmetadata(vi, vn), vn)
istrans(md::Metadata, vn::VarName) = is_flagged(md, vn, "trans")

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

# Need to introduce the _isempty to avoid type piracy of isempty(::NamedTuple).
isempty(vi::VarInfo) = _isempty(vi.metadata)
_isempty(metadata::Metadata) = isempty(metadata.idcs)
_isempty(vnv::VarNamedVector) = isempty(vnv)
@generated function _isempty(metadata::NamedTuple{names}) where {names}
    return Expr(:&&, (:(_isempty(metadata.$f)) for f in names)...)
end

# Specialise link!! without varnames provided for TypedVarInfo. The generic version gets
# the keys of `vi` as a Vector. For TypedVarInfo we can get them as a NamedTuple, which
# helps keep the downstream call to link!! type stable.
function link!!(t::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return link!!(t, vi, all_varnames_namedtuple(vi), model)
end

# X -> R for all variables associated with given sampler
function link!!(
    t::DynamicTransformation,
    vi::VarInfo,
    vns::Union{VarNameCollection,NamedTuple},
    model::Model,
)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _link(model, vi, vns)
    # Call `_link!` instead of `link!` to avoid deprecation warning.
    _link!(vi, vns)
    return vi
end

function link!!(
    t::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameCollection,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, vns, model)
end

function _link!(vi::UntypedVarInfo, vns::VarNameCollection)
    # TODO: Change to a lazy iterator over `vns`
    if ~istrans(vi, vns[1])
        for vn in vns
            f = internal_to_linked_internal_transform(vi, vn)
            _inner_transform!(vi, vn, f)
            settrans!!(vi, true, vn)
        end
    else
        @warn("[DynamicPPL] attempt to link a linked vi")
    end
end

# If we try to _link! a TypedVarInfo with a Tuple or Vector of VarNames, first convert
# it to a NamedTuple that matches the structure of the TypedVarInfo.
function _link!(vi::TypedVarInfo, vns::VarNameCollection)
    return _link!(vi, varname_namedtuple(vns))
end

function _link!(vi::TypedVarInfo, vns::NamedTuple)
    return _link!(vi.metadata, vi, vns)
end

"""
    filter_subsumed(filter_vns, filtered_vns)

Return the subset of `filtered_vns` that are subsumed by any variable in `filter_vns`.
"""
function filter_subsumed(filter_vns, filtered_vns)
    return filter(x -> any(subsumes(y, x) for y in filter_vns), filtered_vns)
end

@generated function _link!(
    ::NamedTuple{metadata_names}, vi, vns::NamedTuple{vns_names}
) where {metadata_names,vns_names}
    expr = Expr(:block)
    for f in metadata_names
        if !(f in vns_names)
            continue
        end
        push!(
            expr.args,
            quote
                f_vns = vi.metadata.$f.vns
                f_vns = filter_subsumed(vns.$f, f_vns)
                if !isempty(f_vns)
                    if !istrans(vi, f_vns[1])
                        # Iterate over all `f_vns` and transform
                        for vn in f_vns
                            f = internal_to_linked_internal_transform(vi, vn)
                            _inner_transform!(vi, vn, f)
                            settrans!!(vi, true, vn)
                        end
                    else
                        @warn("[DynamicPPL] attempt to link a linked vi")
                    end
                end
            end,
        )
    end
    return expr
end

# Specialise invlink!! without varnames provided for TypedVarInfo. The generic version gets
# the keys of `vi` as a Vector. For TypedVarInfo we can get them as a NamedTuple, which
# helps keep the downstream call to invlink!! type stable.
function invlink!!(t::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return invlink!!(t, vi, all_varnames_namedtuple(vi), model)
end

# R -> X for all variables associated with given sampler
function invlink!!(
    t::DynamicTransformation,
    vi::VarInfo,
    vns::Union{VarNameCollection,NamedTuple},
    model::Model,
)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _invlink(model, vi, vns)
    # Call `_invlink!` instead of `invlink!` to avoid deprecation warning.
    _invlink!(vi, vns)
    return vi
end

function invlink!!(
    ::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameCollection,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.invlink!!(vi.varinfo, vns, model)
end

function maybe_invlink_before_eval!!(vi::VarInfo, context::AbstractContext, model::Model)
    # Because `VarInfo` does not contain any information about what the transformation
    # other than whether or not it has actually been transformed, the best we can do
    # is just assume that `default_transformation` is the correct one if `istrans(vi)`.
    t = istrans(vi) ? default_transformation(model, vi) : NoTransformation()
    return maybe_invlink_before_eval!!(t, vi, context, model)
end

function _invlink!(vi::UntypedVarInfo, vns::VarNameCollection)
    if istrans(vi, vns[1])
        for vn in vns
            f = linked_internal_to_internal_transform(vi, vn)
            _inner_transform!(vi, vn, f)
            settrans!!(vi, false, vn)
        end
    else
        @warn("[DynamicPPL] attempt to invlink an invlinked vi")
    end
end

# If we try to _invlink! a TypedVarInfo with a Tuple or Vector of VarNames, first convert
# it to a NamedTuple that matches the structure of the TypedVarInfo.
function _invlink!(vi::TypedVarInfo, vns::VarNameCollection)
    return _invlink!(vi.metadata, vi, varname_namedtuple(vns))
end

function _invlink!(vi::TypedVarInfo, vns::NamedTuple)
    return _invlink!(vi.metadata, vi, vns)
end

@generated function _invlink!(
    ::NamedTuple{metadata_names}, vi, vns::NamedTuple{vns_names}
) where {metadata_names,vns_names}
    expr = Expr(:block)
    for f in metadata_names
        if !(f in vns_names)
            continue
        end

        push!(
            expr.args,
            quote
                f_vns = vi.metadata.$f.vns
                f_vns = filter_subsumed(vns.$f, f_vns)
                if istrans(vi, f_vns[1])
                    # Iterate over all `f_vns` and transform
                    for vn in f_vns
                        f = linked_internal_to_internal_transform(vi, vn)
                        _inner_transform!(vi, vn, f)
                        settrans!!(vi, false, vn)
                    end
                else
                    @warn("[DynamicPPL] attempt to invlink an invlinked vi")
                end
            end,
        )
    end
    return expr
end

function _inner_transform!(vi::VarInfo, vn::VarName, f)
    return _inner_transform!(getmetadata(vi, vn), vi, vn, f)
end

function _inner_transform!(md::Metadata, vi::VarInfo, vn::VarName, f)
    # TODO: Use inplace versions to avoid allocations
    yvec, logjac = with_logabsdet_jacobian(f, getindex_internal(md, vn))
    # Determine the new range.
    start = first(getrange(md, vn))
    # NOTE: `length(yvec)` should never be longer than `getrange(vi, vn)`.
    setrange!(md, vn, start:(start + length(yvec) - 1))
    # Set the new value.
    setval!(md, yvec, vn)
    acclogp!!(vi, -logjac)
    return vi
end

# HACK: We need `SampleFromPrior` to result in ALL values which are in need
# of a transformation to be transformed. `_getvns` will by default return
# an empty iterable for `SampleFromPrior`, so we need to override it here.
# This is quite hacky, but seems safer than changing the behavior of `_getvns`.
_getvns_link(varinfo::VarInfo, spl::AbstractSampler) = _getvns(varinfo, spl)
_getvns_link(varinfo::VarInfo, spl::SampleFromPrior) = nothing
function _getvns_link(varinfo::TypedVarInfo, spl::SampleFromPrior)
    return map(Returns(nothing), varinfo.metadata)
end

function link(
    ::DynamicTransformation, varinfo::VarInfo, vns::VarNameCollection, model::Model
)
    return _link(model, varinfo, vns)
end

function link(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameCollection,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = link(varinfo.varinfo, vns, model)
end

# Specialise link without varnames provided for TypedVarInfo. The generic version gets
# the keys of `vi` as a Vector. For TypedVarInfo we can get them as a NamedTuple, which
# helps keep the downstream call to _link type stable.
function link(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return _link(model, vi, all_varnames_namedtuple(vi))
end

function _link(
    model::Model, varinfo::Union{UntypedVarInfo,VectorVarInfo}, vns::VarNameCollection
)
    varinfo = deepcopy(varinfo)
    return VarInfo(
        _link_metadata!!(model, varinfo, varinfo.metadata, vns),
        Base.Ref(getlogp(varinfo)),
        Ref(get_num_produce(varinfo)),
    )
end

# If we try to _invlink! a TypedVarInfo with a Tuple or Vector of VarNames, first convert
# it to a NamedTuple that matches the structure of the TypedVarInfo.
function _link(model::Model, varinfo::TypedVarInfo, vns::VarNameCollection)
    return _link(model, varinfo, varname_namedtuple(vns))
end

function _link(model::Model, varinfo::TypedVarInfo, vns::NamedTuple)
    varinfo = deepcopy(varinfo)
    md = _link_metadata!(model, varinfo, varinfo.metadata, vns)
    return VarInfo(md, Base.Ref(getlogp(varinfo)), Ref(get_num_produce(varinfo)))
end

@generated function _link_metadata!(
    model::Model,
    varinfo::VarInfo,
    metadata::NamedTuple{metadata_names},
    vns::NamedTuple{vns_names},
) where {metadata_names,vns_names}
    vals = Expr(:tuple)
    for f in metadata_names
        if f in vns_names
            push!(vals.args, :(_link_metadata!!(model, varinfo, metadata.$f, vns.$f)))
        else
            push!(vals.args, :(metadata.$f))
        end
    end

    return :(NamedTuple{$metadata_names}($vals))
end

function _link_metadata!!(::Model, varinfo::VarInfo, metadata::Metadata, target_vns)
    vns = metadata.vns

    # Construct the new transformed values, and keep track of their lengths.
    vals_new = map(vns) do vn
        # Return early if we're already in unconstrained space.
        # HACK: if `target_vns` is `nothing`, we ignore the `target_vns` check.
        if istrans(varinfo, vn) || (target_vns !== nothing && vn ∉ target_vns)
            return metadata.vals[getrange(metadata, vn)]
        end

        # Transform to constrained space.
        x = getindex_internal(metadata, vn)
        dist = getdist(metadata, vn)
        f = internal_to_linked_internal_transform(varinfo, vn, dist)
        y, logjac = with_logabsdet_jacobian(f, x)
        # Vectorize value.
        yvec = tovec(y)
        # Accumulate the log-abs-det jacobian correction.
        acclogp!!(varinfo, -logjac)
        # Mark as transformed.
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

function _link_metadata!!(
    model::Model, varinfo::VarInfo, metadata::VarNamedVector, target_vns
)
    vns = target_vns === nothing ? keys(metadata) : target_vns
    dists = extract_priors(model, varinfo)
    for vn in vns
        # First transform from however the variable is stored in vnv to the model
        # representation.
        transform_to_orig = gettransform(metadata, vn)
        val_old = getindex_internal(metadata, vn)
        val_orig, logjac1 = with_logabsdet_jacobian(transform_to_orig, val_old)
        # Then transform from the model representation to the linked representation.
        transform_from_linked = from_linked_vec_transform(dists[vn])
        transform_to_linked = inverse(transform_from_linked)
        val_new, logjac2 = with_logabsdet_jacobian(transform_to_linked, val_orig)
        # TODO(mhauru) We are calling a !! function but ignoring the return value.
        # Fix this when attending to issue #653.
        acclogp!!(varinfo, -logjac1 - logjac2)
        metadata = setindex_internal!!(metadata, val_new, vn, transform_from_linked)
        settrans!(metadata, true, vn)
    end
    return metadata
end

function invlink(
    ::DynamicTransformation, varinfo::VarInfo, vns::VarNameCollection, model::Model
)
    return _invlink(model, varinfo, vns)
end

function invlink(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameCollection,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = invlink(varinfo.varinfo, vns, model)
end

# Specialise invlink without varnames provided for TypedVarInfo. The generic version gets
# the keys of `vi` as a Vector. For TypedVarInfo we can get them as a NamedTuple, which
# helps keep the downstream call to _invlink type stable.
function invlink(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return _invlink(model, vi, all_varnames_namedtuple(vi))
end

function _invlink(model::Model, varinfo::VarInfo, vns::VarNameCollection)
    varinfo = deepcopy(varinfo)
    return VarInfo(
        _invlink_metadata!!(model, varinfo, varinfo.metadata, vns),
        Base.Ref(getlogp(varinfo)),
        Ref(get_num_produce(varinfo)),
    )
end

# If we try to _invlink a TypedVarInfo with a Tuple or Vector of VarNames, first convert
# it to a NamedTuple that matches the structure of the TypedVarInfo.
function _invlink(model::Model, varinfo::TypedVarInfo, vns::VarNameCollection)
    return _invlink(model, varinfo, varname_namedtuple(vns))
end

function _invlink(model::Model, varinfo::TypedVarInfo, vns::NamedTuple)
    varinfo = deepcopy(varinfo)
    md = _invlink_metadata!(model, varinfo, varinfo.metadata, vns)
    return VarInfo(md, Base.Ref(getlogp(varinfo)), Ref(get_num_produce(varinfo)))
end

@generated function _invlink_metadata!(
    model::Model,
    varinfo::VarInfo,
    metadata::NamedTuple{metadata_names},
    vns::NamedTuple{vns_names},
) where {metadata_names,vns_names}
    vals = Expr(:tuple)
    for f in metadata_names
        if (f in vns_names)
            push!(vals.args, :(_invlink_metadata!!(model, varinfo, metadata.$f, vns.$f)))
        else
            push!(vals.args, :(metadata.$f))
        end
    end

    return :(NamedTuple{$metadata_names}($vals))
end

function _invlink_metadata!!(::Model, varinfo::VarInfo, metadata::Metadata, target_vns)
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
        y = getindex_internal(varinfo, vn)
        dist = getdist(varinfo, vn)
        f = from_linked_internal_transform(varinfo, vn, dist)
        x, logjac = with_logabsdet_jacobian(f, y)
        # Vectorize value.
        xvec = tovec(x)
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

function _invlink_metadata!!(
    ::Model, varinfo::VarInfo, metadata::VarNamedVector, target_vns
)
    vns = target_vns === nothing ? keys(metadata) : target_vns
    for vn in vns
        transform = gettransform(metadata, vn)
        old_val = getindex_internal(metadata, vn)
        new_val, logjac = with_logabsdet_jacobian(transform, old_val)
        # TODO(mhauru) We are calling a !! function but ignoring the return value.
        acclogp!!(varinfo, -logjac)
        new_transform = from_vec_transform(new_val)
        metadata = setindex_internal!!(metadata, tovec(new_val), vn, new_transform)
        settrans!(metadata, false, vn)
    end
    return metadata
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
        push!(out, :(isempty(vns.$f) ? false : istrans(vi, vns.$f[1])))
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
function _nested_setindex_maybe!(
    vi::VarInfo, md::Union{Metadata,VarNamedVector}, val, vn::VarName
)
    # If `vn` is in `vns`, then we can just use the standard `setindex!`.
    vns = Base.keys(md)
    if vn in vns
        setindex!(vi, val, vn)
        return vn
    end

    # Otherwise, we need to check if either of the `vns` subsumes `vn`.
    i = findfirst(Base.Fix2(subsumes, vn), vns)
    i === nothing && return nothing

    vn_parent = vns[i]
    val_parent = getindex(vi, vn_parent)  # TODO: Ensure that we're working with a view here.
    # Split the varname into its tail optic.
    optic = remove_parent_optic(vn_parent, vn)
    # Update the value for the parent.
    val_parent_updated = set!!(val_parent, optic, val)
    setindex!(vi, val_parent_updated, vn_parent)
    return vn_parent
end

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
function getindex(vi::VarInfo, vn::VarName)
    return from_maybe_linked_internal_transform(vi, vn)(getindex_internal(vi, vn))
end

function getindex(vi::VarInfo, vn::VarName, dist::Distribution)
    @assert haskey(vi, vn) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    val = getindex_internal(vi, vn)
    return from_maybe_linked_internal(vi, vn, dist, val)
end

function getindex(vi::VarInfo, vns::Vector{<:VarName})
    vals = map(vn -> getindex(vi, vn), vns)

    et = eltype(vals)
    # This will catch type unstable cases, where vals has mixed types.
    if !isconcretetype(et)
        throw(ArgumentError("All variables must have the same type."))
    end

    if et <: Vector
        all_of_equal_dimension = all(x -> length(x) == length(vals[1]), vals)
        if !all_of_equal_dimension
            throw(ArgumentError("All variables must have the same dimension."))
        end
    end

    # TODO(mhauru) I'm not very pleased with the return type varying like this, even though
    # this should be type stable.
    vec_vals = reduce(vcat, vals)
    if et <: Vector
        # The individual variables are multivariate, and thus we return the values as a
        # matrix.
        return reshape(vec_vals, (:, length(vns)))
    else
        # The individual variables are univariate, and thus we return a vector of scalars.
        return vec_vals
    end
end

function getindex(vi::VarInfo, vns::Vector{<:VarName}, dist::Distribution)
    @assert haskey(vi, vns[1]) "[DynamicPPL] attempted to replay unexisting variables in VarInfo"
    vals_linked = mapreduce(vcat, vns) do vn
        getindex(vi, vn, dist)
    end
    return recombine(dist, vals_linked, length(vns))
end

"""
    getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})

Return the current value(s) of the random variables sampled by `spl` in `vi`.

The value(s) may or may not be transformed to Euclidean space.
"""
getindex(vi::UntypedVarInfo, spl::Sampler) =
    copy(getindex(vi.metadata.vals, _getranges(vi, spl)))
getindex(vi::VarInfo, spl::Sampler) = copy(getindex_internal(vi, _getranges(vi, spl)))
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

# TODO(mhauru) I think the below implementation of setindex! is a mistake. It should be
# called setindex_internal! since it directly writes to the `vals` field of the metadata.
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

Base.haskey(metadata::Metadata, vn::VarName) = haskey(metadata.idcs, vn)

"""
    haskey(vi::VarInfo, vn::VarName)

Check whether `vn` has been sampled in `vi`.
"""
Base.haskey(vi::VarInfo, vn::VarName) = haskey(getmetadata(vi, vn), vn)
function Base.haskey(vi::TypedVarInfo, vn::VarName)
    md_haskey = map(vi.metadata) do metadata
        haskey(metadata, vn)
    end
    return any(md_haskey)
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
    vns = keys(md)

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
        @assert ~(vn in keys(vi)) "[push!!] attempt to add an existing variable $(getsym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gidset"
    elseif vi isa TypedVarInfo
        @assert ~(haskey(vi, vn)) "[push!!] attempt to add an existing variable $(getsym(vn)) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gidset"
    end

    sym = getsym(vn)
    if vi isa TypedVarInfo && ~haskey(vi.metadata, sym)
        # The NamedTuple doesn't have an entry for this variable, let's add one.
        val = tovec(r)
        md = Metadata(
            Dict(vn => 1),
            [vn],
            [1:length(val)],
            val,
            [dist],
            [gidset],
            [get_num_produce(vi)],
            Dict{String,BitVector}("trans" => [false], "del" => [false]),
        )
        vi = Accessors.@set vi.metadata[sym] = md
    else
        meta = getmetadata(vi, vn)
        push!(meta, vn, r, dist, gidset, get_num_produce(vi))
    end

    return vi
end

function Base.push!(vi::VectorVarInfo, vn::VarName, val, args...)
    push!(getmetadata(vi, vn), vn, val, args...)
    return vi
end

function Base.push!(vi::VectorVarInfo, pair::Pair, args...)
    vn, val = pair
    return push!(vi, vn, val, args...)
end

# TODO(mhauru) push! can't be implemented in-place for TypedVarInfo if the symbol doesn't
# exist in the TypedVarInfo already. We could implement it in the cases where it it does
# exist, but that feels a bit pointless. I think we should rather rely on `push!!`.

function Base.push!(meta::Metadata, vn, r, dist, gidset, num_produce)
    val = tovec(r)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals)
    n = length(val)
    push!(meta.ranges, (l + 1):(l + n))
    append!(meta.vals, val)
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, num_produce)
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)
    return meta
end

function Base.delete!(vi::VarInfo, vn::VarName)
    delete!(getmetadata(vi, vn), vn)
    return vi
end

"""
    setorder!(vi::VarInfo, vn::VarName, index::Int)

Set the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe
statements run before sampling `vn`.
"""
function setorder!(vi::VarInfo, vn::VarName, index::Int)
    setorder!(getmetadata(vi, vn), vn, index)
    return vi
end
function setorder!(metadata::Metadata, vn::VarName, index::Int)
    metadata.orders[metadata.idcs[vn]] = index
    return metadata
end
setorder!(vnv::VarNamedVector, ::VarName, ::Int) = vnv

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
function is_flagged(::VarNamedVector, ::VarName, flag::String)
    if flag == "del"
        return true
    else
        throw(ErrorException("Flag $flag not valid for VarNamedVector"))
    end
end

# TODO(mhauru) The "ignorable" argument is a temporary hack while developing VarNamedVector,
# but still having to support the interface based on Metadata too
"""
    unset_flag!(vi::VarInfo, vn::VarName, flag::String, ignorable::Bool=false

Set `vn`'s value for `flag` to `false` in `vi`.

Setting some flags for some `VarInfo` types is not possible, and by default attempting to do
so will error. If `ignorable` is set to `true` then this will silently be ignored instead.
"""
function unset_flag!(vi::VarInfo, vn::VarName, flag::String, ignorable::Bool=false)
    unset_flag!(getmetadata(vi, vn), vn, flag, ignorable)
    return vi
end
function unset_flag!(metadata::Metadata, vn::VarName, flag::String, ignorable::Bool=false)
    metadata.flags[flag][getidx(metadata, vn)] = false
    return metadata
end

function unset_flag!(vnv::VarNamedVector, ::VarName, flag::String, ignorable::Bool=false)
    if ignorable
        return vnv
    end
    if flag == "del"
        throw(ErrorException("The \"del\" flag cannot be unset for VarNamedVector"))
    else
        throw(ErrorException("Flag $flag not valid for VarNamedVector"))
    end
    return vnv
end

"""
    set_retained_vns_del!(vi::VarInfo)

Set the `"del"` flag of variables in `vi` with `order > vi.num_produce[]` to `true`.
"""
function set_retained_vns_del!(vi::UntypedVarInfo)
    idcs = _getidcs(vi)
    if get_num_produce(vi) == 0
        for i in length(idcs):-1:1
            vi.metadata.flags["del"][idcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in idcs && vi.orders[i] > get_num_produce(vi)
                vi.metadata.flags["del"][i] = true
            end
        end
    end
    return nothing
end
function set_retained_vns_del!(vi::TypedVarInfo)
    idcs = _getidcs(vi)
    return _set_retained_vns_del!(vi.metadata, idcs, get_num_produce(vi))
end
@generated function _set_retained_vns_del!(
    metadata, idcs::NamedTuple{names}, num_produce
) where {names}
    expr = Expr(:block)
    for f in names
        f_idcs = :(idcs.$f)
        f_orders = :(metadata.$f.orders)
        f_flags = :(metadata.$f.flags)
        push!(
            expr.args,
            quote
                # Set the flag for variables with symbol `f`
                if num_produce == 0
                    for i in length($f_idcs):-1:1
                        $f_flags["del"][$f_idcs[i]] = true
                    end
                else
                    for i in 1:length($f_orders)
                        if i in $f_idcs && $f_orders[i] > num_produce
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
    keys_strings = map(string, collect_maybe(keys))
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
    return _typed_apply!(kernel!, vi, vi.metadata, values, collect_maybe(keys))
end

@generated function _typed_apply!(
    kernel!, vi::TypedVarInfo, metadata::NamedTuple{names}, values, keys
) where {names}
    updates = map(names) do n
        quote
            for vn in Base.keys(metadata.$n)
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
    string_vns = map(string, collect_maybe(Base.keys(vi)))
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

Note that this does *not* resample the values not provided! It will call
`setflag!(vi, vn, "del")` for variables `vn` for which no values are provided, which means
that the next time we call `model(vi)` these variables will be resampled.

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

julia> var_info = DynamicPPL.VarInfo(rng, m, SampleFromPrior(), DefaultContext(), DynamicPPL.Metadata());  # Checking the setting of "del" flags only makes sense for VarInfo{<:Metadata}. For VarInfo{<:VarNamedVector} the flag is effectively always set.

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

values_as(vi::VectorVarInfo, args...) = values_as(vi.metadata, args...)
values_as(vi::VectorVarInfo, T::Type{Vector}) = values_as(vi.metadata, T)

function values_from_metadata(md::Metadata)
    return (
        # `copy` to avoid accidentally mutation of internal representation.
        vn => copy(
            from_internal_transform(md, vn, getdist(md, vn))(getindex_internal(md, vn))
        ) for vn in md.vns
    )
end

values_from_metadata(md::VarNamedVector) = pairs(md)

# Transforming from internal representation to distribution representation.
# Without `dist` argument: base on `dist` extracted from self.
function from_internal_transform(vi::VarInfo, vn::VarName)
    return from_internal_transform(getmetadata(vi, vn), vn)
end
function from_internal_transform(md::Metadata, vn::VarName)
    return from_internal_transform(md, vn, getdist(md, vn))
end
function from_internal_transform(md::VarNamedVector, vn::VarName)
    return gettransform(md, vn)
end
# With both `vn` and `dist` arguments: base on provided `dist`.
function from_internal_transform(vi::VarInfo, vn::VarName, dist)
    return from_internal_transform(getmetadata(vi, vn), vn, dist)
end
from_internal_transform(::Metadata, ::VarName, dist) = from_vec_transform(dist)
function from_internal_transform(::VarNamedVector, ::VarName, dist)
    return from_vec_transform(dist)
end

# Without `dist` argument: base on `dist` extracted from self.
function from_linked_internal_transform(vi::VarInfo, vn::VarName)
    return from_linked_internal_transform(getmetadata(vi, vn), vn)
end
function from_linked_internal_transform(md::Metadata, vn::VarName)
    return from_linked_internal_transform(md, vn, getdist(md, vn))
end
function from_linked_internal_transform(md::VarNamedVector, vn::VarName)
    return gettransform(md, vn)
end
# With both `vn` and `dist` arguments: base on provided `dist`.
function from_linked_internal_transform(vi::VarInfo, vn::VarName, dist)
    # Dispatch to metadata in case this alters the behavior.
    return from_linked_internal_transform(getmetadata(vi, vn), vn, dist)
end
function from_linked_internal_transform(::Metadata, ::VarName, dist)
    return from_linked_vec_transform(dist)
end
function from_linked_internal_transform(::VarNamedVector, ::VarName, dist)
    return from_linked_vec_transform(dist)
end
