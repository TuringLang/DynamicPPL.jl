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
const UntypedVectorVarInfo = VarInfo{<:VarNamedVector}
const UntypedVarInfo = VarInfo{<:Metadata}
# TODO: TypedVarInfo carries no information about the type of the actual
# metadata i.e. the elements of the NamedTuple. It could be Metadata or it
# could be VarNamedVector. Calling TypedVarInfo(model) will result in a
# TypedVarInfo where the elements are Metadata.
# Resolving this ambiguity would likely require us to replace NamedTuple with
# something which carried both its keys as well as its values' types as type
# parameters.
# Note that below we also define a function TypedVectorVarInfo, which generates
# a TypedVarInfo where the metadata is a NamedTuple of VarNameVectors.
const TypedVarInfo = VarInfo{<:NamedTuple}
const VarInfoOrThreadSafeVarInfo{Tmeta} = Union{
    VarInfo{Tmeta},ThreadSafeVarInfo{<:VarInfo{Tmeta}}
}

# NOTE: This is kind of weird, but it effectively preserves the "old"
# behavior where we're allowed to call `link!` on the same `VarInfo`
# multiple times.
transformation(::VarInfo) = DynamicTransformation()

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

function has_varnamedvector(vi::VarInfo)
    return vi.metadata isa VarNamedVector ||
           (vi isa TypedVarInfo && any(Base.Fix2(isa, VarNamedVector), values(vi.metadata)))
end

########################
# VarInfo constructors #
########################

"""
    UntypedVarInfo([rng, ]model[, sampler, context, metadata])

Return an untyped varinfo object for the given `model` and `context`.

# Arguments
- `rng::Random.AbstractRNG`: The random number generator to use during model evaluation
- `model::Model`: The model for which to create the varinfo object
- `sampler::AbstractSampler`: The sampler to use for the model. Defaults to `SampleFromPrior()`.
- `context::AbstractContext`: The context in which to evaluate the model. Defaults to `DefaultContext()`.
"""
function UntypedVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    varinfo = VarInfo(Metadata())
    context = SamplingContext(rng, sampler, context)
    return last(evaluate!!(model, varinfo, context))
end
function UntypedVarInfo(model::Model, args::Union{AbstractSampler,AbstractContext}...)
    return UntypedVarInfo(Random.default_rng(), model, args...)
end

function TypedVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    return TypedVarInfo(UntypedVarInfo(rng, model, sampler, context))
end
function TypedVarInfo(model::Model, args::Union{AbstractSampler,AbstractContext}...)
    return TypedVarInfo(Random.default_rng(), model, args...)
end

function UntypedVectorVarInfo(vi::UntypedVarInfo)
    md = metadata_to_varnamedvector(vi.metadata)
    lp = getlogp(vi)
    return VarInfo(md, Base.RefValue{eltype(lp)}(lp), Ref(get_num_produce(vi)))
end
function UntypedVectorVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    return UntypedVectorVarInfo(UntypedVarInfo(rng, model, sampler, context))
end
function UntypedVectorVarInfo(model::Model, args::Union{AbstractSampler,AbstractContext}...)
    return UntypedVectorVarInfo(UntypedVarInfo(Random.default_rng(), model, args...))
end

function TypedVectorVarInfo(vi::TypedVarInfo)
    md = map(metadata_to_varnamedvector, vi.metadata)
    lp = getlogp(vi)
    return VarInfo(md, Base.RefValue{eltype(lp)}(lp), Ref(get_num_produce(vi)))
end
function TypedVectorVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    return TypedVectorVarInfo(TypedVarInfo(rng, model, sampler, context))
end
function TypedVectorVarInfo(model::Model, args::Union{AbstractSampler,AbstractContext}...)
    return TypedVectorVarInfo(Random.default_rng(), model, args...)
end

"""
    vector_length(varinfo::VarInfo)

Return the length of the vector representation of `varinfo`.
"""
vector_length(varinfo::VarInfo) = length(varinfo.metadata)
vector_length(varinfo::TypedVarInfo) = sum(length, varinfo.metadata)
vector_length(md::Metadata) = sum(length, md.ranges)

function unflatten(vi::VarInfo, x::AbstractVector)
    md = unflatten_metadata(vi.metadata, x)
    # Note that use of RefValue{eltype(x)} rather than Ref is necessary to deal with cases
    # where e.g. x is a type gradient of some AD backend.
    return VarInfo(
        md,
        Base.RefValue{float_type_with_fallback(eltype(x))}(getlogp(vi)),
        Ref(get_num_produce(vi)),
    )
end

# We would call this `unflatten` if not for `unflatten` having a method for NamedTuples in
# utils.jl.
@generated function unflatten_metadata(
    metadata::NamedTuple{names}, x::AbstractVector
) where {names}
    exprs = []
    offset = :(0)
    for f in names
        mdf = :(metadata.$f)
        len = :(sum(length, $mdf.ranges))
        push!(exprs, :($f = unflatten_metadata($mdf, x[($offset + 1):($offset + $len)])))
        offset = :($offset + $len)
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

function unflatten_metadata(md::Metadata, x::AbstractVector)
    return Metadata(md.idcs, md.vns, md.ranges, x, md.dists, md.orders, md.flags)
end

unflatten_metadata(vnv::VarNamedVector, x::AbstractVector) = unflatten(vnv, x)

# without AbstractSampler
function VarInfo(rng::Random.AbstractRNG, model::Model, context::AbstractContext)
    return VarInfo(rng, model, SampleFromPrior(), context)
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

function subset(varinfo::VarInfo, vns::AbstractVector{<:VarName})
    metadata = subset(varinfo.metadata, vns)
    return VarInfo(metadata, deepcopy(varinfo.logp), deepcopy(varinfo.num_produce))
end

function subset(metadata::NamedTuple, vns::AbstractVector{<:VarName})
    vns_syms = Set(unique(map(getsym, vns)))
    syms = filter(Base.Fix2(in, vns_syms), keys(metadata))
    metadatas = map(syms) do sym
        subset(getfield(metadata, sym), filter(==(sym) ∘ getsym, vns))
    end
    return NamedTuple{syms}(metadatas)
end

# The above method is type unstable since we don't know which symbols are in `vns`.
# In the below special case, when all `vns` have the same symbol, we can write a type stable
# version.

@generated function subset(
    metadata::NamedTuple{names}, vns::AbstractVector{<:VarName{sym}}
) where {names,sym}
    return if (sym in names)
        # TODO(mhauru) Note that this could still generate an empty metadata object if none
        # of the lenses in `vns` are in `metadata`. Not sure if that's okay. Checking for
        # emptiness would make this type unstable again.
        :((; $sym=subset(metadata.$sym, vns)))
    else
        :(NamedTuple{}())
    end
end

function subset(metadata::Metadata, vns_given::AbstractVector{VN}) where {VN<:VarName}
    # TODO: Should we error if `vns` contains a variable that is not in `metadata`?
    # Find all the vns in metadata that are subsumed by one of the given vns.
    vns = filter(vn -> any(subsumes(vn_given, vn) for vn_given in vns_given), metadata.vns)
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
        push!(orders, getorder(metadata_for_vn, vn))
        for k in keys(flags)
            push!(flags[k], is_flagged(metadata_for_vn, vn, k))
        end
    end

    return Metadata(idcs, vns, ranges, vals, dists, orders, flags)
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
getindex_internal(vi::VarInfo, ::Colon) = getindex_internal(vi.metadata, Colon())
# NOTE: `mapreduce` over `NamedTuple` results in worse type-inference.
# See for example https://github.com/JuliaLang/julia/pull/46381.
function getindex_internal(vi::TypedVarInfo, ::Colon)
    return reduce(vcat, map(Base.Fix2(getindex_internal, Colon()), vi.metadata))
end
function getindex_internal(md::Metadata, ::Colon)
    return mapreduce(
        Base.Fix1(getindex_internal, md), vcat, md.vns; init=similar(md.vals, 0)
    )
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

Returns a tuple of the unique symbols of random variables in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(getsym, vi.metadata.vns)))  # get all symbols
syms(vi::TypedVarInfo) = keys(vi.metadata)

_getidcs(vi::UntypedVarInfo) = 1:length(vi.metadata.idcs)
_getidcs(vi::TypedVarInfo) = _getidcs(vi.metadata)

@generated function _getidcs(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findinds(metadata.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

@inline findinds(f_meta::Metadata) = eachindex(f_meta.vns)
findinds(vnv::VarNamedVector) = 1:length(vnv.varnames)

"""
    all_varnames_grouped_by_symbol(vi::TypedVarInfo)

Return a `NamedTuple` of the variables in `vi` grouped by symbol.
"""
all_varnames_grouped_by_symbol(vi::TypedVarInfo) =
    all_varnames_grouped_by_symbol(vi.metadata)

@generated function all_varnames_grouped_by_symbol(md::NamedTuple{names}) where {names}
    expr = Expr(:tuple)
    for f in names
        push!(expr.args, :($f = keys(md.$f)))
    end
    return expr
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

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{LogProbType}(0.0), Ref(0))

function TypedVarInfo(vi::UntypedVectorVarInfo)
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
                sym_idcs, sym_vns, sym_ranges, sym_vals, sym_dists, sym_orders, sym_flags
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

function link!!(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    vns = all_varnames_grouped_by_symbol(vi)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _link(model, vi, vns)
    _link!(vi, vns)
    return vi
end

function link!!(::DynamicTransformation, vi::VarInfo, model::Model)
    vns = keys(vi)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _link(model, vi, vns)
    _link!(vi, vns)
    return vi
end

function link!!(t::DynamicTransformation, vi::ThreadSafeVarInfo{<:VarInfo}, model::Model)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, model)
end

function link!!(::DynamicTransformation, vi::VarInfo, vns::VarNameTuple, model::Model)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _link(model, vi, vns)
    # Call `_link!` instead of `link!` to avoid deprecation warning.
    _link!(vi, vns)
    return vi
end

function link!!(
    t::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.link!!(t, vi.varinfo, vns, model)
end

function _link!(vi::UntypedVarInfo, vns)
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

# If we try to _link! a TypedVarInfo with a Tuple of VarNames, first convert it to a
# NamedTuple that matches the structure of the TypedVarInfo.
function _link!(vi::TypedVarInfo, vns::VarNameTuple)
    return _link!(vi, group_varnames_by_symbol(vns))
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

function invlink!!(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    vns = all_varnames_grouped_by_symbol(vi)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _invlink(model, vi, vns)
    # Call `_invlink!` instead of `invlink!` to avoid deprecation warning.
    _invlink!(vi, vns)
    return vi
end

function invlink!!(::DynamicTransformation, vi::VarInfo, model::Model)
    vns = keys(vi)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _invlink(model, vi, vns)
    _invlink!(vi, vns)
    return vi
end

function invlink!!(t::DynamicTransformation, vi::ThreadSafeVarInfo{<:VarInfo}, model::Model)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.invlink!!(t, vi.varinfo, model)
end

function invlink!!(::DynamicTransformation, vi::VarInfo, vns::VarNameTuple, model::Model)
    # If we're working with a `VarNamedVector`, we always use immutable.
    has_varnamedvector(vi) && return _invlink(model, vi, vns)
    # Call `_invlink!` instead of `invlink!` to avoid deprecation warning.
    _invlink!(vi, vns)
    return vi
end

function invlink!!(
    ::DynamicTransformation,
    vi::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set vi.varinfo = DynamicPPL.invlink!!(vi.varinfo, vns, model)
end

function maybe_invlink_before_eval!!(vi::VarInfo, model::Model)
    # Because `VarInfo` does not contain any information about what the transformation
    # other than whether or not it has actually been transformed, the best we can do
    # is just assume that `default_transformation` is the correct one if `istrans(vi)`.
    t = istrans(vi) ? default_transformation(model, vi) : NoTransformation()
    return maybe_invlink_before_eval!!(t, vi, model)
end

function _invlink!(vi::UntypedVarInfo, vns)
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

# If we try to _invlink! a TypedVarInfo with a Tuple of VarNames, first convert it to a
# NamedTuple that matches the structure of the TypedVarInfo.
function _invlink!(vi::TypedVarInfo, vns::VarNameTuple)
    return _invlink!(vi.metadata, vi, group_varnames_by_symbol(vns))
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

function link(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return _link(model, vi, all_varnames_grouped_by_symbol(vi))
end

function link(::DynamicTransformation, varinfo::VarInfo, model::Model)
    return _link(model, varinfo, keys(varinfo))
end

function link(::DynamicTransformation, varinfo::ThreadSafeVarInfo{<:VarInfo}, model::Model)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = link(varinfo.varinfo, model)
end

function link(::DynamicTransformation, varinfo::VarInfo, vns::VarNameTuple, model::Model)
    return _link(model, varinfo, vns)
end

function link(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`,
    # and so we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = link(varinfo.varinfo, vns, model)
end

function _link(model::Model, varinfo::VarInfo, vns)
    varinfo = deepcopy(varinfo)
    md = _link_metadata!!(model, varinfo, varinfo.metadata, vns)
    return VarInfo(md, Base.Ref(getlogp(varinfo)), Ref(get_num_produce(varinfo)))
end

# If we try to _link a TypedVarInfo with a Tuple of VarNames, first convert it to a
# NamedTuple that matches the structure of the TypedVarInfo.
function _link(model::Model, varinfo::TypedVarInfo, vns::VarNameTuple)
    return _link(model, varinfo, group_varnames_by_symbol(vns))
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

function invlink(::DynamicTransformation, vi::TypedVarInfo, model::Model)
    return _invlink(model, vi, all_varnames_grouped_by_symbol(vi))
end

function invlink(::DynamicTransformation, vi::VarInfo, model::Model)
    return _invlink(model, vi, keys(vi))
end

function invlink(
    ::DynamicTransformation, varinfo::ThreadSafeVarInfo{<:VarInfo}, model::Model
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = invlink(varinfo.varinfo, model)
end

function invlink(::DynamicTransformation, varinfo::VarInfo, vns::VarNameTuple, model::Model)
    return _invlink(model, varinfo, vns)
end

function invlink(
    ::DynamicTransformation,
    varinfo::ThreadSafeVarInfo{<:VarInfo},
    vns::VarNameTuple,
    model::Model,
)
    # By default this will simply evaluate the model with `DynamicTransformationContext`, and so
    # we need to specialize to avoid this.
    return Accessors.@set varinfo.varinfo = invlink(varinfo.varinfo, vns, model)
end

function _invlink(model::Model, varinfo::VarInfo, vns)
    varinfo = deepcopy(varinfo)
    return VarInfo(
        _invlink_metadata!!(model, varinfo, varinfo.metadata, vns),
        Base.Ref(getlogp(varinfo)),
        Ref(get_num_produce(varinfo)),
    )
end

# If we try to _invlink a TypedVarInfo with a Tuple of VarNames, first convert it to a
# NamedTuple that matches the structure of the TypedVarInfo.
function _invlink(model::Model, varinfo::TypedVarInfo, vns::VarNameTuple)
    return _invlink(model, varinfo, group_varnames_by_symbol(vns))
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
        # supposed to touch this `vn`.
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

# TODO(mhauru) The treatment of the case when some variables are linked and others are not
# should be revised. It used to be the case that for UntypedVarInfo `islinked` returned
# whether the first variable was linked. For TypedVarInfo we did an OR over the first
# variables under each symbol. We now more consistently use OR, but I'm not convinced this
# is really the right thing to do.
"""
    islinked(vi::VarInfo)

Check whether `vi` is in the transformed space.

Turing's Hamiltonian samplers use the `link` and `invlink` functions from
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) to map a constrained variable
(for example, one bounded to the space `[0, 1]`) from its constrained space to the set of
real numbers. `islinked` checks if the number is in the constrained space or the real space.

If some but only some of the variables in `vi` are linked, this function will return `true`.
This behavior will likely change in the future.
"""
function islinked(vi::VarInfo)
    return any(istrans(vi, vn) for vn in keys(vi))
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

@inline function findvns(vi, f_vns)
    if length(f_vns) == 0
        throw("Unidentified error, please report this error in an issue.")
    end
    return map(vn -> vi[vn], f_vns)
end

Base.haskey(metadata::Metadata, vn::VarName) = haskey(metadata.idcs, vn)

"""
    haskey(vi::VarInfo, vn::VarName)

Check whether `vn` has a value in `vi`.
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

"""
    push!!(vi::VarInfo, vn::VarName, r, dist::Distribution)

Push a new random variable `vn` with a sampled value `r` from a distribution `dist` to
the `VarInfo` `vi`, mutating if it makes sense.
"""
function BangBang.push!!(vi::VarInfo, vn::VarName, r, dist::Distribution)
    if vi isa UntypedVarInfo
        @assert ~(vn in keys(vi)) "[push!!] attempt to add an existing variable $(getsym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist"
    elseif vi isa TypedVarInfo
        @assert ~(haskey(vi, vn)) "[push!!] attempt to add an existing variable $(getsym(vn)) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist"
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
            [get_num_produce(vi)],
            Dict{String,BitVector}("trans" => [false], "del" => [false]),
        )
        vi = Accessors.@set vi.metadata[sym] = md
    else
        meta = getmetadata(vi, vn)
        push!(meta, vn, r, dist, get_num_produce(vi))
    end

    return vi
end

function Base.push!(vi::UntypedVectorVarInfo, vn::VarName, val, args...)
    push!(getmetadata(vi, vn), vn, val, args...)
    return vi
end

function Base.push!(vi::UntypedVectorVarInfo, pair::Pair, args...)
    vn, val = pair
    return push!(vi, vn, val, args...)
end

# TODO(mhauru) push! can't be implemented in-place for TypedVarInfo if the symbol doesn't
# exist in the TypedVarInfo already. We could implement it in the cases where it it does
# exist, but that feels a bit pointless. I think we should rather rely on `push!!`.

function Base.push!(meta::Metadata, vn, r, dist, num_produce)
    val = tovec(r)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals)
    n = length(val)
    push!(meta.ranges, (l + 1):(l + n))
    append!(meta.vals, val)
    push!(meta.dists, dist)
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
values_as(vi::VarInfo, ::Type{Vector}) = copy(getindex_internal(vi, Colon()))
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

values_as(vi::UntypedVectorVarInfo, args...) = values_as(vi.metadata, args...)
values_as(vi::UntypedVectorVarInfo, T::Type{Vector}) = values_as(vi.metadata, T)

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
