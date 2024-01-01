# Similar to `Metadata` but representing a `Vector` and simpler interface.
# TODO: Should we subtype `AbstractVector` or `AbstractDict`?
"""
    VarNameVector

A container that works like a `Vector` and an `OrderedDict` but is neither.

# Fields
$(FIELDS)
"""
struct VarNameVector{
    K<:VarName,V,TVN<:AbstractVector{K},TVal<:AbstractVector{V},TTrans<:AbstractVector,MData
}
    "mapping from the `VarName` to its integer index in `varnames`, `ranges` and `dists`"
    varname_to_index::OrderedDict{K,Int}

    "vector of identifiers for the random variables, where `varnames[varname_to_index[vn]] == vn`"
    varnames::TVN # AbstractVector{<:VarName}

    "vector of index ranges in `vals` corresponding to `varnames`; each `VarName` `vn` has a single index or a set of contiguous indices in `vals`"
    ranges::Vector{UnitRange{Int}}

    "vector of values of all the univariate, multivariate and matrix variables; the value(s) of `vn` is/are `vals[ranges[varname_to_index[vn]]]`"
    vals::TVal # AbstractVector{<:Real}

    "vector of transformations whose inverse takes us back to the original space"
    transforms::TTrans

    "inactive ranges"
    inactive_ranges::Vector{UnitRange{Int}}

    "metadata associated with the varnames"
    metadata::MData
end

function ==(vnv_left::VarNameVector, vnv_right::VarNameVector)
    return vnv_left.varname_to_index == vnv_right.varname_to_index &&
           vnv_left.varnames == vnv_right.varnames &&
           vnv_left.ranges == vnv_right.ranges &&
           vnv_left.vals == vnv_right.vals &&
           vnv_left.transforms == vnv_right.transforms &&
           vnv_left.inactive_ranges == vnv_right.inactive_ranges &&
           vnv_left.metadata == vnv_right.metadata
end

function VarNameVector(varname_to_index, varnames, ranges, vals, transforms)
    return VarNameVector(
        varname_to_index, varnames, ranges, vals, transforms, UnitRange{Int}[], nothing
    )
end
# TODO: Do we need this?
function VarNameVector{K,V}() where {K,V}
    return VarNameVector(OrderedDict{K,Int}(), K[], UnitRange{Int}[], V[], Any[])
end

# Useful transformation going from the flattened representation.
struct FromVec{Sz}
    sz::Sz
end

FromVec(x::Union{Real,AbstractArray}) = FromVec(size(x))

# TODO: Should we materialize the `reshape`?
(f::FromVec)(x) = reshape(x, f.sz)
(f::FromVec{Tuple{}})(x) = only(x)

Bijectors.with_logabsdet_jacobian(f::FromVec, x) = (f(x), 0)

tovec(x::Real) = [x]
tovec(x::AbstractArray) = vec(x)

Bijectors.inverse(f::FromVec) = tovec
Bijectors.inverse(f::FromVec{Tuple{}}) = tovec

# More convenient constructors.
collect_maybe(x) = collect(x)
collect_maybe(x::AbstractArray) = x

VarNameVector() = VarNameVector{VarName,Real}()
VarNameVector(xs::Pair...) = VarNameVector(OrderedDict(xs...))
VarNameVector(x::AbstractDict) = VarNameVector(keys(x), values(x))
VarNameVector(varnames, vals) = VarNameVector(collect_maybe(varnames), collect_maybe(vals))
function VarNameVector(
    varnames::AbstractVector, vals::AbstractVector, transforms=map(FromVec, vals)
)
    # TODO: Check uniqueness of `varnames`?

    # Convert `vals` into a vector of vectors.
    vals_vecs = map(tovec, vals)

    # TODO: Is this really the way to do this?
    if !(eltype(varnames) <: VarName)
        varnames = convert(Vector{VarName}, varnames)
    end
    varname_to_index = OrderedDict{eltype(varnames),Int}()
    ranges = Vector{UnitRange{Int}}()
    offset = 0
    for (i, (vn, x)) in enumerate(zip(varnames, vals_vecs))
        # Add the varname index.
        push!(varname_to_index, vn => length(varname_to_index) + 1)
        # Add the range.
        r = (offset + 1):(offset + length(x))
        push!(ranges, r)
        # Update the offset.
        offset = r[end]
    end

    return VarNameVector(
        varname_to_index, varnames, ranges, reduce(vcat, vals_vecs), transforms
    )
end

# Some `VarNameVector` specific functions.
getidx(vnv::VarNameVector, vn::VarName) = vnv.varname_to_index[vn]

getrange(vnv::VarNameVector, i::Int) = vnv.ranges[i]
getrange(vnv::VarNameVector, vn::VarName) = getrange(vnv, getidx(vnv, vn))

gettransform(vnv::VarNameVector, vn::VarName) = vnv.transforms[getidx(vnv, vn)]

has_inactive_ranges(vnv::VarNameVector) = !isempty(vnv.inactive_ranges)

# Basic array interface.
Base.eltype(vnv::VarNameVector) = eltype(vnv.vals)
Base.length(vnv::VarNameVector) =
    if isempty(vnv.inactive_ranges)
        length(vnv.vals)
    else
        sum(length, vnv.ranges)
    end
Base.size(vnv::VarNameVector) = (length(vnv),)

# TODO: We should probably remove this
Base.IndexStyle(::Type{<:VarNameVector}) = IndexLinear()

# Dictionary interface.
Base.keys(vnv::VarNameVector) = vnv.varnames

Base.haskey(vnv::VarNameVector, vn::VarName) = haskey(vnv.varname_to_index, vn)

# `getindex` & `setindex!`
Base.getindex(vnv::VarNameVector, i::Int) = getindex_raw(vnv, i)
function Base.getindex(vnv::VarNameVector, vn::VarName)
    x = getindex_raw(vnv, vn)
    f = gettransform(vnv, vn)
    return f(x)
end

getindex_raw(vnv::VarNameVector, i::Int) = vnv.vals[i]
function getindex_raw(vnv::VarNameVector, vn::VarName)
    return vnv.vals[getrange(vnv, vn)]
end

# `getindex` for `Colon`
function Base.getindex(vnv::VarNameVector, ::Colon)
    return if has_inactive_ranges(vnv)
        mapreduce(Base.Fix1(getindex, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

function getindex_raw(vnv::VarNameVector, ::Colon)
    return if has_inactive_ranges(vnv)
        mapreduce(Base.Fix1(getindex_raw, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

# HACK: remove this as soon as possible.
Base.getindex(vnv::VarNameVector, spl::AbstractSampler) = vnv[:]

Base.setindex!(vnv::VarNameVector, val, i::Int) = setindex_raw!(vnv, val, i)
function Base.setindex!(vnv::VarNameVector, val, vn::VarName)
    f = inverse(gettransform(vnv, vn))
    return setindex_raw!(vnv, f(val), vn)
end

setindex_raw!(vnv::VarNameVector, val, i::Int) = vnv.vals[i] = val
function setindex_raw!(vnv::VarNameVector, val::AbstractVector, vn::VarName)
    return vnv.vals[getrange(vnv, vn)] = val
end

# `empty!(!)`
function Base.empty!(vnv::VarNameVector)
    # TODO: Or should the semantics be different, e.g. keeping `varnames`?
    empty!(vnv.varname_to_index)
    empty!(vnv.varnames)
    empty!(vnv.ranges)
    empty!(vnv.vals)
    empty!(vnv.transforms)
    return nothing
end
BangBang.empty!!(vnv::VarNameVector) = (empty!(vnv); return vnv)

# `similar`
similar_metadata(::Nothing) = nothing
similar_metadata(x::Union{AbstractArray,AbstractDict}) = similar(x)
function Base.similar(vnv::VarNameVector)
    # NOTE: Whether or not we should empty the underlying containers or note
    # is somewhat ambiguous. For example, `similar(vnv.varname_to_index)` will
    # result in an empty `AbstractDict`, while the vectors, e.g. `vnv.ranges`,
    # will result in non-empty vectors but with entries as `undef`. But it's
    # much easier to write the rest of the code assuming that `undef` is not
    # present, and so for now we empty the underlying containers, thus differing
    # from the behavior of `similar` for `AbstractArray`s.
    return VarNameVector(
        similar(vnv.varname_to_index),
        similar(vnv.varnames, 0),
        similar(vnv.ranges, 0),
        similar(vnv.vals, 0),
        similar(vnv.transforms, 0),
        similar(vnv.inactive_ranges, 0),
        similar_metadata(vnv.metadata),
    )
end

function nextrange(vnv::VarNameVector, x)
    # NOTE: Need to treat `isempty(vnv.ranges)` separately because `maximum`
    # will error if `vnv.ranges` is empty.
    max_active_range = isempty(vnv.ranges) ? 0 : maximum(last, vnv.ranges)
    # Also need to consider inactive ranges, since we can have scenarios such as
    #
    #     vnv = VarNameVector(@varname(x) => 1, @varname(y) => [2, 3])
    #     update!(vnv, @varname(y), [4]) # => `ranges = [1:1, 2:2], inactive_ranges = [3:3]`
    #
    # Here `nextrange(vnv, [5])` should return `4:4`, _not_ `3:3`.
    # NOTE: We could of course attempt to make use of unused space, e.g. if we have an inactive
    # range which can hold `x`, then we could just use that. Buuut the complexity of this is
    # probably not worth it (at least at the moment).
    max_inactive_range =
        isempty(vnv.inactive_ranges) ? 0 : maximum(last, vnv.inactive_ranges)
    offset = max(max_active_range, max_inactive_range)
    return (offset + 1):(offset + length(x))
end

# `push!` and `push!!`: add a variable to the varname vector.
function Base.push!(vnv::VarNameVector, vn::VarName, val, transform=FromVec(val))
    # Error if we already have the variable.
    haskey(vnv, vn) && throw(ArgumentError("variable name $vn already exists"))
    # NOTE: We need to compute the `nextrange` BEFORE we start mutating
    # the underlying; otherwise we might get some strange behaviors.
    val_vec = tovec(val)
    r_new = nextrange(vnv, val_vec)
    vnv.varname_to_index[vn] = length(vnv.varname_to_index) + 1
    push!(vnv.varnames, vn)
    push!(vnv.ranges, r_new)
    append!(vnv.vals, val_vec)
    push!(vnv.transforms, transform)
    return nothing
end

# `update!` and `update!!`: update a variable in the varname vector.
function update!(vnv::VarNameVector, vn::VarName, val, transform=FromVec(val))
    if !haskey(vnv, vn)
        # Here we just add a new entry.
        return push!(vnv, vn, val, transform)
    end

    # Here we update an existing entry.
    val_vec = tovec(val)
    idx = getidx(vnv, vn)
    r_old = getrange(vnv, idx)
    n_old = length(r_old)
    n_new = length(val_vec)
    # Existing keys needs to be handled differently depending on
    # whether the size of the value is increasing or decreasing.
    if n_new > n_old
        # Add the new range.
        r_new = nextrange(vnv, val_vec)
        vnv.ranges[idx] = r_new
        # Grow the underlying vector to accomodate the new value.
        resize!(vnv.vals, r_new[end])
        # Keep track of the deleted ranges.
        push!(vnv.inactive_ranges, r_old)
    else
        # `n_new <= n_old`
        # Just decrease the current range.
        r_new = r_old[1]:(r_old[1] + n_new - 1)
        vnv.ranges[idx] = r_new
        # And mark the rest as inactive if needed.
        if n_new < n_old
            push!(vnv.inactive_ranges, r_old[n_new]:r_old[end])
        end
    end

    # Update the value.
    vnv.vals[r_new] = val_vec
    # Update the transform.
    vnv.transforms[idx] = transform

    # TODO: Should we maybe sweep over inactive ranges and re-contiguify
    # if we the total number of inactive elements is "large" in some sense?

    return nothing
end

function recontiguify_ranges!(ranges::AbstractVector{<:AbstractRange})
    offset = 0
    for i in 1:length(ranges)
        r_old = ranges[i]
        ranges[i] = (offset + 1):(offset + length(r_old))
        offset += length(r_old)
    end

    return ranges
end

function inactive_ranges_sweep!(vnv::VarNameVector)
    # Extract the re-contiguified values.
    # NOTE: We need to do this before we update the ranges.
    vals = vnv[:]
    # And then we re-contiguify the ranges.
    recontiguify_ranges!(vnv.ranges)
    # Clear the inactive ranges.
    empty!(vnv.inactive_ranges)
    # Now we update the values.
    for (i, r) in enumerate(vnv.ranges)
        vnv.vals[r] = vals[r]
    end
    # And (potentially) shrink the underlying vector.
    resize!(vnv.vals, vnv.ranges[end][end])
    # The rest should be left as is.
    return vnv
end

# Typed version.
function group_by_symbol(vnv::VarNameVector)
    # Group varnames in `vnv` by the symbol.
    d = OrderedDict{Symbol,Vector{VarName}}()
    for vn in vnv.varnames
        push!(get!(d, getsym(vn), Vector{VarName}()), vn)
    end

    # Create a `NamedTuple` from the grouped varnames.
    nt_vals = map(values(d)) do varnames
        # TODO: Do we need to specialize the inputs here?
        VarNameVector(
            map(identity, varnames),
            map(Base.Fix1(getindex, vnv), varnames),
            map(Base.Fix1(gettransform, vnv), varnames),
        )
    end

    return NamedTuple{Tuple(keys(d))}(nt_vals)
end

# `iterate`
# TODO: Maybe implement `iterate` as a vector and then instead implement `pairs`.
function Base.iterate(vnv::VarNameVector, state=nothing)
    res = if state === nothing
        iterate(vnv.varnames)
    else
        iterate(vnv.varnames, state)
    end
    res === nothing && return nothing
    vn, state_new = res
    return vn => getindex(vnv, vn), state_new
end