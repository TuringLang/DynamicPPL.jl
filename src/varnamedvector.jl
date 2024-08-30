"""
    VarNamedVector

A container that works like a `Vector` and an `OrderedDict` but is neither.

# Fields
$(FIELDS)
"""
struct VarNamedVector{
    K<:VarName,V,TVN<:AbstractVector{K},TVal<:AbstractVector{V},TTrans<:AbstractVector
}
    """
    mapping from the `VarName` to its integer index in `varnames`, `ranges` and `transforms`
    """
    varname_to_index::OrderedDict{K,Int}

    """
    vector of identifiers for the random variables, where
    `varnames[varname_to_index[vn]] == vn`
    """
    varnames::TVN # AbstractVector{<:VarName}

    """
    vector of index ranges in `vals` corresponding to `varnames`; each `VarName` `vn` has
    a single index or a set of contiguous indices in `vals`
    """
    ranges::Vector{UnitRange{Int}}

    """
    vector of values of all variables; the value(s) of `vn` is/are
    `vals[ranges[varname_to_index[vn]]]`
    """
    vals::TVal # AbstractVector{<:Real}

    "vector of transformations whose inverse takes us back to the original space"
    transforms::TTrans

    # TODO(mhauru) When do we actually need this? Could this be handled by having
    # `identity` in transforms?
    "specifies whether a variable is transformed or not "
    is_transformed::BitVector

    "additional entries which are considered inactive"
    num_inactive::OrderedDict{Int,Int}
end

function ==(vnv_left::VarNamedVector, vnv_right::VarNamedVector)
    return vnv_left.varname_to_index == vnv_right.varname_to_index &&
           vnv_left.varnames == vnv_right.varnames &&
           vnv_left.ranges == vnv_right.ranges &&
           vnv_left.vals == vnv_right.vals &&
           vnv_left.transforms == vnv_right.transforms &&
           vnv_left.is_transformed == vnv_right.is_transformed &&
           vnv_left.num_inactive == vnv_right.num_inactive
end

function VarNamedVector(
    varname_to_index,
    varnames,
    ranges,
    vals,
    transforms,
    is_transformed=fill!(BitVector(undef, length(varnames)), 0),
)
    return VarNamedVector(
        varname_to_index,
        varnames,
        ranges,
        vals,
        transforms,
        is_transformed,
        OrderedDict{Int,Int}(),
    )
end
# TODO: Do we need this?
function VarNamedVector{K,V}() where {K,V}
    return VarNamedVector(OrderedDict{K,Int}(), K[], UnitRange{Int}[], V[], Any[])
end

# The various constructor(orig...) calls are to create concrete element types. E.g. if
# vnv.vals is a Vector{Real} but all elements are Float64s, [vnv.vals...] will be a
# Vector{Float64}
function VarNamedVector(vnv::VarNamedVector)
    return VarNamedVector(
        OrderedDict(vnv.varname_to_index...),
        [vnv.varnames...],
        copy(vnv.ranges),
        [vnv.vals...],
        [vnv.transforms...],
        copy(vnv.is_transformed),
        copy(vnv.num_inactive),
    )
end

istrans(vnv::VarNamedVector, vn::VarName) = vnv.is_transformed[vnv.varname_to_index[vn]]
function settrans!(vnv::VarNamedVector, val::Bool, vn::VarName)
    return vnv.is_transformed[vnv.varname_to_index[vn]] = val
end

# TODO(mhauru) I would like for this to be VarNamedVector(Union{}, Union{}). This would
# allow expanding the VarName and element types only as necessary, which would help keep
# them concrete. However, making that change here opens some other cans of worms that I
# don't want to deal with right now.
VarNamedVector() = VarNamedVector{VarName,Real}()
VarNamedVector(xs::Pair...) = VarNamedVector(OrderedDict(xs...))
VarNamedVector(x::AbstractDict) = VarNamedVector(keys(x), values(x))
function VarNamedVector(varnames, vals)
    return VarNamedVector(collect_maybe(varnames), collect_maybe(vals))
end
function VarNamedVector(
    varnames::AbstractVector, vals::AbstractVector, transforms=map(from_vec_transform, vals)
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
    for (vn, x) in zip(varnames, vals_vecs)
        # Add the varname index.
        push!(varname_to_index, vn => length(varname_to_index) + 1)
        # Add the range.
        r = (offset + 1):(offset + length(x))
        push!(ranges, r)
        # Update the offset.
        offset = r[end]
    end

    return VarNamedVector(
        varname_to_index, varnames, ranges, reduce(vcat, vals_vecs), transforms
    )
end

"""
    replace_values(vnv::VarNamedVector, vals::AbstractVector)

Replace the values in `vnv` with `vals`.

This is useful when we want to update the entire underlying vector of values
in one go or if we want to change the how the values are stored, e.g. alter the `eltype`.

!!! warning
    This replaces the raw underlying values, and so care should be taken when using this
    function. For example, if `vnv` has any inactive entries, then the provided `vals`
    should also contain the inactive entries to avoid unexpected behavior.

# Example

```jldoctest varnamedvector-replace-values
julia> using DynamicPPL: VarNamedVector, replace_values

julia> vnv = VarNamedVector(@varname(x) => [1.0]);

julia> replace_values(vnv, [2.0])[@varname(x)] == [2.0]
true
```

This is also useful when we want to differentiate wrt. the values
using automatic differentiation, e.g. ForwardDiff.jl.

```jldoctest varnamedvector-replace-values
julia> using ForwardDiff: ForwardDiff

julia> f(x) = sum(abs2, replace_values(vnv, x)[@varname(x)])
f (generic function with 1 method)

julia> ForwardDiff.gradient(f, [1.0])
1-element Vector{Float64}:
 2.0
```
"""
replace_values(vnv::VarNamedVector, vals) = Accessors.@set vnv.vals = vals
replace_values(vnv::VarNamedVector, space, vals) = replace_values(vnv, vals)

function unflatten(vnv::VarNamedVector, vals::AbstractVector)
    new_ranges = deepcopy(vnv.ranges)
    recontiguify_ranges!(new_ranges)
    return VarNamedVector(
        vnv.varname_to_index, vnv.varnames, new_ranges, vals, vnv.transforms
    )
end

# Some `VarNamedVector` specific functions.
getidx(vnv::VarNamedVector, vn::VarName) = vnv.varname_to_index[vn]

getrange(vnv::VarNamedVector, idx::Int) = vnv.ranges[idx]
getrange(vnv::VarNamedVector, vn::VarName) = getrange(vnv, getidx(vnv, vn))

gettransform(vnv::VarNamedVector, vn::VarName) = vnv.transforms[getidx(vnv, vn)]

"""
    has_inactive(vnv::VarNamedVector)

Returns `true` if `vnv` has inactive ranges.
"""
has_inactive(vnv::VarNamedVector) = !isempty(vnv.num_inactive)

"""
    num_inactive(vnv::VarNamedVector)

Return the number of inactive entries in `vnv`.
"""
num_inactive(vnv::VarNamedVector) = sum(values(vnv.num_inactive))

"""
    num_inactive(vnv::VarNamedVector, vn::VarName)

Returns the number of inactive entries for `vn` in `vnv`.
"""
num_inactive(vnv::VarNamedVector, vn::VarName) = num_inactive(vnv, getidx(vnv, vn))
num_inactive(vnv::VarNamedVector, idx::Int) = get(vnv.num_inactive, idx, 0)

"""
    num_allocated(vnv::VarNamedVector)

Returns the number of allocated entries in `vnv`.
"""
num_allocated(vnv::VarNamedVector) = length(vnv.vals)

"""
    num_allocated(vnv::VarNamedVector, vn::VarName)

Returns the number of allocated entries for `vn` in `vnv`.
"""
num_allocated(vnv::VarNamedVector, vn::VarName) = num_allocated(vnv, getidx(vnv, vn))
function num_allocated(vnv::VarNamedVector, idx::Int)
    return length(getrange(vnv, idx)) + num_inactive(vnv, idx)
end

# Basic array interface.
Base.eltype(vnv::VarNamedVector) = eltype(vnv.vals)
Base.length(vnv::VarNamedVector) =
    if isempty(vnv.num_inactive)
        length(vnv.vals)
    else
        sum(length, vnv.ranges)
    end
Base.size(vnv::VarNamedVector) = (length(vnv),)
Base.isempty(vnv::VarNamedVector) = isempty(vnv.varnames)

# TODO: We should probably remove this
Base.IndexStyle(::Type{<:VarNamedVector}) = IndexLinear()

# Dictionary interface.
Base.keys(vnv::VarNamedVector) = vnv.varnames
Base.values(vnv::VarNamedVector) = Iterators.map(Base.Fix1(getindex, vnv), vnv.varnames)
Base.pairs(vnv::VarNamedVector) = (vn => vnv[vn] for vn in keys(vnv))

Base.haskey(vnv::VarNamedVector, vn::VarName) = haskey(vnv.varname_to_index, vn)

# `getindex` & `setindex!`
Base.getindex(vnv::VarNamedVector, i::Int) = getindex_raw(vnv, i)
function Base.getindex(vnv::VarNamedVector, vn::VarName)
    x = getindex_raw(vnv, vn)
    f = gettransform(vnv, vn)
    return f(x)
end

Base.get(vnv::VarNamedVector, vn::VarName) = getvalue(vnv, vn)

function find_range_from_sorted(ranges::AbstractVector{<:AbstractRange}, x)
    # TODO: Assume `ranges` to be sorted and contiguous, and use `searchsortedfirst`
    # for a more efficient approach.
    range_idx = findfirst(Base.Fix1(∈, x), ranges)

    # If we're out of bounds, we raise an error.
    if range_idx === nothing
        throw(ArgumentError("Value $x is not in any of the ranges."))
    end

    return range_idx
end

function adjusted_ranges(vnv::VarNamedVector)
    # Every range following inactive entries needs to be shifted.
    offset = 0
    ranges_adj = similar(vnv.ranges)
    for (idx, r) in enumerate(vnv.ranges)
        # Remove the `offset` in `r` due to inactive entries.
        ranges_adj[idx] = r .- offset
        # Update `offset`.
        offset += get(vnv.num_inactive, idx, 0)
    end

    return ranges_adj
end

function index_to_raw_index(vnv::VarNamedVector, i::Int)
    # If we don't have any inactive entries, there's nothing to do.
    has_inactive(vnv) || return i

    # Get the adjusted ranges.
    ranges_adj = adjusted_ranges(vnv)
    # Determine the adjusted range that the index corresponds to.
    r_idx = find_range_from_sorted(ranges_adj, i)
    r = vnv.ranges[r_idx]
    # Determine how much of the index `i` is used to get to this range.
    i_used = r_idx == 1 ? 0 : sum(length, ranges_adj[1:(r_idx - 1)])
    # Use remainder to index into `r`.
    i_remainder = i - i_used
    return r[i_remainder]
end

getindex_raw(vnv::VarNamedVector, i::Int) = vnv.vals[index_to_raw_index(vnv, i)]
getindex_raw(vnv::VarNamedVector, vn::VarName) = vnv.vals[getrange(vnv, vn)]

# `getindex` for `Colon`
function Base.getindex(vnv::VarNamedVector, ::Colon)
    return if has_inactive(vnv)
        mapreduce(Base.Fix1(getindex, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

function getindex_raw(vnv::VarNamedVector, ::Colon)
    return if has_inactive(vnv)
        mapreduce(Base.Fix1(getindex_raw, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

# HACK: remove this as soon as possible.
function Base.getindex(vnv::VarNamedVector, spl::AbstractSampler)
    throw(ErrorException("Cannot index a VarNamedVector with a sampler."))
end

Base.setindex!(vnv::VarNamedVector, val, i::Int) = setindex_raw!(vnv, val, i)
function Base.setindex!(vnv::VarNamedVector, val, vn::VarName)
    f = inverse(gettransform(vnv, vn))
    return setindex_raw!(vnv, f(val), vn)
end

setindex_raw!(vnv::VarNamedVector, val, i::Int) = vnv.vals[index_to_raw_index(vnv, i)] = val
function setindex_raw!(vnv::VarNamedVector, val::AbstractVector, vn::VarName)
    return vnv.vals[getrange(vnv, vn)] = val
end

function setval!(vnv::VarNamedVector, val, vn::VarName)
    return setindex_raw!(vnv, tovec(val), vn)
end

# set!! is the function defined in utils.jl that tries to do fancy stuff with optics when
# setting the value of a generic container using a VarName. We can bypass all that because
# VarNamedVector handles VarNames natively.
set!!(vnv::VarNamedVector, vn::VarName, val) = update!!(vnv, vn, val)

# `empty!(!)`
function Base.empty!(vnv::VarNamedVector)
    # TODO: Or should the semantics be different, e.g. keeping `varnames`?
    empty!(vnv.varname_to_index)
    empty!(vnv.varnames)
    empty!(vnv.ranges)
    empty!(vnv.vals)
    empty!(vnv.transforms)
    empty!(vnv.num_inactive)
    return nothing
end
BangBang.empty!!(vnv::VarNamedVector) = (empty!(vnv); return vnv)

function Base.merge(left_vnv::VarNamedVector, right_vnv::VarNamedVector)
    # Return early if possible.
    isempty(left_vnv) && return deepcopy(right_vnv)
    isempty(right_vnv) && return deepcopy(left_vnv)

    # Determine varnames.
    vns_left = left_vnv.varnames
    vns_right = right_vnv.varnames
    vns_both = union(vns_left, vns_right)

    # Determine `eltype` of `vals`.
    T_left = eltype(left_vnv.vals)
    T_right = eltype(right_vnv.vals)
    T = promote_type(T_left, T_right)
    # TODO: Is this necessary?
    if !(T <: Real)
        T = Real
    end

    # Determine `eltype` of `varnames`.
    V_left = eltype(left_vnv.varnames)
    V_right = eltype(right_vnv.varnames)
    V = promote_type(V_left, V_right)
    if !(V <: VarName)
        V = VarName
    end

    # Determine `eltype` of `transforms`.
    F_left = eltype(left_vnv.transforms)
    F_right = eltype(right_vnv.transforms)
    F = promote_type(F_left, F_right)

    # Allocate.
    varnames_to_index = OrderedDict{V,Int}()
    ranges = UnitRange{Int}[]
    vals = T[]
    transforms = F[]
    is_transformed = BitVector(undef, length(vns_both))

    # Range offset.
    offset = 0

    for (idx, vn) in enumerate(vns_both)
        # Extract the necessary information from `left` or `right`.
        if vn in vns_left && !(vn in vns_right)
            # `vn` is only in `left`.
            varnames_to_index[vn] = idx
            val = getindex_raw(left_vnv, vn)
            n = length(val)
            r = (offset + 1):(offset + n)
            f = gettransform(left_vnv, vn)
            is_transformed[idx] = istrans(left_vnv, vn)
        else
            # `vn` is either in both or just `right`.
            varnames_to_index[vn] = idx
            val = getindex_raw(right_vnv, vn)
            n = length(val)
            r = (offset + 1):(offset + n)
            f = gettransform(right_vnv, vn)
            is_transformed[idx] = istrans(right_vnv, vn)
        end
        # Update.
        append!(vals, val)
        push!(ranges, r)
        push!(transforms, f)
        # Increment `offset`.
        offset += n
    end

    return VarNamedVector(
        varnames_to_index, vns_both, ranges, vals, transforms, is_transformed
    )
end

"""
    subset(vnv::VarNamedVector, vns::AbstractVector{<:VarName})

Return a new `VarNamedVector` containing the values from `vnv` for variables in `vns`.
"""
function subset(vnv::VarNamedVector, vns_given::AbstractVector{VN}) where {VN<:VarName}
    # NOTE: This does not specialize types when possible.
    vns = mapreduce(vcat, vns_given; init=VN[]) do vn
        filter(Base.Fix1(subsumes, vn), vnv.varnames)
    end
    vnv_new = similar(vnv)
    # Return early if possible.
    isempty(vnv) && return vnv_new

    for vn in vns
        push!(vnv_new, vn, getindex_internal(vnv, vn), gettransform(vnv, vn))
    end

    return vnv_new
end

function Base.similar(vnv::VarNamedVector)
    # NOTE: Whether or not we should empty the underlying containers or not
    # is somewhat ambiguous. For example, `similar(vnv.varname_to_index)` will
    # result in an empty `AbstractDict`, while the vectors, e.g. `vnv.ranges`,
    # will result in non-empty vectors but with entries as `undef`. But it's
    # much easier to write the rest of the code assuming that `undef` is not
    # present, and so for now we empty the underlying containers, thus differing
    # from the behavior of `similar` for `AbstractArray`s.
    return VarNamedVector(
        similar(vnv.varname_to_index),
        similar(vnv.varnames, 0),
        similar(vnv.ranges, 0),
        similar(vnv.vals, 0),
        similar(vnv.transforms, 0),
        BitVector(),
        similar(vnv.num_inactive),
    )
end

"""
    is_contiguous(vnv::VarNamedVector)

Returns `true` if the underlying data of `vnv` is stored in a contiguous array.

This is equivalent to negating [`has_inactive(vnv)`](@ref).
"""
is_contiguous(vnv::VarNamedVector) = !has_inactive(vnv)

"""
    nextrange(vnv::VarNamedVector, x)

Return the range of `length(x)` from the end of current data in `vnv`.
"""
function nextrange(vnv::VarNamedVector, x)
    offset = length(vnv.vals)
    return (offset + 1):(offset + length(x))
end

"""
    push!(vnv::VarNamedVector, vn::VarName, val[, transform])

Add a variable with given value to `vnv`.

By default `transform` is the one that converts the value to a vector, which is how it is
stored in `vnv`.
"""
function Base.push!(
    vnv::VarNamedVector, vn::VarName, val, transform=from_vec_transform(val)
)
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
    push!(vnv.is_transformed, false)
    return nothing
end

function Base.push!(vnv::VarNamedVector, vn, val, dist, gidset, num_produce)
    f = from_vec_transform(dist)
    return push!(vnv, vn, val, f)
end

"""
    loosen_types(vnv::VarNamedVector{K,V,TVN,TVal,TTrans}, ::Type{KNew}, ::Type{TransNew})

Loosen the types of `vnv` to allow varname type `KNew` and transformation type `TransNew`.

If `KNew` is a subtype of `K` and `TransNew` is a subtype of the element type of the `TTrans`
this is a no-op and `vnv` is returned as is. Otherwise a copy of `vnv` is created with the
same data but more abstract types, so that variables of type `KNew` and transformations of
type `TransNew` can be pushed to it.
"""
function loosen_types(
    vnv::VarNamedVector, ::Type{KNew}, ::Type{TransNew}
) where {KNew,TransNew}
    K = eltype(vnv.varnames)
    Trans = eltype(vnv.transforms)
    if KNew <: K && TransNew <: Trans
        return vnv
    else
        vn_type = promote_type(K, KNew)
        transform_type = promote_type(Trans, TransNew)
        return VarNamedVector(
            OrderedDict{vn_type,Int}(vnv.varname_to_index),
            Vector{vn_type}(vnv.varnames),
            vnv.ranges,
            vnv.vals,
            Vector{transform_type}(vnv.transforms),
            vnv.is_transformed,
            vnv.num_inactive,
        )
    end
end

function BangBang.push!!(
    vnv::VarNamedVector, vn::VarName, val, transform=from_vec_transform(val)
)
    vnv = loosen_types(vnv, typeof(vn), typeof(transform))
    push!(vnv, vn, val, transform)
    return vnv
end

function BangBang.push!!(vnv::VarNamedVector, vn, val, dist, gidset, num_produce)
    f = from_vec_transform(dist)
    return push!!(vnv, vn, val, f)
end

"""
    shift_right!(x::AbstractVector{<:Real}, start::Int, n::Int)

Shifts the elements of `x` starting from index `start` by `n` to the right.
"""
function shift_right!(x::AbstractVector{<:Real}, start::Int, n::Int)
    x[(start + n):end] = x[start:(end - n)]
    return x
end

"""
    shift_subsequent_ranges_by!(vnv::VarNamedVector, idx::Int, n)

Shifts the ranges of variables in `vnv` starting from index `idx` by `n`.
"""
function shift_subsequent_ranges_by!(vnv::VarNamedVector, idx::Int, n)
    for i in (idx + 1):length(vnv.ranges)
        vnv.ranges[i] = vnv.ranges[i] .+ n
    end
    return nothing
end

# `update!` and `update!!`: update a variable in the varname vector.
"""
    update!(vnv::VarNamedVector, vn::VarName, val[, transform])

Either add a new entry or update existing entry for `vn` in `vnv` with the value `val`.

If `vn` does not exist in `vnv`, this is equivalent to [`push!`](@ref).

By default `transform` is the one that converts the value to a vector, which is how it is
stored in `vnv`.
"""
function update!(vnv::VarNamedVector, vn::VarName, val, transform=from_vec_transform(val))
    if !haskey(vnv, vn)
        # Here we just add a new entry.
        return push!(vnv, vn, val, transform)
    end

    # Here we update an existing entry.
    val_vec = tovec(val)
    idx = getidx(vnv, vn)
    # Extract the old range.
    r_old = getrange(vnv, idx)
    start_old, end_old = first(r_old), last(r_old)
    n_old = length(r_old)
    # Compute the new range.
    n_new = length(val_vec)
    start_new = start_old
    end_new = start_old + n_new - 1
    r_new = start_new:end_new

    #=
    Suppose we currently have the following:

      | x | x | o | o | o | y | y | y |    <- Current entries

    where 'O' denotes an inactive entry, and we're going to
    update the variable `x` to be of size `k` instead of 2.

    We then have a few different scenarios:
    1. `k > 5`: All inactive entries become active + need to shift `y` to the right.
        E.g. if `k = 7`, then

          | x | x | o | o | o | y | y | y |            <- Current entries
          | x | x | x | x | x | x | x | y | y | y |    <- New entries

    2. `k = 5`: All inactive entries become active.
        Then

          | x | x | o | o | o | y | y | y |            <- Current entries
          | x | x | x | x | x | y | y | y |            <- New entries

    3. `k < 5`: Some inactive entries become active, some remain inactive.
        E.g. if `k = 3`, then

          | x | x | o | o | o | y | y | y |            <- Current entries
          | x | x | x | o | o | y | y | y |            <- New entries

    4. `k = 2`: No inactive entries become active.
        Then

          | x | x | o | o | o | y | y | y |            <- Current entries
          | x | x | o | o | o | y | y | y |            <- New entries

    5. `k < 2`: More entries become inactive.
        E.g. if `k = 1`, then

          | x | x | o | o | o | y | y | y |            <- Current entries
          | x | o | o | o | o | y | y | y |            <- New entries
    =#

    # Compute the allocated space for `vn`.
    had_inactive = haskey(vnv.num_inactive, idx)
    n_allocated = had_inactive ? n_old + vnv.num_inactive[idx] : n_old

    if n_new > n_allocated
        # Then we need to grow the underlying vector.
        n_extra = n_new - n_allocated
        # Allocate.
        resize!(vnv.vals, length(vnv.vals) + n_extra)
        # Shift current values.
        shift_right!(vnv.vals, end_old + 1, n_extra)
        # No more inactive entries.
        had_inactive && delete!(vnv.num_inactive, idx)
        # Update the ranges for all variables after this one.
        shift_subsequent_ranges_by!(vnv, idx, n_extra)
    elseif n_new == n_allocated
        # => No more inactive entries.
        had_inactive && delete!(vnv.num_inactive, idx)
    else
        # `n_new < n_allocated`
        # => Need to update the number of inactive entries.
        vnv.num_inactive[idx] = n_allocated - n_new
    end

    # Update the range for this variable.
    vnv.ranges[idx] = r_new
    # Update the value.
    vnv.vals[r_new] = val_vec
    # Update the transform.
    vnv.transforms[idx] = transform

    # TODO: Should we maybe sweep over inactive ranges and re-contiguify
    # if the total number of inactive elements is "large" in some sense?

    return nothing
end

function update!!(vnv::VarNamedVector, vn::VarName, val, transform=from_vec_transform(val))
    vnv = loosen_types(vnv, typeof(vn), typeof(transform))
    update!(vnv, vn, val, transform)
    return vnv
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

"""
    contiguify!(vnv::VarNamedVector)

Re-contiguify the underlying vector and shrink if possible.
"""
function contiguify!(vnv::VarNamedVector)
    # Extract the re-contiguified values.
    # NOTE: We need to do this before we update the ranges.
    old_vals = copy(vnv.vals)
    old_ranges = copy(vnv.ranges)
    # And then we re-contiguify the ranges.
    recontiguify_ranges!(vnv.ranges)
    # Clear the inactive ranges.
    empty!(vnv.num_inactive)
    # Now we update the values.
    for (old_range, new_range) in zip(old_ranges, vnv.ranges)
        vnv.vals[new_range] = old_vals[old_range]
    end
    # And (potentially) shrink the underlying vector.
    resize!(vnv.vals, vnv.ranges[end][end])
    # The rest should be left as is.
    return vnv
end

"""
    group_by_symbol(vnv::VarNamedVector)

Return a dictionary mapping symbols to `VarNamedVector`s with varnames containing that
symbol.
"""
function group_by_symbol(vnv::VarNamedVector)
    symbols = unique(map(getsym, vnv.varnames))
    nt_vals = map(s -> VarNamedVector(subset(vnv, [VarName(s)])), symbols)
    return OrderedDict(zip(symbols, nt_vals))
end

"""
    shift_index_left!(vnv::VarNamedVector, idx::Int)

Shift the index `idx` to the left by one and update the relevant fields.

!!! warning
    This does not check if index we're shifting to is already occupied.
"""
function shift_index_left!(vnv::VarNamedVector, idx::Int)
    # Shift the index in the lookup table.
    vn = vnv.varnames[idx]
    vnv.varname_to_index[vn] = idx - 1
    # Shift the index in the inactive ranges.
    if haskey(vnv.num_inactive, idx)
        # Done in increasing order =>  don't need to worry about
        # potentially shifting the same index twice.
        vnv.num_inactive[idx - 1] = pop!(vnv.num_inactive, idx)
    end
end

"""
    shift_subsequent_indices_left!(vnv::VarNamedVector, idx::Int)

Shift the indices for all variables after `idx` to the left by one and update
the relevant fields.

This just
"""
function shift_subsequent_indices_left!(vnv::VarNamedVector, idx::Int)
    # Shift the indices for all variables after `idx`.
    for idx_to_shift in (idx + 1):length(vnv.varnames)
        shift_index_left!(vnv, idx_to_shift)
    end
end

function Base.delete!(vnv::VarNamedVector, vn::VarName)
    # Error if we don't have the variable.
    !haskey(vnv, vn) && throw(ArgumentError("variable name $vn does not exist"))

    # Get the index of the variable.
    idx = getidx(vnv, vn)

    # Delete the values.
    r_start = first(getrange(vnv, idx))
    n_allocated = num_allocated(vnv, idx)
    # NOTE: `deleteat!` also results in a `resize!` so we don't need to do that.
    deleteat!(vnv.vals, r_start:(r_start + n_allocated - 1))

    # Delete `vn` from the lookup table.
    delete!(vnv.varname_to_index, vn)

    # Delete any inactive ranges corresponding to `vn`.
    haskey(vnv.num_inactive, idx) && delete!(vnv.num_inactive, idx)

    # Re-adjust the indices for varnames occuring after `vn` so
    # that they point to the correct indices after the deletions below.
    shift_subsequent_indices_left!(vnv, idx)

    # Re-adjust the ranges for varnames occuring after `vn`.
    shift_subsequent_ranges_by!(vnv, idx, -n_allocated)

    # Delete references from vector fields, thus shifting the indices of
    # varnames occuring after `vn` by one to the left, as we adjusted for above.
    deleteat!(vnv.varnames, idx)
    deleteat!(vnv.ranges, idx)
    deleteat!(vnv.transforms, idx)

    return vnv
end

"""
    values_as(vnv::VarNamedVector[, T])

Return the values/realizations in `vnv` as type `T`, if implemented.

If no type `T` is provided, return values as stored in `vnv`.

# Examples

```jldoctest
julia> using DynamicPPL: VarNamedVector

julia> vnv = VarNamedVector(@varname(x) => 1, @varname(y) => [2.0]);

julia> values_as(vnv) == [1.0, 2.0]
true

julia> values_as(vnv, Vector{Float32}) == Vector{Float32}([1.0, 2.0])
true

julia> values_as(vnv, OrderedDict) == OrderedDict(@varname(x) => 1.0, @varname(y) => [2.0])
true

julia> values_as(vnv, NamedTuple) == (x = 1.0, y = [2.0])
true
```
"""
values_as(vnv::VarNamedVector) = values_as(vnv, Vector)
values_as(vnv::VarNamedVector, ::Type{Vector}) = vnv[:]
function values_as(vnv::VarNamedVector, ::Type{Vector{T}}) where {T}
    return convert(Vector{T}, values_as(vnv, Vector))
end
function values_as(vnv::VarNamedVector, ::Type{NamedTuple})
    return NamedTuple(zip(map(Symbol, keys(vnv)), values(vnv)))
end
function values_as(vnv::VarNamedVector, ::Type{D}) where {D<:AbstractDict}
    return ConstructionBase.constructorof(D)(pairs(vnv))
end

# TODO(mhauru) This is tricky to implement in the general case, and the below implementation
# only covers some simple cases. It's probably sufficient in most situations though.
function hasvalue(vnv::VarNamedVector, vn::VarName)
    haskey(vnv, vn) && return true
    any(subsumes(vn, k) for k in keys(vnv)) && return true
    # Handle the easy case where the right symbol isn't even present.
    !any(k -> getsym(k) == getsym(vn), keys(vnv)) && return false

    optic = getoptic(vn)
    if optic isa Accessors.IndexLens || optic isa Accessors.ComposedOptic
        # If vn is of the form @varname(somesymbol[someindex]), we check whether we store
        # @varname(somesymbol) and can index into it with someindex. If we rather have a
        # composed optic with the last part being an index lens, we do a similar check but
        # stripping out the last index lens part. If these pass, the answer is definitely
        # "yes". If not, we still don't know for sure.
        # TODO(mhauru) What about casese where vnv stores both @varname(x) and
        # @varname(x[1]) or @varname(x.a)? Those should probably be banned, but currently
        # aren't.
        head, tail = if optic isa Accessors.ComposedOptic
            decomp_optic = Accessors.decompose(optic)
            first(decomp_optic), Accessors.compose(decomp_optic[2:end]...)
        else
            optic, identity
        end
        parent_varname = VarName{getsym(vn)}(tail)
        if haskey(vnv, parent_varname)
            valvec = getindex(vnv, parent_varname)
            return canview(head, valvec)
        end
    end
    throw(ErrorException("hasvalue has not been fully implemented for this VarName: $(vn)"))
end

# TODO(mhauru) Like hasvalue, this is only partially implemented.
function getvalue(vnv::VarNamedVector, vn::VarName)
    !hasvalue(vnv, vn) && throw(KeyError(vn))
    haskey(vnv, vn) && getindex(vnv, vn)

    subsumed_keys = filter(k -> subsumes(vn, k), keys(vnv))
    if length(subsumed_keys) > 0
        # TODO(mhauru) What happens if getindex returns e.g. matrices, and we vcat them?
        return mapreduce(k -> getindex(vnv, k), vcat, subsumed_keys)
    end

    optic = getoptic(vn)
    # See hasvalue for some comments on the logic of this if block.
    if optic isa Accessors.IndexLens || optic isa Accessors.ComposedOptic
        head, tail = if optic isa Accessors.ComposedOptic
            decomp_optic = Accessors.decompose(optic)
            first(decomp_optic), Accessors.compose(decomp_optic[2:end]...)
        else
            optic, identity
        end
        parent_varname = VarName{getsym(vn)}(tail)
        valvec = getindex(vnv, parent_varname)
        return head(valvec)
    end
    throw(ErrorException("getvalue has not been fully implemented for this VarName: $(vn)"))
end
