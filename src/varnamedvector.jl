"""
    VarNamedVector

A container that stores values in a vectorised form, but indexable by variable names.

A `VarNamedVector` can be thought of as an ordered mapping from `VarName`s to pairs of
`(internal_value, transform)`. Here `internal_value` is a vectorised value for the variable
and `transform` is a function such that `transform(internal_value)` is the "original" value
of the variable, the one that the user sees. For instance, if the variable has a matrix
value, `internal_value` could bea flattened `Vector` of its elements, and `transform` would
be a `reshape` call.

`transform` may implement simply vectorisation, but it may do more. Most importantly, it may
implement linking, where the internal storage of a random variable is in a form where all
values in Euclidean space are valid. This is useful for sampling, because the sampler can
make changes to `internal_value` without worrying about constraints on the space of
the random variable.

The way to access this storage format directly is through the functions `getindex_internal`
and `setindex_internal`. The `transform` argument for `setindex_internal` is optional, by
default it is either the identity, or the existing transform if a value already exists for
this `VarName`.

`VarNamedVector` also provides a `Dict`-like interface that hides away the internal
vectorisation. This can be accessed with `getindex` and `setindex!`. `setindex!` only takes
the value, the transform is automatically set to be a simple vectorisation. The only notable
deviation from the behavior of a `Dict` is that `setindex!` will throw an error if one tries
to set a new value for a variable that lives in a different "space" than the old one (e.g.
is of a different type or size). This is because `setindex!` does not change the transform
of a variable, e.g. preserve linking, and thus the new value must be compatible with the old
transform.

For now, a third value is in fact stored for each `VarName`: a boolean indicating whether
the variable has been transformed to unconstrained Euclidean space or not. This is only in
place temporarily due to the needs of our old Gibbs sampler.

Internally, `VarNamedVector` stores the values of all variables in a single contiguous
vector. This makes some operations more efficient, and means that one can access the entire
contents of the internal storage quickly with `getindex_internal(vnv, :)`. The other fields
of `VarNamedVector` are mostly used to keep track of which part of the internal storage
belongs to which `VarName`.

# Fields

$(FIELDS)

# Extended help

The values for different variables are internally all stored in a single vector. For
instance,
```jldoctest varnamedvector-struct
julia> using DynamicPPL: ReshapeTransform, VarNamedVector, @varname, setindex!, update!, getindex_internal

julia> vnv = VarNamedVector();

julia> setindex!(vnv, [0.0, 0.0, 0.0, 0.0], @varname(x));

julia> setindex!(vnv, reshape(1:6, (2,3)), @varname(y));

julia> vnv.vals
10-element Vector{Real}:
 0.0
 0.0
 0.0
 0.0
 1
 2
 3
 4
 5
 6
```

The `varnames`, `ranges`, and `varname_to_index` fields keep track of which value belongs to
which variable. The `transforms` field stores the transformations that needed to transform
the vectorised internal storage back to its original form:

```jldoctest varnamedvector-struct
julia> vnv.transforms[vnv.varname_to_index[@varname(y)]] == DynamicPPL.ReshapeTransform((6,), (2,3))
true
```

If a variable is updated with a new value that is of a smaller dimension than the old
value, rather than resizing `vnv.vals`, some elements in `vnv.vals` are marked as inactive.

```jldoctest varnamedvector-struct
julia> update!(vnv, [46.0, 48.0], @varname(x))

julia> vnv.vals
10-element Vector{Real}:
 46.0
 48.0
  0.0
  0.0
  1
  2
  3
  4
  5
  6

julia> println(vnv.num_inactive);
OrderedDict(1 => 2)
```

This helps avoid unnecessary memory allocations for values that repeatedly change dimension.
The user does not have to worry about the inactive entries as long as they use functions
like `setindex!` and `getindex!` rather than directly accessing `vnv.vals`.

```jldoctest varnamedvector-struct
julia> vnv[@varname(x)]
2-element Vector{Real}:
 46.0
 48.0

julia> getindex_internal(vnv, :)
8-element Vector{Real}:
 46.0
 48.0
  1
  2
  3
  4
  5
  6
```
"""
struct VarNamedVector{
    K<:VarName,V,TVN<:AbstractVector{K},TVal<:AbstractVector{V},TTrans<:AbstractVector
}
    """
    mapping from a `VarName` to its integer index in `varnames`, `ranges` and `transforms`
    """
    varname_to_index::OrderedDict{K,Int}

    """
    vector of `VarNames` for the variables, where `varnames[varname_to_index[vn]] == vn`
    """
    varnames::TVN # AbstractVector{<:VarName}

    """
    vector of index ranges in `vals` corresponding to `varnames`; each `VarName` `vn` has
    a single index or a set of contiguous indices, such that the values of `vn` can be found
    at `vals[ranges[varname_to_index[vn]]]`
    """
    ranges::Vector{UnitRange{Int}}

    """
    vector of values of all variables; the value(s) of `vn` is/are
    `vals[ranges[varname_to_index[vn]]]`
    """
    vals::TVal # AbstractVector{<:Real}

    """
    vector of transformations, so that `transforms[varname_to_index[vn]]` is a callable
    that transforms the value of `vn` back to its original space, undoing any linking and
    vectorisation
    """
    transforms::TTrans

    """
    vector of booleans indicating whether a variable has been explicitly transformed to
    unconstrained Euclidean space, i.e. whether its domain is all of `ℝ^ⁿ`. If
    `is_unconstrained[varname_to_index[vn]]` is true, it guarantees that the variable
    `vn` is not constrained. However, the converse does not hold: if `is_unconstrained`
    is false, the variable `vn` may still happen to be unconstrained, e.g. if its
    original distribution is itself unconstrained (like a normal distribution).
    """
    is_unconstrained::BitVector

    """
    mapping from a variable index to the number of inactive entries for that variable.
    Inactive entries are elements in `vals` that are not part of the value of any variable.
    They arise when a variable is set to a new value with a different dimension, in-place.
    Inactive entries always come after the last active entry for the given variable.
    See the extended help with `??VarNamedVector` for more details.
    """
    num_inactive::OrderedDict{Int,Int}

    function VarNamedVector(
        varname_to_index,
        varnames::TVN,
        ranges,
        vals::TVal,
        transforms::TTrans,
        is_unconstrained=fill!(BitVector(undef, length(varnames)), 0),
        num_inactive=OrderedDict{Int,Int}(),
    ) where {K,V,TVN<:AbstractVector{K},TVal<:AbstractVector{V},TTrans<:AbstractVector}
        if length(varnames) != length(ranges) ||
            length(varnames) != length(transforms) ||
            length(varnames) != length(is_unconstrained) ||
            length(varnames) != length(varname_to_index)
            msg = (
                "Inputs to VarNamedVector have inconsistent lengths. Got lengths varnames: " *
                "$(length(varnames)), ranges: " *
                "$(length(ranges)), " *
                "transforms: $(length(transforms)), " *
                "is_unconstrained: $(length(is_unconstrained)), " *
                "varname_to_index: $(length(varname_to_index))."
            )
            throw(ArgumentError(msg))
        end

        num_vals = mapreduce(length, (+), ranges; init=0) + sum(values(num_inactive))
        if num_vals != length(vals)
            msg = (
                "The total number of elements in `vals` ($(length(vals))) does not match " *
                "the sum of the lengths of the ranges and the number of inactive entries " *
                "($(num_vals))."
            )
            throw(ArgumentError(msg))
        end

        if Set(values(varname_to_index)) != Set(axes(varnames, 1))
            msg = (
                "The set of values of `varname_to_index` is not the set of valid indices " *
                "for `varnames`."
            )
            throw(ArgumentError(msg))
        end

        if !issubset(Set(keys(num_inactive)), Set(values(varname_to_index)))
            msg = (
                "The keys of `num_inactive` are not a subset of the values of " *
                "`varname_to_index`."
            )
            throw(ArgumentError(msg))
        end

        # Check that the varnames don't overlap. The time cost is quadratic in number of
        # variables. If this ever becomes an issue, we should be able to go down to at least
        # N log N by sorting based on subsumes-order.
        for vn1 in keys(varname_to_index)
            for vn2 in keys(varname_to_index)
                vn1 === vn2 && continue
                if subsumes(vn1, vn2)
                    msg = (
                        "Variables in a VarNamedVector should not subsume each other, " *
                        "but $vn1 subsumes $vn2, i.e. $vn2 describes a subrange of $vn1."
                    )
                    throw(ArgumentError(msg))
                end
            end
        end

        # We could also have a test to check that the ranges don't overlap, but that sounds
        # unlikely to occur, and implementing it in linear time would require a tiny bit of
        # thought.

        return new{K,V,TVN,TVal,TTrans}(
            varname_to_index,
            varnames,
            ranges,
            vals,
            transforms,
            is_unconstrained,
            num_inactive,
        )
    end
end

function VarNamedVector{K,V}() where {K,V}
    return VarNamedVector(OrderedDict{K,Int}(), K[], UnitRange{Int}[], V[], Any[])
end

# TODO(mhauru) I would like for this to be VarNamedVector(Union{}, Union{}). Simlarly the
# transform vector type above could then be Union{}[]. This would allow expanding the
# VarName and element types only as necessary, which would help keep them concrete. However,
# making that change here opens some other cans of worms related to how VarInfo uses
# BangBang, that I don't want to deal with right now.
VarNamedVector() = VarNamedVector{VarName,Real}()
VarNamedVector(xs::Pair...) = VarNamedVector(OrderedDict(xs...))
VarNamedVector(x::AbstractDict) = VarNamedVector(keys(x), values(x))
function VarNamedVector(varnames, vals)
    return VarNamedVector(collect_maybe(varnames), collect_maybe(vals))
end
function VarNamedVector(
    varnames::AbstractVector,
    orig_vals::AbstractVector,
    transforms=fill(identity, length(varnames)),
)
    # Convert `vals` into a vector of vectors.
    vals_vecs = map(tovec, orig_vals)
    transforms = map(
        (t, val) -> _compose_no_identity(t, from_vec_transform(val)), transforms, orig_vals
    )
    # TODO: Is this really the way to do this?
    if !(eltype(varnames) <: VarName)
        varnames = convert(Vector{VarName}, varnames)
    end
    varname_to_index = OrderedDict{eltype(varnames),Int}(
        vn => i for (i, vn) in enumerate(varnames)
    )
    vals = reduce(vcat, vals_vecs)
    # Make the ranges.
    ranges = Vector{UnitRange{Int}}()
    offset = 0
    for x in vals_vecs
        r = (offset + 1):(offset + length(x))
        push!(ranges, r)
        offset = r[end]
    end

    return VarNamedVector(varname_to_index, varnames, ranges, vals, transforms)
end

function ==(vnv_left::VarNamedVector, vnv_right::VarNamedVector)
    return vnv_left.varname_to_index == vnv_right.varname_to_index &&
           vnv_left.varnames == vnv_right.varnames &&
           vnv_left.ranges == vnv_right.ranges &&
           vnv_left.vals == vnv_right.vals &&
           vnv_left.transforms == vnv_right.transforms &&
           vnv_left.is_unconstrained == vnv_right.is_unconstrained &&
           vnv_left.num_inactive == vnv_right.num_inactive
end

getidx(vnv::VarNamedVector, vn::VarName) = vnv.varname_to_index[vn]

getrange(vnv::VarNamedVector, idx::Int) = vnv.ranges[idx]
getrange(vnv::VarNamedVector, vn::VarName) = getrange(vnv, getidx(vnv, vn))

gettransform(vnv::VarNamedVector, idx::Int) = vnv.transforms[idx]
gettransform(vnv::VarNamedVector, vn::VarName) = gettransform(vnv, getidx(vnv, vn))

# TODO(mhauru) Eventually I would like to rename the istrans function to is_unconstrained,
# but that's significantly breaking.
"""
    istrans(vnv::VarNamedVector, vn::VarName)

Return a boolean for whether `vn` is guaranteed to have been transformed so that its domain
is all of Euclidean space.
"""
istrans(vnv::VarNamedVector, vn::VarName) = vnv.is_unconstrained[getidx(vnv, vn)]

"""
    settrans!(vnv::VarNamedVector, val::Bool, vn::VarName)

Set the value for whether `vn` is guaranteed to have been transformed so that all of
Euclidean space is its domain.
"""
function settrans!(vnv::VarNamedVector, val::Bool, vn::VarName)
    return vnv.is_unconstrained[vnv.varname_to_index[vn]] = val
end

function settrans!!(vnv::VarNamedVector, val::Bool, vn::VarName)
    settrans!(vnv, val, vn)
    return vnv
end

"""
    has_inactive(vnv::VarNamedVector)

Returns `true` if `vnv` has inactive entries.

See also: [`num_inactive`](@ref)
"""
has_inactive(vnv::VarNamedVector) = !isempty(vnv.num_inactive)

"""
    num_inactive(vnv::VarNamedVector)

Return the number of inactive entries in `vnv`.

See also: [`has_inactive`](@ref), [`num_allocated`](@ref)
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
    num_allocated(vnv::VarNamedVector[, vn::VarName])
    num_allocated(vnv::VarNamedVector[, idx::Int])

Return the number of allocated entries in `vnv`, both active and inactive.

If either a `VarName` or an `Int` index is specified, only count entries allocated for that
variable.

Allocated entries take up memory in `vnv.vals`, but, if inactive, may not currently hold any
meaningful data. One can remove them with [`contiguify!`](@ref), but doing so may cause more
memory allocations in the future if variables change dimension.
"""
num_allocated(vnv::VarNamedVector) = length(vnv.vals)
num_allocated(vnv::VarNamedVector, vn::VarName) = num_allocated(vnv, getidx(vnv, vn))
function num_allocated(vnv::VarNamedVector, idx::Int)
    return length(getrange(vnv, idx)) + num_inactive(vnv, idx)
end

# Dictionary interface.
Base.isempty(vnv::VarNamedVector) = isempty(vnv.varnames)
Base.length(vnv::VarNamedVector) = length(vnv.varnames)
Base.keys(vnv::VarNamedVector) = vnv.varnames
Base.values(vnv::VarNamedVector) = Iterators.map(Base.Fix1(getindex, vnv), vnv.varnames)
Base.pairs(vnv::VarNamedVector) = (vn => vnv[vn] for vn in keys(vnv))
Base.haskey(vnv::VarNamedVector, vn::VarName) = haskey(vnv.varname_to_index, vn)

# Vector-like interface.
Base.eltype(vnv::VarNamedVector) = eltype(vnv.vals)

"""
    length_internal(vnv::VarNamedVector)

Return the length of the internal storage vector of `vnv`, ignoring inactive entries.
"""
function length_internal(vnv::VarNamedVector)
    if !has_inactive(vnv)
        return length(vnv.vals)
    else
        return sum(length, vnv.ranges)
    end
end

# Getting and setting values

function Base.getindex(vnv::VarNamedVector, vn::VarName)
    x = getindex_internal(vnv, vn)
    f = gettransform(vnv, vn)
    return f(x)
end

"""
    find_containing_range(ranges::AbstractVector{<:AbstractRange}, x)

Find the first range in `ranges` that contains `x`.

Throw an `ArgumentError` if `x` is not in any of the ranges.
"""
function find_containing_range(ranges::AbstractVector{<:AbstractRange}, x)
    # TODO: Assume `ranges` to be sorted and contiguous, and use `searchsortedfirst`
    # for a more efficient approach.
    range_idx = findfirst(Base.Fix1(∈, x), ranges)

    # If we're out of bounds, we raise an error.
    if range_idx === nothing
        throw(ArgumentError("Value $x is not in any of the ranges."))
    end

    return range_idx
end

"""
    adjusted_ranges(vnv::VarNamedVector)

Return what `vnv.ranges` would be if there were no inactive entries.
"""
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

"""
    index_to_vals_index(vnv::VarNamedVector, i::Int)

Convert an integer index that ignores inactive entries to an index that accounts for them.

This is needed when the user wants to index `vnv` like a vector, but shouldn't have to care
about inactive entries in `vnv.vals`.
"""
function index_to_vals_index(vnv::VarNamedVector, i::Int)
    # If we don't have any inactive entries, there's nothing to do.
    has_inactive(vnv) || return i

    # Get the adjusted ranges.
    ranges_adj = adjusted_ranges(vnv)
    # Determine the adjusted range that the index corresponds to.
    r_idx = find_containing_range(ranges_adj, i)
    r = vnv.ranges[r_idx]
    # Determine how much of the index `i` is used to get to this range.
    i_used = r_idx == 1 ? 0 : sum(length, ranges_adj[1:(r_idx - 1)])
    # Use remainder to index into `r`.
    i_remainder = i - i_used
    return r[i_remainder]
end

"""
    getindex_internal(vnv::VarNamedVector, vn::VarName)

Like `getindex`, but returns the values as they are stored in `vnv`, without transforming.
"""
getindex_internal(vnv::VarNamedVector, vn::VarName) = vnv.vals[getrange(vnv, vn)]

"""
    getindex_internal(vnv::VarNamedVector, i::Int)

Gets the `i`th element of the internal storage vector, ignoring inactive entries.
"""
getindex_internal(vnv::VarNamedVector, i::Int) = vnv.vals[index_to_vals_index(vnv, i)]

function getindex_internal(vnv::VarNamedVector, ::Colon)
    return if has_inactive(vnv)
        mapreduce(Base.Fix1(getindex, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

function Base.setindex!(vnv::VarNamedVector, val, vn::VarName)
    if haskey(vnv, vn)
        return update!(vnv, val, vn)
    else
        return insert!(vnv, val, vn)
    end
end

"""
    reset!(vnv::VarNamedVector, val, vn::VarName)

Reset the value of `vn` in `vnv` to `val`.

This differs from `setindex!` in that it will always change the transform of the variable
to be the default vectorisation transform. This undoes any possible linking.

# Examples

```jldoctest varnamedvector-reset
julia> using DynamicPPL: VarNamedVector, @varname, reset!

julia> vnv = VarNamedVector();

julia> vnv[@varname(x)] = reshape(1:9, (3, 3));

julia> setindex!(vnv, 2.0, @varname(x))
ERROR: An error occurred while assigning the value 2.0 to variable x. If you are changing the type or size of a variable you'll need to call reset!
[...]

julia> reset!(vnv, 2.0, @varname(x));

julia> vnv[@varname(x)]
2.0
```
"""
function reset!(vnv::VarNamedVector, val, vn::VarName)
    f = from_vec_transform(val)
    retval = setindex_internal!(vnv, tovec(val), vn, f)
    settrans!(vnv, false, vn)
    return retval
end

"""
    update!(vnv::VarNamedVector, val, vn::VarName)

Update the value of `vn` in `vnv` to `val`.

Like `setindex!`, but errors if the key `vn` doesn't exist.
"""
function update!(vnv::VarNamedVector, val, vn::VarName)
    if !haskey(vnv, vn)
        throw(KeyError(vn))
    end
    f = inverse(gettransform(vnv, vn))
    internal_val = try
        f(val)
    catch
        error(
            "An error occurred while assigning the value $val to variable $vn. " *
            "If you are changing the type or size of a variable you'll need to call " *
            "reset!",
        )
    end
    return setindex_internal!(vnv, internal_val, vn)
end

"""
    insert!(vnv::VarNamedVector, val, vn::VarName)

Add a variable with given value to `vnv`.

Like `setindex!`, but errors if the key `vn` already exists.
"""
function Base.insert!(vnv::VarNamedVector, val, vn::VarName)
    if haskey(vnv, vn)
        throw("Variable $vn already exists in VarNamedVector.")
    end
    return reset!(vnv, val, vn)
end

"""
    push!(vnv::VarNamedVector, pair::Pair)

Add a variable with given value to `vnv`. Pair should be a `VarName` and a value.
"""
function Base.push!(vnv::VarNamedVector, pair::Pair)
    vn, val = pair
    # TODO(mhauru) Or should this rather call `reset!`? It would be more inline with what
    # Dict does, but could also cause confusion.
    return setindex!(vnv, val, vn)
end

"""
    setindex_internal!(vnv::VarNamedVector, val, i::Int)

Sets the `i`th element of the internal storage vector, ignoring inactive entries.
"""
function setindex_internal!(vnv::VarNamedVector, val, i::Int)
    return vnv.vals[index_to_vals_index(vnv, i)] = val
end

"""
    setindex_internal!(vnv::VarNamedVector, val, vn::VarName[, transform])

Like `setindex!`, but sets the values as they are stored internally in `vnv`.

Optionally can set the transformation, such that `transform(val)` is the original value of
the variable. By default, the transform is the identity if creating a new entry in `vnv`, or
the existing transform if updating an existing entry.
"""
function setindex_internal!(
    vnv::VarNamedVector, val::AbstractVector, vn::VarName, transform=nothing
)
    if haskey(vnv, vn)
        return update_internal!(vnv, val, vn, transform)
    else
        return insert_internal!(vnv, val, vn, transform)
    end
end

"""
    insert_internal!(vnv::VarNamedVector, val::AbstractVector, vn::VarName[, transform])

Add a variable with given value to `vnv`.

Like `setindex_internal!`, but errors if the key `vn` already exists.

`transform` should be a function that converts `val` to the original representation. By
default it's `identity`.
"""
function insert_internal!(
    vnv::VarNamedVector, val::AbstractVector, vn::VarName, transform=nothing
)
    if transform === nothing
        transform = identity
    end
    haskey(vnv, vn) && throw(ArgumentError("variable name $vn already exists"))
    # NOTE: We need to compute the `nextrange` BEFORE we start mutating the underlying
    # storage.
    r_new = nextrange(vnv, val)
    vnv.varname_to_index[vn] = length(vnv.varname_to_index) + 1
    push!(vnv.varnames, vn)
    push!(vnv.ranges, r_new)
    append!(vnv.vals, val)
    push!(vnv.transforms, transform)
    push!(vnv.is_unconstrained, false)
    return nothing
end

"""
    update_internal!(vnv::VarNamedVector, vn::VarName, val::AbstractVector[, transform])

Update an existing entry for `vn` in `vnv` with the value `val`.

Like `setindex_internal!`, but errors if the key `vn` doesn't exist.

`transform` should be a function that converts `val` to the original representation. By
default it's the same as the old transform for `vn`.
"""
function update_internal!(
    vnv::VarNamedVector, val::AbstractVector, vn::VarName, transform=nothing
)
    # Here we update an existing entry.
    if !haskey(vnv, vn)
        throw(KeyError(vn))
    end
    idx = getidx(vnv, vn)
    # Extract the old range.
    r_old = getrange(vnv, idx)
    start_old, end_old = first(r_old), last(r_old)
    n_old = length(r_old)
    # Compute the new range.
    n_new = length(val)
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
    vnv.vals[r_new] = val
    if transform !== nothing
        # Update the transform.
        vnv.transforms[idx] = transform
    end

    # TODO: Should we maybe sweep over inactive ranges and re-contiguify
    # if the total number of inactive elements is "large" in some sense?

    return nothing
end

# TODO(mhauru) The num_produce argument is used by Particle Gibbs.
# Remove this method as soon as possible.
function BangBang.push!(vnv::VarNamedVector, vn, val, dist, num_produce)
    f = from_vec_transform(dist)
    return setindex_internal!(vnv, tovec(val), vn, f)
end

# BangBang versions of the above functions.
# The only difference is that update_internal!! and insert_internal!! check whether the
# container types of the VarNamedVector vector need to be expanded to accommodate the new
# values. If so, they create a new instance, otherwise they mutate in place. All the others
# functions, e.g. setindex!!, setindex_internal!!, etc., are carbon copies of the ! versions
# with every ! call replaced with a !! call.

"""
    loosen_types!!(vnv::VarNamedVector{K,V,TVN,TVal,TTrans}, ::Type{KNew}, ::Type{TransNew})

Loosen the types of `vnv` to allow varname type `KNew` and transformation type `TransNew`.

If `KNew` is a subtype of `K` and `TransNew` is a subtype of the element type of the
`TTrans` then this is a no-op and `vnv` is returned as is. Otherwise a new `VarNamedVector`
is returned with the same data but more abstract types, so that variables of type `KNew` and
transformations of type `TransNew` can be pushed to it. Some of the underlying storage is
shared between `vnv` and the return value, and thus mutating one may affect the other.

# See also
[`tighten_types`](@ref)

# Examples

```jldoctest varnamedvector-loosen-types
julia> using DynamicPPL: VarNamedVector, @varname, loosen_types!!, setindex_internal!

julia> vnv = VarNamedVector(@varname(x) => [1.0]);

julia> y_trans(x) = reshape(x, (2, 2));

julia> setindex_internal!(vnv, collect(1:4), @varname(y), y_trans)
ERROR: MethodError: Cannot `convert` an object of type
[...]

julia> vnv_loose = DynamicPPL.loosen_types!!(vnv, typeof(@varname(y)), typeof(y_trans));

julia> setindex_internal!(vnv_loose, collect(1:4), @varname(y), y_trans)

julia> vnv_loose[@varname(y)]
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  4.0
```
"""
function loosen_types!!(
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
            vnv.is_unconstrained,
            vnv.num_inactive,
        )
    end
end

"""
    tighten_types(vnv::VarNamedVector)

Return a copy of `vnv` with the most concrete types possible.

For instance, if `vnv` has its vector of transforms have eltype `Any`, but all the
transforms are actually identity transformations, this function will return a new
`VarNamedVector` with the transforms vector having eltype `typeof(identity)`.

This is a lot like the reverse of [`loosen_types!!`](@ref), but with two notable
differences: Unlike `loosen_types!!`, this function does not mutate `vnv`; it also changes
not only the key and transform eltypes, but also the values eltype.

# See also
[`loosen_types!!`](@ref)

# Examples

```jldoctest varnamedvector-tighten-types
julia> using DynamicPPL: VarNamedVector, @varname, loosen_types!!, setindex_internal!

julia> vnv = VarNamedVector();

julia> setindex!(vnv, [23], @varname(x))

julia> eltype(vnv)
Real

julia> vnv.transforms
1-element Vector{Any}:
 identity (generic function with 1 method)

julia> vnv_tight = DynamicPPL.tighten_types(vnv);

julia> eltype(vnv_tight) == Int
true

julia> vnv_tight.transforms
1-element Vector{typeof(identity)}:
 identity (generic function with 1 method)
```
"""
function tighten_types(vnv::VarNamedVector)
    return VarNamedVector(
        OrderedDict(vnv.varname_to_index...),
        map(identity, vnv.varnames),
        copy(vnv.ranges),
        map(identity, vnv.vals),
        map(identity, vnv.transforms),
        copy(vnv.is_unconstrained),
        copy(vnv.num_inactive),
    )
end

function BangBang.setindex!!(vnv::VarNamedVector, val, vn::VarName)
    if haskey(vnv, vn)
        return update!!(vnv, val, vn)
    else
        return insert!!(vnv, val, vn)
    end
end

function reset!!(vnv::VarNamedVector, val, vn::VarName)
    f = from_vec_transform(val)
    vnv = setindex_internal!!(vnv, tovec(val), vn, f)
    vnv = settrans!!(vnv, false, vn)
    return vnv
end

function update!!(vnv::VarNamedVector, val, vn::VarName)
    if !haskey(vnv, vn)
        throw(KeyError(vn))
    end
    f = inverse(gettransform(vnv, vn))
    internal_val = try
        f(val)
    catch
        error(
            "An error occurred while assigning the value $val to variable $vn. " *
            "If you are changing the type or size of a variable you'll need to either " *
            "`delete!` it first or use `setindex_internal!`",
        )
    end
    return setindex_internal!!(vnv, internal_val, vn)
end

function insert!!(vnv::VarNamedVector, val, vn::VarName)
    if haskey(vnv, vn)
        throw("Variable $vn already exists in VarNamedVector.")
    end
    return reset!!(vnv, val, vn)
end

function setindex_internal!!(
    vnv::VarNamedVector, val::AbstractVector, vn::VarName, transform=nothing
)
    if haskey(vnv, vn)
        return update_internal!!(vnv, val, vn, transform)
    else
        return insert_internal!!(vnv, val, vn, transform)
    end
end

function insert_internal!!(vnv::VarNamedVector, val, vn::VarName, transform=nothing)
    if transform === nothing
        transform = identity
    end
    vnv = loosen_types!!(vnv, typeof(vn), typeof(transform))
    insert_internal!(vnv, val, vn, transform)
    return vnv
end

function update_internal!!(vnv::VarNamedVector, val, vn::VarName, transform=nothing)
    transform_resolved = transform === nothing ? gettransform(vnv, vn) : transform
    vnv = loosen_types!!(vnv, typeof(vn), typeof(transform_resolved))
    update_internal!(vnv, val, vn, transform)
    return vnv
end

function BangBang.push!!(vnv::VarNamedVector, pair::Pair)
    vn, val = pair
    return setindex!!(vnv, val, vn)
end

# TODO(mhauru) The num_produce argument is used by Particle Gibbs.
# Remove this method as soon as possible.
function BangBang.push!!(vnv::VarNamedVector, vn, val, dist, num_produce)
    f = from_vec_transform(dist)
    return setindex_internal!!(vnv, tovec(val), vn, f)
end

function Base.empty!(vnv::VarNamedVector)
    # TODO: Or should the semantics be different, e.g. keeping `varnames`?
    empty!(vnv.varname_to_index)
    empty!(vnv.varnames)
    empty!(vnv.ranges)
    empty!(vnv.vals)
    empty!(vnv.transforms)
    empty!(vnv.is_unconstrained)
    empty!(vnv.num_inactive)
    return nothing
end
BangBang.empty!!(vnv::VarNamedVector) = (empty!(vnv); return vnv)

"""
    replace_raw_storage(vnv::VarNamedVector, vals::AbstractVector)

Replace the values in `vnv` with `vals`, as they are stored internally.

This is useful when we want to update the entire underlying vector of values in one go or if
we want to change the how the values are stored, e.g. alter the `eltype`.

!!! warning
    This replaces the raw underlying values, and so care should be taken when using this
    function. For example, if `vnv` has any inactive entries, then the provided `vals`
    should also contain the inactive entries to avoid unexpected behavior.

# Examples

```jldoctest varnamedvector-replace-raw-storage
julia> using DynamicPPL: VarNamedVector, replace_raw_storage

julia> vnv = VarNamedVector(@varname(x) => [1.0]);

julia> replace_raw_storage(vnv, [2.0])[@varname(x)] == [2.0]
true
```

This is also useful when we want to differentiate wrt. the values using automatic
differentiation, e.g. ForwardDiff.jl.

```jldoctest varnamedvector-replace-raw-storage
julia> using ForwardDiff: ForwardDiff

julia> f(x) = sum(abs2, replace_raw_storage(vnv, x)[@varname(x)])
f (generic function with 1 method)

julia> ForwardDiff.gradient(f, [1.0])
1-element Vector{Float64}:
 2.0
```
"""
replace_raw_storage(vnv::VarNamedVector, vals) = Accessors.@set vnv.vals = vals

vector_length(vnv::VarNamedVector) = length(vnv.vals) - num_inactive(vnv)

"""
    unflatten(vnv::VarNamedVector, vals::AbstractVector)

Return a new instance of `vnv` with the values of `vals` assigned to the variables.

This assumes that `vals` have been transformed by the same transformations that that the
values in `vnv` have been transformed by. However, unlike [`replace_raw_storage`](@ref),
`unflatten` does account for inactive entries in `vnv`, so that the user does not have to
care about them.

This is in a sense the reverse operation of `vnv[:]`.

Unflatten recontiguifies the internal storage, getting rid of any inactive entries.

# Examples

```jldoctest varnamedvector-unflatten
julia> using DynamicPPL: VarNamedVector, unflatten

julia> vnv = VarNamedVector(@varname(x) => [1.0, 2.0], @varname(y) => [3.0]);

julia> unflatten(vnv, vnv[:]) == vnv
true
"""
function unflatten(vnv::VarNamedVector, vals::AbstractVector)
    new_ranges = deepcopy(vnv.ranges)
    recontiguify_ranges!(new_ranges)
    return VarNamedVector(
        vnv.varname_to_index,
        vnv.varnames,
        new_ranges,
        vals,
        vnv.transforms,
        vnv.is_unconstrained,
    )
end

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
    varname_to_index = OrderedDict{V,Int}()
    ranges = UnitRange{Int}[]
    vals = T[]
    transforms = F[]
    is_unconstrained = BitVector(undef, length(vns_both))

    # Range offset.
    offset = 0

    for (idx, vn) in enumerate(vns_both)
        varname_to_index[vn] = idx
        # Extract the necessary information from `left` or `right`.
        if vn in vns_left && !(vn in vns_right)
            # `vn` is only in `left`.
            val = getindex_internal(left_vnv, vn)
            f = gettransform(left_vnv, vn)
            is_unconstrained[idx] = istrans(left_vnv, vn)
        else
            # `vn` is either in both or just `right`.
            # Note that in a `merge` the right value has precedence.
            val = getindex_internal(right_vnv, vn)
            f = gettransform(right_vnv, vn)
            is_unconstrained[idx] = istrans(right_vnv, vn)
        end
        n = length(val)
        r = (offset + 1):(offset + n)
        # Update.
        append!(vals, val)
        push!(ranges, r)
        push!(transforms, f)
        # Increment `offset`.
        offset += n
    end

    return VarNamedVector(
        varname_to_index, vns_both, ranges, vals, transforms, is_unconstrained
    )
end

"""
    subset(vnv::VarNamedVector, vns::AbstractVector{<:VarName})

Return a new `VarNamedVector` containing the values from `vnv` for variables in `vns`.

Which variables to include is determined by the `VarName`'s `subsumes` relation, meaning
that e.g. `subset(vnv, [@varname(x)])` will include variables like `@varname(x.a[1])`.

Preserves the order of variables in `vnv`.

# Examples

```jldoctest varnamedvector-subset
julia> using DynamicPPL: VarNamedVector, @varname, subset

julia> vnv = VarNamedVector(@varname(x) => [1.0, 2.0], @varname(y) => [3.0]);

julia> subset(vnv, [@varname(x)]) == VarNamedVector(@varname(x) => [1.0, 2.0])
true

julia> subset(vnv, [@varname(x[2])]) == VarNamedVector(@varname(x[2]) => [2.0])
true
"""
function subset(vnv::VarNamedVector, vns_given::AbstractVector{<:VarName})
    # NOTE: This does not specialize types when possible.
    vnv_new = similar(vnv)
    # Return early if possible.
    isempty(vnv) && return vnv_new

    for vn in vnv.varnames
        if any(subsumes(vn_given, vn) for vn_given in vns_given)
            insert_internal!(vnv_new, getindex_internal(vnv, vn), vn, gettransform(vnv, vn))
            settrans!(vnv_new, istrans(vnv, vn), vn)
        end
    end

    return vnv_new
end

"""
    similar(vnv::VarNamedVector)

Return a new `VarNamedVector` with the same structure as `vnv`, but with empty values.

In this respect `vnv` behaves more like a dictionary than an array: `similar(vnv)` will
be entirely empty, rather than have `undef` values in it.

# Examples

```julia-doctest-varnamedvector-similar
julia> using DynamicPPL: VarNamedVector, @varname, similar

julia> vnv = VarNamedVector(@varname(x) => [1.0, 2.0], @varname(x[3]) => [3.0]);

julia> similar(vnv) == VarNamedVector{VarName{:x}, Float64}()
true
"""
function Base.similar(vnv::VarNamedVector)
    # NOTE: Whether or not we should empty the underlying containers or not
    # is somewhat ambiguous. For example, `similar(vnv.varname_to_index)` will
    # result in an empty `AbstractDict`, while the vectors, e.g. `vnv.ranges`,
    # will result in non-empty vectors but with entries as `undef`. But it's
    # much easier to write the rest of the code assuming that `undef` is not
    # present, and so for now we empty the underlying containers, thus differing
    # from the behavior of `similar` for `AbstractArray`s.
    return VarNamedVector(
        empty(vnv.varname_to_index),
        similar(vnv.varnames, 0),
        similar(vnv.ranges, 0),
        similar(vnv.vals, 0),
        similar(vnv.transforms, 0),
        BitVector(),
        empty(vnv.num_inactive),
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

# TODO(mhauru) Might add another specialisation to _compose_no_identity, where if
# ReshapeTransforms are composed with each other or with a an UnwrapSingeltonTransform, only
# the latter one would be kept.
"""
    _compose_no_identity(f, g)

Like `f ∘ g`, but if `f` or `g` is `identity` it is omitted.

This helps avoid trivial cases of `ComposedFunction` that would cause unnecessary type
conflicts.
"""
_compose_no_identity(f, g) = f ∘ g
_compose_no_identity(::typeof(identity), g) = g
_compose_no_identity(f, ::typeof(identity)) = f
_compose_no_identity(::typeof(identity), ::typeof(identity)) = identity

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

# set!! is the function defined in utils.jl that tries to do fancy stuff with optics when
# setting the value of a generic container using a VarName. We can bypass all that because
# VarNamedVector handles VarNames natively. However, it's semantics are slightly different
# from setindex!'s: It allows resetting variables that already have a value with values of
# a different type/size.
set!!(vnv::VarNamedVector, vn::VarName, val) = reset!!(vnv, val, vn)

function setval!(vnv::VarNamedVector, val, vn::VarName)
    return setindex_internal!(vnv, tovec(val), vn)
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

# Examples

```jldoctest varnamedvector-contiguify
julia> using DynamicPPL: VarNamedVector, @varname, contiguify!, update!, has_inactive

julia> vnv = VarNamedVector(@varname(x) => [1.0, 2.0, 3.0], @varname(y) => [3.0]);

julia> update!(vnv, [23.0, 24.0], @varname(x));

julia> has_inactive(vnv)
true

julia> length(vnv.vals)
4

julia> contiguify!(vnv);

julia> has_inactive(vnv)
false

julia> length(vnv.vals)
3

julia> vnv[@varname(x)]  # All the values are still there.
2-element Vector{Float64}:
 23.0
 24.0
```
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

# Examples

```jldoctest varnamedvector-group-by-symbol
julia> using DynamicPPL: VarNamedVector, @varname, group_by_symbol

julia> vnv = VarNamedVector(@varname(x) => [1.0], @varname(y) => [2.0], @varname(x[1]) => [3.0]);

julia> d = group_by_symbol(vnv);

julia> collect(keys(d))
[Symbol("x"), Symbol("y")]

julia> d[@varname(x)] == VarNamedVector(@varname(x) => [1.0], @varname(x[1]) => [3.0])
true

julia> d[@varname(y)] == VarNamedVector(@varname(y) => [2.0])
true
"""
function group_by_symbol(vnv::VarNamedVector)
    symbols = unique(map(getsym, vnv.varnames))
    nt_vals = map(s -> tighten_types(subset(vnv, [VarName{s}()])), symbols)
    return OrderedDict(zip(symbols, nt_vals))
end

"""
    shift_index_left!(vnv::VarNamedVector, idx::Int)

Shift the index `idx` to the left by one and update the relevant fields.

This only affects `vnv.varname_to_index` and `vnv.num_inactive` and is only valid as a
helper function for [`shift_subsequent_indices_left!`](@ref).

!!! warning
    This does not check if index we're shifting to is already occupied.
"""
function shift_index_left!(vnv::VarNamedVector, idx::Int)
    # Shift the index in the lookup table.
    vn = vnv.varnames[idx]
    vnv.varname_to_index[vn] = idx - 1
    # Shift the index in the inactive ranges.
    if haskey(vnv.num_inactive, idx)
        # Done in increasing order => don't need to worry about
        # potentially shifting the same index twice.
        vnv.num_inactive[idx - 1] = pop!(vnv.num_inactive, idx)
    end
end

"""
    shift_subsequent_indices_left!(vnv::VarNamedVector, idx::Int)

Shift the indices for all variables after `idx` to the left by one and update the relevant
    fields.

This only affects `vnv.varname_to_index` and `vnv.num_inactive` and is only valid as a
helper function for [`delete!`](@ref).
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
values_as(vnv::VarNamedVector, ::Type{Vector}) = getindex_internal(vnv, :)
function values_as(vnv::VarNamedVector, ::Type{Vector{T}}) where {T}
    return convert(Vector{T}, values_as(vnv, Vector))
end
function values_as(vnv::VarNamedVector, ::Type{NamedTuple})
    return NamedTuple(zip(map(Symbol, keys(vnv)), values(vnv)))
end
function values_as(vnv::VarNamedVector, ::Type{D}) where {D<:AbstractDict}
    return ConstructionBase.constructorof(D)(pairs(vnv))
end

# See the docstring of `getvalue` for the semantics of `hasvalue` and `getvalue`, and how
# they differ from `haskey` and `getindex`. They can be found in src/utils.jl.

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

Base.get(vnv::VarNamedVector, vn::VarName) = getvalue(vnv, vn)
