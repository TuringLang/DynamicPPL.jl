"""
    VarNamedVector

A container that stores values in a vectorised form, but indexable by variable names.

When indexed by integers or `Colon`s, e.g. `vnv[2]` or `vnv[:]`, `VarNamedVector` behaves
like a `Vector`, and returns the values as they are stored. The stored form is always
vectorised, for instance matrix variables have been flattened, and may be further
transformed to achieve linking.

When indexed by `VarName`s, e.g. `vnv[@varname(x)]`, `VarNamedVector` returns the values
in the original space. For instance, a linked matrix variable is first inverse linked and
then reshaped to its original form before returning it to the caller.

`VarNamedVector` also stores a boolean for whether a variable has been transformed to
unconstrained Euclidean space or not.

# Fields
$(FIELDS)
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
    that transformes the value of `vn` back to its original space, undoing any linking and
    vectorisation
    """
    transforms::TTrans

    """
    vector of booleans indicating whether a variable has been transformed to unconstrained
    Euclidean space or not, i.e. whether its domain is all of `ℝ^ⁿ`. Having
    `is_unconstrained[varname_to_index[vn]] == false` does not necessarily mean that a
    variable is constrained, but rather that it's not guaranteed to not be.
    """
    is_unconstrained::BitVector

    """
    mapping from a variable index to the number of inactive entries for that variable.
    Inactive entries are elements in `vals` that are not part of the value of any variable.
    They arise when transformations change the dimension of the value stored. In active
    entries always come after the last active entry for the given variable.
    """
    num_inactive::OrderedDict{Int,Int}

    function VarNamedVector(
        varname_to_index,
        varnames::TVN,
        ranges,
        vals::TVal,
        transforms::TTrans,
        is_unconstrained,
        num_inactive,
    ) where {K,V,TVN<:AbstractVector{K},TVal<:AbstractVector{V},TTrans<:AbstractVector}
        if length(varnames) != length(ranges) ||
            length(varnames) != length(transforms) ||
            length(varnames) != length(is_unconstrained) ||
            length(varnames) != length(varname_to_index)
            msg = """
                Inputs to VarNamedVector have inconsistent lengths. Got lengths \
                varnames: $(length(varnames)), \
                ranges: $(length(ranges)), \
                transforms: $(length(transforms)), \
                is_unconstrained: $(length(is_unconstrained)), \
                varname_to_index: $(length(varname_to_index))."""
            throw(ArgumentError(msg))
        end

        num_vals = mapreduce(length, (+), ranges; init=0) + sum(values(num_inactive))
        if num_vals != length(vals)
            msg = """
            The total number of elements in `vals` ($(length(vals))) does not match the \
                sum of the lengths of the ranges and the number of inactive entries \
                ($(num_vals))."""
            throw(ArgumentError(msg))
        end

        if Set(values(varname_to_index)) != Set(1:length(varnames))
            msg = "The values of `varname_to_index` are not valid indices."
            throw(ArgumentError(msg))
        end

        if !issubset(Set(keys(num_inactive)), Set(values(varname_to_index)))
            msg = "The keys of `num_inactive` are not valid indices."
            throw(ArgumentError(msg))
        end

        # Check that the varnames don't overlap. The time cost is quadratic in number of
        # variables. If this ever becomes an issue, we should be able to go down to at least
        # N log N by sorting based on subsumes-order.
        for vn1 in keys(varname_to_index)
            for vn2 in keys(varname_to_index)
                vn1 === vn2 && continue
                if subsumes(vn1, vn2)
                    msg = """
                    Variables in a VarNamedVector should not subsume each other, \
                    but $vn1 subsumes $vn2"""
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

# Default values for is_unconstrained (all false) and num_inactive (empty).
function VarNamedVector(
    varname_to_index,
    varnames,
    ranges,
    vals,
    transforms,
    is_unconstrained=fill!(BitVector(undef, length(varnames)), 0),
)
    return VarNamedVector(
        varname_to_index,
        varnames,
        ranges,
        vals,
        transforms,
        is_unconstrained,
        OrderedDict{Int,Int}(),
    )
end

# TODO(mhauru) Are we sure we want the last one to be of type Any[]? Might this call
# unnecessary type instability?
function VarNamedVector{K,V}() where {K,V}
    return VarNamedVector(OrderedDict{K,Int}(), K[], UnitRange{Int}[], V[], Any[])
end

# TODO(mhauru) I would like for this to be VarNamedVector(Union{}, Union{}). This would
# allow expanding the VarName and element types only as necessary, which would help keep
# them concrete. However, making that change here opens some other cans of worms related to
# how VarInfo uses BangBang, that I don't want to deal with right now.
VarNamedVector() = VarNamedVector{VarName,Real}()
VarNamedVector(xs::Pair...) = VarNamedVector(OrderedDict(xs...))
VarNamedVector(x::AbstractDict) = VarNamedVector(keys(x), values(x))
function VarNamedVector(varnames, vals)
    return VarNamedVector(collect_maybe(varnames), collect_maybe(vals))
end
function VarNamedVector(
    varnames::AbstractVector,
    vals::AbstractVector,
    transforms=fill(identity, length(varnames)),
)
    # Convert `vals` into a vector of vectors.
    vals_vecs = map(tovec, vals)
    transforms = map(
        (t, val) -> _compose_no_identity(t, from_vec_transform(val)), transforms, vals
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

# Some `VarNamedVector` specific functions.
getidx(vnv::VarNamedVector, vn::VarName) = vnv.varname_to_index[vn]

getrange(vnv::VarNamedVector, idx::Int) = vnv.ranges[idx]
getrange(vnv::VarNamedVector, vn::VarName) = getrange(vnv, getidx(vnv, vn))

gettransform(vnv::VarNamedVector, idx::Int) = vnv.transforms[idx]
gettransform(vnv::VarNamedVector, vn::VarName) = gettransform(vnv, getidx(vnv, vn))

# TODO(mhauru) Eventually I would like to rename the istrans function to is_unconstrained,
# but that's significantly breaking.
"""
    istrans(vnv::VarNamedVector, vn::VarName)

Return a boolean for whether `vn` is guaranteed to have been transformed so that all of
Euclidean space is its domain.
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

Returns the number of allocated entries in `vnv`, both active and inactive.
"""
num_allocated(vnv::VarNamedVector) = length(vnv.vals)

"""
    num_allocated(vnv::VarNamedVector, vn::VarName)

Returns the number of allocated entries for `vn` in `vnv`, both active and inactive.
"""
num_allocated(vnv::VarNamedVector, vn::VarName) = num_allocated(vnv, getidx(vnv, vn))
function num_allocated(vnv::VarNamedVector, idx::Int)
    return length(getrange(vnv, idx)) + num_inactive(vnv, idx)
end

# Basic array interface.
Base.eltype(vnv::VarNamedVector) = eltype(vnv.vals)
Base.length(vnv::VarNamedVector) =
    if !has_inactive(vnv)
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
    getindex_raw(vnv::VarNamedVector, i::Int)
    getindex_raw(vnv::VarNamedVector, vn::VarName)

Like `getindex`, but returns the values as they are stored in `vnv` without transforming.

For integer indices this is the same as `getindex`, but for `VarName`s this is different.
"""
getindex_raw(vnv::VarNamedVector, i::Int) = vnv.vals[index_to_vals_index(vnv, i)]
getindex_raw(vnv::VarNamedVector, vn::VarName) = vnv.vals[getrange(vnv, vn)]

# `getindex` for `Colon`
function Base.getindex(vnv::VarNamedVector, ::Colon)
    return if has_inactive(vnv)
        mapreduce(Base.Fix1(getindex, vnv.vals), vcat, vnv.ranges)
    else
        vnv.vals
    end
end

getindex_raw(vnv::VarNamedVector, ::Colon) = getindex(vnv, Colon())

# TODO(mhauru): Remove this as soon as possible. Only needed because of the old Gibbs
# sampler.
function Base.getindex(vnv::VarNamedVector, spl::AbstractSampler)
    throw(ErrorException("Cannot index a VarNamedVector with a sampler."))
end

Base.setindex!(vnv::VarNamedVector, val, i::Int) = setindex_raw!(vnv, val, i)
function Base.setindex!(vnv::VarNamedVector, val, vn::VarName)
    # Since setindex! does not change the transform, we need to apply it to `val`.
    f = inverse(gettransform(vnv, vn))
    return setindex_raw!(vnv, f(val), vn)
end

"""
    setindex_raw!(vnv::VarNamedVector, val, i::Int)
    setindex_raw!(vnv::VarNamedVector, val, vn::VarName)

Like `setindex!`, but sets the values as they are stored in `vnv` without transforming.

For integer indices this is the same as `setindex!`, but for `VarName`s this is different.
"""
function setindex_raw!(vnv::VarNamedVector, val, i::Int)
    return vnv.vals[index_to_vals_index(vnv, i)] = val
end

function setindex_raw!(vnv::VarNamedVector, val::AbstractVector, vn::VarName)
    return vnv.vals[getrange(vnv, vn)] = val
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
    replace_values(vnv::VarNamedVector, vals::AbstractVector)

Replace the values in `vnv` with `vals`, as they are stored internally.

This is useful when we want to update the entire underlying vector of values in one go or if
we want to change the how the values are stored, e.g. alter the `eltype`.

!!! warning
    This replaces the raw underlying values, and so care should be taken when using this
    function. For example, if `vnv` has any inactive entries, then the provided `vals`
    should also contain the inactive entries to avoid unexpected behavior.

# Examples

```jldoctest varnamedvector-replace-values
julia> using DynamicPPL: VarNamedVector, replace_values

julia> vnv = VarNamedVector(@varname(x) => [1.0]);

julia> replace_values(vnv, [2.0])[@varname(x)] == [2.0]
true
```

This is also useful when we want to differentiate wrt. the values using automatic
differentiation, e.g. ForwardDiff.jl.

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

# TODO(mhauru) The space argument is used by the old Gibbs sampler. To be removed.
function replace_values(vnv::VarNamedVector, ::Val{space}, vals) where {space}
    if length(space) > 0
        msg = "Selecting values in a VarNamedVector with a space is not supported."
        throw(ArgumentError(msg))
    end
    return replace_values(vnv, vals)
end

"""
    unflatten(vnv::VarNamedVector, vals::AbstractVector)

Return a new instance of `vnv` with the values of `vals` assigned to the variables.

This assumes that `vals` have been transformed by the same transformations that that the
values in `vnv` have been transformed by. However, unlike [`replace_values`](@ref),
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
        vnv.varname_to_index, vnv.varnames, new_ranges, vals, vnv.transforms
    )
end

# TODO(mhauru) To be removed once the old Gibbs sampler is removed.
function unflatten(vnv::VarNamedVector, spl::AbstractSampler, vals::AbstractVector)
    if length(getspace(spl)) > 0
        msg = "Selecting values in a VarNamedVector with a space is not supported."
        throw(ArgumentError(msg))
    end
    return unflatten(vnv, vals)
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
            val = getindex_raw(left_vnv, vn)
            f = gettransform(left_vnv, vn)
            is_unconstrained[idx] = istrans(left_vnv, vn)
        else
            # `vn` is either in both or just `right`.
            # Note that in a `merge` the right value has precedence.
            val = getindex_raw(right_vnv, vn)
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

# Examples

```jldoctest varnamedvector-subset
julia> using DynamicPPL: VarNamedVector, @varname, subset

julia> vnv = VarNamedVector(@varname(x) => [1.0, 2.0], @varname(y) => [3.0]);

julia> subset(vnv, [@varname(x)]) == VarNamedVector(@varname(x) => [1.0, 2.0])
true

julia> subset(vnv, [@varname(x[2])]) == VarNamedVector(@varname(x[2]) => [2.0])
true
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
        push!(vnv_new, vn, getindex_raw(vnv, vn), gettransform(vnv, vn))
        settrans!(vnv_new, istrans(vnv, vn), vn)
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
    push!(vnv::VarNamedVector, vn::VarName, val[, transform])

Add a variable with given value to `vnv`.

`transform` should be a function that converts `val` to the original representation, by
default it's `identity`.
"""
function Base.push!(vnv::VarNamedVector, vn::VarName, val, transform=identity)
    # Error if we already have the variable.
    haskey(vnv, vn) && throw(ArgumentError("variable name $vn already exists"))
    # NOTE: We need to compute the `nextrange` BEFORE we start mutating the underlying
    # storage.
    if !(val isa AbstractVector)
        val_vec = tovec(val)
        transform = _compose_no_identity(transform, from_vec_transform(val))
    else
        val_vec = val
    end
    r_new = nextrange(vnv, val_vec)
    vnv.varname_to_index[vn] = length(vnv.varname_to_index) + 1
    push!(vnv.varnames, vn)
    push!(vnv.ranges, r_new)
    append!(vnv.vals, val_vec)
    push!(vnv.transforms, transform)
    push!(vnv.is_unconstrained, false)
    return nothing
end

# TODO(mhauru) The gidset and num_produce arguments are used by the old Gibbs sampler.
# Remove this method as soon as possible.
function Base.push!(vnv::VarNamedVector, vn, val, dist, gidset, num_produce)
    f = from_vec_transform(dist)
    return push!(vnv, vn, tovec(val), f)
end

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
julia> using DynamicPPL: VarNamedVector, @varname, loosen_types!!

julia> vnv = VarNamedVector(@varname(x) => [1.0]);

julia> vnv_new = loosen_types!!(vnv, VarName{:x}, Real);

julia> push!(vnv, @varname(y), Float32[2.0])
ERROR: MethodError: Cannot `convert` an object of type
  VarName{y,typeof(identity)} to an object of type
  VarName{x,typeof(identity)}
[...]

julia> vnv_loose = DynamicPPL.loosen_types!!(vnv, typeof(@varname(y)), Float32);

julia> push!(vnv_loose, @varname(y), Float32[2.0]); vnv_loose  # Passes without issues.
VarNamedVector{VarName{sym, typeof(identity)} where sym, Float64, Vector{VarName{sym, typeof(identity)} where sym}, Vector{Float64}, Vector{Any}}(OrderedDict{VarName{sym, typeof(identity)} where sym, Int64}(x => 1, y => 2), VarName{sym, typeof(identity)} where sym[x, y], UnitRange{Int64}[1:1, 2:2], [1.0, 2.0], Any[identity, identity], Bool[0, 0], OrderedDict{Int64, Int64}())
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

For instance, if `vnv` has element type `Real`, but all the values are actually `Float64`s,
then `tighten_types(vnv)` will have element type `Float64`.

# See also
[`loosen_types!!`](@ref)
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

function BangBang.push!!(vnv::VarNamedVector, vn::VarName, val, transform=identity)
    vnv = loosen_types!!(
        vnv, typeof(vn), typeof(_compose_no_identity(transform, from_vec_transform(val)))
    )
    push!(vnv, vn, val, transform)
    return vnv
end

# TODO(mhauru) The gidset and num_produce arguments are used by the old Gibbs sampler.
# Remove this method as soon as possible.
function BangBang.push!!(vnv::VarNamedVector, vn, val, dist, gidset, num_produce)
    f = from_vec_transform(dist)
    return push!!(vnv, vn, tovec(val), f)
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

"""
    update!(vnv::VarNamedVector, vn::VarName, val[, transform])

Either add a new entry or update existing entry for `vn` in `vnv` with the value `val`.

If `vn` does not exist in `vnv`, this is equivalent to [`push!`](@ref).

`transform` should be a function that converts `val` to the original representation, by
default it's `identity`.
"""
function update!(vnv::VarNamedVector, vn::VarName, val, transform=identity)
    if !haskey(vnv, vn)
        # Here we just add a new entry.
        return push!(vnv, vn, val, transform)
    end

    # Here we update an existing entry.
    if !(val isa AbstractVector)
        val_vec = tovec(val)
        transform = _compose_no_identity(transform, from_vec_transform(val))
    else
        val_vec = val
    end
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

function update!!(vnv::VarNamedVector, vn::VarName, val, transform=identity)
    vnv = loosen_types!!(
        vnv, typeof(vn), typeof(_compose_no_identity(transform, from_vec_transform(val)))
    )
    update!(vnv, vn, val, transform)
    return vnv
end

# set!! is the function defined in utils.jl that tries to do fancy stuff with optics when
# setting the value of a generic container using a VarName. We can bypass all that because
# VarNamedVector handles VarNames natively.
set!!(vnv::VarNamedVector, vn::VarName, val) = update!!(vnv, vn, val)

function setval!(vnv::VarNamedVector, val, vn::VarName)
    return setindex_raw!(vnv, tovec(val), vn)
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

julia> update!(vnv, @varname(x), [23.0, 24.0]);

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
    nt_vals = map(s -> tighten_types(subset(vnv, [VarName(s)])), symbols)
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

Base.get(vnv::VarNamedVector, vn::VarName) = getvalue(vnv, vn)
