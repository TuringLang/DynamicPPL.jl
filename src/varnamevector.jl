# Similar to `Metadata` but representing a `Vector` and simpler interface.
# TODO: Should we subtype `AbstractVector`?
struct VarNameVector{
    TIdcs<:OrderedDict{<:VarName,Int},
    TVN<:AbstractVector{<:VarName},
    TVal<:AbstractVector,
    TTrans<:AbstractVector,
    MData
}
    "mapping from the `VarName` to its integer index in `varnames`, `ranges` and `dists`"
    varname_to_index::TIdcs # Dict{<:VarName,Int}

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

function VarNameVector(varname_to_index, varnames, ranges, vals, transforms)
    return VarNameVector(varname_to_index, varnames, ranges, vals, transforms, UnitRange{Int}[], nothing)
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

VarNameVector(x::AbstractDict) = VarNameVector(keys(x), values(x))
VarNameVector(varnames, vals) = VarNameVector(collect(varnames), collect(vals))
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

    return VarNameVector(varname_to_index, varnames, ranges, reduce(vcat, vals_vecs), transforms)
end

# Basic array interface.
Base.eltype(vnv::VarNameVector) = eltype(vnv.vals)
Base.length(vnv::VarNameVector) = length(vnv.vals)
Base.size(vnv::VarNameVector) = size(vnv.vals)

Base.IndexStyle(::Type{<:VarNameVector}) = IndexLinear()

# Dictionary interface.
Base.keys(vnv::VarNameVector) = vnv.varnames

Base.haskey(vnv::VarNameVector, vn::VarName) = haskey(vnv.varname_to_index, vn)

# `getindex` & `setindex!`
getidx(vnv::VarNameVector, vn::VarName) = vnv.varname_to_index[vn]

getrange(vnv::VarNameVector, i::Int) = vnv.ranges[i]
getrange(vnv::VarNameVector, vn::VarName) = getrange(vnv, getidx(vnv, vn))

gettransform(vnv::VarNameVector, vn::VarName) = vnv.transforms[getidx(vnv, vn)]

Base.getindex(vnv::VarNameVector, ::Colon) = vnv.vals
Base.getindex(vnv::VarNameVector, i::Int) = vnv.vals[i]
function Base.getindex(vnv::VarNameVector, vn::VarName)
    x = vnv.vals[getrange(vnv, vn)]
    f = gettransform(vnv, vn)
    return f(x)
end

# HACK: remove this as soon as possible.
Base.getindex(vnv::VarNameVector, spl::AbstractSampler) = vnv[:]

Base.setindex!(vnv::VarNameVector, val, i::Int) = vnv.vals[i] = val
function Base.setindex!(vnv::VarNameVector, val, vn::VarName)
    f = inverse(gettransform(vnv, vn))
    return vnv.vals[getrange(vnv, vn)] = f(val)
end

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

function nextrange(vnd::VarNameVector, x)
    n = length(vnd)
    return n + 1:n + length(x)
end

# `push!` and `push!!`: add a variable to the varname vector.
function push!(
    vnv::VarNameVector,
    vn::VarName,
    val,
    transform=FromVec(val),
)
    # Error if we already have the variable.
    haskey(vnv, vn) && throw(ArgumentError("variable name $vn already exists"))
    return update!(vnv, vn, val, transform)
end

# `update!` and `update!!`: update a variable in the varname vector.
function update!(
    vnv::VarNameVector,
    vn::VarName,
    val,
    transform=FromVec(val),
)
    val_vec = tovec(val)
    if !haskey(vnv, vn)
        # Here we just add a new entry.
        vnv.varname_to_index[vn] = length(vnv.varname_to_index) + 1
        push!(vnv.varnames, vn)
        push!(vnv.ranges, nextrange(vnv, val_vec))
        append!(vnv.vals, val_vec)
        push!(vnv.transforms, transform)
    else
        # Here we update the existing entry.
        idx = getidx(vnv, vn)
        r_old = getrange(vnv, idx)
        n_old = length(r_old)
        n_new = length(val_vec)
        # Existing keys needs to be handled differently depending on
        # whether the size of the value is increasing or decreasing.
        if n_new > n_old
            # Remove the old range.
            delete!(vnv.ranges, vn)
            # Add the new range.
            r_new = nextrange(vnv, val_vec)
            vnv.varname_to_ranges[vn] = r_new
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
    end

    return vnv
end

function recontiguify_ranges!(ranges::AbstractVector{<:AbstractRange})
    offset = 0
    # NOTE: assumes `ranges` are ordered.
    for i = 1:length(ranges)
        r_old = ranges[i]
        ranges[i] = offset + 1:offset + length(r_old)
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
