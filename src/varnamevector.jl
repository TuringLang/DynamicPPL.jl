# Similar to `Metadata` but representing a `Vector` and simpler interface.
# TODO: Should we subtype `AbstractVector`?
struct VarNameVector{
    TIdcs<:OrderedDict{<:VarName,Int},
    TVN<:AbstractVector{<:VarName},
    TVal<:AbstractVector,
    TTrans<:AbstractVector
}
    "mapping from the `VarName` to its integer index in `vns`, `ranges` and `dists`"
    idcs::TIdcs # Dict{<:VarName,Int}

    "vector of identifiers for the random variables, where `vns[idcs[vn]] == vn`"
    vns::TVN # AbstractVector{<:VarName}

    "vector of index ranges in `vals` corresponding to `vns`; each `VarName` `vn` has a single index or a set of contiguous indices in `vals`"
    ranges::Vector{UnitRange{Int}}

    "vector of values of all the univariate, multivariate and matrix variables; the value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`"
    vals::TVal # AbstractVector{<:Real}

    "vector of transformations whose inverse takes us back to the original space"
    transforms::TTrans
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
VarNameVector(vns, vals) = VarNameVector(collect(vns), collect(vals))
function VarNameVector(
    vns::AbstractVector,
    vals::AbstractVector,
    transforms = map(FromVec, vals)
)
    # TODO: Check uniqueness of `vns`?

    # Convert `vals` into a vector of vectors.
    vals_vecs = map(tovec, vals)

    # TODO: Is this really the way to do this?
    if !(eltype(vns) <: VarName)
        vns = convert(Vector{VarName}, vns)
    end
    idcs = OrderedDict{eltype(vns),Int}()
    ranges = Vector{UnitRange{Int}}()
    offset = 0
    for (i, (vn, x)) in enumerate(zip(vns, vals_vecs))
        # Add the varname index.
        push!(idcs, vn => length(idcs) + 1)
        # Add the range.
        r = (offset + 1):(offset + length(x))
        push!(ranges, r)
        # Update the offset.
        offset = r[end]
    end

    return VarNameVector(idcs, vns, ranges, reduce(vcat, vals_vecs), transforms)
end

# Basic array interface.
Base.eltype(vmd::VarNameVector) = eltype(vmd.vals)
Base.length(vmd::VarNameVector) = length(vmd.vals)
Base.size(vmd::VarNameVector) = size(vmd.vals)

Base.IndexStyle(::Type{<:VarNameVector}) = IndexLinear()

# `getindex` & `setindex!`
getidc(vmd::VarNameVector, vn::VarName) = vmd.idcs[vn]

getrange(vmd::VarNameVector, i::Int) = vmd.ranges[i]
getrange(vmd::VarNameVector, vn::VarName) = getrange(vmd, getidc(vmd, vn))

gettransform(vmd::VarNameVector, vn::VarName) = vmd.transforms[getidc(vmd, vn)]

Base.getindex(vmd::VarNameVector, i::Int) = vmd.vals[i]
function Base.getindex(vmd::VarNameVector, vn::VarName)
    x = vmd.vals[getrange(vmd, vn)]
    f = gettransform(vmd, vn)
    return f(x)
end

Base.setindex!(vmd::VarNameVector, val, i::Int) = vmd.vals[i] = val
function Base.setindex!(vmd::VarNameVector, val, vn::VarName)
    f = inverse(gettransform(vmd, vn))
    vmd.vals[getrange(vmd, vn)] = f(val)
end

# TODO: Re-use some of the show functionality from Base?
function Base.show(io::IO, vmd::VarNameVector)
    print(io, "[")
    for (i, vn) in enumerate(vmd.vns)
        if i > 1
            print(io, ", ")
        end
        print(io, vn, " = ", vmd[vn])
    end
end
