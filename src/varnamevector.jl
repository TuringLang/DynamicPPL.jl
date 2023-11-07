# Similar to `Metadata` but representing a `Vector` and simpler interface.
# TODO: Should we subtype `AbstractVector`?
struct VarNameVector{
    TIdcs<:OrderedDict{<:VarName,Int},
    TVN<:AbstractVector{<:VarName},
    TVal<:AbstractVector,
    TTrans<:AbstractVector,
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
    vns::AbstractVector, vals::AbstractVector, transforms=map(FromVec, vals)
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
Base.eltype(vnv::VarNameVector) = eltype(vnv.vals)
Base.length(vnv::VarNameVector) = length(vnv.vals)
Base.size(vnv::VarNameVector) = size(vnv.vals)

Base.IndexStyle(::Type{<:VarNameVector}) = IndexLinear()

# Dictionary interface.
Base.keys(vnv::VarNameVector) = vnv.vns

Base.haskey(vnv::VarNameVector, vn::VarName) = haskey(vnv.idcs, vn)

# `getindex` & `setindex!`
getidx(vnv::VarNameVector, vn::VarName) = vnv.idcs[vn]

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
    # TODO: Or should the semantics be different, e.g. keeping `vns`?
    empty!(vnv.idcs)
    empty!(vnv.vns)
    empty!(vnv.ranges)
    empty!(vnv.vals)
    empty!(vnv.transforms)
    return nothing
end
BangBang.empty!!(vnv::VarNameVector) = empty!(vnv)

# TODO: Re-use some of the show functionality from Base?
function Base.show(io::IO, vnv::VarNameVector)
    print(io, "[")
    for (i, vn) in enumerate(vnv.vns)
        if i > 1
            print(io, ", ")
        end
        print(io, vn, " = ", vnv[vn])
    end
end

# Typed version.
function group_by_symbol(vnv::VarNameVector)
    # Group varnames in `vnv` by the symbol.
    d = OrderedDict{Symbol,Vector{VarName}}()
    for vn in vnv.vns
        push!(get!(d, getsym(vn), Vector{VarName}()), vn)
    end

    # Create a `NamedTuple` from the grouped varnames.
    nt_vals = map(values(d)) do vns
        # TODO: Do we need to specialize the inputs here?
        VarNameVector(
            map(identity, vns),
            map(Base.Fix1(getindex, vnv), vns),
            map(Base.Fix1(gettransform, vnv), vns),
        )
    end

    return NamedTuple{Tuple(keys(d))}(nt_vals)
end
