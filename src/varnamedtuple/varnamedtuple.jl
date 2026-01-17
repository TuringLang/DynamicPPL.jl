# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using AbstractPPL: AbstractPPL
using Distributions: Distributions, Distribution
using BangBang
using DynamicPPL: DynamicPPL

export VarNamedTuple, vnt_size, map_pairs!!, map_values!!, apply!!

# Currently, keyword arguments are not supported in getindex/_setindex!!. That is because
# `PartialArray` under the hood is backed by `Base.Array`. Thus, if `kw` is not empty, we
# will just error here. However, in principle, this can be expanded by allowing PartialArray
# to wrap generic array types (the 'shadow array' mechanism); see
# https://github.com/TuringLang/DynamicPPL.jl/issues/1194.
function error_kw_indices()
    throw(ArgumentError("Keyword indices in VarNames are not yet supported in DynamicPPL."))
end

include("partial_array.jl")
# The actual definition of the VarNamedTuple struct. Yeah, it needs a better name, I'll sort
# that out.
include("actual.jl")
include("getset.jl")
include("map.jl")

function AbstractPPL.hasvalue(vnt::VarNamedTuple, vn::VarName)
    return haskey(vnt, vn)
end

function AbstractPPL.getvalue(vnt::VarNamedTuple, vn::VarName)
    return getindex(vnt, vn)
end

# TODO(mhauru) The following methods mimic the structure of those in
# AbstractPPLDistributionsExtension, and fall back on converting any PartialArrays to
# dictionaries, and calling the AbstractPPL methods. We should eventually make
# implementations of these directly for PartialArray, and maybe move these methods
# elsewhere. Better yet, once we no longer store VarName values in Dictionaries anywhere,
# and FlexiChains takes over from MCMCChains, this could hopefully all be removed.

# The only case where the Distribution argument makes a difference is if the distribution
# is multivariate and the values are stored in a PartialArray.

function AbstractPPL.hasvalue(
    vnt::VarNamedTuple, vn::VarName, ::Distributions.UnivariateDistribution
)
    return AbstractPPL.hasvalue(vnt, vn)
end

function AbstractPPL.getvalue(
    vnt::VarNamedTuple, vn::VarName, ::Distributions.UnivariateDistribution
)
    return AbstractPPL.getvalue(vnt, vn)
end

function AbstractPPL.hasvalue(vals::VarNamedTuple, vn::VarName, dist::Distribution)
    @warn "`hasvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `hasvalue(vals, vn)`."
    return AbstractPPL.hasvalue(vals, vn)
end

function AbstractPPL.getvalue(vals::VarNamedTuple, vn::VarName, dist::Distribution)
    @warn "`getvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `getvalue(vals, vn)`."
    return AbstractPPL.getvalue(vals, vn)
end

const MV_DIST_TYPES = Union{
    Distributions.MultivariateDistribution,
    Distributions.MatrixDistribution,
    Distributions.LKJCholesky,
}

function AbstractPPL.hasvalue(vnt::VarNamedTuple, vn::VarName, dist::MV_DIST_TYPES)
    if !haskey(vnt, vn)
        # Can't even find the parent VarName, there is no hope.
        return false
    end
    # Note that _getindex_optic, rather than Base.getindex, skips the need to denseify
    # PartialArrays.
    val = _getindex_optic(vnt, vn)
    if !(val isa VarNamedTuple || val isa PartialArray)
        # There is _a_ value. Whether it's the right kind, we do not know, but returning
        # true is no worse than `hasvalue` returning true for e.g. UnivariateDistributions
        # whenever there is at least some value.
        return true
    end
    # Convert to VarName-keyed Dict.
    et = val isa VarNamedTuple ? Any : eltype(val)
    dval = Dict{VarName,et}()
    for k in keys(val)
        # VarNamedTuples have VarNames as keys, PartialArrays have Index optics.
        subvn = val isa VarNamedTuple ? prefix(k, vn) : AbstractPPL.append_optic(vn, k)
        dval[subvn] = getindex(val, k)
    end
    return AbstractPPL.hasvalue(dval, vn, dist)
end

function AbstractPPL.getvalue(vnt::VarNamedTuple, vn::VarName, dist::MV_DIST_TYPES)
    # Note that _getindex_optic, rather than Base.getindex, skips the need to denseify
    # PartialArrays.
    val = _getindex_optic(vnt, vn)
    if !(val isa VarNamedTuple || val isa PartialArray)
        return val
    end
    # Convert to VarName-keyed Dict.
    et = val isa VarNamedTuple ? Any : eltype(val)
    dval = Dict{VarName,et}()
    for k in keys(val)
        # VarNamedTuples have VarNames as keys, PartialArrays have Index optics.
        subvn = val isa VarNamedTuple ? prefix(k, vn) : AbstractPPL.append_optic(vn, k)
        dval[subvn] = getindex(val, k)
    end
    return AbstractPPL.getvalue(dval, vn, dist)
end

end
