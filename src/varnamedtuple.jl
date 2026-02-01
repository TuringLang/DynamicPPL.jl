# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using AbstractPPL: AbstractPPL
using Distributions: Distributions, Distribution
using BangBang
using DynamicPPL: DynamicPPL

export VarNamedTuple,
    vnt_size,
    map_pairs!!,
    map_values!!,
    apply!!,
    templated_setindex!!,
    NoTemplate,
    SkipTemplate,
    @vnt

"""
    NoTemplate

A singleton struct representing the fact that there is no template for a top-level variable.
When `NoTemplate` is used, several things happen:

 - If you attempt to call `make_leaf` with an Index optic, this will error.
 - When recursing into substructures, `NoTemplate` will be propagated.

Collectively this means that you can only set values for variables that only have Property
optics, unless a template is provided.

It might seem more idiomatic to use `nothing` or `missing` for this. However, this causes a
bug with BangBang.setindex!!: https://github.com/JuliaFolds2/BangBang.jl/issues/43 so we use
a dedicated struct instead.
"""
struct NoTemplate end

"""
    SkipTemplate{N}(value)

A struct representing the fact that `value` is the template for the variable `N` levels down
from the top-level variable. In other words, SkipTemplate{0}(value) is equivalent to just
`value`, and SkipTemplate{1}(value) means that `value` is the template for `a` when setting
the variable `@varname(x.a)`.
"""
struct SkipTemplate{N,V}
    value::V
end
SkipTemplate{N}(v) where {N} = SkipTemplate{N,typeof(v)}(v)
SkipTemplate{0}(v) = v
SkipTemplate{N}(::NoTemplate) where {N} = NoTemplate()
SkipTemplate{0}(::NoTemplate) = NoTemplate()
# Decrease the skip level: used when recursing inside make_leaf.
function decrease_skip(st::SkipTemplate{N}) where {N}
    return SkipTemplate{N - 1}(st.value)
end
# Increase the skip level: used when applying a PrefixContext (the template
# must be wrapped by the appropriate number of SkipTemplates).
SkipTemplate{N}(v::SkipTemplate{M}) where {N,M} = SkipTemplate{N + M}(v.value)
SkipTemplate{0}(v::SkipTemplate{M}) where {M} = SkipTemplate{M}(v.value)

include("varnamedtuple/partial_array.jl")
include("varnamedtuple/vnt.jl")
include("varnamedtuple/getset.jl")
include("varnamedtuple/map.jl")
include("varnamedtuple/show.jl")
include("varnamedtuple/macro.jl")

"""
    NamedTuple(vnt::VarNamedTuple)

Convert a `VarNamedTuple` to a standard `NamedTuple`, provided all keys in the
`VarNamedTuple` are `VarName`s with top-level symbols. If any key is a `VarName`
with a non-identity optic (e.g., `@varname(x.a)` or `@varname(x[1])`), this will
throw an `ArgumentError`.

# Examples

```jldoctest
julia> using DynamicPPL, BangBang

julia> vnt = VarNamedTuple(); vnt = setindex!!(vnt, 10, @varname(x))
VarNamedTuple
└─ x => 10

julia> NamedTuple(vnt)
(x = 10,)

julia> vnt2 = setindex!!(vnt, 20, @varname(y.a))
VarNamedTuple
├─ x => 10
└─ y => VarNamedTuple
        └─ a => 20

julia> NamedTuple(vnt2)
ERROR: ArgumentError: Cannot convert VarNamedTuple containing non-identity VarNames to NamedTuple. To create a NamedTuple, all keys in the VarNamedTuple must be top-level symbols.
[...]
```
"""
@generated function NamedTuple(vnt::VarNamedTuple{names,vals}) where {names,vals}
    if isempty(names)
        return :(NamedTuple())
    end
    nt = Expr(:tuple)
    for (n, v) in zip(names, vals.parameters)
        if v <: VarNamedTuple ||
            VarNamedTuple <: v ||
            v <: PartialArray ||
            PartialArray <: v
            throw(
                ArgumentError(
                    "Cannot convert VarNamedTuple containing non-identity VarNames to NamedTuple. To create a NamedTuple, all keys in the VarNamedTuple must be top-level symbols.",
                ),
            )
        end
        push!(nt.args, :($n = vnt.data.$n))
    end
    return nt
end

function AbstractPPL.hasvalue(vnt::VarNamedTuple, vn::VarName)
    return _haskey_optic(vnt, vn)
end

function AbstractPPL.getvalue(vnt::VarNamedTuple, vn::VarName)
    return _getindex_optic(vnt, vn)
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
        dval[subvn] = _getindex_optic(val, k)
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
        dval[subvn] = _getindex_optic(val, k)
    end
    return AbstractPPL.getvalue(dval, vn, dist)
end

end
