# TODO(mhauru) This module should probably be moved to AbstractPPL.
module VarNamedTuples

using AbstractPPL
using AbstractPPL: AbstractPPL
using Distributions: Distributions, Distribution, LKJCholesky
using BangBang
using DynamicPPL: DynamicPPL

export VarNamedTuple,
    vnt_size,
    map_pairs!!,
    map_values!!,
    apply!!,
    templated_setindex!!,
    densify!!,
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

"""
    abstract type SetPermissions end

An abstract type, passed to `_setindex_optic!!`, that controls whether it is allowed to
create a new value or overwrite an old value in a VarNamedTuple.
"""
abstract type SetPermissions end
"""
    AllowAll <: SetPermissions

Allow the creation of new values, and also allow the overwriting of old values.
"""
struct AllowAll <: SetPermissions end
"""
    MustOverwrite(target_vn) <: SetPermissions

Disallow the creation of a new value for `target_vn`, but allow the overwriting of an old
value.

This subtype is a performance optimisation: if it is set to `MustOverwrite`, the function
can assume that the key being set already exists in `collection`. This allows skipping some
code paths, which may have a minor benefit at runtime, but more importantly, allows for
better constant propagation and type stability at compile time.

`permissions` being set to `MustOverwrite` does _not_ guarantee that no new keys will be
added. It only gives the implementation of `_setindex_optic!!` the permission to assume that
the key already exists. Setting it to `MustOverwrite` should be done only when the caller is
sure that the key already exists; anything else is undefined behaviour.
"""
struct MustOverwrite{V<:VarName} <: SetPermissions
    target_vn::V
end
struct MustOverwriteError{V<:VarName} <: Exception
    target_vn::V
    function MustOverwriteError(perm::MustOverwrite)
        return new{typeof(perm.target_vn)}(perm.target_vn)
    end
end
function Base.showerror(io::IO, e::MustOverwriteError)
    # Key doesn't exist yet, so we tried to create it, but we aren't allowed to.
    return print(
        io,
        "MustOverwriteError: Attempted to set a value for $(e.target_vn), but" *
        " `permissions=MustOverwrite` was specified. If you did not attempt" *
        " to call this function yourself, this likely indicates a bug in" *
        " DynamicPPL. Please file an issue at" *
        " https://github.com/TuringLang/DynamicPPL.jl/issues.",
    )
end

"""
    MustNotOverwrite <: SetPermissions

Allow the creation of new values, but disallow the overwriting of old values.
"""
struct MustNotOverwrite{V<:VarName} <: SetPermissions
    target_vn::V
end
struct MustNotOverwriteError{V<:VarName} <: Exception
    target_vn::V
    function MustNotOverwriteError(perm::MustNotOverwrite)
        return new{typeof(perm.target_vn)}(perm.target_vn)
    end
end
function Base.showerror(io::IO, e::MustNotOverwriteError)
    # Key exists already, and we tried to overwrite it, but we aren't allowed to.
    return print(
        io,
        "MustNotOverwriteError: Attempted to set a value for $(e.target_vn), but a value already" *
        " existed. This indicates that a value is being set twice (e.g. if" *
        " the same variable occurs in a model twice).",
    )
end

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
@generated function Base.NamedTuple(vnt::VarNamedTuple{names,vals}) where {names,vals}
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

# TODO(mhauru) The following methods mimic the structure of those in
# AbstractPPLDistributionsExtension, and fall back on converting any PartialArrays to
# dictionaries, and calling the AbstractPPL methods. We should eventually make
# implementations of these directly for PartialArray, and maybe move these methods
# elsewhere. Better yet, once we no longer store VarName values in Dictionaries anywhere,
# and FlexiChains takes over from MCMCChains, this could hopefully all be removed.
#
# NOTE(penelopeysm) See https://github.com/TuringLang/DynamicPPL.jl/issues/1262 for
# explanation of these methods.
AbstractPPL.hasvalue(vnt::VarNamedTuple, vn::VarName) = haskey(vnt, vn)
AbstractPPL.getvalue(vnt::VarNamedTuple, vn::VarName) = vnt[vn]
AbstractPPL.hasvalue(vals::VarNamedTuple, vn::VarName, ::Distribution) = haskey(vals, vn)
AbstractPPL.getvalue(vals::VarNamedTuple, vn::VarName, ::Distribution) = vals[vn]
function AbstractPPL.hasvalue(vnt::VarNamedTuple, vn::VarName, dist::LKJCholesky)
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

function AbstractPPL.getvalue(vnt::VarNamedTuple, vn::VarName, dist::LKJCholesky)
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
