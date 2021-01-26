"""
    VarName(sym[, indexing=()])

A variable identifier for a symbol `sym` and indices `indexing` in the format
returned by [`@vinds`](@ref).  

The Julia variable in the model corresponding to `sym` can refer to a single value or to a
hierarchical array structure of univariate, multivariate or matrix variables. The field `indexing`
stores the indices requires to access the random variable from the Julia variable indicated by `sym`
as a tuple of tuples. Each element of the tuple thereby contains the indices of one indexing
operation.

`VarName`s can be manually constructed using the `VarName(sym, indexing)` constructor, or from an
indexing expression through the [`@varname`](@ref) convenience macro.

# Examples

```jldoctest
julia> vn = @varname(x[:, 1][1+1])
x[Colon(),1][2]

julia> vn.indexing
((Colon(), 1), (2,))
```
"""
struct VarName{sym, T<:Tuple}
    indexing::T
end

VarName(sym::Symbol, indexing::Tuple = ()) = VarName{sym, typeof(indexing)}(indexing)

"""
    VarName(vn::VarName[, indexing=()])

Return a copy of `vn` with a new index `indexing`.
"""
function VarName(vn::VarName, indexing::Tuple = ())
    return VarName{getsym(vn), typeof(indexing)}(indexing)
end


"""
    getsym(vn::VarName)

Return the symbol of the Julia variable used to generate `vn`.
"""
getsym(vn::VarName{sym}) where sym = sym


"""
    getindexing(vn::VarName)

Return the indexing tuple of the Julia variable used to generate `vn`.
"""
getindexing(vn::VarName) = vn.indexing


Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getindexing(vn)), h)
Base.:(==)(x::VarName, y::VarName) = getsym(x) == getsym(y) && getindexing(x) == getindexing(y)

function Base.show(io::IO, vn::VarName)
    print(io, getsym(vn))
    for indices in getindexing(vn)
        print(io, "[")
        join(io, indices, ",")
        print(io, "]")
    end
end


"""
    Symbol(vn::VarName)

Return a `Symbol` represenation of the variable identifier `VarName`.
"""
Base.Symbol(vn::VarName) = Symbol(string(vn))  # simplified symbol


"""
    inspace(vn::Union{VarName, Symbol}, space::Tuple)

Check whether `vn`'s variable symbol is in `space`.
"""
inspace(vn, space::Tuple{}) = true # empty space is treated as universal set
inspace(vn, space::Tuple) = vn in space
inspace(vn::VarName, space::Tuple{}) = true
inspace(vn::VarName, space::Tuple) = any(_in(vn, s) for s in space)

_in(vn::VarName, s::Symbol) = getsym(vn) == s
_in(vn::VarName, s::VarName) = subsumes(s, vn)


"""
    subsumes(u::VarName, v::VarName)

Check whether the variable name `v` describes a sub-range of the variable `u`.  Supported
indexing:

- Scalar: `x` subsumes `x[1, 2]`, `x[1, 2]` subsumes `x[1, 2][3]`, etc.
- Array of scalar: `x[[1, 2], 3]` subsumes `x[1, 3]`, `x[1:3]` subsumes `x[2][1]`, etc.
  (basically everything that fulfills `issubset`).
- Slices: `x[2, :]` subsumes `x[2, 10][1]`, etc.

Currently _not_ supported are: 

- Boolean indexing, literal `CartesianIndex` (these could be added, though)
- Linear indexing of multidimensional arrays: `x[4]` does not subsume `x[2, 2]` for `x` a matrix
- Trailing ones: `x[2, 1]` does not subsume `x[2]` for `x` a vector
"""
function subsumes(u::VarName, v::VarName)
    return getsym(u) == getsym(v) && subsumes(u.indexing, v.indexing)
end

subsumes(::Tuple{}, ::Tuple{}) = true  # x subsumes x
subsumes(::Tuple{}, ::Tuple) = true    # x subsumes x[1]
subsumes(::Tuple, ::Tuple{}) = false   # x[1] does not subsume x
function subsumes(t::Tuple, u::Tuple)  # does x[i]... subsume x[j]...?
    return _issubindex(first(t), first(u)) && subsumes(Base.tail(t), Base.tail(u))
end

const AnyIndex = Union{Int, AbstractVector{Int}, Colon} 
_issubindex_(::Tuple{Vararg{AnyIndex}}, ::Tuple{Vararg{AnyIndex}}) = false
function _issubindex(t::NTuple{N, AnyIndex}, u::NTuple{N, AnyIndex}) where {N}
    return all(_issubrange(j, i) for (i, j) in zip(t, u))
end

const ConcreteIndex = Union{Int, AbstractVector{Int}} # this include all kinds of ranges
"""Determine whether indices `i` are contained in `j`, treating `:` as universal set."""
_issubrange(i::ConcreteIndex, j::ConcreteIndex) = issubset(i, j)
_issubrange(i::Union{ConcreteIndex, Colon}, j::Colon) = true
_issubrange(i::Colon, j::ConcreteIndex) = true



"""
    @varname(expr)

A macro that returns an instance of `VarName` given a symbol or indexing expression.  The `sym`
value is taken from the actual variable name, and the index values are put appropriately into the
constructor (and resolved at runtime).

# Examples

```jldoctest
julia> @varname(x).indexing
()

julia> @varname(x[1]).indexing
((1,),)

julia> @varname(x[:, 1]).indexing
((Colon(), 1),)

julia> @varname(x[:, 1][2]).indexing
((Colon(), 1), (2,))

julia> @varname(x[1,2][1+5][45][3]).indexing
((1, 2), (6,), (45,), (3,))
```

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro varname(expr::Union{Expr, Symbol})
    return esc(varname(expr))
end

varname(expr::Symbol) = VarName(expr)
function varname(expr::Expr)
    if Meta.isexpr(expr, :ref)
        sym, inds = vsym(expr), vinds(expr)
        return :($(DynamicPPL.VarName)($(QuoteNode(sym)), $inds))
    else
        throw("VarName: Mis-formed variable name $(expr)!")
    end
end


"""
    @vsym(expr)

A macro that returns the variable symbol given the input variable expression `expr`. 
For example, `@vsym x[1]` returns `:x`.
"""
macro vsym(expr::Union{Expr, Symbol})
    return QuoteNode(vsym(expr))
end

vsym(expr::Symbol) = expr
function vsym(expr::Expr)
    if Meta.isexpr(expr, :ref)
        return vsym(expr.args[1])
    else
        throw("VarName: Mis-formed variable name $(expr)!")
    end
end

"""
    @vinds(expr)

Returns a tuple of tuples of the indices in `expr`. For example, `@vinds x[1, :][2]` returns
`((1, Colon()), (2,))`.

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro vinds(expr::Union{Expr, Symbol})
    return esc(vinds(expr))
end

vinds(expr::Symbol) = Expr(:tuple)
function vinds(expr::Expr)
    if Meta.isexpr(expr, :ref)
        ex = copy(expr)
        @static if VERSION < v"1.5.0-DEV.666"
            Base.replace_ref_end!(ex)
        else
            Base.replace_ref_begin_end!(ex)
        end
        last = Expr(:tuple, ex.args[2:end]...)
        init = vinds(ex.args[1]).args
        return Expr(:tuple, init..., last)
    else
        throw("VarName: Mis-formed variable name $(expr)!")
    end
end

@generated function inargnames(::VarName{s}, ::Model{_F, argnames}) where {s, argnames, _F}
    return s in argnames
end

@generated function inmissings(::VarName{s}, ::Model{_F, _a, _T, missings}) where {s, missings, _F, _a, _T}
    return s in missings
end
