"""
```
struct VarName{sym, T<:Tuple}
    indexing::T
end
```

A variable identifier. Every variable has a symbol `sym` and indices `indexing` in the format
returned by [`@vinds`](@ref).  The Julia variable in the model corresponding to `sym` can refer to a
single value or to a hierarchical array structure of univariate, multivariate or matrix
variables. `indexing` stores the indices that can access the random variable from the Julia
variable.

Examples:

- `x[1] ~ Normal()` will generate a `VarName` with `sym == :x` and `indexing == "((1,))"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName` with `sym == :x` and
 `indexing == ((Colon(), 1))"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName` with `sym == :x` and
 `indexing == ((Colon(), 1), (2,))`.
"""
struct VarName{sym, T<:Tuple}
    indexing::T
end

VarName(sym::Symbol, indexing::T = ()) where {T} = VarName{sym, T}(indexing)

"""
    VarName(vn::VarName, indexing)

Return a copy of `vn` with a new index `indexing`.
"""
function VarName(vn::VarName{sym}, indexing::T = ()) where {sym, T}
    return VarName{sym, T}(indexing)
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
Base.:(==)(x::VarName{S}, y::VarName{T}) where {S, T} = S == T && getindexing(x) == getindexing(y)

function Base.show(io::IO, vn::VarName)
    print(io, getsym(vn))
    for indices in getindexing(vn)
        print(io, "[")
        join(io, indices, ", ")
        print(io, "]")
    end
end


"""
    Symbol(vn::VarName)

Return a `Symbol` represenation of the variable identifier `VarName`.
"""
Base.Symbol(vn::VarName) = Symbol(string(vn))  # simplified symbol


"""
    inspace(vn::Union{VarName, Symbol, Expr}, space::Tuple)

Check whether `vn`'s variable symbol is in `space`.
"""
inspace(::Union{VarName, Symbol, Expr}, ::Tuple{}) = true
inspace(vn::Union{VarName, Symbol, Expr}, space::Tuple) = any(_ismatch(vn, s) for s in space)

_ismatch(vn, s) = (_name(vn) == _name(s)) && _isprefix(_indexing(s), _indexing(vn))

_isprefix(::Tuple{}, ::Tuple{}) = true
_isprefix(::Tuple{}, ::Tuple) = true
_isprefix(::Tuple, ::Tuple{}) = false
_isprefix(t::Tuple, u::Tuple) = _subsumes(first(t), first(u)) && _isprefix(Base.tail(t), Base.tail(u))

_subsumes(i::Union{Int, UnitRange{Int}}, j::Union{Int, UnitRange{Int}}) = issubset(i, j)
_subsumes(i::Union{Int, UnitRange{Int}, Colon}, j::Colon) = true
_subsumes(i::Colon, ::Union{Int, UnitRange{Int}}) = false

_name(vn::Symbol) = vn
_name(vn::VarName) = getsym(vn)
function _name(vn::Expr)
    if Meta.isexpr(vn, :ref)
        _name(vn.args[1])
    else
        throw("VarName: Mis-formed variable name $(vn)!")
    end
end

_indexing(vn::Symbol) = ()
_indexing(vn::VarName) = getindexing(vn)
function _indexing(vn::Expr)
    if Meta.isexpr(vn, :ref)
        init = _indexing(vn.args[1])
        last = Tuple(vn.args[2:end])
        return (init..., last)
    else
        throw("VarName: Mis-formed variable name $(vn)!")
    end
end


"""
    @varname(expr)

A macro that returns an instance of `VarName` given the symbol or expression of a Julia variable, 
e.g. `@varname x[1,2][1+5][45][3]` returns `VarName{:x}(((1, 2), (6,), (45,), (3,)))`.
"""
macro varname(expr::Union{Expr, Symbol})
    expr |> varname |> esc
end

varname(expr::Symbol) = VarName(expr)
function varname(expr::Expr)
    if Meta.isexpr(expr, :ref)
        sym, inds = vsym(expr), vinds(expr)
        return :(DynamicPPL.VarName($sym, $inds))
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
    expr |> vsym
end

vsym(expr::Symbol) = QuoteNode(expr)
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
"""
macro vinds(expr::Union{Expr, Symbol})
    expr |> vinds |> esc
end

vinds(expr::Symbol) = Expr(:tuple)
function vinds(expr::Expr)
    if Meta.isexpr(expr, :ref)
        last = Expr(:tuple, expr.args[2:end]...)
        init = vinds(expr.args[1]).args
        return Expr(:tuple, init..., last)
    else
        throw("VarName: Mis-formed variable name $(expr)!")
    end
end
