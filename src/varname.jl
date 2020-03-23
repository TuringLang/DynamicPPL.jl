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
    inspace(vn::Union{VarName,Symbol}, space::Tuple)

Check whether `vn`'s symbol is in `space`.
"""
inspace(::VarName, ::Tuple{}) = true
inspace(vn::VarName, space::Tuple) = _inspace(vn, space)
inspace(vn::Union{Symbol, Expr}, space::Tuple) = vn in space

_inspace(vn::VarName, ::Tuple{}) = false
_inspace(vn::VarName{s}, space::Tuple{Symbol, Vararg}) where {s} =
    s == first(space) || _inspace(vn, Base.tail(space))
function _inspace(vn::VarName{s}, space::Tuple{Expr, Vararg}) where {s}
    expr = first(space)
    Meta.isexpr(expr, :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    ip = expr.args[1] == s && isprefix(tuple(expr.args[2:end]), vn.indexing)
    return ip || _inspace(vn, Base.tail(space))
end

isprefix(::Tuple{}, ::Tuple{}) = true
isprefix(t::Tuple{}, u::Tuple) = true
isprefix(::Tuple, u::Tuple{}) = false
isprefix(t::Tuple{<:Any, Vararg}, u::Tuple{<:Any, Vararg}) =
    (first(t) == first(u)) && isprefix(Base.tail(t), Base.tail(u))


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
