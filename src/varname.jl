"""
```
struct VarName{sym}
    indexing  ::    String
end
```

A variable identifier. Every variable has a symbol `sym` and `indices `indexing`. 
The Julia variable in the model corresponding to `sym` can refer to a single value or 
to a hierarchical array structure of univariate, multivariate or matrix variables. `indexing` stores the indices that can access the random variable from the Julia 
variable.

Examples:

- `x[1] ~ Normal()` will generate a `VarName` with `sym == :x` and `indexing == "[1]"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName` with `sym == :x` and
 `indexing == "[Colon(),1]"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName` with `sym == :x` and
 `indexing == "[Colon(),1][2]"`.
"""
struct VarName{sym}
    indexing::String
end



"""
    @varname(var)

A macro that returns an instance of `VarName` given the symbol or expression of a Julia variable, e.g. `@varname x[1,2][1+5][45][3]` returns `VarName{:x}("[1,2][6][45][3]")`.
"""
macro varname(expr::Union{Expr, Symbol})
    expr |> varname |> esc
end
function varname(expr)
    ex = deepcopy(expr)
    ex isa Symbol && return :($(DynamicPPL.VarName){$(QuoteNode(ex))}(""))
    ex.head == :ref || throw("VarName: Mis-formed variable name $(expr)!")
    inds = :(())
    while ex.head == :ref
        if length(ex.args) >= 2
            strs = map(x -> :($x === (:) ? "Colon()" : string($x)), ex.args[2:end])
            pushfirst!(inds.args, :("[" * join($(Expr(:vect, strs...)), ",") * "]"))
        end
        ex = ex.args[1]
        ex isa Symbol && return :($(DynamicPPL.VarName){$(QuoteNode(ex))}(foldl(*, $inds, init = "")))
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end

macro vsym(expr::Union{Expr, Symbol})
    return :(QuoteNode($(vsym(expr))))
end

"""
    vsym(expr::Union{Expr, Symbol})

Returns the variable symbol given the input variable expression `expr`. For example, if the input `expr = :(x[1])`, the output is `:x`.
"""
function vsym(expr::Expr)
    (expr.head == :ref && !isempty(expr.args)) || throw("VarName: Mis-formed variable name $(expr)!")
    vsym(expr.args[1])
end
vsym(sym::Symbol) = sym

"""
    @vinds(expr)

Returns a tuple of tuples of the indices in `expr`. For example, `@vinds x[1,:][2]` returns 
`((1, Colon()), (2,))`.
"""
macro vinds(expr::Union{Expr, Symbol})
    expr |> vinds |> esc
end
function vinds(expr::Union{Expr, Symbol})
    ex = deepcopy(expr)
    inds = Expr(:tuple)
    (ex isa Symbol) && return inds
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    while ex.head == :ref
        pushfirst!(inds.args, Expr(:tuple, ex.args[2:end]...))
        ex = ex.args[1]
        isa(ex, Symbol) && return inds
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end

"""
    split_var_str(var_str, inds_as = Vector)

This function splits a variable string, e.g. `"x[1:3,1:2][3,2]"` to the variable's symbol `"x"` and the indexing `"[1:3,1:2][3,2]"`. If `inds_as = String`, the indices are returned as a string, e.g. `"[1:3,1:2][3,2]"`. If `inds_as = Vector`, the indices are returned as a vector of vectors of strings, e.g. `[["1:3", "1:2"], ["3", "2"]]`.
"""
function split_var_str(var_str, inds_as = Vector)
    ind = findfirst(c -> c == '[', var_str)
    if inds_as === String
        if ind === nothing
            return var_str, ""
        else
            return var_str[1:ind-1], var_str[ind:end]
        end
    end
    @assert inds_as === Vector
    inds = Vector{String}[]
    if ind === nothing
        return var_str, inds
    end
    sym = var_str[1:ind-1]
    ind = length(sym)
    while ind < length(var_str)
        ind += 1
        @assert var_str[ind] == '['
        push!(inds, String[])
        while var_str[ind] != ']'
            ind += 1
            if var_str[ind] == '['
                ind2 = findnext(c -> c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2]))
                ind = ind2+1
            else
                ind2 = findnext(c -> c == ',' || c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2-1]))
                ind = ind2
            end
        end
    end
    return sym, inds
end


@generated function inargnames(::VarName{s}, ::Model{_F, argnames}) where {s, argnames, _F}
    return s in argnames
end

@generated function inmissings(::VarName{s}, ::Model{_F, _a, _T, missings}) where {s, missings, _F, _a, _T}
    return s in missings
end
