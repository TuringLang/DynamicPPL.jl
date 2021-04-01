"""
    inargnames(varname::VarName, model::Model)

Statically check whether the variable of name `varname` is an argument of the `model`.

Possibly existing indices of `varname` are neglected.
"""
@generated function inargnames(::VarName{s}, ::Model{_F, argnames}) where {s, argnames, _F}
    return s in argnames
end


"""
    inmissings(varname::VarName, model::Model)

Statically check whether the variable of name `varname` is a statically declared unobserved variable
of the `model`.

Possibly existing indices of `varname` are neglected.
"""
@generated function inmissings(::VarName{s}, ::Model{_F, _a, _T, missings}) where {s, missings, _F, _a, _T}
    return s in missings
end
