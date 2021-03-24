"""
    inargnames(varname, model)

Statically check whether the `getsym(varname)` is among the model's argument names.
"""
@generated function inargnames(::VarName{s}, ::Model{_F, argnames}) where {s, argnames, _F}
    return s in argnames
end


"""
    inmissings(varname, model)

Statically check whether the `getsym(varname)` is among the model's missing variable names.
"""
@generated function inmissings(::VarName{s}, ::Model{_F, _a, _T, missings}) where {s, missings, _F, _a, _T}
    return s in missings
end
