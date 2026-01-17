"""
    inargnames(varname::VarName, model::Model)

Statically check whether the variable of name `varname` is an argument of the `model`.

Possibly existing indices of `varname` are neglected.
"""
@generated function inargnames(
    ::VarName{s}, ::Model{_F,argnames,defaultnames}
) where {s,argnames,defaultnames,_F}
    return s in argnames || s in defaultnames
end

"""
    inmissings(varname::VarName, model::Model)

Statically check whether the variable of name `varname` is a statically declared unobserved variable
of the `model`.

Possibly existing indices of `varname` are neglected.
"""
@generated function inmissings(
    ::VarName{s}, ::Model{_F,_a,_T,missings}
) where {s,missings,_F,_a,_T}
    return s in missings
end

# TODO(mhauru) This should probably be Base.size(::VarName) in AbstractPPL.
"""
    varnamesize(vn::VarName)

Return the size of the object referenced by this VarName.

```jldoctest
julia> varnamesize(@varname(a))
()

julia> varnamesize(@varname(b[1:3, 2]))
(3,)

julia> varnamesize(@varname(c.d[4].e[3, 2:5, 2, 1:4, 1]))
(4, 4)
"""
function varnamesize(vn::VarName)
    l = AbstractPPL._last(vn.optic)
    if l isa Accessors.IndexLens
        return reduce((x, y) -> tuple(x..., y...), map(size, l.indices))
    else
        return ()
    end
end
