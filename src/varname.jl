"""
    subsumes_string(u::String, v::String[, u_indexing])

Check whether stringified variable name `v` describes a sub-range of stringified variable `u`.

This is a very restricted version `subumes(u::VarName, v::VarName)` only really supporting:
- Scalar: `x` subsumes `x[1, 2]`, `x[1, 2]` subsumes `x[1, 2][3]`, etc.

## Note
- To get same matching capabilities as `AbstractPPL.subumes(u::VarName, v::VarName)`
  for strings, one can always do `eval(varname(Meta.parse(u))` to get `VarName` of `u`,
  and similarly to `v`. But this is slow.
"""
function subsumes_string(u::String, v::String, u_indexing=u * "[")
    return u == v || startswith(v, u_indexing)
end

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

function remove_trailing_index(vn::VarName{sym,Optic}) where {sym,Optic}
    return if Optic === typeof(identity)
        vn
    elseif Optic <: Accessors.IndexLens
        VarName{sym}()
    else
        AbstractPPL.prefix(
            remove_trailing_index(AbstractPPL.unprefix(vn, VarName{sym}())), VarName{sym}()
        )
    end
end

function split_trailing_index(vn::VarName{sym,Optic}) where {sym,Optic}
    return if Optic === typeof(identity)
        (vn, identity)
    elseif Optic <: Accessors.IndexLens
        (VarName{sym}(), getoptic(vn))
    else
        (pref, index) = split_trailing_index(AbstractPPL.unprefix(vn, VarName{sym}()))
        (AbstractPPL.prefix(pref, VarName{sym}()), index)
    end
end
