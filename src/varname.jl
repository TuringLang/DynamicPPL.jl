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

