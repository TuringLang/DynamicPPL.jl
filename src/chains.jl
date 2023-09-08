"""
    supports_varname_indexing(chain::AbstractChains)

Return `true` if `chain` supports indexing using `VarName` in place of the
variable name index.
"""
supports_varname_indexing(::AbstractChains) = false

"""
    getindex_varname(chain::AbstractChains, sample_idx, varname::VarName, chain_idx)

Return the value of `varname` in `chain` at `sample_idx` and `chain_idx`.

Whether this method is implemented for `chains` is indicated by [`supports_varname_indexing`](@ref).
"""
function getindex_varname end

"""
    varnames(chains::AbstractChains)

Return an iterator over the varnames present in `chains`.

Whether this method is implemented for `chains` is indicated by [`supports_varname_indexing`](@ref).
"""
function varnames end
