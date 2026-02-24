"""
    varnames_in_chain(model:::Model, chain)
    varnames_in_chain(varinfo::VarInfo, chain)

Return `true` if all variable names in `model`/`varinfo` are in `chain`.
"""
varnames_in_chain(model::Model, chain) = varnames_in_chain(VarInfo(model), chain)
function varnames_in_chain(varinfo::VarInfo, chain)
    return all(vn -> varname_in_chain(varinfo, vn, chain, 1, 1), keys(varinfo))
end

"""
    varnames_in_chain!(model::Model, chain, out)
    varnames_in_chain!(varinfo::VarInfo, chain, out)

Return `out` with `true` for all variable names in `model` that are in `chain`.
"""
function varnames_in_chain!(model::Model, chain, out)
    return varnames_in_chain!(VarInfo(model), chain, out)
end
function varnames_in_chain!(varinfo::VarInfo, chain, out)
    for vn in keys(varinfo)
        varname_in_chain!(varinfo, vn, chain, 1, 1, out)
    end

    return out
end

"""
    varname_in_chain(model::Model, vn, chain, chain_idx, iteration_idx)
    varname_in_chain(varinfo::VarInfo, vn, chain, chain_idx, iteration_idx)

Return `true` if `vn` is in `chain` at `chain_idx` and `iteration_idx`.
"""
function varname_in_chain(model::Model, vn, chain, chain_idx, iteration_idx)
    return varname_in_chain(VarInfo(model), vn, chain, chain_idx, iteration_idx)
end

function varname_in_chain(varinfo::AbstractVarInfo, vn, chain, chain_idx, iteration_idx)
    !haskey(varinfo, vn) && return false
    return varname_in_chain(varinfo[vn], vn, chain, chain_idx, iteration_idx)
end

function varname_in_chain(x, vn, chain, chain_idx, iteration_idx)
    out = OrderedDict{VarName,Bool}()
    varname_in_chain!(x, vn, chain, chain_idx, iteration_idx, out)
    return all(values(out))
end

"""
    varname_in_chain!(model::Model, vn, chain, chain_idx, iteration_idx, out)
    varname_in_chain!(varinfo::VarInfo, vn, chain, chain_idx, iteration_idx, out)

Return a dictionary mapping the varname `vn` to `true` if `vn` is in `chain` at
`chain_idx` and `iteration_idx`.

If `chain_idx` and `iteration_idx` are not provided, then they default to `1`.

This differs from [`varname_in_chain`](@ref) in that it returns a dictionary
rather than a single boolean. This can be quite useful for debugging purposes.
"""
function varname_in_chain!(model::Model, vn, chain, chain_idx, iteration_idx, out)
    return varname_in_chain!(VarInfo(model), vn, chain, chain_idx, iteration_idx, out)
end

function varname_in_chain!(
    vi::AbstractVarInfo, vn_parent, chain, chain_idx, iteration_idx, out
)
    return varname_in_chain!(vi[vn_parent], vn_parent, chain, chain_idx, iteration_idx, out)
end

function varname_in_chain!(x, vn_parent, chain, chain_idx, iteration_idx, out)
    sym = Symbol(vn_parent)
    out[vn_parent] = sym ∈ names(chain) && !ismissing(chain[iteration_idx, sym, chain_idx])
    return out
end

function varname_in_chain!(
    x::AbstractArray, vn_parent::VarName{sym}, chain, chain_idx, iteration_idx, out
) where {sym}
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the optic.
    # This way we can use `getoptic(vn)` to extract the value from `x` and use `getoptic(vn) ∘ vn_parent`
    # to extract the value from the `chain`.
    for vn in AbstractPPL.varname_leaves(VarName{sym}(), x)
        # Update `out`, possibly in place, and return.
        l = AbstractPPL.getoptic(vn)
        varname_in_chain!(x, l ∘ vn_parent, chain, chain_idx, iteration_idx, out)
    end
    return out
end

"""
    values_from_chain(model::Model, chain, chain_idx, iteration_idx)
    values_from_chain(varinfo::VarInfo, chain, chain_idx, iteration_idx)

Return a dictionary mapping each variable name in `model`/`varinfo` to its
value in `chain` at `chain_idx` and `iteration_idx`.
"""
function values_from_chain(x, vn_parent, chain, chain_idx, iteration_idx)
    # HACK: If it's not an array, we fall back to just returning the first value.
    return only(chain[iteration_idx, Symbol(vn_parent), chain_idx])
end
function values_from_chain(
    x::AbstractArray, vn_parent::VarName{sym}, chain, chain_idx, iteration_idx
) where {sym}
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the optic.
    # This way we can use `getoptic(vn)` to extract the value from `x` and use `getoptic(vn) ∘ vn_parent`
    # to extract the value from the `chain`.
    out = similar(x)
    for vn in AbstractPPL.varname_leaves(VarName{sym}(), x)
        # Update `out`, possibly in place, and return.
        l = AbstractPPL.getoptic(vn)
        out = Accessors.set(
            out,
            BangBang.prefermutation(l),
            chain[iteration_idx, Symbol(l ∘ vn_parent), chain_idx],
        )
    end

    return out
end
function values_from_chain(vi::AbstractVarInfo, vn_parent, chain, chain_idx, iteration_idx)
    # Use the value `vi[vn_parent]` to obtain a buffer.
    return values_from_chain(vi[vn_parent], vn_parent, chain, chain_idx, iteration_idx)
end

"""
    values_from_chain!(model::Model, chain, chain_idx, iteration_idx, out)
    values_from_chain!(varinfo::VarInfo, chain, chain_idx, iteration_idx, out)

Mutate `out` to map each variable name in `model`/`varinfo` to its value in
`chain` at `chain_idx` and `iteration_idx`.
"""
function values_from_chain!(model::Model, chain, chain_idx, iteration_idx, out)
    return values_from_chain(VarInfo(model), chain, chain_idx, iteration_idx, out)
end

function values_from_chain!(vi::AbstractVarInfo, chain, chain_idx, iteration_idx, out)
    for vn in keys(vi)
        out[vn] = values_from_chain(vi, vn, chain, chain_idx, iteration_idx)
    end
    return out
end
