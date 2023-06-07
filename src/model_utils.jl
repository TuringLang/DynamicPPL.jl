"""
    varnames_in_chain(model:::Model, chain)
    varnames_in_chain(varinfo::VarInfo, chain)

Return `true` if all variable names in `model`/`varinfo` are in `chain`.
"""
varnames_in_chain(model::Model, chain) = varnames_in_chain(VarInfo(model), chain)
function varnames_in_chain(varinfo::VarInfo, chain)
    return all(vn -> varname_in_chain(varinfo, vn, chain), keys(varinfo))
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
    varname_in_chain!(x, vn, chain, out, chain_idx, iteration_idx)
    return all(values(out))
end

"""
    varname_in_chain!(model::Model, vn, chain, out, chain_idx, iteration_idx)
    varname_in_chain!(varinfo::VarInfo, vn, chain, out, chain_idx, iteration_idx)

Return a dictionary mapping the varname `vn` to `true` if `vn` is in `chain` at
`chain_idx` and `iteration_idx`.

If `chain_idx` and `iteration_idx` are not provided, then they default to `1`.

This differs from [`varname_in_chain`](@ref) in that it returns a dictionary
rather than a single boolean. This can be quite useful for debugging purposes.
"""
function varname_in_chain!(model::Model, vn, chain, out, chain_idx, iteration_idx)
    return varname_in_chain!(VarInfo(model), vn, chain, chain_idx, iteration_idx, out)
end

function varname_in_chain!(
    vi::AbstractVarInfo, vn_parent, chain, out, chain_idx, iteration_idx
)
    return varname_in_chain!(vi[vn_parent], vn_parent, chain, out, chain_idx, iteration_idx)
end

function varname_in_chain!(x, vn_parent, chain, out, chain_idx, iteration_idx)
    sym = Symbol(vn_parent)
    out[vn_parent] = sym ∈ names(chain) && !ismissing(chain[iteration_idx, sym, chain_idx])
    return out
end

function varname_in_chain!(
    x::AbstractArray, vn_parent::VarName{sym}, chain, out, chain_idx, iteration_idx
) where {sym}
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the lens.
    # This way we can use `getlens(vn)` to extract the value from `x` and use `vn_parent ∘ getlens(vn)`
    # to extract the value from the `chain`.
    for vn in varname_leaves(VarName{sym}(), x)
        # Update `out`, possibly in place, and return.
        l = AbstractPPL.getlens(vn)
        varname_in_chain!(x, vn_parent ∘ l, chain, out, chain_idx, iteration_idx)
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
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the lens.
    # This way we can use `getlens(vn)` to extract the value from `x` and use `vn_parent ∘ getlens(vn)`
    # to extract the value from the `chain`.
    out = similar(x)
    for vn in varname_leaves(VarName{sym}(), x)
        # Update `out`, possibly in place, and return.
        l = AbstractPPL.getlens(vn)
        out = Setfield.set(
            out,
            BangBang.prefermutation(l),
            chain[iteration_idx, Symbol(vn_parent ∘ l), chain_idx],
        )
    end

    return out
end
function values_from_chain(vi::AbstractVarInfo, vn_parent, chain, chain_idx, iteration_idx)
    # Use the value `vi[vn_parent]` to obtain a buffer.
    return values_from_chain(vi[vn_parent], vn_parent, chain, chain_idx, iteration_idx)
end

"""
    values_from_chain!(model::Model, chain, out, chain_idx, iteration_idx)
    values_from_chain!(varinfo::VarInfo, chain, out, chain_idx, iteration_idx)

Mutate `out` to map each variable name in `model`/`varinfo` to its value in
`chain` at `chain_idx` and `iteration_idx`.
"""
function values_from_chain!(model::DynamicPPL.Model, chain, out, chain_idx, iteration_idx)
    return values_from_chain(VarInfo(model), chain, out, chain_idx, iteration_idx)
end

function values_from_chain!(vi::AbstractVarInfo, chain, out, chain_idx, iteration_idx)
    for vn in keys(vi)
        out[vn] = values_from_chain(vi, vn, chain, chain_idx, iteration_idx)
    end
    return out
end

"""
    value_iterator_from_chain(model::Model, chain)
    value_iterator_from_chain(varinfo::AbstractVarInfo, chain)

Return an iterator over the values in `chain` for each variable in `model`/`varinfo`.

# Example
```jldoctest
julia> using MCMCChains, DynamicPPL, Distributions, StableRNGs

julia> rng = StableRNG(42);

julia> @model function demo_model(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           for i in eachindex(x)
               x[i] ~ Normal(m, sqrt(s))
           end

           return s, m
       end
demo_model (generic function with 2 methods)

julia> model = demo_model([1.0, 2.0]);

julia> chain = Chains(rand(rng, 10, 2, 3), [:s, :m]);

julia> iter = value_iterator_from_chain(model, chain);

julia> first(iter)
OrderedDict{VarName, Any} with 2 entries:
  s => 0.580515
  m => 0.739328

julia> collect(iter)
10×3 Matrix{OrderedDict{VarName, Any}}:
 OrderedDict(s=>0.580515, m=>0.739328)  …  OrderedDict(s=>0.186047, m=>0.402423)
 OrderedDict(s=>0.191241, m=>0.627342)     OrderedDict(s=>0.776277, m=>0.166342)
 OrderedDict(s=>0.971133, m=>0.637584)     OrderedDict(s=>0.651655, m=>0.712044)
 OrderedDict(s=>0.74345, m=>0.110359)      OrderedDict(s=>0.469214, m=>0.104502)
 OrderedDict(s=>0.170969, m=>0.598514)     OrderedDict(s=>0.853546, m=>0.185399)
 OrderedDict(s=>0.704776, m=>0.322111)  …  OrderedDict(s=>0.638301, m=>0.853802)
 OrderedDict(s=>0.441044, m=>0.162285)     OrderedDict(s=>0.852959, m=>0.0956922)
 OrderedDict(s=>0.803972, m=>0.643369)     OrderedDict(s=>0.245049, m=>0.871985)
 OrderedDict(s=>0.772384, m=>0.646323)     OrderedDict(s=>0.906603, m=>0.385502)
 OrderedDict(s=>0.70882, m=>0.253105)      OrderedDict(s=>0.413222, m=>0.953288)

julia> # This can be used to `condition` a `Model`.
       conditioned_model = model | first(iter);

julia> conditioned_model()  # <= results in same values as the `first(iter)` above
(0.5805148626851955, 0.7393275279160691)
```
"""
function value_iterator_from_chain(model::DynamicPPL.Model, chain)
    return value_iterator_from_chain(VarInfo(model), chain)
end

function value_iterator_from_chain(vi::AbstractVarInfo, chain)
    return Iterators.map(
        Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    ) do (iteration_idx, chain_idx)
        values_from_chain!(vi, chain, OrderedDict{VarName,Any}(), chain_idx, iteration_idx)
    end
end
