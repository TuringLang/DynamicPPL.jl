### Yong ############################################################## 
# Yong added the below new functions on 2023-07-04, they are doing the some functionalities as Tor's functions. Some redundancy needs to be removed?
using Turing, Distributions, DynamicPPL, MCMCChains, Test

#### 1. varname_in_chain ####
# here we just check if vn and its leaves are present in the chain; we are not checking its presence in model. So we don't need to pass model or varinfo to this function.
"""
    varname_in_chain(vn::VarName, chain, chain_idx, iteration_idx)

Return `true` if `vn` or any of `vn_child` is in `chain` at `chain_idx` and `iteration_idx`; also returned is the dictionary containing the names related to `vn` presented in the chain, if any.
"""
function varname_in_chain(vn::VarName, chain, chain_idx=1, iteration_idx=1)
    out = OrderedDict{Symbol,Bool}()
    for vn_child in namesingroup(chain, Symbol(vn)) # namesingroup: https://github.com/TuringLang/MCMCChains.jl/blob/master/src/chains.jl
        # print("\n $vn_child of $vn is in chain")
        out[vn_child] = Symbol(vn_child) ∈ names(chain) && !ismissing(chain[iteration_idx, Symbol(vn_child), chain_idx])
    end
    return !isempty(out), out
end

#### 2. varnames_in_chain ####
# we iteratively test whether each of keys(VarInfo(model)) is present in the chain or not
"""
    varnames_in_chain(model:::Model, chain)
    varnames_in_chain(varinfo::VarInfo, chain)

Return `true` if all variable names in `model`/`varinfo` are in `chain`; also returned is the dictionary containing the names related to `vn` presented in the chain, if any.
"""
varnames_in_chain(model::Model, chain) = varnames_in_chain(VarInfo(model), chain)
function varnames_in_chain(varinfo::VarInfo, chain)
    out_logical = OrderedDict()
    out = OrderedDict()
    for vn in keys(varinfo)
        out_logical[Symbol(vn)], out[Symbol(vn)] = varname_in_chain(vn, chain, 1, 1)
    end
    return all(values(out_logical)), out
end

#### 3. values_from_chain ####
"""
    vn_values_from_chain(vn, chain, chain_idx, iteration_idx)

Return `true` if `vn` or any of its leaves is in `chain`; also returned is the dictionary containing the names related to `vn` presented in the chain, if any.
"""
function vn_values_from_chain(vn::VarName, chain, chain_idx, iteration_idx)
    out = OrderedDict()
    # no need to test if varname_in_chain(vn, chain)[1] - if vn is not in chain, then out will be empty.
    for vn_child in namesingroup(chain, Symbol(vn))
        try
            out[vn_child] = chain[iteration_idx, Symbol(vn_child), chain_idx]
        catch
            println("Error: retrieve value for $vn_child using chain[$iteration_idx, Symbol($vn_child), $chain_idx] not successful!")
        end
    end
    return !isempty(out), out
end

"""
    values_from_chain(model:::Model, chain)
    values_from_chain(varinfo::VarInfo, chain)

Return a dictionary containing the values of all variables in `model`/`varinfo` presented in `chain`, if any.
"""
values_from_chain(model::Model, chain, chain_idx, iteration_idx) = values_from_chain(VarInfo(model), chain, chain_idx, iteration_idx)
function values_from_chain(varinfo::VarInfo, chain, chain_idx, iteration_idx)
    out = OrderedDict()
    for vn in keys(varinfo)
        _, out_vn = vn_values_from_chain(vn, chain, chain_idx, iteration_idx)
        merge!(out, out_vn)
    end
    return out
end

"""
    values_from_chain(varinfo, chain, chain_idx, iteration_idx_range)

Return a dictionary containing the values of all variables in `model`/`varinfo` presented in `chain`,  as per iteration_idx_range.
"""
values_from_chain(model::Model, chain, chain_idx_range, iteration_idx_range) = values_from_chain(VarInfo(model), chain, chain_idx_range, iteration_idx_range)
function values_from_chain(varinfo::VarInfo, chain, chain_idx_range::UnitRange, iteration_idx_range::UnitRange)
    all_out = OrderedDict()
    for chain_idx in chain_idx_range
        out = OrderedDict()
        for vn in keys(varinfo)
            for iteration_idx in iteration_idx_range
                _, out_vn = vn_values_from_chain(vn, chain, chain_idx, iteration_idx)
                for key in keys(out_vn)
                    if haskey(out, key)
                        out[key] = vcat(out[key], out_vn[key])
                    else
                        out[key] = out_vn[key]
                    end
                end
            end
        end
        all_out["chain_idx_"*string(chain_idx)] = out
    end
    return all_out
end
function values_from_chain(varinfo::VarInfo, chain, chain_idx_range::Int, iteration_idx_range::UnitRange)
    return values_from_chain(varinfo, chain, chain_idx_range:chain_idx_range, iteration_idx_range)
end
function values_from_chain(varinfo::VarInfo, chain, chain_idx_range::UnitRange, iteration_idx_range::Int)
    return values_from_chain(varinfo, chain, chain_idx_range, iteration_idx_range:iteration_idx_range)
end
function values_from_chain(varinfo::VarInfo, chain, chain_idx_range::Int, iteration_idx_range::Int) #  this is equivalent to values_from_chain(varinfo::VarInfo, chain, chain_idx, iteration_idx)
    return values_from_chain(varinfo, chain, chain_idx_range:chain_idx_range, iteration_idx_range:iteration_idx_range)
end
## if either chain_idx_range and iteration_idx_range not specified, then all chains will be included.
function values_from_chain(varinfo::VarInfo, chain, chain_idx_range, iteration_idx_range)
    if chain_idx_range === nothing
        print("chain_idx_range is missing!")
        chain_idx_range = 1:size(chain)[3]
    end
    if iteration_idx_range === nothing
        print("iteration_idx_range is missing!")    
        iteration_idx_range = 1:size(chain)[1]
    end
    return values_from_chain(varinfo, chain, chain_idx_range, iteration_idx_range)
end

### Tor ##############################################################
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
```julia
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
