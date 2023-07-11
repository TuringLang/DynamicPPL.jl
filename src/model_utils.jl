### Yong ############################################################## 
# Yong added the below new functions on 2023-07-04, they are doing the some functionalities as Tor's functions. Some redundancy needs to be removed?
# using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test
using Distributions, Random, Test

#### 1. varname_in_chain ####
# here we just check if vn and its leaves are present in the chain; we are not checking its presence in model. So we don't need to pass model or varinfo to this function.
"""
    varname_in_chain(vn::VarName, chain, chain_idx, iteration_idx)

Return two outputs:
    - first output: logical `true` if `vn` or ANY of `vn_child` is in `chain` at `chain_idx` and `iteration_idx`; 
    - second output: a dictionary containing all leaf names of `vn` from the chain, if any.

# Example
```julia
julia> using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test

julia> Random.seed!(111)
MersenneTwister(111)

julia> @model function test_model(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           x ~ Normal(m, sqrt(s))
       end
test_model (generic function with 2 methods)

julia> model = test_model(1.5)

julia> chain = sample(model, NUTS(), 100)

julia> varname_in_chain(VarName(:s), chain, 1, 1)
(true, OrderedDict{Symbol, Bool}(:s => 1))

julia> varname_in_chain(VarName(:m), chain, 1, 1)
(true, OrderedDict{Symbol, Bool}(:m => 1))

julia> varname_in_chain(VarName(:x), chain, 1, 1)
(false, OrderedDict{Symbol, Bool}())

```
"""
function varname_in_chain(vn::VarName, chain, chain_idx=1, iteration_idx=1)
    out = OrderedDict{Symbol,Bool}()
    for vn_child in namesingroup(chain, Symbol(vn)) # namesingroup: https://github.com/TuringLang/MCMCChains.jl/blob/master/src/chains.jl
        # print("\n $vn_child of $vn is in chain")
        out[vn_child] =
            Symbol(vn_child) ∈ names(chain) &&
            !ismissing(chain[iteration_idx, Symbol(vn_child), chain_idx])
    end
    return !isempty(out), out
end

#### 2. varnames_in_chain ####
# we iteratively check whether each of keys(VarInfo(model)) is present in the chain or not using `varname_in_chain`.
"""
    varnames_in_chain(model:::Model, chain)
    varnames_in_chain(varinfo::VarInfo, chain)

Return two outputs:
    - first output: logical `true` if ALL variable names in `model`/`varinfo` are in `chain`; 
    - second output: a dictionary containing all leaf names of `vn` presented in the chain, if any.

# Example
```julia
julia> using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test

julia> Random.seed!(111)
MersenneTwister(111)

julia> @model function test_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
end
test_model (generic function with 2 methods

julia> model = test_model(1.5)

julia> chain = sample(model, NUTS(), 100)

julia> varnames_in_chain(model, chain)
(true, OrderedDict{Any, Any}(:s => OrderedDict{Symbol, Bool}(:s => 1), :m => OrderedDict{Symbol, Bool}(:m => 1)))

```
"""
varnames_in_chain(model::Model, chain) = varnames_in_chain(VarInfo(model), chain)
function varnames_in_chain(varinfo::VarInfo, chain)
    out_logical = OrderedDict()
    out = OrderedDict()
    for vn in keys(varinfo)
        out_logical[Symbol(vn)], out[Symbol(vn)] = varname_in_chain(vn, chain, 1, 1) # by default, we check the first chain and the first iteration.
    end
    return all(values(out_logical)), out
end

#### 3. values_from_chain ####
"""
    vn_values_from_chain(vn, chain, chain_idx, iteration_idx)

Return two outputs:
    - first output: logical `true` if `vn` or ANY of its leaves is in `chain`
    - second output: a dictionary containing all leaf names (if any) of `vn` and their values at `chain_idx`, `iteration_idx` from the chain.

# Example
```julia
julia> using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test

julia> Random.seed!(111)
MersenneTwister(111)

julia> @model function test_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
end
test_model (generic function with 2 methods)

julia> model = test_model(1.5)

julia> chain = sample(model, NUTS(), 100)

julia> vn_values_from_chain(VarName(:s), chain, 1, 1)
(true, OrderedDict{Any, Any}(:s => 1.385664578516751))

julia> vn_values_from_chain(VarName(:m), chain, 1, 1)
(true, OrderedDict{Any, Any}(:m => 0.9529550916018266))

julia> vn_values_from_chain(VarName(:x), chain, 1, 1)
(false, OrderedDict{Any, Any}())

```
"""
function vn_values_from_chain(vn::VarName, chain, chain_idx, iteration_idx)
    out = OrderedDict()
    # no need to check if varname_in_chain(vn, chain)[1] - if vn is not in chain, then out will be empty.
    for vn_child in namesingroup(chain, Symbol(vn))
        try
            out[vn_child] = chain[iteration_idx, Symbol(vn_child), chain_idx]
        catch
            println(
                "Error: retrieve value for $vn_child using chain[$iteration_idx, Symbol($vn_child), $chain_idx] not successful!",
            )
        end
    end
    return !isempty(out), out
end

"""
    values_from_chain(model:::Model, chain)
    values_from_chain(varinfo::VarInfo, chain)

Return one output:
    - a dictionary containing the (leaves_name, value) pair of ALL parameters in `model`/`varinfo` at `chain_idx`, `iteration_idx` from the chain (if ANY of the leaves of `vn` is present in the chain).

# Example
```julia

julia> using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test

julia> Random.seed!(111)
MersenneTwister(111)

julia> @model function test_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
end

julia> model = test_model(1.5)

julia> chain = sample(model, NUTS(), 100)

julia> values_from_chain(model, chain, 1, 1)
OrderedDict{Any, Any} with 2 entries:
  :s => 1.38566
  :m => 0.952955

```
"""
function values_from_chain(model::Model, chain, chain_idx, iteration_idx)
    return values_from_chain(VarInfo(model), chain, chain_idx, iteration_idx)
end
function values_from_chain(varinfo::VarInfo, chain, chain_idx, iteration_idx)
    out = OrderedDict()
    for vn in keys(varinfo)
        _, out_vn = vn_values_from_chain(vn, chain, chain_idx, iteration_idx)
        merge!(out, out_vn)
    end
    return out
end

"""
    values_from_chain(varinfo, chain, chain_idx_range, iteration_idx_range)

Return one output:
    - a dictionary containing the values of all leaf names of all parameters in `model`/`varinfo` within `chain_idx_range` and `iteration_idx_range``.

# Example
```julia

julia> using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test

julia> Random.seed!(111)
MersenneTwister(111)

julia> @model function test_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
end

julia> model = test_model(1.5)

julia> chain = sample(model, NUTS(), 100)

julia> values_from_chain(model, chain, 1:2, 1:10)
Error: retrieve value for s using chain[1, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[2, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[3, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[4, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[5, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[6, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[7, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[8, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[9, Symbol(s), 2] not successful!
Error: retrieve value for s using chain[10, Symbol(s), 2] not successful!
Error: retrieve value for m using chain[1, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[2, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[3, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[4, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[5, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[6, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[7, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[8, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[9, Symbol(m), 2] not successful!
Error: retrieve value for m using chain[10, Symbol(m), 2] not successful!
OrderedDict{Any, Any} with 2 entries:
  "chain_idx_1" => OrderedDict{Any, Any}(:s=>[1.38566, 1.6544, 1.36912, 1.18434, 1.33485, 1.966…
  "chain_idx_2" => OrderedDict{Any, Any}()

```
"""
function values_from_chain(model::Model, chain, chain_idx_range, iteration_idx_range)
    return values_from_chain(VarInfo(model), chain, chain_idx_range, iteration_idx_range)
end
function values_from_chain(
    varinfo::VarInfo, chain, chain_idx_range::UnitRange, iteration_idx_range::UnitRange
)
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
        all_out["chain_idx_" * string(chain_idx)] = out
    end
    return all_out
end
function values_from_chain(
    varinfo::VarInfo, chain, chain_idx_range::Int, iteration_idx_range::UnitRange
)
    return values_from_chain(
        varinfo, chain, chain_idx_range:chain_idx_range, iteration_idx_range
    )
end
function values_from_chain(
    varinfo::VarInfo, chain, chain_idx_range::UnitRange, iteration_idx_range::Int
)
    return values_from_chain(
        varinfo, chain, chain_idx_range, iteration_idx_range:iteration_idx_range
    )
end
function values_from_chain(
    varinfo::VarInfo, chain, chain_idx_range::Int, iteration_idx_range::Int
) #  this is equivalent to values_from_chain(varinfo::VarInfo, chain, chain_idx, iteration_idx)
    return values_from_chain(
        varinfo,
        chain,
        chain_idx_range:chain_idx_range,
        iteration_idx_range:iteration_idx_range,
    )
end
# if either chain_idx_range or iteration_idx_range is specified as `nothing`, then all chains will be included.
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