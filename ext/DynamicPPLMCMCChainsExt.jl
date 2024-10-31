module DynamicPPLMCMCChainsExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using MCMCChains: MCMCChains
else
    using ..DynamicPPL: DynamicPPL
    using ..MCMCChains: MCMCChains
end

# Load state from a `Chains`: By convention, it is stored in `:samplerstate` metadata
function DynamicPPL.loadstate(chain::MCMCChains.Chains)
    if !haskey(chain.info, :samplerstate)
        throw(
            ArgumentError(
                "The chain object does not contain the final state of the sampler: Metadata `:samplerstate` missing.",
            ),
        )
    end
    return chain.info[:samplerstate]
end

_has_varname_to_symbol(info::NamedTuple{names}) where {names} = :varname_to_symbol in names

function DynamicPPL.supports_varname_indexing(chain::MCMCChains.Chains)
    return _has_varname_to_symbol(chain.info)
end

function _check_varname_indexing(c::MCMCChains.Chains)
    return DynamicPPL.supports_varname_indexing(c) ||
           error("Chains do not support indexing using `VarName`s.")
end

function DynamicPPL.getindex_varname(
    c::MCMCChains.Chains, sample_idx, vn::DynamicPPL.VarName, chain_idx
)
    _check_varname_indexing(c)
    return c[sample_idx, c.info.varname_to_symbol[vn], chain_idx]
end
function DynamicPPL.varnames(c::MCMCChains.Chains)
    _check_varname_indexing(c)
    return keys(c.info.varname_to_symbol)
end

"""
    returned_quantities(model::Model, chain::MCMCChains.Chains)

Execute `model` for each of the samples in `chain` and return an array of the values
returned by the `model` for each sample.

# Examples
## General
Often you might have additional quantities computed inside the model that you want to
inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()
    return interesting_quantity(θ, x)
end
m = demo(data)
chain = sample(m, alg, n)
# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples
# from the posterior/`chain`:
returned_quantities(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using DynamicPPL, Turing

julia> @model function demo(xs)
           s ~ InverseGamma(2, 3)
           m_shifted ~ Normal(10, √s)
           m = m_shifted - 10

           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end

           return (m, )
       end
demo (generic function with 1 method)

julia> model = demo(randn(10));

julia> chain = sample(model, MH(), 10);

julia> returned_quantities(model, chain)
10×1 Array{Tuple{Float64},2}:
 (2.1964758025119338,)
 (2.1964758025119338,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.043088571494005024,)
 (-0.16489786710222099,)
 (-0.16489786710222099,)
```
"""
function DynamicPPL.returned_quantities(
    model::DynamicPPL.Model, chain_full::MCMCChains.Chains
)
    chain = MCMCChains.get_sections(chain_full, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        if DynamicPPL.supports_varname_indexing(chain)
            varname_pairs = _varname_pairs_with_varname_indexing(
                chain, varinfo, sample_idx, chain_idx
            )
        else
            varname_pairs = _varname_pairs_without_varname_indexing(
                chain, varinfo, sample_idx, chain_idx
            )
        end
        fixed_model = DynamicPPL.fix(model, Dict(varname_pairs))
        return fixed_model()
    end
end

"""
    _varname_pairs_with_varname_indexing(
        chain::MCMCChains.Chains, varinfo, sample_idx, chain_idx
    )

Get pairs of `VarName => value` for all the variables in the `varinfo`, picking the values
from the chain.

This implementation assumes `chain` can be indexed using variable names, and is the
preffered implementation.
"""
function _varname_pairs_with_varname_indexing(
    chain::MCMCChains.Chains, varinfo, sample_idx, chain_idx
)
    vns = DynamicPPL.varnames(chain)
    vn_parents = Iterators.map(vns) do vn
        # The call nested_setindex_maybe! is used to handle cases where vn is not
        # the variable name used in the model, but rather subsumed by one. Except
        # for the subsumption part, this could be
        # vn => getindex_varname(chain, sample_idx, vn, chain_idx)
        # TODO(mhauru) This call to nested_setindex_maybe! is unintuitive.
        DynamicPPL.nested_setindex_maybe!(
            varinfo, DynamicPPL.getindex_varname(chain, sample_idx, vn, chain_idx), vn
        )
    end
    varname_pairs = Iterators.map(Iterators.filter(!isnothing, vn_parents)) do vn_parent
        vn_parent => varinfo[vn_parent]
    end
    return varname_pairs
end

"""
Check which keys in `key_strings` are subsumed by `vn_string` and return the their values.

The subsumption check is done with `DynamicPPL.subsumes_string`, which is quite weak, and
won't catch all cases. We should get rid of this if we can.
"""
# TODO(mhauru) See docstring above.
function _vcat_subsumed_values(vn_string, values, key_strings)
    indices = findall(Base.Fix1(DynamicPPL.subsumes_string, vn_string), key_strings)
    return !isempty(indices) ? reduce(vcat, values[indices]) : nothing
end

"""
    _varname_pairs_without_varname_indexing(
        chain::MCMCChains.Chains, varinfo, sample_idx, chain_idx
    )

Get pairs of `VarName => value` for all the variables in the `varinfo`, picking the values
from the chain.

This implementation does not assume that `chain` can be indexed using variable names. It is
thus not guaranteed to work in cases where the variable names have complex subsumption
patterns, such as if the model has a variable `x` but the chain stores `x.a[1]`.
"""
function _varname_pairs_without_varname_indexing(
    chain::MCMCChains.Chains, varinfo, sample_idx, chain_idx
)
    values = chain.value[sample_idx, :, chain_idx]
    keys = Base.keys(chain)
    keys_strings = map(string, keys)
    varname_pairs = [
        vn => _vcat_subsumed_values(string(vn), values, keys_strings) for
        vn in Base.keys(varinfo)
    ]
    return varname_pairs
end

end
