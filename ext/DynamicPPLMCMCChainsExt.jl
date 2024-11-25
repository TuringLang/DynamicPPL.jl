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
    returned(model::Model, chain::MCMCChains.Chains)

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
returned(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using Turing

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

julia> returned(model, chain)
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
function DynamicPPL.returned(
    model::DynamicPPL.Model, chain_full::MCMCChains.Chains
)
    chain = MCMCChains.get_sections(chain_full, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        # TODO: Use `fix` once we've addressed https://github.com/TuringLang/DynamicPPL.jl/issues/702.
        # Update the varinfo with the current sample and make variables not present in `chain`
        # to be sampled.
        DynamicPPL.setval_and_resample!(varinfo, chain, sample_idx, chain_idx)
        # NOTE: Some of the varialbes can be a view into the `varinfo`, so we need to
        # `deepcopy` the `varinfo` before passing it to the `model`.
        model(deepcopy(varinfo))
    end
end

end
