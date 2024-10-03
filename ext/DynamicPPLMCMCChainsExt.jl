module DynamicPPLMCMCChainsExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL, Random
    using MCMCChains: MCMCChains
else
    using ..DynamicPPL: DynamicPPL, Random
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

function DynamicPPL.generated_quantities(
    model::DynamicPPL.Model, chain_full::MCMCChains.Chains
)
    chain = MCMCChains.get_sections(chain_full, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        # Update the varinfo with the current sample and make variables not present in `chain`
        # to be sampled.
        DynamicPPL.setval_and_resample!(varinfo, chain, sample_idx, chain_idx)

        # TODO: Some of the variables can be a view into the `varinfo`, so we need to
        # `deepcopy` the `varinfo` before passing it to `model`.
        model(deepcopy(varinfo))
    end
end

function DynamicPPL.predict(
    model::DynamicPPL.Model, chain::MCMCChains.Chains; include_all=false
)
    return predict(Random.default_rng(), model, chain; include_all=include_all)
end
function DynamicPPL.predict(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    include_all=false,
)
    params_only_chain = MCMCChains.get_sections(chain, :parameters)

    varname_to_symbol = if :varname_to_symbol in keys(params_only_chain.info)
        # the mapping is introduced in Turing by
        # https://github.com/TuringLang/Turing.jl/commit/8d8416ac6c7363c6003ee6ea1fbaac26b4fc8dc3
        params_only_chain.info[:varname_to_symbol]
    else
        # if not using Turing, then we need to construct the mapping ourselves
        Dict{DynamicPPL.VarName,Symbol}([
            DynamicPPL.@varname($sym) => sym for
            sym in params_only_chain.name_map.parameters
        ])
    end

    num_of_chains = size(params_only_chain, 3)
    # num_of_params = 
    num_of_samples = size(params_only_chain, 1)

    predictions = []
    for chain_idx in 1:num_of_chains
        predictions_single_chain = []
        for sample_idx in 1:num_of_samples
            d_to_fix = OrderedDict{DynamicPPL.VarName,Any}()

            # construct the dictionary to fix the model
            for (vn, sym) in varname_to_symbol
                d_to_fix[vn] = params_only_chain[sample_idx, sym, chain_idx]
            end

            # fix the model and sample from it
            fixed_model = DynamicPPL.fix(model, d_to_fix)
            predictive_sample = rand(rng, fixed_model)

            # TODO: Turing version uses `Transition` and `bundle_samples` to form new chains: is it worth it to move Transition to AbstractMCMC?
            push!(predictions, predictive_sample)
        end
        push!(predictions, predictions_single_chain)
    end

    return predictions
end

end
