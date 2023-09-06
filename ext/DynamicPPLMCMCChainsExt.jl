module DynamicPPLMCMCChainsExt

using DynamicPPL: DynamicPPL
using MCMCChains: MCMCChains

_has_varname_to_symbol(info::NamedTuple{names}) where {names} = :varname_to_symbol in names
function DynamicPPL.supports_varname_indexing(chain::MCMCChains.Chains)
    return _has_varname_to_symbol(chain.info)
end

function DynamicPPL.generated_quantities(model::DynamicPPL.Model, chain::MCMCChains.Chains)
    chain_parameters = MCMCChains.get_sections(chain, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        if DynamicPPL.supports_varname_indexing(chain)
            for vn in keys(chain.info.varname_to_symbol)
                # FIXME: Make it so we can support `chain[sample_idx, vn, chain_idx]`
                # indexing instead of the `chain[vn][sample_idx, chain_idx]` below.
                DynamicPPL.nested_setindex!(varinfo, chain[vn][sample_idx, chain_idx], vn)
            end
        else
            # NOTE: This can be quite unreliable (but will warn the uesr in that case).
            # Hence the above path is much more preferable.
            DynamicPPL.setval_and_resample!(varinfo, chain, sample_idx, chain_idx)
        end
        # TODO: Some of the variables can be a view into the `varinfo`, so we need to
        # `deepcopy` the `varinfo` before passing it to `model`.
        model(deepcopy(varinfo))
    end
end

end
