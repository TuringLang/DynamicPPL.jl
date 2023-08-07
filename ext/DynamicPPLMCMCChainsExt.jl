module DynamicPPLMCMCChainsExt

using DynamicPPL: DynamicPPL
using MCMCChains: MCMCChains

function DynamicPPL.generated_quantities(
    model::DynamicPPL.Model, chain::MCMCChains.Chains
)
    chain_parameters = MCMCChains.get_sections(chain, :parameters)
    varinfo = DynamicPPL.VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        DynamicPPL.setval_and_resample!(varinfo, chain_parameters, sample_idx, chain_idx)
        model(varinfo)
    end
end

end
