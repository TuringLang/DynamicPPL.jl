module DynamicPPLMCMCChainsExt

using DynamicPPL: DynamicPPL, AbstractPPL, AbstractMCMC, Random
using MCMCChains: MCMCChains

function getindex_varname(
    c::MCMCChains.Chains, sample_idx, vn::DynamicPPL.VarName, chain_idx
)
    return c[sample_idx, c.info.varname_to_symbol[vn], chain_idx]
end
function get_varnames(c::MCMCChains.Chains)
    haskey(c.info, :varname_to_symbol) ||
        error("This `Chains` object does not support indexing using `VarName`s.")
    return keys(c.info.varname_to_symbol)
end

"""
    AbstractMCMC.from_samples(
        ::Type{MCMCChains.Chains},
        params_and_stats::AbstractMatrix{<:ParamsWithStats}
    )

Convert an array of `DynamicPPL.ParamsWithStats` to an `MCMCChains.Chains` object.
"""
function AbstractMCMC.from_samples(
    ::Type{MCMCChains.Chains},
    params_and_stats::AbstractMatrix{<:DynamicPPL.ParamsWithStats},
)
    # Handle parameters
    all_vn_leaves = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()
    split_dicts = map(params_and_stats) do ps
        # Separate into individual VarNames.
        vn_leaves_and_vals = if isempty(ps.params)
            Tuple{DynamicPPL.VarName,Any}[]
        else
            iters = map(
                AbstractPPL.varname_and_value_leaves,
                keys(ps.params),
                values(ps.params),
            )
            mapreduce(collect, vcat, iters)
        end
        vn_leaves = map(first, vn_leaves_and_vals)
        vals = map(last, vn_leaves_and_vals)
        for vn_leaf in vn_leaves
            push!(all_vn_leaves, vn_leaf)
        end
        DynamicPPL.OrderedCollections.OrderedDict(zip(vn_leaves, vals))
    end
    vn_leaves = collect(all_vn_leaves)
    param_vals = [
        get(split_dicts[i, j], key, missing) for i in eachindex(axes(split_dicts, 1)),
        key in vn_leaves, j in eachindex(axes(split_dicts, 2))
    ]
    param_symbols = map(Symbol, vn_leaves)
    # Handle statistics
    stat_keys = DynamicPPL.OrderedCollections.OrderedSet{Symbol}()
    for ps in params_and_stats
        for k in keys(ps.stats)
            push!(stat_keys, k)
        end
    end
    stat_keys = collect(stat_keys)
    stat_vals = [
        get(params_and_stats[i, j].stats, key, missing) for
        i in eachindex(axes(params_and_stats, 1)), key in stat_keys,
        j in eachindex(axes(params_and_stats, 2))
    ]
    # Construct name map and info
    name_map = (internals=stat_keys,)
    info = (
        varname_to_symbol=DynamicPPL.OrderedCollections.OrderedDict(
            zip(all_vn_leaves, param_symbols)
        ),
    )
    # Concatenate parameter and statistic values
    vals = cat(param_vals, stat_vals; dims=2)
    symbols = vcat(param_symbols, stat_keys)
    return MCMCChains.Chains(MCMCChains.concretize(vals), symbols, name_map; info=info)
end

"""
    AbstractMCMC.to_samples(
        ::Type{DynamicPPL.ParamsWithStats},
        chain::MCMCChains.Chains
    )

Convert an `MCMCChains.Chains` object to an array of `DynamicPPL.ParamsWithStats`.

For this to work, `chain` must contain the `varname_to_symbol` mapping in its `info` field.
"""
function AbstractMCMC.to_samples(
    ::Type{DynamicPPL.ParamsWithStats}, chain::MCMCChains.Chains
)
    idxs = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    # Get parameters
    params_matrix = map(idxs) do (sample_idx, chain_idx)
        d = DynamicPPL.OrderedCollections.OrderedDict{DynamicPPL.VarName,Any}()
        for vn in get_varnames(chain)
            d[vn] = getindex_varname(chain, sample_idx, vn, chain_idx)
        end
        d
    end
    # Statistics
    stats_matrix = if :internals in MCMCChains.sections(chain)
        internals_chain = MCMCChains.get_sections(chain, :internals)
        map(idxs) do (sample_idx, chain_idx)
            get(internals_chain[sample_idx, :, chain_idx], keys(internals_chain); flatten=true)
        end
    else
        fill(NamedTuple(), size(idxs))
    end
    # Bundle them together
    return map(idxs) do (sample_idx, chain_idx)
        DynamicPPL.ParamsWithStats(
            params_matrix[sample_idx, chain_idx], stats_matrix[sample_idx, chain_idx]
        )
    end
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:DynamicPPL.ParamsWithStats},
    model::DynamicPPL.Model,
    spl::AbstractMCMC.AbstractSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state=false,
    stats=missing,
    sort_chain=false,
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    bare_chain = AbstractMCMC.from_samples(MCMCChains.Chains, reshape(ts, :, 1))

    # Add additional MCMC-specific info
    info = bare_chain.info
    if save_state
        info = merge(info, (model=model, sampler=spl, samplerstate=state))
    end
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Reconstruct the chain with the extra information
    # Yeah, this is quite ugly. Blame MCMCChains.
    chain = MCMCChains.Chains(
        bare_chain.value.data,
        names(bare_chain),
        bare_chain.name_map;
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )
    return sort_chain ? sort(chain) : chain
end

function DynamicPPL.predict(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    include_all=false,
)
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)
    accs = (
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
        DynamicPPL.ValuesAsInModelAccumulator(false),
    )
    predictions = map(
        DynamicPPL.ParamsWithStats ∘ last,
        DynamicPPL.reevaluate_with_chain(
            rng, model, parameter_only_chain, accs, DynamicPPL.InitFromPrior()
        ),
    )
    chain_result = AbstractMCMC.from_samples(MCMCChains.Chains, predictions)
    parameter_names = if include_all
        MCMCChains.names(chain_result, :parameters)
    else
        filter(
            k -> !(k in MCMCChains.names(parameter_only_chain, :parameters)),
            names(chain_result, :parameters),
        )
    end
    return chain_result[parameter_names]
end
function DynamicPPL.predict(
    model::DynamicPPL.Model, chain::MCMCChains.Chains; include_all=false
)
    return DynamicPPL.predict(Random.default_rng(), model, chain; include_all=include_all)
end
end

function DynamicPPL.pointwise_logdensities(
    model::DynamicPPL.Model, chain::MCMCChains.Chains, ::Val{whichlogprob}=Val(:both)
) where {whichlogprob}
    acc = DynamicPPL.PointwiseLogProbAccumulator{whichlogprob}()
    accname = DynamicPPL.accumulator_name(acc)
    parameter_only_chain = MCMCChains.get_sections(chain, :parameters)
    pointwise_logps =
        map(reevaluate_with_chain(model, parameter_only_chain, (acc,), nothing)) do (_, vi)
            DynamicPPL.getacc(vi, Val(accname)).logps
        end
    # pointwise_logps is a matrix of OrderedDicts
    all_keys = DynamicPPL.OrderedCollections.OrderedSet{DynamicPPL.VarName}()
    for d in pointwise_logps
        union!(all_keys, DynamicPPL.OrderedCollections.OrderedSet(keys(d)))
    end
    # this is a 3D array: (iterations, variables, chains)
    new_data = [
        get(pointwise_logps[iter, chain], k, missing) for
        iter in 1:size(pointwise_logps, 1), k in all_keys,
        chain in 1:size(pointwise_logps, 2)
    ]
    return MCMCChains.Chains(new_data, Symbol.(collect(all_keys)))
end
