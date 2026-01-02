module FlexiChainsDynamicPPLExt

using FlexiChains:
    FlexiChains, FlexiChain, VarName, Parameter, Extra, ParameterOrExtra, VNChain
using FlexiChains: DimensionalData as DD
using DynamicPPL: DynamicPPL, AbstractPPL, AbstractMCMC, Distributions, OrderedCollections
using Random: Random

##################################################
# AbstractMCMC.{to,from}_samples implementations #
##################################################

"""
    AbstractMCMC.from_samples(
        ::Type{<:VNChain},
        params_and_stats::AbstractMatrix{<:DynamicPPL.ParamsWithStats}
    )::VNChain

Convert a matrix of [`DynamicPPL.ParamsWithStats`](@extref) to a `VNChain`.
"""
function AbstractMCMC.from_samples(
    ::Type{<:VNChain}, params_and_stats::AbstractMatrix{<:DynamicPPL.ParamsWithStats}
)::VNChain
    # Just need to convert the `ParamsWithStats` to Dicts of ParameterOrExtra.
    dicts = map(params_and_stats) do ps
        # Parameters
        d = OrderedCollections.OrderedDict{ParameterOrExtra{<:VarName},Any}(
            Parameter(vn) => val for (vn, val) in ps.params
        )
        # Stats
        for (stat_vn, stat_val) in pairs(ps.stats)
            d[Extra(stat_vn)] = stat_val
        end
        d
    end
    return VNChain(size(params_and_stats, 1), size(params_and_stats, 2), dicts)
end

"""
    AbstractMCMC.to_samples(
        ::Type{DynamicPPL.ParamsWithStats},
        chain::VNChain
    )::DimensionalData.DimMatrix{DynamicPPL.ParamsWithStats}

Convert a `VNChain` to a `DimMatrix` of [`DynamicPPL.ParamsWithStats`](@extref).

The axes of the `DimMatrix` are the same as those of the input `VNChain`.
"""
function AbstractMCMC.to_samples(
    ::Type{DynamicPPL.ParamsWithStats}, chain::FlexiChain{T}
)::DD.DimMatrix{<:DynamicPPL.ParamsWithStats} where {T<:VarName}
    dicts = FlexiChains.values_at(chain, :, :)
    return map(dicts) do d
        # Need to separate parameters and stats.
        param_dict = OrderedCollections.OrderedDict{T,Any}(
            vn_param.name => val for (vn_param, val) in d if vn_param isa Parameter{<:T}
        )
        stats_nt = NamedTuple(
            Symbol(extra_param.name) => val for
            (extra_param, val) in d if extra_param isa Extra
        )
        DynamicPPL.ParamsWithStats(param_dict, stats_nt)
    end
end

# This method will make `bundle_samples` 'just work'
function FlexiChains.to_varname_dict(
    transition::DynamicPPL.ParamsWithStats
)::OrderedCollections.OrderedDict{ParameterOrExtra{<:VarName},Any}
    d = OrderedCollections.OrderedDict{ParameterOrExtra{<:VarName},Any}()
    for (varname, value) in pairs(transition.params)
        d[Parameter(varname)] = value
    end
    # add in the transition stats (if available)
    for (key, value) in pairs(transition.stats)
        d[Extra(key)] = value
    end
    return d
end

############################
# InitFromParams extension #
############################
"""
    DynamicPPL.InitFromParams(
        chn::FlexiChain{<:VarName},
        iter::Union{Int,DD.At},
        chain::Union{Int,DD.At},
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )::DynamicPPL.InitFromParams

Use the parameters stored in a FlexiChain as an initialisation strategy.
"""
function DynamicPPL.InitFromParams(
    chn::FlexiChain{<:VarName},
    iter::Union{Int,DD.At},
    chain::Union{Int,DD.At},
    fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=DynamicPPL.InitFromPrior(),
)
    # Note that this is functionally the same as `InitFromFlexiChainUnsafe` but it is more
    # flexible because it allows `DD.At` indices, and it also allows for split-VarNames
    # (although that's an unlikely situation). I think conceptually, the difference is that
    # `InitFromParams` isn't meant to be used in tight loops / performance-sensitive code,
    # and can thus give more guarantees about flexibility, whereas
    # `InitFromFlexiChainUnsafe` is really meant for internal use only.
    return DynamicPPL.InitFromParams(FlexiChains.parameters_at(chn, iter, chain), fallback)
end

##################################
# Optimisation for parameters_at #
##################################
"""
    InitFromFlexiChainUnsafe(
        chain::FlexiChain, iter_index::Int, chain_index::Int, fallback=nothing
    )

A DynamicPPL initialisation strategy that obtains values from the given `FlexiChain` at the
specified iteration and chain indices.

In order for `InitFromFlexiChainUnsafe` to work correctly, two things must be ensured:

1. The variables being asked for must **exactly** match those stored in the FlexiChain. That
   is, if the chain contains `@varname(y)` and the model asks for `@varname(y)`, this will
   either error (if no fallback is provided) or silently use the fallback.

2. The `iter_index` and `chain_index` arguments must be 1-based indices.

These requirements allow us to skip the usual `getindex` method when retrieving values from
the `FlexiChain`, and instead index directly into the data storage, which is much faster.

These conditions, especially (1), can be guaranteed if and only if the chain used to
re-evaluate the model was generated from the same model (or a model with the same
structure).

`fallback` provides the same functionality as it does in `DynamicPPL.InitFromParams`, that
is, if a variable is not found in the `FlexiChain`, the fallback strategy is used to
generate its value. This is necessary for `predict`.
"""
struct InitFromFlexiChainUnsafe{
    C<:FlexiChains.VNChain,S<:Union{DynamicPPL.AbstractInitStrategy,Nothing}
} <: DynamicPPL.AbstractInitStrategy
    chain::C
    iter_index::Int
    chain_index::Int
    fallback::S
end
function DynamicPPL.init(
    rng::Random.AbstractRNG,
    vn::VarName,
    dist::Distributions.Distribution,
    strategy::InitFromFlexiChainUnsafe,
)
    param = FlexiChains.Parameter(vn)
    # This skips the `getindex` faff and index directly into underlying data. Of course,
    # this is a bit more prone to breaking. But the performance gains are huge, so this is
    # really worth it.
    if haskey(strategy.chain._data, param)
        x = strategy.chain._data[param][strategy.iter_index, strategy.chain_index]
        return (x, DynamicPPL.typed_identity)
    elseif strategy.fallback !== nothing
        return DynamicPPL.init(rng, vn, dist, strategy.fallback)
    else
        error("Variable $(vn) not found in FlexiChain and no fallback strategy provided.")
    end
end

###########################################
# DynamicPPL: predict, returned, logjoint #
###########################################

function DynamicPPL.reevaluate_with_chain(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::FlexiChain{<:VarName},
    accs::NTuple{N,DynamicPPL.AbstractAccumulator},
    fallback_strategy::Union{DynamicPPL.AbstractInitStrategy,Nothing}=nothing,
)::DD.DimMatrix{<:Tuple{<:Any,<:DynamicPPL.AbstractVarInfo}} where {N}
    niters, nchains = size(chain)
    tuples = Iterators.product(1:niters, 1:nchains)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.AccumulatorTuple(accs))
    retvals_and_varinfos = map(tuples) do (i, j)
        DynamicPPL.init!!(
            rng, model, vi, InitFromFlexiChainUnsafe(chain, i, j, fallback_strategy)
        )
    end
    return FlexiChains._raw_to_user_data(chain, retvals_and_varinfos)
end

# Overloaded here because we want to make sure that any DimArray return values are stacked
# together with the iter/chain dimensions. This is achieved with an extra call to
# _raw_to_user_data.
# https://github.com/penelopeysm/FlexiChains.jl/issues/91
function DynamicPPL.returned(
    model::DynamicPPL.Model, chain::FlexiChain{<:VarName}
)::DD.DimArray
    return FlexiChains._raw_to_user_data(
        chain, map(first, DynamicPPL.reevaluate_with_chain(model, chain))
    )
end

function DynamicPPL.predict(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::FlexiChain{<:VarName};
    include_all::Bool=true,
)::FlexiChain{VarName}
    existing_parameters = Set(FlexiChains.parameters(chain))
    accs = _default_reevaluate_accs()
    fallback = DynamicPPL.InitFromPrior()
    param_dicts = map(reevaluate(rng, model, chain, accs, fallback)) do (_, vi)
        vn_dict = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
        # ^ that is OrderedDict{VarName}
        p_dict = OrderedCollections.OrderedDict{ParameterOrExtra{<:VarName},Any}(
            Parameter(vn) => val for
            (vn, val) in vn_dict if (include_all || !(vn in existing_parameters))
        )
        # Tack on the probabilities
        p_dict[FlexiChains._LOGPRIOR_KEY] = DynamicPPL.getlogprior(vi)
        p_dict[FlexiChains._LOGJOINT_KEY] = DynamicPPL.getlogjoint(vi)
        p_dict[FlexiChains._LOGLIKELIHOOD_KEY] = DynamicPPL.getloglikelihood(vi)
        p_dict
    end
    ni, nc = size(chain)
    predictions_chain = FlexiChain{VarName}(
        ni,
        nc,
        param_dicts;
        iter_indices=FlexiChains.iter_indices(chain),
        chain_indices=FlexiChains.chain_indices(chain),
    )
    old_extras_chain = FlexiChains.subset_extras(chain)
    return merge(old_extras_chain, predictions_chain)
end
function DynamicPPL.predict(
    model::DynamicPPL.Model, chain::FlexiChain{<:VarName}; include_all::Bool=true
)::FlexiChain{VarName}
    return DynamicPPL.predict(Random.default_rng(), model, chain; include_all=include_all)
end

"""
    DynamicPPL.pointwise_logdensities(
        model::Model,
        chain::FlexiChain{T},
        ::Val{whichlogprob}=Val(:both),
    )::FlexiChain{T} where {T<:VarName,whichlogprob}

Calculate the log probability density associated with each variable in the model, for each
iteration in the `FlexiChain`.

The `whichlogprob` argument controls which log probabilities are calculated and stored. It can take the values `:prior`, `:likelihood`, or `:both` (the default).

Returns a new `FlexiChain` with the same structure as the input `chain`, mapping the
variables to their log probabilities.
"""
function DynamicPPL.pointwise_logdensities(
    model::DynamicPPL.Model, chain::FlexiChain{<:VarName}, ::Val{whichlogprob}=Val(:both)
) where {whichlogprob}
    AccType = DynamicPPL.PointwiseLogProbAccumulator{whichlogprob}
    pld_dicts = map(reevaluate(model, chain, (AccType(),))) do (_, vi)
        logps = DynamicPPL.getacc(vi, Val(DynamicPPL.accumulator_name(AccType))).logps
        OrderedCollections.OrderedDict{ParameterOrExtra{<:VarName},Any}(
            Parameter(vn) => only(val) for (vn, val) in logps
        )
    end
    return FlexiChain{VarName}(
        FlexiChains.niters(chain),
        FlexiChains.nchains(chain),
        pld_dicts;
        iter_indices=FlexiChains.iter_indices(chain),
        chain_indices=FlexiChains.chain_indices(chain),
    )
end

#######################
# Precompile workload #
#######################

using DynamicPPL: DynamicPPL, Distributions, AbstractMCMC, @model, VarInfo, ParamsWithStats
using FlexiChains: VNChain, summarystats
using FlexiChains: PrecompileTools

# dummy, needed to satisfy interface of bundle_samples
struct NotASampler <: AbstractMCMC.AbstractSampler end
PrecompileTools.@setup_workload begin
    @model function f()
        x ~ Distributions.MvNormal(zeros(2), [1.0 0.5; 0.5 1.0])
        return y ~ Distributions.Normal()
    end
    model = f()
    transitions = [ParamsWithStats(VarInfo(model), model) for _ in 1:10]
    PrecompileTools.@compile_workload begin
        chn = AbstractMCMC.bundle_samples(
            transitions, model, NotASampler(), nothing, VNChain
        )
        summarystats(chn)
    end
end

end # module FlexiChainsDynamicPPLExt
