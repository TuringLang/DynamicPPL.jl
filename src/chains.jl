"""
    ParamsWithStats

A struct which contains parameter values extracted from a `VarInfo`, along with any
statistics associated with the VarInfo. The statistics are provided as a NamedTuple and are
optional.
"""
struct ParamsWithStats{P<:VarNamedTuple,S<:NamedTuple}
    params::P
    stats::S
end

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        model::Model,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `model` with the provided `varinfo`.
Re-evaluation of the model is often necessary to obtain correct parameter values as well as
log probabilities. This is especially true when using linked VarInfos, i.e., when variables
have been transformed to unconstrained space, and if this is not done, subtle correctness
bugs may arise: see, e.g., https://github.com/TuringLang/Turing.jl/issues/2195.

`include_colon_eq` controls whether variables on the left-hand side of `:=` are included in
the resulting parameters.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo,
    model::DynamicPPL.Model,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.RawValueAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.RawValueAccumulator(include_colon_eq),)
    end
    oavi = OnlyAccsVarInfo(accs)
    init = InitFromParams(varinfo.values, nothing)
    oavi = last(DynamicPPL.init!!(model, oavi, init, UnlinkAll()))
    params = get_raw_values(oavi)
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(oavi),
                loglikelihood=DynamicPPL.getloglikelihood(oavi),
                logjoint=DynamicPPL.getlogjoint(oavi),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

There is one case where re-evaluation is not necessary, which is when the VarInfos all
already contain `DynamicPPL.RawValueAccumulator`. This accumulator stores values
as seen during the model evaluation, so the values can be simply read off. In this case,
the `model` argument can be omitted, and no re-evaluation will be performed. However, it is
the caller's responsibility to ensure that `RawValueAccumulator` is indeed present
inside `varinfo`.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo, stats::NamedTuple=NamedTuple(); include_log_probs::Bool=true
)
    params = get_raw_values(varinfo)
    if include_log_probs
        has_prior_acc = DynamicPPL.hasacc(varinfo, Val(:LogPrior))
        has_likelihood_acc = DynamicPPL.hasacc(varinfo, Val(:LogLikelihood))
        if has_prior_acc
            stats = merge(stats, (logprior=DynamicPPL.getlogprior(varinfo),))
        end
        if has_likelihood_acc
            stats = merge(stats, (loglikelihood=DynamicPPL.getloglikelihood(varinfo),))
        end
        if has_prior_acc && has_likelihood_acc
            stats = merge(stats, (logjoint=DynamicPPL.getlogjoint(varinfo),))
        end
    end
    return ParamsWithStats(params, stats)
end

"""
    ParamsWithStats(
        param_vector::AbstractVector,
        ldf::DynamicPPL.LogDensityFunction,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `ldf` with the provided
`param_vector`.

This method is intended to replace the old method of obtaining parameters and statistics
via `unflatten!!` plus re-evaluation. It is faster for two reasons:

1. It does not rely on `deepcopy`-ing the VarInfo object (this used to be mandatory as
   otherwise re-evaluation would mutate the VarInfo, rendering it unusable for subsequent
   MCMC iterations).
2. The re-evaluation is faster as it uses `OnlyAccsVarInfo`.
"""
function ParamsWithStats(
    param_vector::AbstractVector,
    ldf::DynamicPPL.LogDensityFunction,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    strategy = InitFromParams(
        VectorWithRanges(ldf._varname_ranges, param_vector, ldf.transform_strategy), nothing
    )
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.RawValueAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.RawValueAccumulator(include_colon_eq),)
    end
    # UnlinkAll() actually doesn't have any impact here, because there isn't even a
    # LogJacobianAccumulator; consequently, it doesn't matter whether we interpret the
    # parameters as being in linked space or not. However, we just include it for clarity.
    _, vi = DynamicPPL.init!!(
        ldf.model, OnlyAccsVarInfo(AccumulatorTuple(accs)), strategy, UnlinkAll()
    )
    params = get_raw_values(vi)
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(vi),
                loglikelihood=DynamicPPL.getloglikelihood(vi),
                logjoint=DynamicPPL.getlogjoint(vi),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end
