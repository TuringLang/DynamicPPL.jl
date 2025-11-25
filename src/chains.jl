"""
    supports_varname_indexing(chain::AbstractChains)

Return `true` if `chain` supports indexing using `VarName` in place of the
variable name index.
"""
supports_varname_indexing(::AbstractChains) = false

"""
    getindex_varname(chain::AbstractChains, sample_idx, varname::VarName, chain_idx)

Return the value of `varname` in `chain` at `sample_idx` and `chain_idx`.

Whether this method is implemented for `chains` is indicated by [`supports_varname_indexing`](@ref).
"""
function getindex_varname end

"""
    varnames(chains::AbstractChains)

Return an iterator over the varnames present in `chains`.

Whether this method is implemented for `chains` is indicated by [`supports_varname_indexing`](@ref).
"""
function varnames end

"""
    ParamsWithStats

A struct which contains parameter values extracted from a `VarInfo`, along with any
statistics associated with the VarInfo. The statistics are provided as a NamedTuple and are
optional.
"""
struct ParamsWithStats{P<:OrderedDict{<:VarName,<:Any},S<:NamedTuple}
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
    varinfo = maybe_to_typed_varinfo(varinfo)
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),)
    end
    varinfo = DynamicPPL.setaccs!!(varinfo, accs)
    varinfo = last(DynamicPPL.evaluate!!(model, varinfo))
    params = DynamicPPL.getacc(varinfo, Val(:ValuesAsInModel)).values
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(varinfo),
                loglikelihood=DynamicPPL.getloglikelihood(varinfo),
                lp=DynamicPPL.getlogjoint(varinfo),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end

# Re-evaluating the model is unconscionably slow for untyped VarInfo. It's much faster to
# convert it to a typed varinfo first, hence this method.
# https://github.com/TuringLang/Turing.jl/issues/2604
maybe_to_typed_varinfo(vi::UntypedVarInfo) = typed_varinfo(vi)
maybe_to_typed_varinfo(vi::UntypedVectorVarInfo) = typed_vector_varinfo(vi)
maybe_to_typed_varinfo(vi::AbstractVarInfo) = vi

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

There is one case where re-evaluation is not necessary, which is when the VarInfos all
already contain `DynamicPPL.ValuesAsInModelAccumulator`. This accumulator stores values
as seen during the model evaluation, so the values can be simply read off. In this case,
the `model` argument can be omitted, and no re-evaluation will be performed. However, it is
the caller's responsibility to ensure that `ValuesAsInModelAccumulator` is indeed present
inside `varinfo`.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo, stats::NamedTuple=NamedTuple(); include_log_probs::Bool=true
)
    params = DynamicPPL.getacc(varinfo, Val(:ValuesAsInModel)).values
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
via `unflatten` plus re-evaluation. It is faster for two reasons:

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
        VectorWithRanges(ldf._iden_varname_ranges, ldf._varname_ranges, param_vector),
        nothing,
    )
    accs = if include_log_probs
        (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),
        )
    else
        (DynamicPPL.ValuesAsInModelAccumulator(include_colon_eq),)
    end
    _, vi = DynamicPPL.init!!(ldf.model, OnlyAccsVarInfo(AccumulatorTuple(accs)), strategy)
    params = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(vi),
                loglikelihood=DynamicPPL.getloglikelihood(vi),
                lp=DynamicPPL.getlogjoint(vi),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end
