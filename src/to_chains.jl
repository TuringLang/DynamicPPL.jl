"""
    ParamsWithStats

A struct which contains parameter values extracted from a `VarInfo`, along with any
statistics associated with the VarInfo. The statistics are provided as a NamedTuple and are
optional.

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

    ParamsWithStats(
        varinfo::AbstractVarInfo,
        ::Nothing,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

There is one case where re-evaluation is not necessary, which is when the VarInfos all
already contain `DynamicPPL.ValuesAsInModelAccumulator`. This accumulator stores values
as seen during the model evaluation, so the values can be simply read off. In this case,
`model` can be set to `nothing`, and no re-evaluation will be performed. However, it is the
caller's responsibility to ensure that `ValuesAsInModelAccumulator` is indeed
present.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
struct ParamsWithStats{P<:OrderedDict{VarName,Any},S<:NamedTuple}
    params::P
    stats::S

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
        return new{typeof(params),typeof(stats)}(params, stats)
    end

    function ParamsWithStats(
        varinfo::AbstractVarInfo,
        ::Nothing,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
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
        return new{typeof(params),typeof(stats)}(params, stats)
    end
end

# Re-evaluating the model is unconscionably slow for untyped VarInfo. It's much faster to
# convert it to a typed varinfo first, hence this method.
# https://github.com/TuringLang/Turing.jl/issues/2604
maybe_to_typed_varinfo(vi::VarInfo{<:Metadata}) = typed_varinfo(vi)
maybe_to_typed_varinfo(vi::AbstractVarInfo) = vi

"""
    to_chains(
        Tout::Type{<:AbstractChains},
        params_and_stats::AbstractArray{<:ParamsWithStats}
    )

Convert an array of `ParamsWithStats` to a chains object of type `Tout`.

This function is not implemented here but rather in package extensions for individual chains
packages.
"""
function to_chains end
