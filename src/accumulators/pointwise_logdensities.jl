import PosteriorStats

"""
    PointwiseLogProb{Prior,Likelihood}

A callable struct that computes the log probability of a given value under a distribution.
The `Prior` and `Likelihood` type parameters are used to control whether the log probability
is computed for prior or likelihood terms, respectively.

This struct is used in conjunction with `VNTAccumulator`, via

    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood}())

where `Prior` and `Likelihood` are the boolean type parameters. This accumulator will then
store the log-probabilities for all tilde-statements in the model.
"""
struct PointwiseLogProb{Prior,Likelihood,Factorised} end
function (plp::PointwiseLogProb{Prior,Likelihood,Factorised})(
    val, tval, logjac, vn, dist
) where {Prior,Likelihood,Factorised}
    return if Prior
        if Factorised && hasmethod(
            PosteriorStats.pointwise_conditional_loglikelihoods,
            Tuple{typeof(val),typeof([dist])},
        )
            dropdims(
                PosteriorStats.pointwise_conditional_loglikelihoods(val, [dist]); dims=1
            )
        else
            logpdf(dist, val)
        end
    else
        return DoNotAccumulate()
    end
end
const POINTWISE_ACCNAME = :PointwiseLogProb

# Not exported
function get_pointwise_logprobs(varinfo::AbstractVarInfo)
    return getacc(varinfo, Val(POINTWISE_ACCNAME)).values
end

# Have to overload accumulate_assume!! since VNTAccumulator by default does not track
# observe statements.
function accumulate_observe!!(
    acc::VNTAccumulator{POINTWISE_ACCNAME,PointwiseLogProb{Prior,Likelihood,Factorised}},
    right,
    left,
    vn,
    template,
) where {Prior,Likelihood,Factorised}
    # vn could be `nothing`, in which case we can't store it in a VNT.
    return if Likelihood && vn isa VarName
        logp =
            if Factorised && hasmethod(
                PosteriorStats.pointwise_conditional_loglikelihoods,
                Tuple{typeof(left),typeof([right])},
            )
                dropdims(
                    PosteriorStats.pointwise_conditional_loglikelihoods(left, [right]);
                    dims=1,
                )
            else
                logpdf(right, left)
            end
        new_values = DynamicPPL.templated_setindex!!(acc.values, logp, vn, template)
        return VNTAccumulator{POINTWISE_ACCNAME}(acc.f, new_values)
    else
        # No need to accumulate likelihoods.
        acc
    end
end

"""
    _pointwise_logdensities(
        model::Model,
        varinfo::AbstractVarInfo,
        ::Val{Prior}=Val(true),
        ::Val{Likelihood}=Val(true);
        factorize=false
    ) where {Prior,Likelihood}

Shared internal function that computes pointwise log-densities (either priors, likelihoods,
or both).
"""
function _pointwise_logdensities(
    model::Model,
    varinfo::AbstractVarInfo,
    ::Val{Prior}=Val(true),
    ::Val{Likelihood}=Val(true);
    factorize=false,
) where {Prior,Likelihood}
    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood,factorize}())
    oavi = OnlyAccsVarInfo(acc)
    init_strategy = InitFromParams(varinfo.values, nothing)
    oavi = last(init!!(model, oavi, init_strategy, UnlinkAll()))
    return get_pointwise_logprobs(oavi)
end

function pointwise_logdensities(model::Model, varinfo::AbstractVarInfo; factorize=false)
    return _pointwise_logdensities(
        model, varinfo, Val(true), Val(true); factorize=factorize
    )
end

function pointwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo; factorize=false)
    return _pointwise_logdensities(
        model, varinfo, Val(false), Val(true); factorize=factorize
    )
end

function pointwise_prior_logdensities(
    model::Model, varinfo::AbstractVarInfo; factorize=false
)
    return _pointwise_logdensities(
        model, varinfo, Val(true), Val(false); factorize=factorize
    )
end
