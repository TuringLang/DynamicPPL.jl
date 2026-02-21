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
struct PointwiseLogProb{Prior,Likelihood} end
function (plp::PointwiseLogProb{Prior,Likelihood})(
    val, tval, logjac, vn, dist
) where {Prior,Likelihood}
    if Prior
        return logpdf(dist, val)
    else
        return DoNotAccumulate()
    end
end
const POINTWISE_ACCNAME = :PointwiseLogProbAccumulator

# Not exported
function get_pointwise_logprobs(varinfo::AbstractVarInfo)
    return getacc(varinfo, Val(POINTWISE_ACCNAME)).values
end

# Have to overload accumulate_assume!! since VNTAccumulator by default does not track
# observe statements.
function accumulate_observe!!(
    acc::VNTAccumulator{POINTWISE_ACCNAME,PointwiseLogProb{Prior,Likelihood}},
    right,
    left,
    vn,
    template,
) where {Prior,Likelihood}
    # vn could be `nothing`, in which case we can't store it in a VNT.
    return if Likelihood && vn isa VarName
        logp = logpdf(right, left)
        new_values = DynamicPPL.templated_setindex!!(acc.values, logp, vn, template)
        return VNTAccumulator{POINTWISE_ACCNAME}(acc.f, new_values)
    else
        # No need to accumulate likelihoods.
        acc
    end
end

function pointwise_logdensities(
    model::Model,
    varinfo::AbstractVarInfo,
    ::Val{Prior}=Val(true),
    ::Val{Likelihood}=Val(true),
) where {Prior,Likelihood}
    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood}())
    oavi = OnlyAccsVarInfo(acc)
    init_strategy = InitFromParams(varinfo.values, nothing)
    oavi = last(init!!(model, oavi, init_strategy, UnlinkAll()))
    return get_pointwise_logprobs(oavi)
end

function pointwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo)
    return pointwise_logdensities(model, varinfo, Val(false), Val(true))
end

function pointwise_prior_logdensities(model::Model, varinfo::AbstractVarInfo)
    return pointwise_logdensities(model, varinfo, Val(true), Val(false))
end
