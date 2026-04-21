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
Base.copy(plp::PointwiseLogProb) = plp
function (plp::PointwiseLogProb{Prior,Likelihood})(
    val, tval, logjac, vn, dist
) where {Prior,Likelihood}
    if Prior
        return logpdf(dist, val)
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

"""
    _pointwise_logdensities(
        model::Model,
        init_strat::AbstractInitStrategy,
        ::Val{Prior}=Val(true),
        ::Val{Likelihood}=Val(true),
    ) where {Prior,Likelihood}

Shared internal function that computes pointwise log-densities (either priors, likelihoods,
or both).
"""
function _pointwise_logdensities(
    model::Model,
    init_strat::AbstractInitStrategy,
    ::Val{Prior}=Val(true),
    ::Val{Likelihood}=Val(true),
) where {Prior,Likelihood}
    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood}())
    oavi = OnlyAccsVarInfo(acc)
    oavi = last(init!!(model, oavi, init_strat, UnlinkAll()))
    return get_pointwise_logprobs(oavi)
end

"""
    DynamicPPL.pointwise_logdensities(
        model::Model,
        init_strat::AbstractInitStrategy
    )

Calculate the pointwise log-densities for the parameters obtained by evaluating the model
with the given initialisation strategy. The resulting VarNamedTuple will contain both
log-prior probabilities (for random variables) and log-likelihoods (for observed variables).
"""
function pointwise_logdensities(model::Model, init_strat::AbstractInitStrategy)
    return _pointwise_logdensities(model, init_strat, Val(true), Val(true))
end

"""
    DynamicPPL.pointwise_loglikelihoods(
        model::Model,
        init_strat::AbstractInitStrategy
    )

Same as `pointwise_logdensities`, but only returns the log-likelihoods for observed variables.
"""
function pointwise_loglikelihoods(model::Model, init_strat::AbstractInitStrategy)
    return _pointwise_logdensities(model, init_strat, Val(false), Val(true))
end

"""
    DynamicPPL.pointwise_prior_logdensities(
        model::Model,
        init_strat::AbstractInitStrategy
    )

Same as `pointwise_logdensities`, but only returns the log-densities for random variables
(i.e. the priors).
"""
function pointwise_prior_logdensities(model::Model, init_strat::AbstractInitStrategy)
    return _pointwise_logdensities(model, init_strat, Val(true), Val(false))
end
