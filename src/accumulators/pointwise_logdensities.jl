import PartitionedDistributions

# Force specialisation on D and V.
function _maybe_pointwise_logpdf(dist::D, value::V, ::Val{true}) where {D<:Distribution,V}
    return if hasmethod(
        PartitionedDistributions.pointwise_conditional_logpdfs,
        Tuple{typeof(dist),typeof(value)},
    )
        PartitionedDistributions.pointwise_conditional_logpdfs(dist, value)
    else
        logpdf(dist, value)
    end
end
function _maybe_pointwise_logpdf(dist::Distribution, value, ::Val{false})
    return logpdf(dist, value)
end

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
Base.copy(plp::PointwiseLogProb) = plp
function (plp::PointwiseLogProb{Prior,Likelihood,Factorised})(
    val, tval, logjac, vn, dist
) where {Prior,Likelihood,Factorised}
    return if Prior
        _maybe_pointwise_logpdf(dist, val, Val{Factorised}())
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
        logp = _maybe_pointwise_logpdf(right, left, Val{Factorised}())
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
        ::Val{Likelihood}=Val(true);
        factorize=false
    ) where {Prior,Likelihood}

Shared internal function that computes pointwise log-densities (either priors, likelihoods,
or both).
"""
function _pointwise_logdensities(
    model::Model,
    init_strat::AbstractInitStrategy,
    ::Val{Prior}=Val(true),
    ::Val{Likelihood}=Val(true);
    factorize=false,
) where {Prior,Likelihood}
    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood,factorize}())
    oavi = OnlyAccsVarInfo(acc)
    oavi = last(init!!(model, oavi, init_strat, UnlinkAll()))
    return get_pointwise_logprobs(oavi)
end

"""
    DynamicPPL.pointwise_logdensities(
        model::Model,
        init_strat::AbstractInitStrategy;
        factorize=false 
    )

Calculate the pointwise log-densities for the parameters obtained by evaluating the model
with the given initialisation strategy. The resulting VarNamedTuple will contain both
log-prior probabilities (for random variables) and log-likelihoods (for observed variables).

If `factorize=true`, additionally attempt to provide factorised log-densities for
distributions that can be partitioned into blocks, using PartitionedDistributions.jl. For
example, if `factorize=true`, then `y ~ MvNormal(...)` will return a vector of
log-densities, one for each element of `y`. If `factorize=false`, then the log-density for
`y ~ MvNormal(...)` will be a single scalar.
"""
function pointwise_logdensities(
    model::Model, init_strat::AbstractInitStrategy; factorize=false
)
    return _pointwise_logdensities(
        model, init_strat, Val(true), Val(true); factorize=factorize
    )
end

"""
    DynamicPPL.pointwise_loglikelihoods(
        model::Model,
        init_strat::AbstractInitStrategy;
        factorize=false
    )

Same as `pointwise_logdensities`, but only returns the log-likelihoods for observed variables.
"""
function pointwise_loglikelihoods(
    model::Model, init_strat::AbstractInitStrategy; factorize=false
)
    return _pointwise_logdensities(
        model, init_strat, Val(false), Val(true); factorize=factorize
    )
end

"""
    DynamicPPL.pointwise_prior_logdensities(
        model::Model,
        init_strat::AbstractInitStrategy;
        factorize=false
    )

Same as `pointwise_logdensities`, but only returns the log-densities for random variables
(i.e. the priors).
"""
function pointwise_prior_logdensities(
    model::Model, init_strat::AbstractInitStrategy; factorize=false
)
    return _pointwise_logdensities(
        model, init_strat, Val(true), Val(false); factorize=factorize
    )
end
