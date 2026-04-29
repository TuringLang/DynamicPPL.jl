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

const _FACTORIZE_KWARG_DOC = """
If `factorize=true`, additionally attempt to provide factorised log-densities for
distributions that can be partitioned into blocks, using PartitionedDistributions.jl.

For example, if `factorize=true`, then `y ~ MvNormal(...)` will return a vector of
log-densities, one for each element of `y`. The `i`-th element of this vector will be the
conditional log-probability of `y[i]` given all the other elements of `y` (often denoted
`log p(y_{i} | y_{-i})`): in particular this is exactly the log-density required for
leave-one-out cross-validation.

In contrast, if `factorize=false`, then the log-density for `y ~ MvNormal(...)` will be a
single scalar corresponding to `logpdf(MvNormal(...), y)`.

Note that the sum of the factorised log-densities may not, in general, be equal to the
log-density of the full distribution: they will only be equal if the original distribution
can be completely factorised into independent components. For example, if `y ~ MvNormal(μ,
Σ)` where `Σ` is diagonal, then each element of `y` is independent and the sum of the
factorised log-densities will be equal to the log-density of the full distribution. In
contrast, if `Σ` has off-diagonal entries, then the elements of `y` are not independent.
"""

"""
    PointwiseLogProb{Prior,Likelihood,Factorised}

A callable struct that computes the log probability of a given value under a distribution.
The `Prior` and `Likelihood` type parameters are used to control whether the log probability
is computed for prior or likelihood terms, respectively. The `Factorised` type parameter
controls whether to attempt to factorise the log-densities.

This struct is used in conjunction with `VNTAccumulator`, via

    acc = VNTAccumulator{POINTWISE_ACCNAME}(PointwiseLogProb{Prior,Likelihood,Factorised}())

where `Prior`, `Likelihood`, and `Factorised` are the boolean type parameters. This
accumulator will then store the log-probabilities for all tilde-statements in the model.
"""
struct PointwiseLogProb{Prior,Likelihood,Factorised} end
function PointwiseLogProb{Prior,Likelihood}() where {Prior,Likelihood}
    # Default definition to preserve backwards compatibility
    return PointwiseLogProb{Prior,Likelihood,false}()
end
Base.copy(plp::PointwiseLogProb) = plp
function (plp::PointwiseLogProb{Prior,Likelihood,Factorised})(
    val, tval, logjac, vn, dist
) where {Prior,Likelihood,Factorised}
    return if Prior
        _maybe_pointwise_logpdf(dist, val, Val{Factorised}())
    else
        DoNotAccumulate()
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
        VNTAccumulator{POINTWISE_ACCNAME}(acc.f, new_values)
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
    )::VarNamedTuple

Calculate the pointwise log-densities for the parameters obtained by evaluating the model
with the given initialisation strategy. The resulting VarNamedTuple will contain both
log-prior probabilities (for random variables) and log-likelihoods (for observed variables).

$(_FACTORIZE_KWARG_DOC)
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
    )::VarNamedTuple

Calculate the pointwise log-likelihoods for observed variables, using parameters obtained
from the given initialisation strategy.

$(_FACTORIZE_KWARG_DOC)
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
    )::VarNamedTuple

Calculate the pointwise log-prior probabilities for random variables, using parameters
obtained from the given initialisation strategy.

$(_FACTORIZE_KWARG_DOC)
"""
function pointwise_prior_logdensities(
    model::Model, init_strat::AbstractInitStrategy; factorize=false
)
    return _pointwise_logdensities(
        model, init_strat, Val(true), Val(false); factorize=factorize
    )
end
