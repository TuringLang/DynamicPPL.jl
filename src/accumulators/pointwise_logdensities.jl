"""
    PointwiseLogProbAccumulator{whichlogprob} <: AbstractAccumulator

An accumulator that stores the log-probabilities of each variable in a model.

Internally this accumulator stores the log-probabilities in a dictionary, where the keys are
the variable names and the values are log-probabilities.

`whichlogprob` is a symbol that can be `:both`, `:prior`, or `:likelihood`, and specifies
which log-probabilities to store in the accumulator.
"""
struct PointwiseLogProbAccumulator{whichlogprob} <: AbstractAccumulator
    logps::OrderedDict{VarName,LogProbType}

    function PointwiseLogProbAccumulator{whichlogprob}(
        d::OrderedDict{VarName,LogProbType}=OrderedDict{VarName,LogProbType}()
    ) where {whichlogprob}
        return new{whichlogprob}(d)
    end
end

function Base.:(==)(
    acc1::PointwiseLogProbAccumulator{wlp1}, acc2::PointwiseLogProbAccumulator{wlp2}
) where {wlp1,wlp2}
    return (wlp1 == wlp2 && acc1.logps == acc2.logps)
end

function Base.copy(acc::PointwiseLogProbAccumulator{whichlogprob}) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}(copy(acc.logps))
end

function accumulator_name(
    ::Type{<:PointwiseLogProbAccumulator{whichlogprob}}
) where {whichlogprob}
    return Symbol("PointwiseLogProbAccumulator{$whichlogprob}")
end

function _zero(::PointwiseLogProbAccumulator{whichlogprob}) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}()
end
reset(acc::PointwiseLogProbAccumulator) = _zero(acc)
split(acc::PointwiseLogProbAccumulator) = _zero(acc)
function combine(
    acc::PointwiseLogProbAccumulator{whichlogprob},
    acc2::PointwiseLogProbAccumulator{whichlogprob},
) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}(mergewith(+, acc.logps, acc2.logps))
end

function accumulate_assume!!(
    acc::PointwiseLogProbAccumulator{whichlogprob}, val, tval, logjac, vn, right, template
) where {whichlogprob}
    if whichlogprob == :both || whichlogprob == :prior
        acc.logps[vn] = logpdf(right, val)
    end
    return acc
end

function accumulate_observe!!(
    acc::PointwiseLogProbAccumulator{whichlogprob}, right, left, vn
) where {whichlogprob}
    # If `vn` is nothing the LHS of ~ is a literal and we don't have a name to attach this
    # acc to, and thus do nothing.
    if vn === nothing
        return acc
    end
    if whichlogprob == :both || whichlogprob == :likelihood
        acc.logps[vn] = loglikelihood(right, left)
    end
    return acc
end

function pointwise_logdensities(
    model::Model, varinfo::AbstractVarInfo, ::Val{whichlogprob}=Val(:both)
) where {whichlogprob}
    AccType = PointwiseLogProbAccumulator{whichlogprob}
    oavi = OnlyAccsVarInfo((AccType(),))
    init_strategy = InitFromParams(varinfo.values, nothing)
    oavi = last(init!!(model, oavi, init_strategy, UnlinkAll()))
    return getacc(oavi, Val(accumulator_name(AccType))).logps
end

function pointwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo)
    return pointwise_logdensities(model, varinfo, Val(:likelihood))
end

function pointwise_prior_logdensities(model::Model, varinfo::AbstractVarInfo)
    return pointwise_logdensities(model, varinfo, Val(:prior))
end
