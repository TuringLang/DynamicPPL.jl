abstract type AtomicLogProbAccumulator{T} <: LogProbAccumulator{T} end

function acclogp(acc::AtomicLogProbAccumulator, val)
    @atomic :monotonic acc.logp + val
    return acc
end
logp(acc::AtomicLogProbAccumulator) = acc.logp

mutable struct AtomicLogPriorAccumulator{T<:Real} <: AtomicLogProbAccumulator{T}
    @atomic logp::T
end
accumulator_name(::Type{<:AtomicLogPriorAccumulator}) = :LogPrior
function accumulate_assume!!(acc::AtomicLogPriorAccumulator, val, logjac, vn, right)
    return acclogp(acc, logpdf(right, val))
end
accumulate_observe!!(acc::AtomicLogPriorAccumulator, right, left, vn) = acc

mutable struct AtomicLogJacobianAccumulator{T<:Real} <: AtomicLogProbAccumulator{T}
    @atomic logp::T
end
accumulator_name(::Type{<:AtomicLogJacobianAccumulator}) = :LogJacobian
function accumulate_assume!!(acc::AtomicLogJacobianAccumulator, val, logjac, vn, right)
    return acclogp(acc, logjac)
end
accumulate_observe!!(acc::AtomicLogJacobianAccumulator, right, left, vn) = acc

mutable struct AtomicLogLikelihoodAccumulator{T<:Real} <: AtomicLogProbAccumulator{T}
    @atomic logp::T
end
accumulator_name(::Type{<:AtomicLogLikelihoodAccumulator}) = :LogLikelihood
accumulate_assume!!(acc::AtomicLogLikelihoodAccumulator, val, logjac, vn, right) = acc
function accumulate_observe!!(acc::AtomicLogLikelihoodAccumulator, right, left, vn)
    return acclogp(acc, Distributions.loglikelihood(right, left))
end

function default_atomic_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        AtomicLogPriorAccumulator{FloatT}(),
        AtomicLogJacobianAccumulator{FloatT}(),
        AtomicLogLikelihoodAccumulator{FloatT}(),
    )
end
