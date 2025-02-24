abstract type AbstractAccumulator end

accumulator_name(acc::AbstractAccumulator) = accumulator_name(typeof(acc))

"""
    AccumulatorTuple{N,T<:NamedTuple}

A collection of accumulators, stored as a `NamedTuple`.

This is defined as a separate type to be able to dispatch on it cleanly and without method
ambiguities or conflicts with other `NamedTuple` types. We also use this type to enforce the
constraint the name in the tuple for each accumulator `acc` must be `accumulator_name(acc)`.
"""
struct AccumulatorTuple{N,T<:NamedTuple}
    nt::T

    function AccumulatorTuple(t::T) where {N,T<:NTuple{N,AbstractAccumulator}}
        names = accumulator_name.(t)
        nt = NamedTuple{names}(t)
        return new{N,typeof(nt)}(nt)
    end
end

AccumulatorTuple(accs::Vararg{AbstractAccumulator}) = AccumulatorTuple(accs)
AccumulatorTuple(nt::NamedTuple) = AccumulatorTuple(tuple(nt...))

Base.getindex(at::AccumulatorTuple, idx) = at.nt[idx]
Base.length(::AccumulatorTuple{N}) where {N} = N
Base.iterate(at::AccumulatorTuple, args...) = iterate(at.nt, args...)

function setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)
    return Accessors.@set at.nt[accumulator_name(acc)] = acc
end

function getacc(at::AccumulatorTuple, ::Type{AccType}) where {AccType}
    return at[accumulator_name(AccType)]
end

function accumulate_assume!!(at::AccumulatorTuple, r, logp, vn, right)
    return AccumulatorTuple(map(acc -> accumulate_assume!!(acc, r, logp, vn, right), at.nt))
end

function accumulate_observe!!(at::AccumulatorTuple, left, right)
    return AccumulatorTuple(map(acc -> accumulate_observe!!(acc, left, right), at.nt))
end

function acc!!(at::AccumulatorTuple, ::Type{AccType}, args...) where {AccType}
    accname = accumulator_name(AccType)
    return Accessors.@set at.nt[accname] = acc!!(at[accname], args...)
end

struct LogPrior{T} <: AbstractAccumulator
    logp::T
end

LogPrior{T}() where {T} = LogPrior(zero(T))

struct LogLikelihood{T} <: AbstractAccumulator
    logp::T
end

LogLikelihood{T}() where {T} = LogLikelihood(zero(T))

struct NumProduce{T<:Integer} <: AbstractAccumulator
    num::T
end

NumProduce{T}() where {T} = NumProduce(zero(T))

accumulator_name(::Type{<:LogPrior}) = :LogPrior
accumulator_name(::Type{<:LogLikelihood}) = :LogLikelihood
accumulator_name(::Type{<:NumProduce}) = :NumProduce

split(::LogPrior{T}) where {T} = LogPrior(zero(T))
split(::LogLikelihood{T}) where {T} = LogLikelihood(zero(T))
split(acc::NumProduce) = acc

combine(acc::LogPrior, acc2::LogPrior) = LogPrior(acc.logp + acc2.logp)
combine(acc::LogLikelihood, acc2::LogLikelihood) = LogLikelihood(acc.logp + acc2.logp)
function combine(acc::NumProduce, acc2::NumProduce)
    return NumProduce(max(acc.num, acc2.num))
end

acc!!(acc::LogPrior, logp) = LogPrior(acc.logp + logp)
acc!!(acc::LogLikelihood, logp) = LogLikelihood(acc.logp + logp)
acc!!(acc::NumProduce, n) = NumProduce(acc.num + n)

function accumulate_assume!!(acc::LogPrior, val, logjac, vn, right)
    return acc!!(acc, logpdf(right, val) + logjac)
end
accumulate_observe!!(acc::LogPrior, left, right) = acc

accumulate_assume!!(acc::LogLikelihood, val, logjac, vn, right) = acc
function accumulate_observe!!(acc::LogLikelihood, left, right)
    return acc!!(acc, logpdf(right, left))
end

accumulate_assume!!(acc::NumProduce, val, logjac, vn, right) = acc
accumulate_observe!!(acc::NumProduce, left, right) = acc!!(acc, 1)
