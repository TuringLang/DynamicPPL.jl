"""
    LogPriorAccumulator{T<:Real} <: AbstractAccumulator

An accumulator that tracks the cumulative log prior during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogPriorAccumulator{T<:Real} <: AbstractAccumulator
    "the scalar log prior value"
    logp::T
end

"""
    LogPriorAccumulator{T}()

Create a new `LogPriorAccumulator` accumulator with the log prior initialized to zero.
"""
LogPriorAccumulator{T}() where {T<:Real} = LogPriorAccumulator(zero(T))
LogPriorAccumulator() = LogPriorAccumulator{LogProbType}()

"""
    LogLikelihoodAccumulator{T<:Real} <: AbstractAccumulator

An accumulator that tracks the cumulative log likelihood during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogLikelihoodAccumulator{T<:Real} <: AbstractAccumulator
    "the scalar log likelihood value"
    logp::T
end

"""
    LogLikelihoodAccumulator{T}()

Create a new `LogLikelihoodAccumulator` accumulator with the log likelihood initialized to zero.
"""
LogLikelihoodAccumulator{T}() where {T<:Real} = LogLikelihoodAccumulator(zero(T))
LogLikelihoodAccumulator() = LogLikelihoodAccumulator{LogProbType}()

"""
    NumProduceAccumulator{T} <: AbstractAccumulator

An accumulator that tracks the number of observations during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct NumProduceAccumulator{T<:Integer} <: AbstractAccumulator
    "the number of observations"
    num::T
end

"""
    NumProduceAccumulator{T<:Integer}()

Create a new `NumProduceAccumulator` accumulator with the number of observations initialized to zero.
"""
NumProduceAccumulator{T}() where {T<:Integer} = NumProduceAccumulator(zero(T))
NumProduceAccumulator() = NumProduceAccumulator{Int}()

function Base.show(io::IO, acc::LogPriorAccumulator)
    return print(io, "LogPriorAccumulator($(repr(acc.logp)))")
end
function Base.show(io::IO, acc::LogLikelihoodAccumulator)
    return print(io, "LogLikelihoodAccumulator($(repr(acc.logp)))")
end
function Base.show(io::IO, acc::NumProduceAccumulator)
    return print(io, "NumProduceAccumulator($(repr(acc.num)))")
end

accumulator_name(::Type{<:LogPriorAccumulator}) = :LogPrior
accumulator_name(::Type{<:LogLikelihoodAccumulator}) = :LogLikelihood
accumulator_name(::Type{<:NumProduceAccumulator}) = :NumProduce

split(::LogPriorAccumulator{T}) where {T} = LogPriorAccumulator(zero(T))
split(::LogLikelihoodAccumulator{T}) where {T} = LogLikelihoodAccumulator(zero(T))
split(acc::NumProduceAccumulator) = acc

function combine(acc::LogPriorAccumulator, acc2::LogPriorAccumulator)
    return LogPriorAccumulator(acc.logp + acc2.logp)
end
function combine(acc::LogLikelihoodAccumulator, acc2::LogLikelihoodAccumulator)
    return LogLikelihoodAccumulator(acc.logp + acc2.logp)
end
function combine(acc::NumProduceAccumulator, acc2::NumProduceAccumulator)
    return NumProduceAccumulator(max(acc.num, acc2.num))
end

function Base.:+(acc1::LogPriorAccumulator, acc2::LogPriorAccumulator)
    return LogPriorAccumulator(acc1.logp + acc2.logp)
end
function Base.:+(acc1::LogLikelihoodAccumulator, acc2::LogLikelihoodAccumulator)
    return LogLikelihoodAccumulator(acc1.logp + acc2.logp)
end
increment(acc::NumProduceAccumulator) = NumProduceAccumulator(acc.num + oneunit(acc.num))

Base.zero(acc::LogPriorAccumulator) = LogPriorAccumulator(zero(acc.logp))
Base.zero(acc::LogLikelihoodAccumulator) = LogLikelihoodAccumulator(zero(acc.logp))
Base.zero(acc::NumProduceAccumulator) = NumProduceAccumulator(zero(acc.num))

function accumulate_assume!!(acc::LogPriorAccumulator, val, logjac, vn, right)
    return acc + LogPriorAccumulator(logpdf(right, val) + logjac)
end
accumulate_observe!!(acc::LogPriorAccumulator, right, left, vn) = acc

accumulate_assume!!(acc::LogLikelihoodAccumulator, val, logjac, vn, right) = acc
function accumulate_observe!!(acc::LogLikelihoodAccumulator, right, left, vn)
    # Note that it's important to use the loglikelihood function here, not logpdf, because
    # they handle vectors differently:
    # https://github.com/JuliaStats/Distributions.jl/issues/1972
    return acc + LogLikelihoodAccumulator(Distributions.loglikelihood(right, left))
end

accumulate_assume!!(acc::NumProduceAccumulator, val, logjac, vn, right) = acc
accumulate_observe!!(acc::NumProduceAccumulator, right, left, vn) = increment(acc)

function Base.convert(::Type{LogPriorAccumulator{T}}, acc::LogPriorAccumulator) where {T}
    return LogPriorAccumulator(convert(T, acc.logp))
end
function Base.convert(
    ::Type{LogLikelihoodAccumulator{T}}, acc::LogLikelihoodAccumulator
) where {T}
    return LogLikelihoodAccumulator(convert(T, acc.logp))
end
function Base.convert(
    ::Type{NumProduceAccumulator{T}}, acc::NumProduceAccumulator
) where {T}
    return NumProduceAccumulator(convert(T, acc.num))
end

# TODO(mhauru)
# We ignore the convert_eltype calls for NumProduceAccumulator, by letting them fallback on
# convert_eltype(::AbstractAccumulator, ::Type). This is because they are only used to
# deal with dual number types of AD backends, which shouldn't concern NumProduceAccumulator. This is
# horribly hacky and should be fixed. See also comment in `unflatten` in `src/varinfo.jl`.
function convert_eltype(::Type{T}, acc::LogPriorAccumulator) where {T}
    return LogPriorAccumulator(convert(T, acc.logp))
end
function convert_eltype(::Type{T}, acc::LogLikelihoodAccumulator) where {T}
    return LogLikelihoodAccumulator(convert(T, acc.logp))
end

function default_accumulators()
    return AccumulatorTuple(
        LogPriorAccumulator{LogProbType}(),
        LogLikelihoodAccumulator{LogProbType}(),
        NumProduceAccumulator{Int}(),
    )
end
