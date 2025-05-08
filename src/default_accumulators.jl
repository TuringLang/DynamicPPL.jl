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
    VariableOrderAccumulator{T} <: AbstractAccumulator

An accumulator that tracks the order of variables in a `VarInfo`.

This doesn't track the full ordering, but rather how many observations have taken place
before the assume statement for each variable. This is needed for particle methods, where
the model is segmented into parts by each observation, and we need to know which part each
assume statement is in.

# Fields
$(TYPEDFIELDS)
"""
struct VariableOrderAccumulator{Eltype<:Integer,VNType<:VarName} <: AbstractAccumulator
    "the number of observations"
    num_produce::Eltype
    "mapping of variable names to their order in the model"
    order::OrderedDict{VNType, Eltype}
end

"""
    VariableOrderAccumulator{T<:Integer}(n=zero(T))

Create a new `VariableOrderAccumulator` accumulator with the number of observations set to n
"""
VariableOrderAccumulator{T}(n=zero(T)) where {T<:Integer} = VariableOrderAccumulator(convert(T, n), OrderedDict{VarName, T}())
VariableOrderAccumulator(n) = VariableOrderAccumulator{typeof(n)}(n)
VariableOrderAccumulator() = VariableOrderAccumulator{Int}()

function Base.show(io::IO, acc::LogPriorAccumulator)
    return print(io, "LogPriorAccumulator($(repr(acc.logp)))")
end
function Base.show(io::IO, acc::LogLikelihoodAccumulator)
    return print(io, "LogLikelihoodAccumulator($(repr(acc.logp)))")
end
function Base.show(io::IO, acc::VariableOrderAccumulator)
    return print(io, "VariableOrderAccumulator($(repr(acc.num_produce)), $(repr(acc.order)))")
end

accumulator_name(::Type{<:LogPriorAccumulator}) = :LogPrior
accumulator_name(::Type{<:LogLikelihoodAccumulator}) = :LogLikelihood
accumulator_name(::Type{<:VariableOrderAccumulator}) = :VariableOrder

split(::LogPriorAccumulator{T}) where {T} = LogPriorAccumulator(zero(T))
split(::LogLikelihoodAccumulator{T}) where {T} = LogLikelihoodAccumulator(zero(T))
split(acc::VariableOrderAccumulator) = acc

function combine(acc::LogPriorAccumulator, acc2::LogPriorAccumulator)
    return LogPriorAccumulator(acc.logp + acc2.logp)
end
function combine(acc::LogLikelihoodAccumulator, acc2::LogLikelihoodAccumulator)
    return LogLikelihoodAccumulator(acc.logp + acc2.logp)
end
function combine(acc::VariableOrderAccumulator, acc2::VariableOrderAccumulator)
    # Note that assumptions are not allowed within in parallelised blocks, and thus the
    # dictionaries should be identical.
    return VariableOrderAccumulator(max(acc.num_produce, acc2.num_produce), merge(acc.order, acc2.order))
end

function Base.:+(acc1::LogPriorAccumulator, acc2::LogPriorAccumulator)
    return LogPriorAccumulator(acc1.logp + acc2.logp)
end
function Base.:+(acc1::LogLikelihoodAccumulator, acc2::LogLikelihoodAccumulator)
    return LogLikelihoodAccumulator(acc1.logp + acc2.logp)
end
increment(acc::VariableOrderAccumulator) = VariableOrderAccumulator(acc.num_produce + oneunit(acc.num_produce), acc.order)

Base.zero(acc::LogPriorAccumulator) = LogPriorAccumulator(zero(acc.logp))
Base.zero(acc::LogLikelihoodAccumulator) = LogLikelihoodAccumulator(zero(acc.logp))

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

function accumulate_assume!!(acc::VariableOrderAccumulator, val, logjac, vn, right)
    acc.order[vn] = acc.num_produce
    return acc
end
accumulate_observe!!(acc::VariableOrderAccumulator, right, left, vn) = increment(acc)

function Base.convert(::Type{LogPriorAccumulator{T}}, acc::LogPriorAccumulator) where {T}
    return LogPriorAccumulator(convert(T, acc.logp))
end
function Base.convert(
    ::Type{LogLikelihoodAccumulator{T}}, acc::LogLikelihoodAccumulator
) where {T}
    return LogLikelihoodAccumulator(convert(T, acc.logp))
end
function Base.convert(
    ::Type{VariableOrderAccumulator{ElType, VnType}}, acc::VariableOrderAccumulator
) where {ElType, VnType}
    order = OrderedDict{VnType, ElType}()
    for (k, v) in acc.order
        order[convert(VnType, k)] = convert(ElType, v)
    end
    return VariableOrderAccumulator(convert(ElType, acc.num_produce), order)
end

# TODO(mhauru)
# We ignore the convert_eltype calls for VariableOrderAccumulator, by letting them fallback on
# convert_eltype(::AbstractAccumulator, ::Type). This is because they are only used to
# deal with dual number types of AD backends, which shouldn't concern VariableOrderAccumulator. This is
# horribly hacky and should be fixed. See also comment in `unflatten` in `src/varinfo.jl`.
function convert_eltype(::Type{T}, acc::LogPriorAccumulator) where {T}
    return LogPriorAccumulator(convert(T, acc.logp))
end
function convert_eltype(::Type{T}, acc::LogLikelihoodAccumulator) where {T}
    return LogLikelihoodAccumulator(convert(T, acc.logp))
end

function default_accumulators(
    ::Type{FloatT}=LogProbType, ::Type{IntT}=Int
) where {FloatT,IntT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        VariableOrderAccumulator{IntT}(),
    )
end
