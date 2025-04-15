"""
    AbstractAccumulator

An abstract type for accumulators.

An accumulator is an object that may change its value at every tilde_assume or tilde_observe
call based on the value of the random variable in question. The obvious examples of
accumulators or the log prior and log likelihood. Others examples might be a variable that
counts the number of observations in a trace, or the names of random variables seen so far.

An accumulator must implement the following methods:
- `accumulator_name(acc::AbstractAccumulator)`: returns a Symbol by which accumulators of
this type are identified. This name is unique in the sense that a `VarInfo` can only have
one accumulator for each name. Often the name is just the name of the type.
- `accumulate_observe!!(acc::AbstractAccumulator, right, left, vn)`: updates `acc` based on
observing the random variable `vn` with value `left`, with `right` being the distribution on
the RHS of the tilde statement. `accumulate_observe!!` may mutate `acc`, but not any of the
other arguments. `vn` is `nothing` in the case of literal observations like
`0.0 ~ Normal()`. `accumulate_observe!!` is called within `tilde_observe!!` for each
accumulator in the current `VarInfo`.
- `accumulate_assume!!(acc::AbstractAccumulator, val, logjac, vn, right)`: updates `acc`
at when a `tilde_assume` call is made. `vn` is the name of the variable being assumed
"""
abstract type AbstractAccumulator end

accumulator_name(acc::AbstractAccumulator) = accumulator_name(typeof(acc))

"""
    AccumulatorTuple{N,T<:NamedTuple}

A collection of accumulators, stored as a `NamedTuple` of length `N`

This is defined as a separate type to be able to dispatch on it cleanly and without method
ambiguities or conflicts with other `NamedTuple` types. We also use this type to enforce the
constraint that the name in the tuple for each accumulator `acc` must be
`accumulator_name(acc)`, and these names must be unique.

The constructor can be called with a tuple or a `VarArgs` of `AbstractAccumulators`. The
names will be generated automatically. One can also call the constructor with a `NamedTuple`
but the names in the argument will be discarded in favour of the generated ones.

# Fields
$(TYPEDFIELDS)
"""
struct AccumulatorTuple{N,T<:NamedTuple}
    nt::T

    function AccumulatorTuple(t::T) where {N,T<:NTuple{N,AbstractAccumulator}}
        names = map(accumulator_name, t)
        nt = NamedTuple{names}(t)
        return new{N,typeof(nt)}(nt)
    end
end

AccumulatorTuple(accs::Vararg{AbstractAccumulator}) = AccumulatorTuple(accs)
AccumulatorTuple(nt::NamedTuple) = AccumulatorTuple(tuple(nt...))

Base.getindex(at::AccumulatorTuple, idx) = at.nt[idx]
Base.length(::AccumulatorTuple{N}) where {N} = N
Base.iterate(at::AccumulatorTuple, args...) = iterate(at.nt, args...)

"""
    setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)

Add `acc` to `at`. Returns a new `AccumulatorTuple`.

If an `AbstractAccumulator` with the same `accumulator_name` already exists in `at` it is
replaced. `at` will never be mutated, but the name has the `!!` for consistency with the
corresponding function for `AbstractVarInfo`.
"""
function setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)
    return Accessors.@set at.nt[accumulator_name(acc)] = acc
end

function getacc(at::AccumulatorTuple, ::Val{accname}) where {accname}
    return at[accname]
end

function accumulate_assume!!(at::AccumulatorTuple, r, logjac, vn, right)
    return AccumulatorTuple(
        map(acc -> accumulate_assume!!(acc, r, logjac, vn, right), at.nt)
    )
end

function accumulate_observe!!(at::AccumulatorTuple, right, left, vn)
    return AccumulatorTuple(map(acc -> accumulate_observe!!(acc, right, left, vn), at.nt))
end

function acc!!(at::AccumulatorTuple, ::Val{accname}, args...) where {accname}
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

resetacc!!(acc::LogPrior) = LogPrior(zero(acc.logp))
resetacc!!(acc::LogLikelihood) = LogLikelihood(zero(acc.logp))
# TODO(mhauru) How to handle reset for NumProduce? Do we need to define different types of
# resets?
resetacc!!(acc::NumProduce) = acc

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
accumulate_observe!!(acc::LogPrior, right, left, vn) = acc

accumulate_assume!!(acc::LogLikelihood, val, logjac, vn, right) = acc
function accumulate_observe!!(acc::LogLikelihood, right, left, vn)
    return acc!!(acc, logpdf(right, left))
end

accumulate_assume!!(acc::NumProduce, val, logjac, vn, right) = acc
accumulate_observe!!(acc::NumProduce, right, left, vn) = acc!!(acc, 1)
