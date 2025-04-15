"""
    AbstractAccumulator

An abstract type for accumulators.

An accumulator is an object that may change its value at every tilde_assume or tilde_observe
call based on the value of the random variable in question. The obvious examples of
accumulators are the log prior and log likelihood. Others examples might be a variable that
counts the number of observations in a trace, or a list of the names of random variables
seen so far.

An accumulator type `T` must implement the following methods:
- `accumulator_name(acc::T)`
- `accumulate_observe!!(acc::T, right, left, vn)`
- `accumulate_assume!!(acc::T, val, logjac, vn, right)`

To be able to work with multi-threading, it should also implement:
- `split(acc::T)`
- `combine(acc::T, acc2::T)`

It may also want to implement
- `acc!!(acc::T, args...)`

See the documentation for each of these functions for more details.
"""
abstract type AbstractAccumulator end

# TODO(mhauru) Add to the above docstring stuff about resets.

"""
    accumulator_name(acc::AbstractAccumulator)

Return a Symbol which can be used as a name for `acc`.

The name has to be unique in the sense that a `VarInfo` can only have one accumulator for
each name. The most typical case, and the default implementation, is that the name only
depends on the type of `acc`, not on its value.
"""
accumulator_name(acc::AbstractAccumulator) = accumulator_name(typeof(acc))

"""
    accumulate_observe!!(acc::AbstractAccumulator, right, left, vn)

Update `acc` in a `tilde_observe` call. Returns the updated `acc`.

`vn` is the name of the variable being observed, `left` is the value of the variable, and
`right` is the distribution on the RHS of the tilde statement. `vn` is `nothing` in the case
of literal observations like `0.0 ~ Normal()`.

`accumulate_observe!!` may mutate `acc`, but not any of the other arguments.

See also: [`accumulate_assume!!`](@ref)
"""
function accumulate_observe!! end

"""
    accumulate_assume!!(acc::AbstractAccumulator, val, logjac, vn, right)

Update `acc` in a `tilde_assume` call. Returns the updated `acc`.

`vn` is the name of the variable being assumed, `val` is the value of the variable, and
`right` is the distribution on the RHS of the tilde statement. `logjac` is the log
determinant of the Jacobian of the transformation that was done to convert the value of `vn`
as it was given (e.g. by sampler operating in linked space) to `val`.

`accumulate_assume!!` may mutate `acc`, but not any of the other arguments.

See also: [`accumulate_observe!!`](@ref)
"""
function accumulate_assume!! end

"""
    split(acc::AbstractAccumulator)

Return a new accumulator like `acc` but empty.

The precise meaning of "empty" is that that the returned value should be such that
`combine(acc, split(acc))` is equal to `acc`. This is used in the context of multi-threading
where different threads may accumulate independently and the results are the combined.

See also: [`combine`](@ref)
"""
function split end

"""
    combine(acc::AbstractAccumulator, acc2::AbstractAccumulator)

Combine two accumulators of the same type. Returns a new accumulator.

See also: [`split`](@ref)
"""
function combine end

"""
    acc!!(acc::AbstractAccumulator, args...)

Update `acc` with the values in `args`. Returns the updated `acc`.

What this means depends greatly on the type of `acc`. For example, for `LogPrior` `args`
would be just `logp`. The utility of this function is that one can call
`acc!!(varinfo::AbstractVarinfo, Val(accname), args...)`, and this call will be propagated
to a call on the particular accumulator.
"""
function acc!! end

# END ABSTRACT ACCUMULATOR, BEGIN ACCUMULATOR TUPLE

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

"""
    getacc(at::AccumulatorTuple, ::Val{accname})

Get the accumulator with name `accname` from `at`.
"""
function getacc(at::AccumulatorTuple, ::Val{accname}) where {accname}
    return at[accname]
end

"""
    accumulate_assume!!(at::AccumulatorTuple, r, logjac, vn, right)

Call `accumulate_assume!!` on each accumulator in `at`.

Returns a new AccumulatorTuple.
"""
function accumulate_assume!!(at::AccumulatorTuple, r, logjac, vn, right)
    return AccumulatorTuple(
        map(acc -> accumulate_assume!!(acc, r, logjac, vn, right), at.nt)
    )
end

"""
    accumulate_observe!!(at::AccumulatorTuple, right, left, vn)

Call `accumulate_observe!!` on each accumulator in `at`.

Returns a new AccumulatorTuple.
"""
function accumulate_observe!!(at::AccumulatorTuple, right, left, vn)
    return AccumulatorTuple(map(acc -> accumulate_observe!!(acc, right, left, vn), at.nt))
end

function acc!!(at::AccumulatorTuple, ::Val{accname}, args...) where {accname}
    return Accessors.@set at.nt[accname] = acc!!(at[accname], args...)
end

# END ACCUMULATOR TUPLE, BEGIN LOG PROB AND NUM PRODUCE ACCUMULATORS

"""
    LogPrior{T} <: AbstractAccumulator

An accumulator that tracks the cumulative log prior during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogPrior{T} <: AbstractAccumulator
    logp::T
end

"""
    LogPrior{T}() where {T}

Create a new `LogPrior` accumulator with the log prior initialized to zero.
"""
LogPrior{T}() where {T} = LogPrior(zero(T))

"""
    LogLikelihood{T} <: AbstractAccumulator

An accumulator that tracks the cumulative log likelihood during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogLikelihood{T} <: AbstractAccumulator
    logp::T
end

"""
    LogLikelihood{T}() where {T}

Create a new `LogLikelihood` accumulator with the log likelihood initialized to zero.
"""
LogLikelihood{T}() where {T} = LogLikelihood(zero(T))

"""
    NumProduce{T} <: AbstractAccumulator

An accumulator that tracks the number of observations during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct NumProduce{T<:Integer} <: AbstractAccumulator
    num::T
end

"""
    NumProduce{T}() where {T<:Integer}

Create a new `NumProduce` accumulator with the number of observations initialized to zero.
"""
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
