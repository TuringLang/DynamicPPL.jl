"""
    AbstractAccumulator

An abstract type for accumulators.

An accumulator is an object that may change its value at every tilde_assume!! or
tilde_observe!! call based on the random variable in question. The obvious examples of
accumulators are the log prior and log likelihood. Other examples might be a variable that
counts the number of observations in a trace, or a list of the names of random variables
seen so far.

An accumulator type `T <: AbstractAccumulator` must implement the following methods:
- `accumulator_name(acc::T)` or `accumulator_name(::Type{T})`
- `accumulate_observe!!(acc::T, right, left, vn)`
- `accumulate_assume!!(acc::T, val, logjac, vn, right)`

To be able to work with multi-threading, it should also implement:
- `split(acc::T)`
- `combine(acc::T, acc2::T)`

See the documentation for each of these functions for more details.
"""
abstract type AbstractAccumulator end

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

Update `acc` in a `tilde_observe!!` call. Returns the updated `acc`.

`vn` is the name of the variable being observed, `left` is the value of the variable, and
`right` is the distribution on the RHS of the tilde statement. `vn` is `nothing` in the case
of literal observations like `0.0 ~ Normal()`.

`accumulate_observe!!` may mutate `acc`, but not any of the other arguments.

See also: [`accumulate_assume!!`](@ref)
"""
function accumulate_observe!! end

"""
    accumulate_assume!!(acc::AbstractAccumulator, val, logjac, vn, right)

Update `acc` in a `tilde_assume!!` call. Returns the updated `acc`.

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

# TODO(mhauru) The existence of this function makes me sad. See comment in unflatten in
# src/varinfo.jl.
"""
    convert_eltype(::Type{T}, acc::AbstractAccumulator)

Convert `acc` to use element type `T`.

What "element type" means depends on the type of `acc`. By default this function does
nothing. Accumulator types that need to hold differentiable values, such as dual numbers
used by various AD backends, should implement a method for this function.
"""
convert_eltype(::Type, acc::AbstractAccumulator) = acc

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

# When showing with text/plain, leave out information about the wrapper AccumulatorTuple.
Base.show(io::IO, mime::MIME"text/plain", at::AccumulatorTuple) = show(io, mime, at.nt)
Base.getindex(at::AccumulatorTuple, idx) = at.nt[idx]
Base.length(::AccumulatorTuple{N}) where {N} = N
Base.iterate(at::AccumulatorTuple, args...) = iterate(at.nt, args...)
function Base.haskey(at::AccumulatorTuple, ::Val{accname}) where {accname}
    # @inline to ensure constant propagation can resolve this to a compile-time constant.
    @inline return haskey(at.nt, accname)
end
Base.keys(at::AccumulatorTuple) = keys(at.nt)

function Base.convert(::Type{AccumulatorTuple{N,T}}, accs::AccumulatorTuple{N}) where {N,T}
    return AccumulatorTuple(convert(T, accs.nt))
end

"""
    setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)

Add `acc` to `at`. Returns a new `AccumulatorTuple`.

If an `AbstractAccumulator` with the same `accumulator_name` already exists in `at` it is
replaced. `at` will never be mutated, but the name has the `!!` for consistency with the
corresponding function for `AbstractVarInfo`.
"""
function setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)
    accname = accumulator_name(acc)
    new_nt = merge(at.nt, NamedTuple{(accname,)}((acc,)))
    return AccumulatorTuple(new_nt)
end

"""
    getacc(at::AccumulatorTuple, ::Val{accname})

Get the accumulator with name `accname` from `at`.
"""
function getacc(at::AccumulatorTuple, ::Val{accname}) where {accname}
    return at[accname]
end

function Base.map(func::Function, at::AccumulatorTuple)
    return AccumulatorTuple(map(func, at.nt))
end

"""
    map_accumulator(func::Function, at::AccumulatorTuple, ::Val{accname})

Update the accumulator with name `accname` in `at` by calling `func` on it.

Returns a new `AccumulatorTuple`.
"""
function map_accumulator(
    func::Function, at::AccumulatorTuple, ::Val{accname}
) where {accname}
    # Would like to write this as
    # return Accessors.@set at.nt[accname] = func(at[accname], args...)
    # for readability, but that one isn't type stable due to
    # https://github.com/JuliaObjects/Accessors.jl/issues/198
    new_val = func(at[accname])
    new_nt = merge(at.nt, NamedTuple{(accname,)}((new_val,)))
    return AccumulatorTuple(new_nt)
end
