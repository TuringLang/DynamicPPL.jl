"""
    LogProbAccumulator{T} <: AbstractAccumulator

An abstract type for accumulators that hold a single scalar log probability value.

Every subtype of `LogProbAccumulator` must implement
* A method for `logp` that returns the scalar log probability value that defines it.
* A single-argument constructor that takes a `logp` value.
* `accumulator_name`, `accumulate_assume!!`, and `accumulate_observe!!` methods like any
    other accumulator.

`LogProbAccumulator` provides implementations for other common functions, like convenience
constructors, `copy`, `show`, `==`, `isequal`, `hash`, `split`, and `combine`.

This type has no great conceptual significance, it just reduces code duplication between
types like LogPriorAccumulator, LogJacobianAccumulator, and LogLikelihoodAccumulator.
"""
abstract type LogProbAccumulator{T<:Real} <: AbstractAccumulator end

# The first of the below methods sets AccType{T}() = AccType(zero(T)) for any
# AccType <: LogProbAccumulator{T}. The second one sets LogProbType as the default eltype T
# when calling AccType().
"""
    LogProbAccumulator{T}()

Create a new `LogProbAccumulator` accumulator with the log prior initialized to zero.
"""
(::Type{AccType})() where {T<:Real,AccType<:LogProbAccumulator{T}} = AccType(zero(T))
(::Type{AccType})() where {AccType<:LogProbAccumulator} = AccType{LogProbType}()

Base.copy(acc::LogProbAccumulator) = acc

function Base.show(io::IO, acc::LogProbAccumulator)
    return print(io, "$(string(basetypeof(acc)))($(repr(logp(acc))))")
end

# Note that == and isequal are different, and equality under the latter should imply
# equality of hashes. Both of the below implementations are also different from the default
# implementation for structs.
function Base.:(==)(acc1::LogProbAccumulator, acc2::LogProbAccumulator)
    return accumulator_name(acc1) === accumulator_name(acc2) && logp(acc1) == logp(acc2)
end

function Base.isequal(acc1::LogProbAccumulator, acc2::LogProbAccumulator)
    return basetypeof(acc1) === basetypeof(acc2) && isequal(logp(acc1), logp(acc2))
end

Base.hash(acc::T, h::UInt) where {T<:LogProbAccumulator} = hash((T, logp(acc)), h)

_zero(::Tacc) where {Tlogp,Tacc<:LogProbAccumulator{Tlogp}} = Tacc(zero(Tlogp))
reset(acc::LogProbAccumulator) = _zero(acc)
split(acc::LogProbAccumulator) = _zero(acc)
function combine(acc::LogProbAccumulator, acc2::LogProbAccumulator)
    if basetypeof(acc) !== basetypeof(acc2)
        msg = "Cannot combine accumulators of different types: $(basetypeof(acc)) and $(basetypeof(acc2))"
        throw(ArgumentError(msg))
    end
    return basetypeof(acc)(logp(acc) + logp(acc2))
end

acclogp(acc::LogProbAccumulator, val) = basetypeof(acc)(logp(acc) + val)

function Base.convert(
    ::Type{AccType}, acc::LogProbAccumulator
) where {T,AccType<:LogProbAccumulator{T}}
    return AccType(convert(T, logp(acc)))
end

function convert_eltype(::Type{T}, acc::LogProbAccumulator) where {T}
    return basetypeof(acc)(convert(T, logp(acc)))
end

"""
    LogPriorAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log prior during model execution.

Note that the log prior stored in here is always calculated based on unlinked
parameters, i.e., the value of `logp` is independent of whether tha VarInfo is
linked or not.

# Fields
$(TYPEDFIELDS)
"""
struct LogPriorAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the scalar log prior value"
    logp::T
end

logp(acc::LogPriorAccumulator) = acc.logp

accumulator_name(::Type{<:LogPriorAccumulator}) = :LogPrior

function accumulate_assume!!(
    acc::LogPriorAccumulator, val, tval, logjac, vn, right, template
)
    return acclogp(acc, logpdf(right, val))
end
accumulate_observe!!(acc::LogPriorAccumulator, right, left, vn) = acc

"""
    LogJacobianAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log Jacobian (technically,
log(abs(det(J)))) during model execution. Specifically, J refers to the
Jacobian of the _link transform_, i.e., from the space of the original
distribution to unconstrained space.

!!! note
    This accumulator is only incremented if the variable is transformed by a
    link function, i.e., if the VarInfo is linked (for the particular
    variable that is currently being accumulated). If the variable is not
    linked, the log Jacobian term will be 0.

    In general, for the forward Jacobian ``\\mathbf{J}`` corresponding to the
    function ``\\mathbf{y} = f(\\mathbf{x})``,

    ```math
    \\log(q(\\mathbf{y})) = \\log(p(\\mathbf{x})) - \\log (|\\mathbf{J}|)
    ```

    and correspondingly:

    ```julia
    getlogjoint_internal(vi) = getlogjoint(vi) - getlogjac(vi)
    ```

# Fields
$(TYPEDFIELDS)
"""
struct LogJacobianAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the logabsdet of the link transform Jacobian"
    logjac::T
end

logp(acc::LogJacobianAccumulator) = acc.logjac

accumulator_name(::Type{<:LogJacobianAccumulator}) = :LogJacobian

function accumulate_assume!!(
    acc::LogJacobianAccumulator, val, tval, logjac, vn, right, template
)
    return acclogp(acc, logjac)
end
accumulate_observe!!(acc::LogJacobianAccumulator, right, left, vn) = acc

"""
    LogLikelihoodAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log likelihood during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogLikelihoodAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the scalar log likelihood value"
    logp::T
end

logp(acc::LogLikelihoodAccumulator) = acc.logp

accumulator_name(::Type{<:LogLikelihoodAccumulator}) = :LogLikelihood

function accumulate_assume!!(
    acc::LogLikelihoodAccumulator, val, tval, logjac, vn, right, template
)
    return acc
end
function accumulate_observe!!(acc::LogLikelihoodAccumulator, right, left, vn)
    # Note that it's important to use the loglikelihood function here, not logpdf, because
    # they handle vectors differently:
    # https://github.com/JuliaStats/Distributions.jl/issues/1972
    return acclogp(acc, Distributions.loglikelihood(right, left))
end

function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
    )
end
