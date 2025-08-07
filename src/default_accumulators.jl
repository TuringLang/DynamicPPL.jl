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

split(::AccType) where {T,AccType<:LogProbAccumulator{T}} = AccType(zero(T))

function combine(acc::LogProbAccumulator, acc2::LogProbAccumulator)
    if basetypeof(acc) !== basetypeof(acc2)
        msg = "Cannot combine accumulators of different types: $(basetypeof(acc)) and $(basetypeof(acc2))"
        throw(ArgumentError(msg))
    end
    return basetypeof(acc)(logp(acc) + logp(acc2))
end

acclogp(acc::LogProbAccumulator, val) = basetypeof(acc)(logp(acc) + val)

Base.zero(acc::T) where {T<:LogProbAccumulator} = T(zero(logp(acc)))

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

function accumulate_assume!!(acc::LogPriorAccumulator, val, logjac, vn, right)
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

function accumulate_assume!!(acc::LogJacobianAccumulator, val, logjac, vn, right)
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

accumulate_assume!!(acc::LogLikelihoodAccumulator, val, logjac, vn, right) = acc
function accumulate_observe!!(acc::LogLikelihoodAccumulator, right, left, vn)
    # Note that it's important to use the loglikelihood function here, not logpdf, because
    # they handle vectors differently:
    # https://github.com/JuliaStats/Distributions.jl/issues/1972
    return acclogp(acc, Distributions.loglikelihood(right, left))
end

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
    order::Dict{VNType,Eltype}
end

"""
    VariableOrderAccumulator{T<:Integer}(n=zero(T))

Create a new `VariableOrderAccumulator` with the number of observations set to `n`.
"""
VariableOrderAccumulator{T}(n=zero(T)) where {T<:Integer} =
    VariableOrderAccumulator(convert(T, n), Dict{VarName,T}())
VariableOrderAccumulator(n) = VariableOrderAccumulator{typeof(n)}(n)
VariableOrderAccumulator() = VariableOrderAccumulator{Int}()

function Base.copy(acc::VariableOrderAccumulator)
    return VariableOrderAccumulator(acc.num_produce, copy(acc.order))
end

function Base.show(io::IO, acc::VariableOrderAccumulator)
    return print(
        io, "VariableOrderAccumulator($(string(acc.num_produce)), $(repr(acc.order)))"
    )
end

function Base.:(==)(acc1::VariableOrderAccumulator, acc2::VariableOrderAccumulator)
    return acc1.num_produce == acc2.num_produce && acc1.order == acc2.order
end

function Base.isequal(acc1::VariableOrderAccumulator, acc2::VariableOrderAccumulator)
    return isequal(acc1.num_produce, acc2.num_produce) && isequal(acc1.order, acc2.order)
end

function Base.hash(acc::VariableOrderAccumulator, h::UInt)
    return hash((VariableOrderAccumulator, acc.num_produce, acc.order), h)
end

accumulator_name(::Type{<:VariableOrderAccumulator}) = :VariableOrder

split(acc::VariableOrderAccumulator) = copy(acc)

function combine(acc::VariableOrderAccumulator, acc2::VariableOrderAccumulator)
    # Note that assumptions are not allowed in parallelised blocks, and thus the
    # dictionaries should be identical.
    return VariableOrderAccumulator(
        max(acc.num_produce, acc2.num_produce), merge(acc.order, acc2.order)
    )
end

function increment(acc::VariableOrderAccumulator)
    return VariableOrderAccumulator(acc.num_produce + oneunit(acc.num_produce), acc.order)
end

function accumulate_assume!!(acc::VariableOrderAccumulator, val, logjac, vn, right)
    acc.order[vn] = acc.num_produce
    return acc
end
accumulate_observe!!(acc::VariableOrderAccumulator, right, left, vn) = increment(acc)

function Base.convert(
    ::Type{VariableOrderAccumulator{ElType,VnType}}, acc::VariableOrderAccumulator
) where {ElType,VnType}
    order = Dict{VnType,ElType}()
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

function default_accumulators(
    ::Type{FloatT}=LogProbType, ::Type{IntT}=Int
) where {FloatT,IntT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        VariableOrderAccumulator{IntT}(),
    )
end

function subset(acc::VariableOrderAccumulator, vns::AbstractVector{<:VarName})
    order = filter(pair -> any(subsumes(vn, first(pair)) for vn in vns), acc.order)
    return VariableOrderAccumulator(acc.num_produce, order)
end

"""
    merge(acc1::VariableOrderAccumulator, acc2::VariableOrderAccumulator)

Merge two `VariableOrderAccumulator` instances.

The `num_produce` field of the return value is the `num_produce` of `acc2`.
"""
function Base.merge(acc1::VariableOrderAccumulator, acc2::VariableOrderAccumulator)
    return VariableOrderAccumulator(acc2.num_produce, merge(acc1.order, acc2.order))
end
