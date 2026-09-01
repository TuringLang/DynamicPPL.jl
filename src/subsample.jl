abstract type AbstractIndependentDistribution{N,S<:Distributions.ValueSupport} <:
              Distribution{Distributions.ArrayLikeVariate{N},S} end

struct IndependentDistribution{
    N,M,S<:Distributions.ValueSupport,D<:Distribution{Distributions.ArrayLikeVariate{N},S}
} <: AbstractIndependentDistribution{M,S}
    dist::D

    function IndependentDistribution{N,M,S,D}(dist::D) where {N,M,S,D}
        M == N + 1 ||
            throw(ArgumentError("an independent distribution must add one dimension"))
        return new{N,M,S,D}(dist)
    end
end

struct IndexedIndependentDistribution{
    N,M,S<:Distributions.ValueSupport,F,D<:Distribution{Distributions.ArrayLikeVariate{N},S}
} <: AbstractIndependentDistribution{M,S}
    make_distribution::F
    n::Int
    prototype::D

    function IndexedIndependentDistribution{N,M,S,F,D}(
        make_distribution::F, n::Int, prototype::D
    ) where {N,M,S,F,D}
        M == N + 1 ||
            throw(ArgumentError("an independent distribution must add one dimension"))
        return new{N,M,S,F,D}(make_distribution, n, prototype)
    end
end

"""
    independent_distribution(dist::Distribution)
    independent_distribution(make_distribution, n::Integer)

Treat observations as conditionally independent draws.

For a value containing `n` observations, `logpdf`, `loglikelihood`, and `insupport` behave
as for `product_distribution(Fill(dist, n))`. Ordinary model evaluation imposes no
subsampling constraints. [`subsample`](@ref) additionally verifies the model structure
required for valid subsampling.

The conditioned value must have exactly one observation axis after the axes of a single
draw from `dist`. For example, repeated univariate draws form a vector, while repeated
draws from a `k`-dimensional multivariate distribution form a `k × n` matrix. Because `n`
is inferred from the conditioned value, `independent_distribution(dist)` cannot be sampled
before that value is known, and its complete `size` is unavailable until then.

Use `independent_distribution(make_distribution, n)` when observation `i` has distribution
`make_distribution(i)`. Outside subsampling, this behaves as
`product_distribution(map(make_distribution, 1:n))`. Subsampling constructs
`O(batch_size)` distributions rather than all `n`. The factory must be side-effect-free;
construction calls it at index `1` to determine the variate shape.
"""
function independent_distribution(
    dist::D
) where {
    N,S<:Distributions.ValueSupport,D<:Distribution{Distributions.ArrayLikeVariate{N},S}
}
    return IndependentDistribution{N,N + 1,S,D}(dist)
end

function independent_distribution(make_distribution, n::Integer)
    n isa Bool && throw(ArgumentError("the number of observations must not be a boolean"))
    n > 0 || throw(ArgumentError("the number of observations must be positive"))
    n <= typemax(Int) || throw(ArgumentError("the number of observations is too large"))
    prototype = make_distribution(1)
    return _indexed_independent_distribution(make_distribution, Int(n), prototype)
end
function _indexed_independent_distribution(
    make_distribution::F, n::Int, prototype::D
) where {
    F,N,S<:Distributions.ValueSupport,D<:Distribution{Distributions.ArrayLikeVariate{N},S}
}
    return IndexedIndependentDistribution{N,N + 1,S,F,D}(make_distribution, n, prototype)
end
function _indexed_independent_distribution(make_distribution, n::Int, prototype)
    throw(
        ArgumentError(
            "`make_distribution(1)` must return a distribution whose draws have a statically known number of dimensions; got $(typeof(prototype))",
        ),
    )
end
function independent_distribution(dist::Distribution)
    throw(
        ArgumentError(
            "`independent_distribution` requires a distribution whose draws have a statically known number of dimensions; got $(typeof(dist)).",
        ),
    )
end

function Base.size(::IndependentDistribution)
    throw(
        ArgumentError(
            "the size of `independent_distribution(dist)` depends on its conditioned value and is not known from the distribution alone",
        ),
    )
end
Base.eltype(::Type{<:IndependentDistribution{N,M,S,D}}) where {N,M,S,D} = eltype(D)
function Base.eltype(::Type{<:IndexedIndependentDistribution{N,M,S,F,D}}) where {N,M,S,F,D}
    return eltype(D)
end

function _indexed_product(dist::IndexedIndependentDistribution, indices)
    distributions = map(indices) do i
        return i == 1 ? dist.prototype : dist.make_distribution(i)
    end
    return product_distribution(distributions)
end
Base.size(dist::IndexedIndependentDistribution) = (size(dist.prototype)..., dist.n)

for f in (
    :(Bijectors.VectorBijectors.from_linked_vec),
    :(Bijectors.VectorBijectors.to_linked_vec),
    :(Bijectors.VectorBijectors.from_vec),
    :(Bijectors.VectorBijectors.to_vec),
    :(Bijectors.VectorBijectors.vec_length),
    :(Bijectors.VectorBijectors.linked_vec_length),
    :(Bijectors.VectorBijectors.optic_vec),
    :(Bijectors.VectorBijectors.linked_optic_vec),
)
    @eval $f(dist::IndexedIndependentDistribution) = $f(_indexed_product(dist, 1:(dist.n)))
end

function Base.rand(::Random.AbstractRNG, ::IndependentDistribution)
    throw(
        ArgumentError(
            "`independent_distribution(dist)` is an observed-only marker and cannot be sampled. Fully condition its left-hand side before evaluating the model.",
        ),
    )
end
function Base.rand(rng::Random.AbstractRNG, dist::IndexedIndependentDistribution)
    return rand(rng, _indexed_product(dist, 1:(dist.n)))
end

function _validate_independent_value(
    dist::AbstractIndependentDistribution{M}, value::AbstractArray{<:Real}
) where {M}
    expected_ndims = M
    event_ndims = M - 1
    ndims(value) == expected_ndims || throw(
        DimensionMismatch(
            "an observation under `independent_distribution` must have $expected_ndims dimensions: $event_ndims event dimension(s) followed by one observation dimension; got $(ndims(value)) dimensions",
        ),
    )
    value_size = _independent_size(value)
    n = value_size[expected_ndims]
    prototype = _independent_prototype(dist)
    expected_size = (size(prototype)..., n)
    value_size == expected_size || throw(
        DimensionMismatch(
            "the conditioned value has size $value_size, but $n draws from $(typeof(prototype)) require size $expected_size",
        ),
    )
    return n
end
_independent_size(value) = size(value)
_independent_size(value, dimension) = size(value, dimension)
_independent_prototype(dist::IndependentDistribution) = dist.dist
_independent_prototype(dist::IndexedIndependentDistribution) = dist.prototype

function _independent_product(
    dist::IndependentDistribution, value::AbstractArray{<:Real}, indices=nothing
)
    n = _validate_independent_value(dist, value)
    indices === nothing ||
        length(indices) == n ||
        throw(
            DimensionMismatch(
                "the batch contains $(length(indices)) indices but $n observations"
            ),
        )
    return product_distribution(Fill(dist.dist, n))
end
function _independent_product(
    dist::IndexedIndependentDistribution, value::AbstractArray{<:Real}, indices=nothing
)
    n = _validate_independent_value(dist, value)
    selected = indices === nothing ? (1:(dist.n)) : indices
    length(selected) == n || throw(
        DimensionMismatch(
            "the batch contains $(length(selected)) indices but $n observations"
        ),
    )
    all(i -> 1 <= i <= dist.n, selected) ||
        throw(ArgumentError("independent-distribution indices must lie in `1:$(dist.n)`"))
    return _indexed_product(dist, selected)
end
function _independent_product(dist::AbstractIndependentDistribution, value)
    return _throw_invalid_independent_value(dist, value)
end
function _throw_invalid_independent_value(dist::AbstractIndependentDistribution, value)
    throw(
        ArgumentError(
            "the conditioned value for `independent_distribution` must be an array of real numbers; got $(typeof(value))",
        ),
    )
end

function Distributions.logpdf(
    dist::AbstractIndependentDistribution{N}, value::AbstractArray{<:Real,M}
) where {N,M}
    return logpdf(_independent_product(dist, value), value)
end
# This unreachable dimension resolves an ambiguity with Distributions' univariate fallback.
function Distributions.logpdf(
    dist::AbstractIndependentDistribution{0}, value::AbstractArray{<:Real,0}
)
    return logpdf(_independent_product(dist, value), value)
end
function Distributions.logpdf(dist::AbstractIndependentDistribution{1}, value::Real)
    return _throw_invalid_independent_value(dist, value)
end
function Distributions.logpdf(
    dist::AbstractIndependentDistribution{N},
    value::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return _throw_invalid_independent_value(dist, value)
end
function Distributions.loglikelihood(
    dist::AbstractIndependentDistribution{N}, value::AbstractArray{<:Real,M}
) where {N,M}
    return logpdf(dist, value)
end
function Distributions.loglikelihood(dist::AbstractIndependentDistribution{1}, value::Real)
    return logpdf(dist, value)
end
function Distributions.loglikelihood(
    dist::AbstractIndependentDistribution{N},
    value::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return logpdf(dist, value)
end
function Distributions.insupport(
    dist::AbstractIndependentDistribution{N}, value::AbstractArray{<:Real,M}
) where {N,M}
    return insupport(_independent_product(dist, value), value)
end
# This unreachable dimension resolves an ambiguity with Distributions' univariate fallback.
function Distributions.insupport(
    dist::AbstractIndependentDistribution{0}, value::AbstractArray{<:Real,0}
)
    return insupport(_independent_product(dist, value), value)
end
function Distributions.insupport(
    dist::AbstractIndependentDistribution{1}, value::AbstractMatrix
)
    return insupport(_independent_product(dist, value), value)
end
function Distributions.insupport(dist::AbstractIndependentDistribution{1}, value::Real)
    return _throw_invalid_independent_value(dist, value)
end

struct IndependentLogPriorAccumulator{T<:Real,V<:VarName,Evaluate} <: LogProbAccumulator{T}
    logp::T
    observation::V
    seen_observation::Bool
end

function IndependentLogPriorAccumulator(
    observation::V, ::Val{Evaluate}
) where {V<:VarName,Evaluate}
    return IndependentLogPriorAccumulator{LogProbType,V,Evaluate}(
        zero(LogProbType), observation, false
    )
end

logp(acc::IndependentLogPriorAccumulator) = acc.logp
accumulator_name(::Type{<:IndependentLogPriorAccumulator}) = :LogPrior
function reset(acc::IndependentLogPriorAccumulator{T,V,Evaluate}) where {T,V,Evaluate}
    return IndependentLogPriorAccumulator{T,V,Evaluate}(zero(T), acc.observation, false)
end
function accumulate_assume!!(
    acc::IndependentLogPriorAccumulator{T,V,Evaluate},
    value,
    tvalue,
    logjac,
    vn,
    dist,
    template,
) where {T,V,Evaluate}
    value_logp = _independent_logprior(Val(Evaluate), acc, dist, value)
    return acclogp(acc, value_logp)
end
function _independent_logprior(
    ::Val{true}, acc::IndependentLogPriorAccumulator, dist, value
)
    return logpdf(dist, value)
end
function _independent_logprior(
    ::Val{false}, acc::IndependentLogPriorAccumulator, dist, value
)
    return zero(logp(acc))
end
function accumulate_observe!!(
    acc::IndependentLogPriorAccumulator{T,V,Evaluate}, dist, value, vn, template
) where {T,V,Evaluate}
    return IndependentLogPriorAccumulator{T,V,Evaluate}(logp(acc), acc.observation, true)
end
function acclogp(
    acc::IndependentLogPriorAccumulator{T,V,Evaluate}, value
) where {T,V,Evaluate}
    !acc.seen_observation || throw(
        ArgumentError(
            "a prior contribution occurs after the independent observation `$(acc.observation)`; the independent observation must be the last probability-bearing statement",
        ),
    )
    new_logp = logp(acc) + value
    return IndependentLogPriorAccumulator{typeof(new_logp),V,Evaluate}(
        new_logp, acc.observation, false
    )
end

struct IndependentLogJacobianAccumulator{T<:Real,V<:VarName} <: LogProbAccumulator{T}
    logjac::T
    observation::V
    seen_observation::Bool
end

function IndependentLogJacobianAccumulator(observation::V) where {V<:VarName}
    return IndependentLogJacobianAccumulator(zero(LogProbType), observation, false)
end

logp(acc::IndependentLogJacobianAccumulator) = acc.logjac
accumulator_name(::Type{<:IndependentLogJacobianAccumulator}) = :LogJacobian
function reset(acc::IndependentLogJacobianAccumulator{T}) where {T}
    return IndependentLogJacobianAccumulator(zero(T), acc.observation, false)
end
function accumulate_assume!!(
    acc::IndependentLogJacobianAccumulator, value, tvalue, logjac, vn, dist, template
)
    return acclogp(acc, logjac)
end
function accumulate_observe!!(
    acc::IndependentLogJacobianAccumulator, dist, value, vn, template
)
    return IndependentLogJacobianAccumulator(logp(acc), acc.observation, true)
end
function acclogp(acc::IndependentLogJacobianAccumulator, value)
    !acc.seen_observation || throw(
        ArgumentError(
            "a log-Jacobian contribution occurs after the independent observation `$(acc.observation)`; the independent observation must be the last probability-bearing statement",
        ),
    )
    return IndependentLogJacobianAccumulator(logp(acc) + value, acc.observation, false)
end

struct IndependentLogLikelihoodAccumulator{T<:Real,V<:VarName,S<:Real,I,Evaluate} <:
       LogProbAccumulator{T}
    logp::T
    observation::V
    scale::S
    expected_nobs::Int
    population_size::Int
    indices::I
    count::Int
end

function IndependentLogLikelihoodAccumulator(
    observation::V,
    scale::S,
    expected_nobs::Int,
    population_size::Int,
    indices::I,
    ::Val{Evaluate},
) where {V<:VarName,S<:Real,I,Evaluate}
    return IndependentLogLikelihoodAccumulator{LogProbType,V,S,I,Evaluate}(
        zero(LogProbType), observation, scale, expected_nobs, population_size, indices, 0
    )
end

logp(acc::IndependentLogLikelihoodAccumulator) = acc.logp
accumulator_name(::Type{<:IndependentLogLikelihoodAccumulator}) = :LogLikelihood
function reset(
    acc::IndependentLogLikelihoodAccumulator{T,V,S,I,Evaluate}
) where {T,V,S,I,Evaluate}
    return IndependentLogLikelihoodAccumulator{T,V,S,I,Evaluate}(
        zero(T),
        acc.observation,
        acc.scale,
        acc.expected_nobs,
        acc.population_size,
        acc.indices,
        0,
    )
end

function accumulate_assume!!(
    acc::IndependentLogLikelihoodAccumulator, value, tvalue, logjac, vn, dist, template
)
    acc.count == 0 || throw(
        ArgumentError(
            "a probability-bearing statement for `$vn` occurs after the independent observation `$(acc.observation)`; the independent observation must be last",
        ),
    )
    return acc
end

function accumulate_observe!!(
    acc::IndependentLogLikelihoodAccumulator{T,V,S,I,Evaluate}, dist, value, vn, template
) where {T,V,S,I,Evaluate}
    acc.count == 0 || throw(
        ArgumentError(
            "the model contains more than one observed statement; `$(acc.observation)` must be the only observation",
        ),
    )
    subsumes(acc.observation, vn) || throw(
        ArgumentError(
            "the model observed `$vn`, but subsampling requires `$(acc.observation)` to be the only observation",
        ),
    )
    dist isa AbstractIndependentDistribution || throw(
        ArgumentError(
            "subsampling requires the observation `$(acc.observation)` to have `independent_distribution(dist)` on the right-hand side; got $(typeof(dist))",
        ),
    )
    n = _independent_size(value, ndims(value))
    if n != acc.expected_nobs
        throw(
            DimensionMismatch(
                "the independent observation `$(acc.observation)` contains $n observations, but $(acc.expected_nobs) were expected",
            ),
        )
    end
    value_logp = _independent_loglikelihood(Val(Evaluate), acc, dist, value)
    return IndependentLogLikelihoodAccumulator{
        typeof(logp(acc) + value_logp),V,S,I,Evaluate
    }(
        logp(acc) + value_logp,
        acc.observation,
        acc.scale,
        acc.expected_nobs,
        acc.population_size,
        acc.indices,
        1,
    )
end

function _independent_loglikelihood(
    ::Val{true}, acc::IndependentLogLikelihoodAccumulator, dist, value
)
    _validate_independent_population(acc, dist)
    product = _independent_product(dist, value, acc.indices)
    return acc.scale * Distributions.loglikelihood(product, value)
end
function _independent_loglikelihood(
    ::Val{false}, acc::IndependentLogLikelihoodAccumulator, dist, value
)
    _validate_independent_value(dist, value)
    _validate_independent_population(acc, dist)
    if acc.indices !== nothing
        length(acc.indices) == acc.expected_nobs || throw(
            DimensionMismatch(
                "the batch contains $(length(acc.indices)) indices, but $(acc.expected_nobs) were expected",
            ),
        )
        dist isa IndexedIndependentDistribution &&
            !all(i -> 1 <= i <= dist.n, acc.indices) &&
            throw(
                ArgumentError("independent-distribution indices must lie in `1:$(dist.n)`")
            )
    end
    return zero(logp(acc))
end

_validate_independent_population(acc::IndependentLogLikelihoodAccumulator, dist) = nothing
function _validate_independent_population(
    acc::IndependentLogLikelihoodAccumulator, dist::IndexedIndependentDistribution
)
    dist.n == acc.population_size || throw(
        DimensionMismatch(
            "the independent distribution contains $(dist.n) observations, but the dataset contains $(acc.population_size)",
        ),
    )
    return nothing
end

function acclogp(::IndependentLogLikelihoodAccumulator, value)
    throw(
        ArgumentError(
            "subsampling does not support likelihood contributions through `@addlogprob!` or `accloglikelihood!!`; the independent observation must be the complete likelihood.",
        ),
    )
end

struct IndependentLogJoint{R}
    reference_ranges::R
end
IndependentLogJoint() = IndependentLogJoint(nothing)

function _check_independent_accumulators(vi::AbstractVarInfo)
    prior = hasacc(vi, Val(:LogPrior)) ? getacc(vi, Val(:LogPrior)) : nothing
    prior isa IndependentLogPriorAccumulator || throw(
        ArgumentError(
            "the subsampling log-prior accumulator was removed or replaced; got $(typeof(prior))",
        ),
    )
    jacobian = hasacc(vi, Val(:LogJacobian)) ? getacc(vi, Val(:LogJacobian)) : nothing
    jacobian isa IndependentLogJacobianAccumulator || throw(
        ArgumentError(
            "the subsampling log-Jacobian accumulator was removed or replaced; got $(typeof(jacobian))",
        ),
    )
    likelihood = if hasacc(vi, Val(:LogLikelihood))
        getacc(vi, Val(:LogLikelihood))
    else
        nothing
    end
    likelihood isa IndependentLogLikelihoodAccumulator || throw(
        ArgumentError(
            "the subsampling log-likelihood accumulator was removed or replaced; got $(typeof(likelihood))",
        ),
    )
    return likelihood
end

function _check_independent_evaluation_layout(
    reference::VarNamedTuple, candidate::VarNamedTuple
)
    reference_vns = keys(reference)
    candidate_vns = keys(candidate)
    reference_vns == candidate_vns || throw(
        DimensionMismatch(
            "model evaluation changed the latent variables from $(collect(reference_vns)) to $(collect(candidate_vns))",
        ),
    )
    offset = 1
    for vn in reference_vns
        reference_rat = reference[vn]
        candidate_value = candidate[vn]
        candidate_length = length(get_internal_value(candidate_value))
        candidate_range = offset:(offset + candidate_length - 1)
        reference_rat.range == candidate_range || throw(
            DimensionMismatch(
                "model evaluation changed the parameter range for `$vn` from $(reference_rat.range) to $candidate_range",
            ),
        )
        candidate_transform = get_transform(candidate_value)
        isequal(reference_rat.transform, candidate_transform) || throw(
            ArgumentError(
                "model evaluation changed the transform for `$vn` from $(reference_rat.transform) to $candidate_transform",
            ),
        )
        offset += candidate_length
    end
    return nothing
end

function (getlogjoint::IndependentLogJoint)(vi::AbstractVarInfo)
    likelihood = _check_independent_accumulators(vi)
    likelihood.count == 1 || throw(
        ArgumentError(
            "the model did not encounter the conditioned independent observation `$(likelihood.observation)` exactly once",
        ),
    )
    if getlogjoint.reference_ranges !== nothing
        vector_values = if hasacc(vi, Val(VECTORVAL_ACCNAME))
            getacc(vi, Val(VECTORVAL_ACCNAME))
        else
            nothing
        end
        vector_values isa VNTAccumulator{VECTORVAL_ACCNAME,typeof(_get_vector_tval)} ||
            throw(
                ArgumentError(
                    "the subsampling latent-layout accumulator was removed or replaced; got $(typeof(vector_values))",
                ),
            )
        _check_independent_evaluation_layout(
            getlogjoint.reference_ranges, vector_values.values
        )
    end
    return getlogprior(vi) + getloglikelihood(vi) - getlogjac(vi)
end

function _independent_accumulators(
    observation::VarName,
    scale::Real,
    expected_nobs::Int,
    population_size::Int,
    indices=nothing,
    evaluate::Val=Val(true),
)
    return AccumulatorTuple((
        IndependentLogPriorAccumulator(observation, evaluate),
        IndependentLogJacobianAccumulator(observation),
        IndependentLogLikelihoodAccumulator(
            observation, scale, expected_nobs, population_size, indices, evaluate
        ),
    ))
end

function _probe_independent_model(
    rng::Random.AbstractRNG,
    model::Model,
    observation::VarName,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy,
    scale::Real,
    expected_nobs::Int,
    population_size::Int,
    indices=nothing,
)
    accumulators = AccumulatorTuple((
        VectorValueAccumulator(),
        _independent_accumulators(
            observation, scale, expected_nobs, population_size, indices, Val(false)
        )...,
    ))
    vi = OnlyAccsVarInfo(accumulators)
    _, vi = init!!(rng, model, vi, init_strategy, transform_strategy)
    IndependentLogJoint()(vi)
    return LogDensityFunction(model, getlogjoint_internal, get_vector_values(vi))
end

function _strict_independent_ldf(
    model::Model,
    observation::VarName,
    ranges::VarNamedTuple,
    sample::AbstractVector{<:Real},
    scale::Real,
    expected_nobs::Int,
    population_size::Int,
    indices=nothing,
)
    accumulators = AccumulatorTuple((
        VectorValueAccumulator(),
        _independent_accumulators(
            observation, scale, expected_nobs, population_size, indices
        )...,
    ))
    return LogDensityFunction(
        model, IndependentLogJoint(ranges), ranges, sample, accumulators
    )
end

_unprefix_observation(::AbstractContext, observation::VarName) = observation
function _unprefix_observation(context::PrefixContext, observation::VarName)
    unprefixed = try
        AbstractPPL.unprefix(observation, context.vn_prefix)
    catch e
        e isa ArgumentError || rethrow()
        observation
    end
    return _unprefix_observation(childcontext(context), unprefixed)
end
function _unprefix_observation(context::AbstractParentContext, observation::VarName)
    return _unprefix_observation(childcontext(context), observation)
end

function _conditioned_independent_observation(model::Model)
    values = conditioned(model)
    length(values) == 1 || throw(
        ArgumentError(
            "the model must have exactly one fully conditioned variable; found $(collect(keys(values)))",
        ),
    )
    observation = only(keys(values))
    unprefixed_observation = _unprefix_observation(model.context, observation)
    inargnames(unprefixed_observation, model) && throw(
        ArgumentError(
            "the independent observation `$observation` must be supplied through `condition`, not as a model argument",
        ),
    )
    data = values[observation]
    data isa AbstractArray{<:Real} || throw(
        ArgumentError(
            "the independent observation `$observation` must be fully conditioned on an array of real numbers; got $(typeof(data))",
        ),
    )
    ndims(data) > 0 || throw(
        ArgumentError(
            "the conditioned value for the independent observation `$observation` must have an observation dimension",
        ),
    )
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("the conditioned data must be nonempty"))
    return observation, data
end

function _check_independent_layout(
    reference::LogDensityFunction, candidate::LogDensityFunction
)
    reference_ranges = get_all_ranges_and_transforms(reference)
    candidate_ranges = get_all_ranges_and_transforms(candidate)
    reference_vns = collect(keys(reference_ranges))
    candidate_vns = collect(keys(candidate_ranges))
    reference_vns == candidate_vns || throw(
        DimensionMismatch(
            "subsampling changed the latent variables from $reference_vns to $candidate_vns",
        ),
    )
    for vn in reference_vns
        reference_rat = reference_ranges[vn]
        candidate_rat = candidate_ranges[vn]
        reference_rat.range == candidate_rat.range || throw(
            DimensionMismatch(
                "subsampling changed the parameter range for `$vn` from $(reference_rat.range) to $(candidate_rat.range)",
            ),
        )
        isequal(reference_rat.transform, candidate_rat.transform) || throw(
            ArgumentError(
                "subsampling changed the transform for `$vn` from $(reference_rat.transform) to $(candidate_rat.transform)",
            ),
        )
    end
    LogDensityProblems.dimension(reference) == LogDensityProblems.dimension(candidate) ||
        throw(DimensionMismatch("subsampling changed the log-density dimension"))
    get_input_vector_type(reference) === get_input_vector_type(candidate) || throw(
        ArgumentError(
            "subsampling changed the parameter vector type from $(get_input_vector_type(reference)) to $(get_input_vector_type(candidate))",
        ),
    )
    return nothing
end

struct SubsamplingState{P,V<:VarName,D}
    ldf::P
    observation::V
    full_data::D
end

abstract type AbstractSubsamplingData{T,N} <: AbstractArray{T,N} end

struct SubsamplingShape{T,N,A<:AbstractArray{T,N},R<:AbstractArray{T}} <:
       AbstractSubsamplingData{T,N}
    data::A
    root::R
end
function SubsamplingShape(data::A) where {T,N,A<:AbstractArray{T,N}}
    return SubsamplingShape{T,N,A,A}(data, data)
end

Base.IndexStyle(::Type{<:SubsamplingShape{T,N,A,R}}) where {T,N,A,R} = IndexStyle(A)
function _throw_full_data_access()
    throw(
        ArgumentError(
            "subsampling cannot inspect the full conditioned data or its shape while validating model structure; express observation-specific distributions with `independent_distribution(make_distribution, n)`",
        ),
    )
end
Base.size(::SubsamplingShape) = _throw_full_data_access()
Base.size(::SubsamplingShape, ::Integer) = _throw_full_data_access()
Base.axes(::SubsamplingShape) = _throw_full_data_access()
Base.axes(::SubsamplingShape, ::Integer) = _throw_full_data_access()
Base.length(::SubsamplingShape) = _throw_full_data_access()
function VarNamedTuples._haskey_optic(
    data::SubsamplingShape, optic::VarNamedTuples.IndexWithoutChild
)
    return checkbounds(Bool, data.data, optic.ix...; optic.kw...)
end
function Base.getindex(data::SubsamplingShape, indices...)
    return SubsamplingShape(view(data.data, indices...), data.root)
end
_independent_size(data::SubsamplingShape) = size(data.data)
_independent_size(data::SubsamplingShape, dimension) = size(data.data, dimension)

struct SubsampledData{T,N,A<:AbstractArray{T,N},R<:AbstractArray{T}} <:
       AbstractSubsamplingData{T,N}
    data::A
    root::R
end
function SubsampledData(data::A) where {T,N,A<:AbstractArray{T,N}}
    return SubsampledData{T,N,A,A}(data, data)
end

Base.IndexStyle(::Type{<:SubsampledData{T,N,A,R}}) where {T,N,A,R} = IndexStyle(A)
Base.size(data::SubsampledData) = size(data.data)
Base.axes(data::SubsampledData) = axes(data.data)
function Base.getindex(data::SubsampledData, indices...)
    value = getindex(data.data, indices...)
    return value isa AbstractArray ? SubsampledData(value, data.root) : value
end

# Restore the conditioned parent when lowering writes a subsumed observation into a local.
function BangBang.setindex!!(
    data::AbstractArray, value::SubsamplingShape, indices...; kwargs...
)
    return SubsamplingShape(value.root)
end
function BangBang.setindex!!(
    data::AbstractArray, value::SubsampledData, indices...; kwargs...
)
    return value.root
end

_subsampling_probe_rng() = Random.Xoshiro(0)

function _subsampling_problem(
    model::Model, dataset_size::Integer, transform_strategy::AbstractTransformStrategy
)
    dataset_size isa Bool &&
        throw(ArgumentError("the dataset size must be an integer, not a boolean"))
    dataset_size > 0 || throw(ArgumentError("the dataset size must be positive"))
    requires_threadsafe(model) && throw(
        ArgumentError(
            "subsampling does not support models marked for thread-safe evaluation because their probability-bearing statements have no verifiable global order",
        ),
    )
    observation, data = _conditioned_independent_observation(model)
    observed_size = size(data, ndims(data))
    observed_size == dataset_size || throw(
        DimensionMismatch(
            "the conditioned data contain $observed_size observations, but dataset size $dataset_size was specified",
        ),
    )
    scale = dataset_size//dataset_size
    probe_model = condition(
        decondition(model, observation), observation => SubsamplingShape(data)
    )
    layout = _probe_independent_model(
        _subsampling_probe_rng(),
        probe_model,
        observation,
        InitFromPrior(),
        transform_strategy,
        scale,
        Int(dataset_size),
        Int(dataset_size),
    )
    return SubsamplingState(layout, observation, data)
end

"""
    subsample(
        [rng::AbstractRNG,]
        model::Model,
        batch_size::Integer,
        dataset_size::Integer;
        transform_strategy::AbstractTransformStrategy=UnlinkAll(),
    )

    subsample(
        [rng::AbstractRNG,]
        model::Model,
        indices::AbstractVector{<:Integer},
        dataset_size::Integer;
        transform_strategy::AbstractTransformStrategy=UnlinkAll(),
    )

    subsample(
        [rng::AbstractRNG,]
        model::Model,
        resampler,
        dataset_size::Integer;
        transform_strategy::AbstractTransformStrategy=UnlinkAll(),
    )

Create a scaled log-density problem for one minibatch.

The model must contain one fully conditioned independent observation. `dataset_size` must equal the
size of its final dimension. An integer `batch_size` draws that many observations uniformly
without replacement. Pass `indices` to use an externally selected batch, or pass a
`resampler` that returns indices when called as `resampler(rng, dataset_size)`. The returned
`LogDensityFunction` keeps that batch fixed. Its likelihood is scaled by the ratio of
`dataset_size` to the number of selected indices.

For unbiased scaling, let `Cᵢ` be the multiplicity of dataset element `i` in the returned
indices and let `n = length(indices)`. A custom resampler must satisfy
`E[Cᵢ / n] = 1 / dataset_size`. For fixed `n`, this reduces to
`E[Cᵢ] = n / dataset_size`.

# Examples

```julia
using Distributions, DynamicPPL

@model function location_model()
    μ ~ Normal()
    x ~ independent_distribution(Normal(μ))
end

data = [-1.0, 0.5, 1.0]
random_batch = subsample(location_model() | (x=data,), 2, length(data))
selected_batch = subsample(location_model() | (x=data,), [1, 3], length(data))
```
"""
function subsample(
    rng::Random.AbstractRNG,
    model::Model,
    batch_size::Integer,
    dataset_size::Integer;
    transform_strategy::T=UnlinkAll(),
) where {T<:AbstractTransformStrategy}
    batch_size isa Bool &&
        throw(ArgumentError("the batch size must be an integer, not a boolean"))
    dataset_size isa Bool &&
        throw(ArgumentError("the dataset size must be an integer, not a boolean"))
    dataset_size > 0 || throw(ArgumentError("the dataset size must be positive"))
    batch_size > 0 || throw(ArgumentError("the batch size must be positive"))
    batch_size <= dataset_size || throw(
        ArgumentError("the batch size $batch_size exceeds the dataset size $dataset_size"),
    )
    problem = _subsampling_problem(model, dataset_size, transform_strategy)
    indices = _sample_without_replacement(
        rng, size(problem.full_data, ndims(problem.full_data)), Int(batch_size)
    )
    return _batch_ldf(problem, indices)
end

function subsample(
    rng::Random.AbstractRNG,
    model::Model,
    indices::AbstractVector{<:Integer},
    dataset_size::Integer;
    transform_strategy::T=UnlinkAll(),
) where {T<:AbstractTransformStrategy}
    problem = _subsampling_problem(model, dataset_size, transform_strategy)
    return _batch_ldf(problem, indices)
end

function subsample(
    rng::Random.AbstractRNG,
    model::Model,
    resampler,
    dataset_size::Integer;
    transform_strategy::T=UnlinkAll(),
) where {T<:AbstractTransformStrategy}
    problem = _subsampling_problem(model, dataset_size, transform_strategy)
    observed_size = size(problem.full_data, ndims(problem.full_data))
    indices = resampler(rng, observed_size)
    indices isa AbstractVector{<:Integer} || throw(
        ArgumentError(
            "the resampler must return a vector of integer indices; got $(typeof(indices))",
        ),
    )
    return _batch_ldf(problem, indices)
end

function subsample(
    model::Model,
    batch_size_or_resampler,
    dataset_size::Integer;
    transform_strategy::T=UnlinkAll(),
) where {T<:AbstractTransformStrategy}
    return subsample(
        Random.default_rng(),
        model,
        batch_size_or_resampler,
        dataset_size;
        transform_strategy,
    )
end

function _select_batch(data::AbstractArray, batch::AbstractVector{<:Integer})
    isempty(batch) && throw(ArgumentError("the batch must be nonempty"))
    eltype(batch) <: Bool &&
        throw(ArgumentError("batch entries must be integer indices, not booleans"))
    n = size(data, ndims(data))
    all(i -> 1 <= i <= n, batch) ||
        throw(ArgumentError("batch indices must lie in `1:$n`; got $(collect(batch))"))
    indices = ntuple(i -> i == ndims(data) ? batch : Colon(), ndims(data))
    return copy(Base.maybeview(data, indices...))
end

function _sample_without_replacement(
    rng::Random.AbstractRNG, dataset_size::Int, batch_size::Int
)
    indices = Vector{Int}(undef, batch_size)
    selected = Set{Int}()
    offset = dataset_size - batch_size
    for i in 1:batch_size
        upper = offset + i
        candidate = rand(rng, 1:upper)
        index = candidate in selected ? upper : candidate
        push!(selected, index)
        indices[i] = index
    end
    return indices
end

function _batch_ldf(problem::SubsamplingState, batch::AbstractVector{<:Integer})
    batch_data = _select_batch(problem.full_data, batch)
    batch = collect(Int, batch)
    batch_model = condition(
        decondition(problem.ldf.model, problem.observation),
        problem.observation => SubsampledData(batch_data),
    )
    ranges = get_all_ranges_and_transforms(problem.ldf)
    sample = get_sample_input_vector(problem.ldf)
    batch_size = length(batch)
    full_size = size(problem.full_data, ndims(problem.full_data))
    scale = full_size//batch_size
    init_strategy = InitFromVector(sample, ranges, problem.ldf.transform_strategy)
    batch_layout = _probe_independent_model(
        _subsampling_probe_rng(),
        batch_model,
        problem.observation,
        init_strategy,
        problem.ldf.transform_strategy,
        scale,
        batch_size,
        full_size,
        batch,
    )
    _check_independent_layout(problem.ldf, batch_layout)

    return _strict_independent_ldf(
        batch_model,
        problem.observation,
        ranges,
        sample,
        scale,
        batch_size,
        full_size,
        batch,
    )
end
