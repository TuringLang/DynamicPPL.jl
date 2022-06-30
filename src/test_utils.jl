module TestUtils

using AbstractMCMC
using DynamicPPL
using LinearAlgebra
using Distributions
using Test

"""
    logprior_true(model, θ)

Return the `logprior` of `model` for `θ`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`loglikelihood_true`](@ref).
"""
function logprior_true end

"""
    loglikelihood_true(model, θ)

Return the `loglikelihood` of `model` for `θ`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`logprior_true`](@ref).
"""
function loglikelihood_true end

"""
    logjoint_true(model, θ)

Return the `logjoint` of `model` for `θ`.

Defaults to `logprior_true(model, θ) + loglikelihood_true(model, θ)`.

This should generally be implemented by hand for every specific `model`
so that the returned value can be used as a ground-truth for testing things like:

1. Validity of evaluation of `model` using a particular implementation of `AbstractVarInfo`.
2. Validity of a sampler when combined with DynamicPPL by running the sampler twice: once targeting ground-truth functions, e.g. `logjoint_true`, and once targeting `model`.

And more.

See also: [`logprior_true`](@ref), [`loglikelihood_true`](@ref).
"""
function logjoint_true(model::Model, args...)
    return logprior_true(model, args...) + loglikelihood_true(model, args...)
end

"""
    demo_dynamic_constraint()

A model with variables `m` and `x` with `x` having support depending on `m`.
"""
@model function demo_dynamic_constraint()
    m ~ Normal()
    x ~ truncated(Normal(), m, Inf)

    return (m=m, x=x)
end

function logprior_true(model::Model{typeof(demo_dynamic_constraint)}, m, x)
    return logpdf(Normal(), m) + logpdf(truncated(Normal(), m, Inf))
end
function loglikelihood_true(model::Model{typeof(demo_dynamic_constraint)}, m, x)
    return zero(float(eltype(m)))
end
function Base.keys(model::Model{typeof(demo_dynamic_constraint)})
    return [@varname(m), @varname(x)]
end

# A collection of models for which the mean-of-means for the posterior should
# be same.
@model function demo_dot_assume_dot_observe(
    x=[10.0, 10.0], ::Type{TV}=Vector{Float64}
) where {TV}
    # `dot_assume` and `observe`
    m = TV(undef, length(x))
    m .~ Normal()
    x ~ MvNormal(m, 0.25 * I)
    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_dot_observe)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_dot_observe)}, m)
    return loglikelihood(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_dot_assume_dot_observe)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function demo_assume_index_observe(
    x=[10.0, 10.0], ::Type{TV}=Vector{Float64}
) where {TV}
    # `assume` with indexing and `observe`
    m = TV(undef, length(x))
    for i in eachindex(m)
        m[i] ~ Normal()
    end
    x ~ MvNormal(m, 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_index_observe)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_index_observe)}, m)
    return logpdf(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_assume_index_observe)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function demo_assume_multivariate_observe(x=[10.0, 10.0])
    # Multivariate `assume` and `observe`
    m ~ MvNormal(zero(x), I)
    x ~ MvNormal(m, 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_multivariate_observe)}, m)
    return logpdf(MvNormal(zero(model.args.x), I), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_multivariate_observe)}, m)
    return logpdf(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_assume_multivariate_observe)})
    return [@varname(m)]
end

@model function demo_dot_assume_observe_index(
    x=[10.0, 10.0], ::Type{TV}=Vector{Float64}
) where {TV}
    # `dot_assume` and `observe` with indexing
    m = TV(undef, length(x))
    m .~ Normal()
    for i in eachindex(x)
        x[i] ~ Normal(m[i], 0.5)
    end

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_index)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_observe_index)}, m)
    return sum(logpdf.(Normal.(m, 0.5), model.args.x))
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_index)})
    return [@varname(m[1]), @varname(m[2])]
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function demo_assume_dot_observe(x=[10.0])
    # `assume` and `dot_observe`
    m ~ Normal()
    x .~ Normal(m, 0.5)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_dot_observe)}, m)
    return logpdf(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_dot_observe)}, m)
    return sum(logpdf.(Normal.(m, 0.5), model.args.x))
end
function Base.keys(model::Model{typeof(demo_assume_dot_observe)})
    return [@varname(m)]
end

@model function demo_assume_observe_literal()
    # `assume` and literal `observe`
    m ~ MvNormal(zeros(2), I)
    [10.0, 10.0] ~ MvNormal(m, 0.25 * I)

    return (; m=m, x=[10.0, 10.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_observe_literal)}, m)
    return logpdf(MvNormal(zeros(2), I), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_observe_literal)}, m)
    return logpdf(MvNormal(m, 0.25 * I), [10.0, 10.0])
end
function Base.keys(model::Model{typeof(demo_assume_observe_literal)})
    return [@varname(m)]
end

@model function demo_dot_assume_observe_index_literal(::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    m = TV(undef, 2)
    m .~ Normal()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end

    return (; m=m, x=fill(10.0, length(m)), logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_index_literal)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_observe_index_literal)}, m)
    return sum(logpdf.(Normal.(m, 0.5), fill(10.0, length(m))))
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_index_literal)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function demo_assume_literal_dot_observe()
    # `assume` and literal `dot_observe`
    m ~ Normal()
    [10.0] .~ Normal(m, 0.5)

    return (; m=m, x=[10.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_literal_dot_observe)}, m)
    return logpdf(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_literal_dot_observe)}, m)
    return logpdf(Normal(m, 0.5), 10.0)
end
function Base.keys(model::Model{typeof(demo_assume_literal_dot_observe)})
    return [@varname(m)]
end

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, 2)
    m .~ Normal()

    return m
end

@model function demo_assume_submodel_observe_index_literal()
    # Submodel prior
    @submodel m = _prior_dot_assume()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end

    return (; m=m, x=[10.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_submodel_observe_index_literal)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, m
)
    return sum(logpdf.(Normal.(m, 0.5), 10.0))
end
function Base.keys(model::Model{typeof(demo_assume_submodel_observe_index_literal)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function _likelihood_dot_observe(m, x)
    return x ~ MvNormal(m, 0.25 * I)
end

@model function demo_dot_assume_observe_submodel(
    x=[10.0, 10.0], ::Type{TV}=Vector{Float64}
) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Submodel likelihood
    @submodel _likelihood_dot_observe(m, x)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_submodel)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_observe_submodel)}, m)
    return logpdf(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_submodel)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function demo_dot_assume_dot_observe_matrix(
    x=fill(10.0, 2, 1), ::Type{TV}=Vector{Float64}
) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Dotted observe for `Matrix`.
    x .~ MvNormal(m, 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_dot_observe_matrix)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_dot_observe_matrix)}, m)
    return loglikelihood(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_dot_assume_dot_observe_matrix)})
    return [@varname(m[1]), @varname(m[2])]
end

@model function demo_dot_assume_matrix_dot_observe_matrix(
    x=fill(10.0, 2, 1), ::Type{TV}=Array{Float64}
) where {TV}
    d = length(x) ÷ 2
    m = TV(undef, d, 2)
    m .~ MvNormal(zeros(d), I)

    # Dotted observe for `Matrix`.
    x .~ MvNormal(vec(m), 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}, m)
    return loglikelihood(Normal(), vec(m))
end
function loglikelihood_true(
    model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}, m
)
    return loglikelihood(MvNormal(vec(m), 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)})
    return [@varname(m[:, 1]), @varname(m[:, 2])]
end

@model function demo_dot_assume_array_dot_observe(
    x=[10.0, 10.0], ::Type{TV}=Vector{Float64}
) where {TV}
    # `dot_assume` and `observe`
    m = TV(undef, length(x))
    m .~ [Normal() for _ in 1:length(x)]
    x ~ MvNormal(m, 0.25 * I)
    return (; m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_array_dot_observe)}, m)
    return loglikelihood(Normal(), m)
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_array_dot_observe)}, m)
    return loglikelihood(MvNormal(m, 0.25 * I), model.args.x)
end
function Base.keys(model::Model{typeof(demo_dot_assume_array_dot_observe)})
    return [@varname(m[1]), @varname(m[2])]
end

const DEMO_MODELS = (
    demo_dot_assume_dot_observe(),
    demo_assume_index_observe(),
    demo_assume_multivariate_observe(),
    demo_dot_assume_observe_index(),
    demo_assume_dot_observe(),
    demo_assume_observe_literal(),
    demo_dot_assume_observe_index_literal(),
    demo_assume_literal_dot_observe(),
    demo_assume_submodel_observe_index_literal(),
    demo_dot_assume_observe_submodel(),
    demo_dot_assume_dot_observe_matrix(),
    demo_dot_assume_matrix_dot_observe_matrix(),
    demo_dot_assume_array_dot_observe(),
)

# TODO: Is this really the best/most convenient "default" test method?
"""
    test_sampler_demo_models(meanfunction, sampler, args...; kwargs...)

Test that `sampler` produces the correct marginal posterior means on all models in `demo_models`.

In short, this method iterators through `demo_models`, calls `AbstractMCMC.sample` on the
`model` and `sampler` to produce a `chain`, and then checks `meanfunction(chain)` against `target`
provided in `kwargs...`.

# Arguments
- `meanfunction`: A callable which computes the mean of the marginal means from the
  chain resulting from the `sample` call.
- `sampler`: The `AbstractMCMC.AbstractSampler` to test.
- `args...`: Arguments forwarded to `sample`.

# Keyword arguments
- `target`: Value to compare result of `meanfunction(chain)` to.
- `atol=1e-1`: Absolute tolerance used in `@test`.
- `rtol=1e-3`: Relative tolerance used in `@test`.
- `kwargs...`: Keyword arguments forwarded to `sample`.
"""
function test_sampler_demo_models(
    meanfunction,
    sampler::AbstractMCMC.AbstractSampler,
    args...;
    target=8.0,
    atol=1e-1,
    rtol=1e-3,
    kwargs...,
)
    @testset "$(nameof(typeof(sampler))) on $(nameof(m))" for model in DEMO_MODELS
        chain = AbstractMCMC.sample(model, sampler, args...; kwargs...)
        μ = meanfunction(chain)
        @test μ ≈ target atol = atol rtol = rtol
    end
end

"""
    test_sampler_continuous([meanfunction, ]sampler, args...; kwargs...)

Test that `sampler` produces the correct marginal posterior means on all models in `demo_models`.

As of right now, this is just an alias for [`test_sampler_demo_models`](@ref).
"""
function test_sampler_continuous(
    meanfunction, sampler::AbstractMCMC.AbstractSampler, args...; kwargs...
)
    return test_sampler_demo_models(meanfunction, sampler, args...; kwargs...)
end

function test_sampler_continuous(sampler::AbstractMCMC.AbstractSampler, args...; kwargs...)
    # Default for `MCMCChains.Chains`.
    return test_sampler_continuous(sampler, args...; kwargs...) do chain
        mean(Array(chain))
    end
end

end
