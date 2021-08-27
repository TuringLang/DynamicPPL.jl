module TestUtils

using AbstractMCMC
using DynamicPPL
using Distributions
using Test

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

@model function demo_assume_multivariate_observe_index(x=[10.0, 10.0])
    # Multivariate `assume` and `observe`
    m ~ MvNormal(zero(x), I)
    x ~ MvNormal(m, 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
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

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function demo_assume_dot_observe(x=[10.0])
    # `assume` and `dot_observe`
    m ~ Normal()
    x .~ Normal(m, 0.5)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end

@model function demo_assume_observe_literal()
    # `assume` and literal `observe`
    m ~ MvNormal(zeros(2), I)
    [10.0, 10.0] ~ MvNormal(m, 0.25 * I)

    return (; m=m, x=[10.0, 10.0], logp=getlogp(__varinfo__))
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

@model function demo_assume_literal_dot_observe()
    # `assume` and literal `dot_observe`
    m ~ Normal()
    [10.0] .~ Normal(m, 0.5)

    return (; m=m, x=[10.0], logp=getlogp(__varinfo__))
end

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, 2)
    m .~ Normal()

    return m
end

@model function demo_assume_submodel_observe_index_literal()
    # Submodel prior
    m = @submodel _prior_dot_assume()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end

    return (; m=m, x=[10.0], logp=getlogp(__varinfo__))
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

@model function demo_dot_assume_dot_observe_matrix(
    x=fill(10.0, 2, 1), ::Type{TV}=Vector{Float64}
) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Dotted observe for `Matrix`.
    x .~ MvNormal(m, 0.25 * I)

    return (; m=m, x=x, logp=getlogp(__varinfo__))
end

const DEMO_MODELS = (
    demo_dot_assume_dot_observe(),
    demo_assume_index_observe(),
    demo_assume_multivariate_observe_index(),
    demo_dot_assume_observe_index(),
    demo_assume_dot_observe(),
    demo_assume_observe_literal(),
    demo_dot_assume_observe_index_literal(),
    demo_assume_literal_dot_observe(),
    demo_assume_submodel_observe_index_literal(),
    demo_dot_assume_observe_submodel(),
    demo_dot_assume_dot_observe_matrix(),
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
    @testset "$(nameof(typeof(sampler))) on $(m.name)" for model in DEMO_MODELS
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
