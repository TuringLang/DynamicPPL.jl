module TestUtils

using AbstractMCMC
using DynamicPPL
using Distributions
using Test

# A collection of models for which the mean-of-means for the posterior should
# be same.
@model function demo1(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and `observe`
    m = TV(undef, length(x))
    m .~ Normal()
    x ~ MvNormal(m, 0.5 * ones(length(x)))
    return (; m, x, logp=getlogp(__varinfo__))
end

@model function demo2(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `assume` with indexing and `observe`
    m = TV(undef, length(x))
    for i in eachindex(m)
        m[i] ~ Normal()
    end
    x ~ MvNormal(m, 0.5 * ones(length(x)))

    return (; m, x, logp=getlogp(__varinfo__))
end

@model function demo3(x=10 * ones(2))
    # Multivariate `assume` and `observe`
    m ~ MvNormal(length(x), 1.0)
    x ~ MvNormal(m, 0.5 * ones(length(x)))

    return (; m, x, logp=getlogp(__varinfo__))
end

@model function demo4(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and `observe` with indexing
    m = TV(undef, length(x))
    m .~ Normal()
    for i in eachindex(x)
        x[i] ~ Normal(m[i], 0.5)
    end

    return (; m, x, logp=getlogp(__varinfo__))
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function demo5(x=10 * ones(1))
    # `assume` and `dot_observe`
    m ~ Normal()
    x .~ Normal(m, 0.5)

    return (; m, x, logp=getlogp(__varinfo__))
end

@model function demo6()
    # `assume` and literal `observe`
    m ~ MvNormal(2, 1.0)
    [10.0, 10.0] ~ MvNormal(m, 0.5 * ones(2))

    return (; m, x=[10.0, 10.0], logp=getlogp(__varinfo__))
end

@model function demo7(::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    m = TV(undef, 2)
    m .~ Normal()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end

    return (; m, x=10 * ones(length(m)), logp=getlogp(__varinfo__))
end

@model function demo8()
    # `assume` and literal `dot_observe`
    m ~ Normal()
    [10.0] .~ Normal(m, 0.5)

    return (; m, x=[10.0], logp=getlogp(__varinfo__))
end

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, 2)
    m .~ Normal()

    return m
end

@model function demo9()
    # Submodel prior
    m = @submodel _prior_dot_assume()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end

    return (; m, x=[10.0], logp=getlogp(__varinfo__))
end

@model function _likelihood_dot_observe(m, x)
    return x ~ MvNormal(m, 0.5 * ones(length(m)))
end

@model function demo10(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Submodel likelihood
    @submodel _likelihood_dot_observe(m, x)

    return (; m, x, logp=getlogp(__varinfo__))
end

@model function demo11(x=10 * ones(2, 1), ::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Dotted observe for `Matrix`.
    return x .~ MvNormal(m, 0.5)
end

const demo_models = (
    demo1(),
    demo2(),
    demo3(),
    demo4(),
    demo5(),
    demo6(),
    demo7(),
    demo8(),
    demo9(),
    demo10(),
    demo11(),
)

# TODO: Is this really the best "default"?
function test_sampler_demo_models(
    meanf,
    spl::AbstractMCMC.AbstractSampler,
    args...;
    target=8.0,
    atol=1e-1,
    rtol=1 - 1,
    kwargs...,
)
    @testset "$(nameof(typeof(spl))) on $(m.name)" for m in demo_models
        chain = AbstractMCMC.sample(m, spl, args...)
        μ = meanf(chain)
        @test μ ≈ target atol = atol rtol = rtol
    end
end

function test_sampler_continuous(
    meanf, spl::AbstractMCMC.AbstractSampler, args...; kwargs...
)
    return test_sampler_demo_models(meanf, spl, args...; kwargs...)
end

function test_sampler_continuous(spl::AbstractMCMC.AbstractSampler, args...; kwargs...)
    # Default for `MCMCChains.Chains`.
    return test_sampler_continuous(spl, args...; kwargs...) do chain
        mean(Array(chain))
    end
end

end
