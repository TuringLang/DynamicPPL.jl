module TestUtils

using AbstractMCMC
using DynamicPPL
using LinearAlgebra
using Distributions
using Test

using Random: Random
using Bijectors: Bijectors
using Setfield: Setfield

"""
    varnames(vn::VarName, val)

Return iterator over all varnames that are represented by `vn` on `val`,
e.g. `varnames(@varname(x), rand(2))` results in an iterator over `[@varname(x[1]), @varname(x[2])]`.
"""
varnames(vn::VarName, val::Real) = [vn]
function varnames(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        VarName(vn, DynamicPPL.getlens(vn) ∘ Setfield.IndexLens(Tuple(I))) for
        I in CartesianIndices(val)
    )
end
function varnames(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varnames(
            VarName(vn, DynamicPPL.getlens(vn) ∘ Setfield.IndexLens(Tuple(I))), val[I]
        ) for I in CartesianIndices(val)
    )
end

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
    logjoint_true(model, args...)

Return the `logjoint` of `model` for `args...`.

Defaults to `logprior_true(model, args...) + loglikelihood_true(model, args..)`.

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
    logjoint_true_with_logabsdet_jacobian(model::Model, args...)

Return a tuple `(args_unconstrained, logjoint)` of `model` for `args...`.

Unlike [`logjoint_true`](@ref), the returned logjoint computation includes the
log-absdet-jacobian adjustment, thus computing logjoint for the unconstrained variables.

Note that `args` are assumed be in the support of `model`, while `args_unconstrained`
will be unconstrained.

This should generally not be implemented directly, instead one should implement
[`logprior_true_with_logabsdet_jacobian`](@ref) for a given `model`.

See also: [`logjoint_true`](@ref), [`logprior_true_with_logabsdet_jacobian`](@ref).
"""
function logjoint_true_with_logabsdet_jacobian(model::Model, args...)
    args_unconstrained, lp = logprior_true_with_logabsdet_jacobian(model, args...)
    return args_unconstrained, lp + loglikelihood_true(model, args...)
end

"""
    logprior_true_with_logabsdet_jacobian(model::Model, args...)

Return a tuple `(args_unconstrained, logprior_unconstrained)` of `model` for `args...`.

Unlike [`logprior_true`](@ref), the returned logprior computation includes the
log-absdet-jacobian adjustment, thus computing logprior for the unconstrained variables.

Note that `args` are assumed be in the support of `model`, while `args_unconstrained`
will be unconstrained.

See also: [`logprior_true`](@ref).
"""
function logprior_true_with_logabsdet_jacobian end

"""
    example_values(model::Model)

Return a `NamedTuple` compatible with `keys(model)` with values in support of `model`.
"""
example_values(model::Model) = example_values(Random.GLOBAL_RNG, model)

"""
    posterior_mean_values(model::Model)

Return a `NamedTuple` compatible with `keys(model)` where the values represent
the posterior mean under `model`.
"""
function posterior_mean_values end

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
    return logpdf(Normal(), m) + logpdf(truncated(Normal(), m, Inf), x)
end
function loglikelihood_true(model::Model{typeof(demo_dynamic_constraint)}, m, x)
    return zero(float(eltype(m)))
end
function Base.keys(model::Model{typeof(demo_dynamic_constraint)})
    return [@varname(m), @varname(x)]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dynamic_constraint)}
)
    m = rand(rng, Normal())
    return (m=m, x=rand(rng, truncated(Normal(), m, Inf)))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dynamic_constraint)}, m, x
)
    b_x = Bijectors.bijector(truncated(Normal(), m, Inf))
    x_unconstrained, Δlogp = Bijectors.with_logabsdet_jacobian(b_x, x)
    return (m=m, x=x_unconstrained), logprior_true(model, m, x) - Δlogp
end

# A collection of models for which the posterior should be "similar".
# Some utility methods for these.
function _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
    b = Bijectors.bijector(InverseGamma(2, 3))
    s_unconstrained = b.(s)
    Δlogp = sum(Base.Fix1(Bijectors.logabsdetjac, b).(s))
    return (s=s_unconstrained, m=m), logprior_true(model, s, m) - Δlogp
end

@model function demo_dot_assume_dot_observe(
    x=[1.5, 1.5], ::Type{TV}=Vector{Float64}
) where {TV}
    # `dot_assume` and `observe`
    s = TV(undef, length(x))
    m = TV(undef, length(x))
    s .~ InverseGamma(2, 3)
    m .~ Normal.(0, sqrt.(s))

    x ~ MvNormal(m, Diagonal(s))
    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_dot_observe)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_dot_observe)}, s, m)
    return loglikelihood(MvNormal(m, Diagonal(s)), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_dot_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_dot_observe)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_dot_observe)}
)
    n = length(model.args.x)
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_dot_assume_dot_observe)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_assume_index_observe(
    x=[1.5, 1.5], ::Type{TV}=Vector{Float64}
) where {TV}
    # `assume` with indexing and `observe`
    s = TV(undef, length(x))
    for i in eachindex(s)
        s[i] ~ InverseGamma(2, 3)
    end
    m = TV(undef, length(x))
    for i in eachindex(m)
        m[i] ~ Normal(0, sqrt(s[i]))
    end
    x ~ MvNormal(m, Diagonal(s))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_index_observe)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(model::Model{typeof(demo_assume_index_observe)}, s, m)
    return logpdf(MvNormal(m, Diagonal(s)), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_index_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_index_observe)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_assume_index_observe)}
)
    n = length(model.args.x)
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_assume_index_observe)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_assume_multivariate_observe(x=[1.5, 1.5])
    # Multivariate `assume` and `observe`
    s ~ product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m ~ MvNormal(zero(x), Diagonal(s))
    x ~ MvNormal(m, Diagonal(s))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_multivariate_observe)}, s, m)
    s_dist = product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m_dist = MvNormal(zero(model.args.x), Diagonal(s))
    return logpdf(s_dist, s) + logpdf(m_dist, m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_multivariate_observe)}, s, m)
    return logpdf(MvNormal(m, Diagonal(s)), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_multivariate_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_multivariate_observe)})
    return [@varname(s), @varname(m)]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_assume_multivariate_observe)}
)
    s = rand(rng, product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)]))
    return (s=s, m=rand(rng, MvNormal(zero(model.args.x), Diagonal(s))))
end
function posterior_mean_values(model::Model{typeof(demo_assume_multivariate_observe)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_dot_assume_observe_index(
    x=[1.5, 1.5], ::Type{TV}=Vector{Float64}
) where {TV}
    # `dot_assume` and `observe` with indexing
    s = TV(undef, length(x))
    s .~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    m .~ Normal.(0, sqrt.(s))
    for i in eachindex(x)
        x[i] ~ Normal(m[i], sqrt(s[i]))
    end

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_index)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_observe_index)}, s, m)
    return sum(logpdf.(Normal.(m, sqrt.(s)), model.args.x))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_observe_index)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_index)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_observe_index)}
)
    n = length(model.args.x)
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_dot_assume_observe_index)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function demo_assume_dot_observe(x=[1.5])
    # `assume` and `dot_observe`
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x .~ Normal(m, sqrt(s))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_dot_observe)}, s, m)
    return logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_dot_observe)}, s, m)
    return sum(logpdf.(Normal.(m, sqrt.(s)), model.args.x))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_dot_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_dot_observe)})
    return [@varname(s), @varname(m)]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_assume_dot_observe)}
)
    s = rand(rng, InverseGamma(2, 3))
    m = rand(rng, Normal(0, sqrt(s)))
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_assume_dot_observe)})
    return (s=2.375, m=0.75)
end

@model function demo_assume_observe_literal()
    # `assume` and literal `observe`
    s ~ product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m ~ MvNormal(zeros(2), Diagonal(s))
    [1.5, 1.5] ~ MvNormal(m, Diagonal(s))

    return (; s=s, m=m, x=[1.5, 1.5], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_observe_literal)}, s, m)
    s_dist = product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m_dist = MvNormal(zeros(2), Diagonal(s))
    return logpdf(s_dist, s) + logpdf(m_dist, m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_observe_literal)}, s, m)
    return logpdf(MvNormal(m, Diagonal(s)), [1.5, 1.5])
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_observe_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_observe_literal)})
    return [@varname(s), @varname(m)]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_assume_observe_literal)}
)
    s = rand(rng, product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)]))
    return (s=s, m=rand(rng, MvNormal(zeros(2), Diagonal(s))))
end
function posterior_mean_values(model::Model{typeof(demo_assume_observe_literal)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_dot_assume_observe_index_literal(::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    s = TV(undef, 2)
    m = TV(undef, 2)
    s .~ InverseGamma(2, 3)
    m .~ Normal.(0, sqrt.(s))

    for i in eachindex(m)
        1.5 ~ Normal(m[i], sqrt(s[i]))
    end

    return (; s=s, m=m, x=fill(1.5, length(m)), logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(
    model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m
)
    return sum(logpdf.(Normal.(m, sqrt.(s)), fill(1.5, length(m))))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_index_literal)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_observe_index_literal)}
)
    n = 2
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_dot_assume_observe_index_literal)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_assume_literal_dot_observe()
    # `assume` and literal `dot_observe`
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    [1.5] .~ Normal(m, sqrt(s))

    return (; s=s, m=m, x=[1.5], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_literal_dot_observe)}, s, m)
    return logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_literal_dot_observe)}, s, m)
    return logpdf(Normal(m, sqrt(s)), 1.5)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_literal_dot_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_literal_dot_observe)})
    return [@varname(s), @varname(m)]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_assume_literal_dot_observe)}
)
    s = rand(rng, InverseGamma(2, 3))
    m = rand(rng, Normal(0, sqrt(s)))
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_assume_literal_dot_observe)})
    return (s=2.375, m=0.75)
end

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    s = TV(undef, 2)
    s .~ InverseGamma(2, 3)
    m = TV(undef, 2)
    m .~ Normal.(0, sqrt.(s))

    return s, m
end

@model function demo_assume_submodel_observe_index_literal()
    # Submodel prior
    @submodel s, m = _prior_dot_assume()
    for i in eachindex(m, s)
        1.5 ~ Normal(m[i], sqrt(s[i]))
    end

    return (; s=s, m=m, x=[1.5, 1.5], logp=getlogp(__varinfo__))
end
function logprior_true(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return sum(logpdf.(Normal.(m, sqrt.(s)), 1.5))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_assume_submodel_observe_index_literal)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG,
    model::Model{typeof(demo_assume_submodel_observe_index_literal)},
)
    n = 2
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}
)
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function _likelihood_mltivariate_observe(s, m, x)
    return x ~ MvNormal(m, Diagonal(s))
end

@model function demo_dot_assume_observe_submodel(
    x=[1.5, 1.5], ::Type{TV}=Vector{Float64}
) where {TV}
    s = TV(undef, length(x))
    s .~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    m .~ Normal.(0, sqrt.(s))

    # Submodel likelihood
    @submodel _likelihood_mltivariate_observe(s, m, x)

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_submodel)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_observe_submodel)}, s, m)
    return logpdf(MvNormal(m, Diagonal(s)), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_observe_submodel)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_observe_submodel)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_observe_submodel)}
)
    n = length(model.args.x)
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_dot_assume_observe_submodel)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_dot_assume_dot_observe_matrix(
    x=fill(1.5, 2, 1), ::Type{TV}=Vector{Float64}
) where {TV}
    s = TV(undef, length(x))
    s .~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    m .~ Normal.(0, sqrt.(s))

    # Dotted observe for `Matrix`.
    x .~ MvNormal(m, Diagonal(s))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_dot_observe_matrix)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(model::Model{typeof(demo_dot_assume_dot_observe_matrix)}, s, m)
    return sum(logpdf.(Normal.(m, sqrt.(s)), model.args.x))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_dot_observe_matrix)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_dot_observe_matrix)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_dot_observe_matrix)}
)
    n = length(model.args.x)
    s = rand(rng, InverseGamma(2, 3), n)
    m = similar(s)
    for i in eachindex(m, s)
        m[i] = rand(rng, Normal(0, sqrt(s[i])))
    end
    return (s=s, m=m)
end
function posterior_mean_values(model::Model{typeof(demo_dot_assume_dot_observe_matrix)})
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
end

@model function demo_dot_assume_matrix_dot_observe_matrix(
    x=fill(1.5, 2, 1), ::Type{TV}=Array{Float64}
) where {TV}
    n = length(x)
    d = length(x) ÷ 2
    s = TV(undef, d, 2)
    s .~ product_distribution([InverseGamma(2, 3) for _ in 1:d])
    s_vec = vec(s)
    m ~ MvNormal(zeros(n), Diagonal(s_vec))

    # Dotted observe for `Matrix`.
    x .~ MvNormal(m, Diagonal(s_vec))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(
    model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}, s, m
)
    n = length(model.args.x)
    s_vec = vec(s)
    return loglikelihood(InverseGamma(2, 3), s_vec) + logpdf(MvNormal(zeros(n), s_vec), m)
end
function loglikelihood_true(
    model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}, s, m
)
    return loglikelihood(MvNormal(m, Diagonal(vec(s))), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function Base.keys(model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)})
    return [@varname(s[:, 1]), @varname(s[:, 2]), @varname(m[1]), @varname(m[2])]
end
function example_values(
    rng::Random.AbstractRNG, model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}
)
    n = length(model.args.x)
    d = n ÷ 2
    s = rand(rng, product_distribution([InverseGamma(2, 3) for _ in 1:d]), 2)
    m = rand(rng, MvNormal(zeros(n), Diagonal(vec(s))))
    return (s=s, m=m)
end
function posterior_mean_values(
    model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)}
)
    vals = example_values(model)
    vals.s .= 2.375
    vals.m .= 0.75
    return vals
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
)

# TODO: Is this really the best/most convenient "default" test method?
"""
    test_sampler_demo_models(meanfunction, sampler, args...; kwargs...)

Test that `sampler` produces the correct marginal posterior means on all models in `demo_models`.

In short, this method iterators through `demo_models`, calls `AbstractMCMC.sample` on the
`model` and `sampler` to produce a `chain`, and then checks `meanfunction(chain, vn)`
for every (leaf) varname `vn` against the corresponding value returned by
[`posterior_mean_values`](@ref) for each model.

# Arguments
- `meanfunction`: A callable which computes the mean of the marginal means from the
  chain resulting from the `sample` call.
- `sampler`: The `AbstractMCMC.AbstractSampler` to test.
- `args...`: Arguments forwarded to `sample`.

# Keyword arguments
- `atol=1e-1`: Absolute tolerance used in `@test`.
- `rtol=1e-3`: Relative tolerance used in `@test`.
- `kwargs...`: Keyword arguments forwarded to `sample`.
"""
function test_sampler_demo_models(
    meanfunction,
    sampler::AbstractMCMC.AbstractSampler,
    args...;
    atol=1e-1,
    rtol=1e-3,
    kwargs...,
)
    @testset "$(typeof(sampler)) on $(nameof(model))" for model in DEMO_MODELS
        chain = AbstractMCMC.sample(model, sampler, args...; kwargs...)
        target_values = posterior_mean_values(model)
        for vn in keys(model)
            # We want to compare elementwise which can be achieved by
            # extracting the leaves of the `VarName` and the corresponding value.
            for vn_leaf in varnames(vn, get(target_values, vn))
                target_value = get(target_values, vn_leaf)
                chain_mean_value = meanfunction(chain, vn_leaf)
                @test chain_mean_value ≈ target_value atol = atol rtol = rtol
            end
        end
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
    return test_sampler_continuous(sampler, args...; kwargs...) do chain, vn
        # HACK(torfjelde): This assumes that we can index into `chain` with `Symbol(vn)`.
        mean(Array(chain[Symbol(vn)]))
    end
end

end
