# models.jl
# ---------
#
# Contains a list of DynamicPPL models, each containing implementations of
# `logprior_true`, `loglikelihood_true`, `varnames`, and
# `logprior_true_with_logabsdet_jacobian` for testing purposes.
#
# Some additionally contain an implementation of `rand_prior_true`.

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
function varnames(model::Model{typeof(demo_dynamic_constraint)})
    return [@varname(m), @varname(x)]
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dynamic_constraint)}, m, x
)
    b_x = Bijectors.bijector(truncated(Normal(), m, Inf))
    x_unconstrained, Δlogp = Bijectors.with_logabsdet_jacobian(b_x, x)
    return (m=m, x=x_unconstrained), logprior_true(model, m, x) - Δlogp
end

"""
    demo_one_variable_multiple_constraints()

A model with a single multivariate `x` whose components have multiple different constraints.

# Model
```julia
x[1] ~ Normal()
x[2] ~ InverseGamma(2, 3)
x[3] ~ truncated(Normal(), -5, 20)
x[4:5] ~ Dirichlet([1.0, 2.0])
```
"""
@model function demo_one_variable_multiple_constraints(
    ::Type{TV}=Vector{Float64}
) where {TV}
    x = TV(undef, 5)
    x[1] ~ Normal()
    x[2] ~ InverseGamma(2, 3)
    x[3] ~ truncated(Normal(), -5, 20)
    x[4:5] ~ Dirichlet([1.0, 2.0])
    return (x=x,)
end
function logprior_true(model::Model{typeof(demo_one_variable_multiple_constraints)}, x)
    return (
        logpdf(Normal(), x[1]) +
        logpdf(InverseGamma(2, 3), x[2]) +
        logpdf(truncated(Normal(), -5, 20), x[3]) +
        logpdf(Dirichlet([1.0, 2.0]), x[4:5])
    )
end
function loglikelihood_true(model::Model{typeof(demo_one_variable_multiple_constraints)}, x)
    return zero(float(eltype(x)))
end
function varnames(model::Model{typeof(demo_one_variable_multiple_constraints)})
    return [@varname(x[1]), @varname(x[2]), @varname(x[3]), @varname(x[4:5])]
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_one_variable_multiple_constraints)}, x
)
    b_x2 = Bijectors.bijector(InverseGamma(2, 3))
    b_x3 = Bijectors.bijector(truncated(Normal(), -5, 20))
    b_x4 = Bijectors.bijector(Dirichlet([1.0, 2.0]))
    x_unconstrained = vcat(x[1], b_x2(x[2]), b_x3(x[3]), b_x4(x[4:5]))
    Δlogp = (
        Bijectors.logabsdetjac(b_x2, x[2]) +
        Bijectors.logabsdetjac(b_x3, x[3]) +
        Bijectors.logabsdetjac(b_x4, x[4:5])
    )
    return (x=x_unconstrained,), logprior_true(model, x) - Δlogp
end
function rand_prior_true(
    rng::Random.AbstractRNG, model::Model{typeof(demo_one_variable_multiple_constraints)}
)
    x = Vector{Float64}(undef, 5)
    x[1] = rand(rng, Normal())
    x[2] = rand(rng, InverseGamma(2, 3))
    x[3] = rand(rng, truncated(Normal(), -5, 20))
    x[4:5] = rand(rng, Dirichlet([1.0, 2.0]))
    return (x=x,)
end

"""
    demo_lkjchol(d=2)

A model with a single variable `x` with support on the Cholesky factor of a
LKJ distribution.

# Model
```julia
x ~ LKJCholesky(d, 1.0)
```
"""
@model function demo_lkjchol(d::Int=2)
    x ~ LKJCholesky(d, 1.0)
    return (x=x,)
end

function logprior_true(model::Model{typeof(demo_lkjchol)}, x)
    return logpdf(LKJCholesky(model.args.d, 1.0), x)
end

function loglikelihood_true(model::Model{typeof(demo_lkjchol)}, x)
    return zero(float(eltype(x)))
end

function varnames(model::Model{typeof(demo_lkjchol)})
    return [@varname(x)]
end

function logprior_true_with_logabsdet_jacobian(model::Model{typeof(demo_lkjchol)}, x)
    b_x = Bijectors.bijector(LKJCholesky(model.args.d, 1.0))
    x_unconstrained, Δlogp = Bijectors.with_logabsdet_jacobian(b_x, x)
    return (x=x_unconstrained,), logprior_true(model, x) - Δlogp
end

function rand_prior_true(rng::Random.AbstractRNG, model::Model{typeof(demo_lkjchol)})
    x = rand(rng, LKJCholesky(model.args.d, 1.0))
    return (x=x,)
end

# Model to test `StaticTransformation` with.
"""
    demo_static_transformation()

Simple model for which [`default_transformation`](@ref) returns a [`StaticTransformation`](@ref).
"""
@model function demo_static_transformation()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))

    return (; s, m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end

function DynamicPPL.default_transformation(::Model{typeof(demo_static_transformation)})
    b = Bijectors.Stacked(Bijectors.elementwise(exp), identity)
    return DynamicPPL.StaticTransformation(b)
end

posterior_mean(::Model{typeof(demo_static_transformation)}) = (s=49 / 24, m=7 / 6)
function logprior_true(::Model{typeof(demo_static_transformation)}, s, m)
    return logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)
end
function loglikelihood_true(::Model{typeof(demo_static_transformation)}, s, m)
    return logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_static_transformation)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end

function rand_prior_true(
    rng::Random.AbstractRNG, model::Model{typeof(demo_static_transformation)}
)
    s = rand(rng, InverseGamma(2, 3))
    m = rand(rng, Normal(0, sqrt(s)))
    return (s=s, m=m)
end

# A collection of models for which the posterior should be "similar".
# Some utility methods for these.
function _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
    b = Bijectors.bijector(InverseGamma(2, 3))
    s_unconstrained = b.(s)
    Δlogp = sum(Base.Fix1(Bijectors.logabsdetjac, b), s)
    return (s=s_unconstrained, m=m), logprior_true(model, s, m) - Δlogp
end

@model function demo_dot_assume_dot_observe(
    x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
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
function varnames(model::Model{typeof(demo_dot_assume_dot_observe)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function demo_assume_index_observe(
    x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
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
function varnames(model::Model{typeof(demo_assume_index_observe)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function demo_assume_multivariate_observe(x=[1.5, 2.0])
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
function varnames(model::Model{typeof(demo_assume_multivariate_observe)})
    return [@varname(s), @varname(m)]
end

@model function demo_dot_assume_observe_index(
    x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
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
function varnames(model::Model{typeof(demo_dot_assume_observe_index)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function demo_assume_dot_observe(x=[1.5, 2.0])
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
function varnames(model::Model{typeof(demo_assume_dot_observe)})
    return [@varname(s), @varname(m)]
end

@model function demo_assume_observe_literal()
    # `assume` and literal `observe`
    s ~ product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m ~ MvNormal(zeros(2), Diagonal(s))
    [1.5, 2.0] ~ MvNormal(m, Diagonal(s))

    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_observe_literal)}, s, m)
    s_dist = product_distribution([InverseGamma(2, 3), InverseGamma(2, 3)])
    m_dist = MvNormal(zeros(2), Diagonal(s))
    return logpdf(s_dist, s) + logpdf(m_dist, m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_observe_literal)}, s, m)
    return logpdf(MvNormal(m, Diagonal(s)), [1.5, 2.0])
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_observe_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function varnames(model::Model{typeof(demo_assume_observe_literal)})
    return [@varname(s), @varname(m)]
end

@model function demo_dot_assume_observe_index_literal(::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    s = TV(undef, 2)
    m = TV(undef, 2)
    s .~ InverseGamma(2, 3)
    m .~ Normal.(0, sqrt.(s))

    1.5 ~ Normal(m[1], sqrt(s[1]))
    2.0 ~ Normal(m[2], sqrt(s[2]))

    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(
    model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m
)
    return sum(logpdf.(Normal.(m, sqrt.(s)), [1.5, 2.0]))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_dot_assume_observe_index_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function varnames(model::Model{typeof(demo_dot_assume_observe_index_literal)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function demo_assume_literal_dot_observe()
    # `assume` and literal `dot_observe`
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    [1.5, 2.0] .~ Normal(m, sqrt(s))

    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_literal_dot_observe)}, s, m)
    return logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)
end
function loglikelihood_true(model::Model{typeof(demo_assume_literal_dot_observe)}, s, m)
    return loglikelihood(Normal(m, sqrt(s)), [1.5, 2.0])
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_literal_dot_observe)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function varnames(model::Model{typeof(demo_assume_literal_dot_observe)})
    return [@varname(s), @varname(m)]
end

# Only used as a submodel
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
    1.5 ~ Normal(m[1], sqrt(s[1]))
    2.0 ~ Normal(m[2], sqrt(s[2]))

    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end
function logprior_true(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return loglikelihood(InverseGamma(2, 3), s) + sum(logpdf.(Normal.(0, sqrt.(s)), m))
end
function loglikelihood_true(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return sum(logpdf.(Normal.(m, sqrt.(s)), [1.5, 2.0]))
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_submodel_observe_index_literal)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function varnames(model::Model{typeof(demo_assume_submodel_observe_index_literal)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function _likelihood_mltivariate_observe(s, m, x)
    return x ~ MvNormal(m, Diagonal(s))
end

@model function demo_dot_assume_observe_submodel(
    x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
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
function varnames(model::Model{typeof(demo_dot_assume_observe_submodel)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function demo_dot_assume_dot_observe_matrix(
    x=transpose([1.5 2.0;]), ::Type{TV}=Vector{Float64}
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
function varnames(model::Model{typeof(demo_dot_assume_dot_observe_matrix)})
    return [@varname(s[1]), @varname(s[2]), @varname(m[1]), @varname(m[2])]
end

@model function demo_dot_assume_matrix_dot_observe_matrix(
    x=transpose([1.5 2.0;]), ::Type{TV}=Array{Float64}
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
    return loglikelihood(InverseGamma(2, 3), s_vec) +
           logpdf(MvNormal(zeros(n), Diagonal(s_vec)), m)
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
function varnames(model::Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)})
    s = zeros(1, 2) # used for varname concretization only
    return [@varname(s[:, 1], true), @varname(s[:, 2], true), @varname(m)]
end

@model function demo_assume_matrix_dot_observe_matrix(
    x=transpose([1.5 2.0;]), ::Type{TV}=Array{Float64}
) where {TV}
    n = length(x)
    d = n ÷ 2
    s ~ reshape(product_distribution(fill(InverseGamma(2, 3), n)), d, 2)
    s_vec = vec(s)
    m ~ MvNormal(zeros(n), Diagonal(s_vec))

    # Dotted observe for `Matrix`.
    x .~ MvNormal(m, Diagonal(s_vec))

    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end
function logprior_true(model::Model{typeof(demo_assume_matrix_dot_observe_matrix)}, s, m)
    n = length(model.args.x)
    s_vec = vec(s)
    return loglikelihood(InverseGamma(2, 3), s_vec) +
           logpdf(MvNormal(zeros(n), Diagonal(s_vec)), m)
end
function loglikelihood_true(
    model::Model{typeof(demo_assume_matrix_dot_observe_matrix)}, s, m
)
    return loglikelihood(MvNormal(m, Diagonal(vec(s))), model.args.x)
end
function logprior_true_with_logabsdet_jacobian(
    model::Model{typeof(demo_assume_matrix_dot_observe_matrix)}, s, m
)
    return _demo_logprior_true_with_logabsdet_jacobian(model, s, m)
end
function varnames(model::Model{typeof(demo_assume_matrix_dot_observe_matrix)})
    return [@varname(s), @varname(m)]
end

const DemoModels = Union{
    Model{typeof(demo_dot_assume_dot_observe)},
    Model{typeof(demo_assume_index_observe)},
    Model{typeof(demo_assume_multivariate_observe)},
    Model{typeof(demo_dot_assume_observe_index)},
    Model{typeof(demo_assume_dot_observe)},
    Model{typeof(demo_assume_literal_dot_observe)},
    Model{typeof(demo_assume_observe_literal)},
    Model{typeof(demo_dot_assume_observe_index_literal)},
    Model{typeof(demo_assume_submodel_observe_index_literal)},
    Model{typeof(demo_dot_assume_observe_submodel)},
    Model{typeof(demo_dot_assume_dot_observe_matrix)},
    Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)},
    Model{typeof(demo_assume_matrix_dot_observe_matrix)},
}

const UnivariateAssumeDemoModels = Union{
    Model{typeof(demo_assume_dot_observe)},Model{typeof(demo_assume_literal_dot_observe)}
}
function posterior_mean(model::UnivariateAssumeDemoModels)
    return (s=49 / 24, m=7 / 6)
end
function likelihood_optima(::UnivariateAssumeDemoModels)
    return (s=1 / 16, m=7 / 4)
end
function posterior_optima(::UnivariateAssumeDemoModels)
    # TODO: Figure out exact for `s`.
    return (s=0.907407, m=7 / 6)
end
function rand_prior_true(rng::Random.AbstractRNG, model::UnivariateAssumeDemoModels)
    s = rand(rng, InverseGamma(2, 3))
    m = rand(rng, Normal(0, sqrt(s)))

    return (s=s, m=m)
end

const MultivariateAssumeDemoModels = Union{
    Model{typeof(demo_dot_assume_dot_observe)},
    Model{typeof(demo_assume_index_observe)},
    Model{typeof(demo_assume_multivariate_observe)},
    Model{typeof(demo_dot_assume_observe_index)},
    Model{typeof(demo_assume_observe_literal)},
    Model{typeof(demo_dot_assume_observe_index_literal)},
    Model{typeof(demo_assume_submodel_observe_index_literal)},
    Model{typeof(demo_dot_assume_observe_submodel)},
    Model{typeof(demo_dot_assume_dot_observe_matrix)},
    Model{typeof(demo_dot_assume_matrix_dot_observe_matrix)},
}
function posterior_mean(model::MultivariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    vals.s[1] = 19 / 8
    vals.m[1] = 3 / 4

    vals.s[2] = 8 / 3
    vals.m[2] = 1

    return vals
end
function likelihood_optima(model::MultivariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    # NOTE: These are "as close to zero as we can get".
    vals.s[1] = 1e-32
    vals.s[2] = 1e-32

    vals.m[1] = 1.5
    vals.m[2] = 2.0

    return vals
end
function posterior_optima(model::MultivariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    # TODO: Figure out exact for `s[1]`.
    vals.s[1] = 0.890625
    vals.s[2] = 1
    vals.m[1] = 3 / 4
    vals.m[2] = 1

    return vals
end
function rand_prior_true(rng::Random.AbstractRNG, model::MultivariateAssumeDemoModels)
    # Get template values from `model`.
    retval = model(rng)
    vals = (s=retval.s, m=retval.m)
    # Fill containers with realizations from prior.
    for i in LinearIndices(vals.s)
        vals.s[i] = rand(rng, InverseGamma(2, 3))
        vals.m[i] = rand(rng, Normal(0, sqrt(vals.s[i])))
    end

    return vals
end

const MatrixvariateAssumeDemoModels = Union{
    Model{typeof(demo_assume_matrix_dot_observe_matrix)}
}
function posterior_mean(model::MatrixvariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    vals.s[1, 1] = 19 / 8
    vals.m[1] = 3 / 4

    vals.s[1, 2] = 8 / 3
    vals.m[2] = 1

    return vals
end
function likelihood_optima(model::MatrixvariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    # NOTE: These are "as close to zero as we can get".
    vals.s[1, 1] = 1e-32
    vals.s[1, 2] = 1e-32

    vals.m[1] = 1.5
    vals.m[2] = 2.0

    return vals
end
function posterior_optima(model::MatrixvariateAssumeDemoModels)
    # Get some containers to fill.
    vals = rand_prior_true(model)

    # TODO: Figure out exact for `s[1]`.
    vals.s[1, 1] = 0.890625
    vals.s[1, 2] = 1
    vals.m[1] = 3 / 4
    vals.m[2] = 1

    return vals
end
function rand_prior_true(rng::Random.AbstractRNG, model::MatrixvariateAssumeDemoModels)
    # Get template values from `model`.
    retval = model(rng)
    vals = (s=retval.s, m=retval.m)
    # Fill containers with realizations from prior.
    for i in LinearIndices(vals.s)
        vals.s[i] = rand(rng, InverseGamma(2, 3))
        vals.m[i] = rand(rng, Normal(0, sqrt(vals.s[i])))
    end

    return vals
end

"""
A collection of models corresponding to the posterior distribution defined by
the generative process

    s ~ InverseGamma(2, 3)
    m ~ Normal(0, √s)
    1.5 ~ Normal(m, √s)
    2.0 ~ Normal(m, √s)

or by

    s[1] ~ InverseGamma(2, 3)
    s[2] ~ InverseGamma(2, 3)
    m[1] ~ Normal(0, √s)
    m[2] ~ Normal(0, √s)
    1.5 ~ Normal(m[1], √s[1])
    2.0 ~ Normal(m[2], √s[2])

These are examples of a Normal-InverseGamma conjugate prior with Normal likelihood,
for which the posterior is known in closed form.

In particular, for the univariate model (the former one):

    mean(s) == 49 / 24
    mean(m) == 7 / 6

And for the multivariate one (the latter one):

    mean(s[1]) == 19 / 8
    mean(m[1]) == 3 / 4
    mean(s[2]) == 8 / 3
    mean(m[2]) == 1

"""
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
    demo_assume_matrix_dot_observe_matrix(),
)
