module TestUtils

using AbstractMCMC
using DynamicPPL
using LinearAlgebra
using Distributions
using Test

using Random: Random
using Bijectors: Bijectors
using Setfield: Setfield

# For backwards compat.
using DynamicPPL: varname_leaves

"""
    update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)

Return instance similar to `vi` but with `vns` set to values from `vals`.
"""
function update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)
    for vn in vns
        vi = DynamicPPL.setindex!!(vi, get(vals, vn), vn)
    end
    return vi
end

"""
    test_values(vi::AbstractVarInfo, vals::NamedTuple, vns)

Test that `vi[vn]` corresponds to the correct value in `vals` for every `vn` in `vns`.
"""
function test_values(vi::AbstractVarInfo, vals::NamedTuple, vns; isequal=isequal, kwargs...)
    for vn in vns
        @test isequal(vi[vn], get(vals, vn); kwargs...)
    end
end

"""
    setup_varinfos(model::Model, example_values::NamedTuple, varnames)

Return a tuple of instances for different implementations of `AbstractVarInfo` with
each `vi`, supposedly, satisfying `vi[vn] == get(example_values, vn)` for `vn` in `varnames`.
"""
function setup_varinfos(model::Model, example_values::NamedTuple, varnames)
    # <:VarInfo
    vi_untyped = VarInfo()
    model(vi_untyped)
    vi_typed = DynamicPPL.TypedVarInfo(vi_untyped)
    # <:SimpleVarInfo
    svi_typed = SimpleVarInfo(example_values)
    svi_untyped = SimpleVarInfo(OrderedDict())

    return map((vi_untyped, vi_typed, svi_typed, svi_untyped)) do vi
        # Set them all to the same values.
        update_values!!(vi, example_values, varnames)
    end
end

"""
    logprior_true(model, args...)

Return the `logprior` of `model` for `args`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`loglikelihood_true`](@ref).
"""
function logprior_true end

"""
    loglikelihood_true(model, args...)

Return the `loglikelihood` of `model` for `args`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`logprior_true`](@ref).
"""
function loglikelihood_true end

"""
    logjoint_true(model, args...)

Return the `logjoint` of `model` for `args`.

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

Return a tuple `(args_unconstrained, logjoint)` of `model` for `args`.

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
    varnames(model::Model)

Return a collection of `VarName` as they are expected to appear in the model.

Even though it is recommended to implement this by hand for a particular `Model`,
a default implementation using [`SimpleVarInfo{<:Dict}`](@ref) is provided.
"""
function varnames(model::Model)
    return collect(
        keys(last(DynamicPPL.evaluate!!(model, SimpleVarInfo(Dict()), SamplingContext())))
    )
end

"""
    posterior_mean(model::Model)

Return a `NamedTuple` compatible with `varnames(model)` where the values represent
the posterior mean under `model`.

"Compatible" means that a `varname` from `varnames(model)` can be used to extract the
corresponding value using `get`, e.g. `get(posterior_mean(model), varname)`.
"""
function posterior_mean end

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
    return [@varname(s[:, 1]), @varname(s[:, 2]), @varname(m)]
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
}

# We require demo models to have explict impleentations of `rand` since we want
# these to be considered as ground truth.
function Random.rand(rng::Random.AbstractRNG, ::Type{NamedTuple}, model::DemoModels)
    return error("demo models requires explicit implementation of rand")
end

const UnivariateAssumeDemoModels = Union{
    Model{typeof(demo_assume_dot_observe)},Model{typeof(demo_assume_literal_dot_observe)}
}
function posterior_mean(model::UnivariateAssumeDemoModels)
    return (s=49 / 24, m=7 / 6)
end
function Random.rand(
    rng::Random.AbstractRNG, ::Type{NamedTuple}, model::UnivariateAssumeDemoModels
)
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
    vals = Random.rand(model)

    vals.s[1] = 19 / 8
    vals.m[1] = 3 / 4

    vals.s[2] = 8 / 3
    vals.m[2] = 1

    return vals
end
function Random.rand(
    rng::Random.AbstractRNG, ::Type{NamedTuple}, model::MultivariateAssumeDemoModels
)
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
)

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

"""
    marginal_mean_of_samples(chain, varname)

Return the mean of variable represented by `varname` in `chain`.
"""
marginal_mean_of_samples(chain, varname) = mean(Array(chain[Symbol(varname)]))

"""
    test_sampler(models, sampler, args...; kwargs...)

Test that `sampler` produces correct marginal posterior means on each model in `models`.

In short, this method iterates through `models`, calls `AbstractMCMC.sample` on the
`model` and `sampler` to produce a `chain`, and then checks `marginal_mean_of_samples(chain, vn)`
for every (leaf) varname `vn` against the corresponding value returned by
[`posterior_mean`](@ref) for each model.

To change how comparison is done for a particular `chain` type, one can overload
[`marginal_mean_of_samples`](@ref) for the corresponding type.

# Arguments
- `models`: A collection of instaces of [`DynamicPPL.Model`](@ref) to test on.
- `sampler`: The `AbstractMCMC.AbstractSampler` to test.
- `args...`: Arguments forwarded to `sample`.

# Keyword arguments
- `varnames_filter`: A filter to apply to `varnames(model)`, allowing comparison for only
    a subset of the varnames.
- `atol=1e-1`: Absolute tolerance used in `@test`.
- `rtol=1e-3`: Relative tolerance used in `@test`.
- `kwargs...`: Keyword arguments forwarded to `sample`.
"""
function test_sampler(
    models,
    sampler::AbstractMCMC.AbstractSampler,
    args...;
    varnames_filter=Returns(true),
    atol=1e-1,
    rtol=1e-3,
    kwargs...,
)
    @testset "$(typeof(sampler)) on $(nameof(model))" for model in models
        chain = AbstractMCMC.sample(model, sampler, args...; kwargs...)
        target_values = posterior_mean(model)
        for vn in filter(varnames_filter, varnames(model))
            # We want to compare elementwise which can be achieved by
            # extracting the leaves of the `VarName` and the corresponding value.
            for vn_leaf in varname_leaves(vn, get(target_values, vn))
                target_value = get(target_values, vn_leaf)
                chain_mean_value = marginal_mean_of_samples(chain, vn_leaf)
                @test chain_mean_value ≈ target_value atol = atol rtol = rtol
            end
        end
    end
end

"""
    test_sampler_on_demo_models(meanfunction, sampler, args...; kwargs...)

Test `sampler` on every model in [`DEMO_MODELS`](@ref).

This is just a proxy for `test_sampler(meanfunction, DEMO_MODELS, sampler, args...; kwargs...)`.
"""
function test_sampler_on_demo_models(
    sampler::AbstractMCMC.AbstractSampler, args...; kwargs...
)
    return test_sampler(DEMO_MODELS, sampler, args...; kwargs...)
end

"""
    test_sampler_continuous(sampler, args...; kwargs...)

Test that `sampler` produces the correct marginal posterior means on all models in `demo_models`.

As of right now, this is just an alias for [`test_sampler_on_demo_models`](@ref).
"""
function test_sampler_continuous(sampler::AbstractMCMC.AbstractSampler, args...; kwargs...)
    return test_sampler_on_demo_models(sampler, args...; kwargs...)
end

end
