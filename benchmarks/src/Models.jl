"""
Models for benchmarking Turing.jl.

Each model returns a NamedTuple of all the random variables in the model that are not
observed (this is used for constructing SimpleVarInfos).
"""
module Models

using DynamicPPL: @model, to_submodel
using Distributions:
    Categorical,
    Dirichlet,
    Exponential,
    Gamma,
    LKJCholesky,
    MatrixBeta,
    Normal,
    logpdf,
    product_distribution,
    truncated

export simple_assume_observe_non_model,
    simple_assume_observe, smorgasbord, loop_univariate, multivariate, parent, dynamic, lda

# This one is like simple_assume_observe, but explicitly does not use DynamicPPL.
# Other runtimes are normalised by this one's runtime.
function simple_assume_observe_non_model(x, obs)
    logp = logdf(x, Normal())
    logp += logpdf(obs, Normal(x, 1))
    return logp
end

@model function simple_assume_observe(obs)
    x ~ Normal()
    obs ~ Normal(x, 1)
    return (; x=x)
end

@model function smorgasbord(x, y, ::Type{TV}=Vector{Float64}) where {TV}
    @assert length(x) == length(y)
    m ~ truncated(Normal(); lower=0)
    means ~ product_distribution(fill(Exponential(m), length(x)))
    stds = TV(undef, length(x))
    stds .~ Gamma(1, 1)
    for i in 1:length(x)
        x[i] ~ Normal(means[i], stds[i])
    end
    y ~ product_distribution([Normal(means[i], stds[i]) for i in 1:length(x)])
    0.0 ~ Normal(sum(y), 1)
    return (; m=m, means=means, stds=stds)
end

@model function loop_univariate(num_dims, ::Type{TV}=Vector{Float64}) where {TV}
    a = TV(undef, num_dims)
    o = TV(undef, num_dims)
    for i in 1:num_dims
        a[i] ~ Normal(0, 1)
    end
    m = sum(a)
    for i in 1:num_dims
        o[i] ~ Normal(m, 1)
    end
    return (; a=a)
end

@model function multivariate(num_dims, ::Type{TV}=Vector{Float64}) where {TV}
    a = TV(undef, num_dims)
    o = TV(undef, num_dims)
    a ~ product_distribution(fill(Normal(0, 1), num_dims))
    m = sum(a)
    o ~ product_distribution(fill(Normal(m, 1), num_dims))
    return (; a=a)
end

@model function sub()
    x ~ Normal()
    return x
end

@model function parent(y)
    x ~ to_submodel(sub())
    y ~ Normal(x, 1)
    return (; x=x)
end

@model function dynamic(::Type{T}=Vector{Float64}) where {T}
    eta ~ truncated(Normal(); lower=0.0)
    mat1 ~ LKJCholesky(4, eta)
    mat2 ~ MatrixBeta(5, 6.0, 8.0)
    dim = eta > 0.2 ? 2 : 3
    vec = T(undef, dim)
    vec .~ truncated(Exponential(0.5); lower=0.0, upper=1.0)
    return (; eta=eta, mat1=mat1, mat2=mat2, vec=vec)
end

@model function lda(K, d, w)
    V = length(unique(w))
    D = length(unique(d))
    N = length(d)
    @assert length(w) == N

    ϕ = Vector{Vector{Real}}(undef, K)
    for i in 1:K
        ϕ[i] ~ Dirichlet(ones(V) / V)
    end

    θ = Vector{Vector{Real}}(undef, D)
    for i in 1:D
        θ[i] ~ Dirichlet(ones(K) / K)
    end

    z = zeros(Int, N)

    for i in 1:N
        z[i] ~ Categorical(θ[d[i]])
        w[i] ~ Categorical(ϕ[d[i]])
    end
    return (; ϕ=ϕ, θ=θ, z=z)
end

end
