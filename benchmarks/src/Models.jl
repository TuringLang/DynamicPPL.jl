"""
Models for benchmarking Turing.jl.

Each model returns a NamedTuple of all the random variables in the model that are not
observed (this is used for constructing SimpleVarInfos).
"""
module Models

using Distributions:
    Categorical,
    Dirichlet,
    Exponential,
    Gamma,
    LKJCholesky,
    InverseWishart,
    Normal,
    logpdf,
    product_distribution,
    truncated
using DynamicPPL: DynamicPPL, @model, to_submodel
using LinearAlgebra: cholesky

export simple_assume_observe_non_model,
    simple_assume_observe, smorgasbord, loop_univariate, multivariate, parent, dynamic, lda

# This one is like simple_assume_observe, but explicitly does not use DynamicPPL.
# Other runtimes are normalised by this one's runtime.
function simple_assume_observe_non_model(obs)
    x = rand(Normal())
    logp = logpdf(Normal(), x)
    logp += logpdf(Normal(x, 1), obs)
    return (; logp=logp, x=x)
end

"""
A simple model that does one scalar assumption and one scalar observation.
"""
@model function simple_assume_observe(obs)
    x ~ Normal()
    obs ~ Normal(x, 1)
    return (; x=x)
end

"""
A short model that tries to cover many DynamicPPL features.

Includes scalar, vector univariate, and multivariate variables; ~, .~, and loops; allocating
a variable vector; observations passed as arguments, and as literals.
"""
@model function smorgasbord(x, y, (::Type{TV})=Vector{Float64}) where {TV}
    @assert length(x) == length(y)
    m ~ truncated(Normal(); lower=0)
    means ~ product_distribution(fill(Exponential(m), length(x)))
    stds = TV(undef, length(x))
    stds .~ Gamma(1, 1)
    for i in 1:length(x)
        x[i] ~ Normal(means[i], stds[i])
    end
    y ~ product_distribution(map((mean, std) -> Normal(mean, std), means, stds))
    0.0 ~ Normal(sum(y), 1)
    return (; m=m, means=means, stds=stds)
end

"""
A model that loops over two vectors of univariate normals of length `num_dims`.

The second variable, `o`, is meant to be conditioned on after model instantiation.

See `multivariate` for a version that uses `product_distribution` rather than loops.
"""
@model function loop_univariate(num_dims, (::Type{TV})=Vector{Float64}) where {TV}
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

"""
A model with two multivariate normal distributed variables of dimension `num_dims`.

The second variable, `o`, is meant to be conditioned on after model instantiation.

See `loop_univariate` for a version that uses loops rather than `product_distribution`.
"""
@model function multivariate(num_dims, (::Type{TV})=Vector{Float64}) where {TV}
    a = TV(undef, num_dims)
    o = TV(undef, num_dims)
    a ~ product_distribution(fill(Normal(0, 1), num_dims))
    m = sum(a)
    o ~ product_distribution(fill(Normal(m, 1), num_dims))
    return (; a=a)
end

"""
A submodel for `parent`. Not exported.
"""
@model function sub()
    x ~ Normal()
    return x
end

"""
Like simple_assume_observe, but with a submodel for the assumed random variable.
"""
@model function parent(obs)
    x ~ to_submodel(sub())
    obs ~ Normal(x, 1)
    return (; x=x)
end

"""
A model with random variables that have changing support under linking, or otherwise
complicated bijectors.
"""
@model function dynamic((::Type{T})=Vector{Float64}) where {T}
    eta ~ truncated(Normal(); lower=0.0, upper=0.1)
    mat1 ~ LKJCholesky(4, eta)
    mat2 ~ InverseWishart(3.2, cholesky([1.0 0.5; 0.5 1.0]))
    return (; eta=eta, mat1=mat1, mat2=mat2)
end

"""
A simple Linear Discriminant Analysis model.
"""
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
