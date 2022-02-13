using Distributions: Distributions
using Bijectors: Bijectors
using Distributions: Univariate, Multivariate, Matrixvariate

"""
A named distribution that carries the name of the random variable with it.
"""
struct NamedDist{variate,support,Td<:Distribution{variate,support},Tv<:VarName} <:
       Distribution{variate,support}
    dist::Td
    name::Tv
end

NamedDist(dist::Distribution, name::Symbol) = NamedDist(dist, VarName{name}())

Distributions.logpdf(dist::NamedDist, x::Real) = Distributions.logpdf(dist.dist, x)
function Distributions.logpdf(dist::NamedDist, x::AbstractArray{<:Real})
    return Distributions.logpdf(dist.dist, x)
end
function Distributions.loglikelihood(dist::NamedDist, x::Real)
    return Distributions.loglikelihood(dist.dist, x)
end
function Distributions.loglikelihood(dist::NamedDist, x::AbstractArray{<:Real})
    return Distributions.loglikelihood(dist.dist, x)
end

struct NoDist{variate,support,Td<:Distribution{variate,support}} <:
       Distribution{variate,support}
    dist::Td
end
NoDist(dist::NamedDist) = NamedDist(NoDist(dist.dist), dist.name)

Distributions.rand(rng::Random.AbstractRNG, d::NoDist) = rand(rng, d.dist)
Distributions.logpdf(d::NoDist{<:Univariate}, ::Real) = 0
Distributions.logpdf(d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}) = 0
function Distributions.logpdf(d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(Int, size(x, 2))
end
Distributions.logpdf(d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}) = 0
Distributions.minimum(d::NoDist) = minimum(d.dist)
Distributions.maximum(d::NoDist) = maximum(d.dist)

Bijectors.logpdf_with_trans(d::NoDist{<:Univariate}, ::Real) = 0
Bijectors.logpdf_with_trans(d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}) = 0
function Bijectors.logpdf_with_trans(d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(Int, size(x, 2))
end
Bijectors.logpdf_with_trans(d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}) = 0
