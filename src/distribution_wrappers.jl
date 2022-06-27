using Distributions: Distributions
using Bijectors: Bijectors
using Distributions: Univariate, Multivariate, Matrixvariate

"""
Base type for distribution wrappers.
"""
abstract type WrappedDistribution{variate,support,Td<:Distribution{variate,support}} <:
              Distribution{variate,support}
end

wrapped_dist_type(::Type{<:WrappedDistribution{<:Any,<:Any,Td}}) where {Td} = Td
wrapped_dist_type(d::WrappedDistribution) = wrapped_dist_type(d)

wrapped_dist(d::WrappedDistribution) = d.dist

Base.length(d::WrappedDistribution{<:Multivariate}) = length(wrapped_dist(d))
Base.size(d::WrappedDistribution{<:Multivariate}) = size(wrapped_dist(d))
Base.eltype(::Type{T}) where {T<:WrappedDistribution} = eltype(wrapped_dist_type(T))
Base.eltype(d::WrappedDistribution) = eltype(wrapped_dist_type(d))

function Distributions.rand(rng::Random.AbstractRNG, d::WrappedDistribution)
    rand(rng, wrapped_dist(d))
end
Distributions.minimum(d::WrappedDistribution) = minimum(wrapped_dist(d))
Distributions.maximum(d::WrappedDistribution) = maximum(wrapped_dist(d))

Bijectors.bijector(d::WrappedDistribution) = bijector(wrapped_dist(d))

"""
A named distribution that carries the name of the random variable with it.
"""
struct NamedDist{variate,support,Td<:Distribution{variate,support},Tv<:VarName} <:
       WrappedDistribution{variate,support,Td}
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

"""
Wrapper around distribution `Td` that suppresses `logpdf()` calculation.

Note that *SampleFromPrior* would still sample from `Td`.
"""
struct NoDist{variate,support,Td<:Distribution{variate,support}} <:
       WrappedDistribution{variate,support,Td}
    dist::Td
end
NoDist(dist::NamedDist) = NamedDist(NoDist(dist.dist), dist.name)

nodist(dist::Distribution) = NoDist(dist)
nodist(dists::AbstractArray) = nodist.(dists)

Distributions.rand(rng::Random.AbstractRNG, d::NoDist) = rand(rng, d.dist)
Distributions.logpdf(d::NoDist{<:Univariate}, ::Real) = 0
Distributions.logpdf(d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}) = 0
function Distributions.logpdf(d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(Int, size(x, 2))
end
Distributions.logpdf(d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}) = 0

Bijectors.logpdf_with_trans(d::NoDist{<:Univariate}, ::Real, ::Bool) = 0
function Bijectors.logpdf_with_trans(
    d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}, ::Bool
)
    return 0
end
function Bijectors.logpdf_with_trans(
    d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real}, ::Bool
)
    return zeros(Int, size(x, 2))
end
function Bijectors.logpdf_with_trans(
    d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}, ::Bool
)
    return 0
end
