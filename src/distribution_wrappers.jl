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

Base.length(dist::NamedDist) = Base.length(dist.dist)
Base.size(dist::NamedDist) = Base.size(dist.dist)

Distributions.logpdf(dist::NamedDist, x::Real) = Distributions.logpdf(dist.dist, x)
function Distributions.logpdf(dist::NamedDist, x::AbstractArray{<:Real,0})
    # extract the singleton value from 0-dimensional array
    return Distributions.logpdf(dist.dist, first(x))
end
function Distributions.logpdf(dist::NamedDist, x::AbstractArray{<:Real})
    return Distributions.logpdf(dist.dist, x)
end
function Distributions.loglikelihood(dist::NamedDist, x::Real)
    return Distributions.loglikelihood(dist.dist, x)
end
function Distributions.loglikelihood(dist::NamedDist, x::AbstractArray{<:Real})
    return Distributions.loglikelihood(dist.dist, x)
end

Bijectors.bijector(d::NamedDist) = Bijectors.bijector(d.dist)

struct NoDist{variate,support,Td<:Distribution{variate,support}} <:
       Distribution{variate,support}
    dist::Td
end
NoDist(dist::NamedDist) = NamedDist(NoDist(dist.dist), dist.name)

nodist(dist::Distribution) = NoDist(dist)
nodist(dists::AbstractArray) = nodist.(dists)

Base.length(dist::NoDist) = Base.length(dist.dist)
Base.size(dist::NoDist) = Base.size(dist.dist)

Distributions.rand(rng::Random.AbstractRNG, d::NoDist) = rand(rng, d.dist)
# NOTE(torfjelde): Need this to avoid stack overflow.
function Distributions.rand!(
    rng::Random.AbstractRNG,
    d::NoDist{Distributions.ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,N},
) where {N}
    return Distributions.rand!(rng, d.dist, x)
end
Distributions.logpdf(::NoDist{<:Univariate}, x::Real) = zero(eltype(x))
Distributions.logpdf(::NoDist{<:Multivariate}, x::AbstractVector{<:Real}) = zero(eltype(x))
function Distributions.logpdf(::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(eltype(x), size(x, 2))
end
Distributions.logpdf(::NoDist{<:Matrixvariate}, x::AbstractMatrix{<:Real}) = zero(eltype(x))
Distributions.minimum(d::NoDist) = minimum(d.dist)
Distributions.maximum(d::NoDist) = maximum(d.dist)

Bijectors.logpdf_with_trans(::NoDist{<:Univariate}, x::Real, ::Bool) = zero(eltype(x))
function Bijectors.logpdf_with_trans(
    ::NoDist{<:Multivariate}, x::AbstractVector{<:Real}, ::Bool
)
    return zero(eltype(x))
end
function Bijectors.logpdf_with_trans(
    ::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real}, ::Bool
)
    return zeros(eltype(x), size(x, 2))
end
function Bijectors.logpdf_with_trans(
    ::NoDist{<:Matrixvariate}, x::AbstractMatrix{<:Real}, ::Bool
)
    return zero(eltype(x))
end

Bijectors.bijector(d::NoDist) = Bijectors.bijector(d.dist)
