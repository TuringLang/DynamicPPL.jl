using Bijectors: Bijectors
using Distributions:
    Distributions,
    Univariate,
    Multivariate,
    Matrixvariate,
    product_distribution,
    Product,
    UnivariateDistribution
using FillArrays: Fill

"""
    NamedDist(dist::Distribution, name::Symbol)

A named distribution that carries the name of the random variable with it.

# Fields

$(TYPEDFIELDS)
"""
struct NamedDist{variate,support,Td<:Distribution{variate,support},Tv<:VarName} <:
       Distribution{variate,support}
    "the underlying distribution"
    dist::Td
    "the name of the random variable"
    name::Tv
end

NamedDist(dist::Distribution, name::Symbol) = NamedDist(dist, VarName{name}())

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

struct NoDist{variate,support,Td<:Distribution{variate,support}} <:
       Distribution{variate,support}
    dist::Td
end
NoDist(dist::NamedDist) = NamedDist(NoDist(dist.dist), dist.name)

nodist(dist::Distribution) = NoDist(dist)
nodist(dists::AbstractArray) = nodist.(dists)

Distributions.rand(rng::Random.AbstractRNG, d::NoDist) = rand(rng, d.dist)
# NOTE(torfjelde): Need this to avoid stack overflow.
function Distributions.rand!(
    rng::Random.AbstractRNG,
    d::NoDist{Distributions.ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,N},
) where {N}
    return Distributions.rand!(rng, d.dist, x)
end
function Distributions.logpdf(::NoDist{<:Univariate}, x::Real)
    return zero(LogProbType)
end
function Distributions.logpdf(::NoDist{<:Multivariate}, x::AbstractVector{<:Real})
    return zero(LogProbType)
end
function Distributions.logpdf(::NoDist{<:Matrixvariate}, x::AbstractMatrix{<:Real})
    return zero(LogProbType)
end

for f in (
    :(Bijectors.VectorBijectors.from_linked_vec),
    :(Bijectors.VectorBijectors.to_linked_vec),
    :(Bijectors.VectorBijectors.from_vec),
    :(Bijectors.VectorBijectors.to_vec),
    :(Bijectors.VectorBijectors.vec_length),
    :(Bijectors.VectorBijectors.linked_vec_length),
    :(Bijectors.VectorBijectors.optic_vec),
    :(Bijectors.VectorBijectors.linked_optic_vec),
    :(Base.length),
    :(Base.size),
    :(Distributions.minimum),
    :(Distributions.maximum),
)
    @eval $f(d::NoDist) = $f(d.dist)
    @eval $f(d::NamedDist) = $f(d.dist)
end

"""
    filldist(d::Distribution, ns...)

Create a product distribution from a single distribution and a list of dimension sizes. If
`size(d)` is `(d1, d2, ...)` and `ns` is `(n1, n2, ...)`, then the resulting distribution
will have size `(d1, d2, ..., n1, n2, ...)`.

When sampling from the resulting distribution, the output will be an array where each
element is sampled from the original distribution `d`.

This is a convenient wrapper around `product_distribution(FillArrays.Fill(d, ns...))`.

!!! note
    `filldist` used to be defined in DistributionsAD.jl. The definition here is semantically
    equivalent, but removes a lot of custom code that is no longer needed nowadays.
"""
function filldist(dist::Distribution, dim::Int, dims::Int...)
    return product_distribution(Fill(dist, dim, dims...))
end

"""
    arraydist(dists::AbstractArray{<:Distribution})

Create a product distribution from an array of sub-distributions. Each element of `dists`
should have the same size. If the size of each element is `(d1, d2, ...)`, and `size(dists)`
is `(n1, n2, ...)`, then the resulting distribution will have size `(d1, d2, ..., n1, n2,
...)`.

This is equivalent to `product_distribution(dists)`, but can be more performant in some
instances (specifically when `dists` is a vector of `Normal`s).

!!! note
    `arraydist` used to be defined in DistributionsAD.jl. The definition here is
    semantically equivalent, but removes a lot of custom code that is no longer needed
    nowadays.
"""
function arraydist(dists::AbstractArray{<:Distribution})
    return product_distribution(dists)
end
function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    return Product(dists)
end