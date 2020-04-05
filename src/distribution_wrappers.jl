import Distributions
import Bijectors
using Distributions: Univariate,
                     Multivariate,
                     Matrixvariate


"""
A named distribution that carries the name of the random variable with it.
"""
struct NamedDist{
    variate, 
    support, 
    Td <: Distribution{variate, support},
    name
} <: Distribution{variate, support}
    dist::Td
    name::VarName{name}
end

NamedDist(dist::Distribution, name::Symbol) = NamedDist(dist, VarName(name))


struct NoDist{
    variate, 
    support, 
    Td <: Distribution{variate, support}
} <: Distribution{variate, support}
    dist::Td
end
NoDist(dist::NamedDist) = NamedDist(NoDist(dist.dist), dist.name)

Distributions.rand(d::NoDist) = rand(d.dist)
Distributions.logpdf(d::NoDist{<:Univariate}, ::Real) = 0
Distributions.logpdf(d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}) = 0
function Distributions.logpdf(d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(Int, size(x, 2))
end
Distributions.logpdf(d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}) = 0

Bijectors.logpdf_with_trans(d::NoDist{<:Univariate}, ::Real) = 0
Bijectors.logpdf_with_trans(d::NoDist{<:Multivariate}, ::AbstractVector{<:Real}) = 0
function Bijectors.logpdf_with_trans(d::NoDist{<:Multivariate}, x::AbstractMatrix{<:Real})
    return zeros(Int, size(x, 2))
end
Bijectors.logpdf_with_trans(d::NoDist{<:Matrixvariate}, ::AbstractMatrix{<:Real}) = 0
