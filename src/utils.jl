# singleton for indicating if no default arguments are present
struct NoDefault end
const NO_DEFAULT = NoDefault()

"""
    @addlogprob!(ex)

Add the result of the evaluation of `ex` to the joint log probability.
"""
macro addlogprob!(ex)
    return quote
        acclogp!($(esc(:(__varinfo__))), $(esc(ex)))
    end
end

"""
    getargs_dottilde(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L .~ R` or
`(~).(L, R)`, or `nothing` otherwise.
"""
getargs_dottilde(x) = nothing
function getargs_dottilde(expr::Expr)
    return MacroTools.@match expr begin
        (.~)(L_, R_) => (L, R)
        (~).(L_, R_) => (L, R)
        # Julia 1.6: see https://github.com/TuringLang/Turing.jl/issues/1525
        (L_ .~ R_) => (L, R)
        x_ => nothing
    end
end

"""
    getargs_tilde(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L ~ R`, or `nothing`
otherwise.
"""
getargs_tilde(x) = nothing
function getargs_tilde(expr::Expr)
    return MacroTools.@match expr begin
        (~)(L_, R_) => (L, R)
        x_ => nothing
    end
end

############################################
# Julia 1.2 temporary fix - Julia PR 33303 #
############################################
function to_namedtuple_expr(syms, vals=syms)
    if length(syms) == 0
        nt = :(NamedTuple())
    else
        nt_type = Expr(:curly, :NamedTuple, 
            Expr(:tuple, QuoteNode.(syms)...), 
            Expr(:curly, :Tuple, [:(Core.Typeof($x)) for x in vals]...)
        )
        nt = Expr(:call, :($(DynamicPPL.namedtuple)), nt_type, Expr(:tuple, vals...))
    end
    return nt
end


if VERSION == v"1.2"
    @eval function namedtuple(::Type{NamedTuple{names, T}}, args::Tuple) where {names, T <: Tuple}
        if length(args) != length(names)
            throw(ArgumentError("Wrong number of arguments to named tuple constructor."))
        end
        # Note T(args) might not return something of type T; e.g.
        # Tuple{Type{Float64}}((Float64,)) returns a Tuple{DataType}
        $(Expr(:splatnew, :(NamedTuple{names,T}), :(T(args))))
    end
else
    function namedtuple(::Type{NamedTuple{names, T}}, args::Tuple) where {names, T <: Tuple}
        return NamedTuple{names, T}(args)
    end
end

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d::UnivariateDistribution, r::Real) = [r]
vectorize(d::MultivariateDistribution, r::AbstractVector{<:Real}) = copy(r)
vectorize(d::MatrixDistribution, r::AbstractMatrix{<:Real}) = copy(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
reconstruct(d::UnivariateDistribution, val::AbstractVector) = val[1]
reconstruct(d::MultivariateDistribution, val::AbstractVector) = copy(val)
function reconstruct(d::MatrixDistribution, val::AbstractVector)
    return reshape(copy(val), size(d))
end
function reconstruct!(r, d::Distribution, val::AbstractVector)
    return reconstruct!(r, d, val)
end
function reconstruct!(r, d::MultivariateDistribution, val::AbstractVector)
    r .= val
    return r
end
function reconstruct(d::Distribution, val::AbstractVector, n::Int)
    return reconstruct(d, val, n)
end
function reconstruct(d::UnivariateDistribution, val::AbstractVector, n::Int)
    return copy(val)
end
function reconstruct(d::MultivariateDistribution, val::AbstractVector, n::Int)
    return copy(reshape(val, size(d)[1], n))
end
function reconstruct(d::MatrixDistribution, val::AbstractVector, n::Int)
    tmp = reshape(val, size(d)[1], size(d)[2], n)
    orig = [tmp[:, :, i] for i in 1:size(tmp, 3)]
    return orig
end
function reconstruct!(r, d::Distribution, val::AbstractVector, n::Int)
    return reconstruct!(r, d, val, n)
end
function reconstruct!(r, d::MultivariateDistribution, val::AbstractVector, n::Int)
    r .= val
    return r
end

# Uniform random numbers with range 4 for robust initializations
# Reference: https://mc-stan.org/docs/2_19/reference-manual/initialization.html
randrealuni(rng::Random.AbstractRNG) = 4 * rand(rng) - 2
randrealuni(rng::Random.AbstractRNG, args...) = 4 .* rand(rng, args...) .- 2

const Transformable = Union{PositiveDistribution,UnitDistribution,TransformDistribution,
                            SimplexDistribution,PDMatDistribution}
istransformable(dist) = false
istransformable(::Transformable) = true

#################################
# Single-sample initialisations #
#################################

inittrans(rng, dist::UnivariateDistribution) = invlink(dist, randrealuni(rng))
inittrans(rng, dist::MultivariateDistribution) = invlink(dist, randrealuni(rng, size(dist)[1]))
inittrans(rng, dist::MatrixDistribution) = invlink(dist, randrealuni(rng, size(dist)...))

################################
# Multi-sample initialisations #
################################

inittrans(rng, dist::UnivariateDistribution, n::Int) = invlink(dist, randrealuni(rng, n))
function inittrans(rng, dist::MultivariateDistribution, n::Int)
    return invlink(dist, randrealuni(rng, size(dist)[1], n))
end
function inittrans(rng, dist::MatrixDistribution, n::Int)
    return invlink(dist, [randrealuni(rng, size(dist)...) for _ in 1:n])
end


#######################
# Convenience methods #
#######################
collectmaybe(x) = x
collectmaybe(x::Base.AbstractSet) = collect(x)

_getindex(x, inds::Tuple) = _getindex(x[first(inds)...], Base.tail(inds))
_getindex(x, inds::Tuple{}) = x
