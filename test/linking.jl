using Bijectors

# Simple transformations which alters the "dimension" of the variable.
struct TrilToVec{S}
    size::S
end

struct TrilFromVec{S}
    size::S
end

Bijectors.inverse(f::TrilToVec) = TrilFromVec(f.size)
Bijectors.inverse(f::TrilFromVec) = TrilToVec(f.size)

function (v::TrilToVec)(x)
    mask = tril(trues(v.size))
    return vec(x[mask])
end
function (v::TrilFromVec)(y)
    mask = tril(trues(v.size))
    x = similar(y, v.size)
    x[mask] .= y
    return LowerTriangular(x)
end

# Just some dummy values so we can make sure that the log-prob computation
# has been altered correctly.
Bijectors.with_logabsdet_jacobian(f::TrilToVec, x) = (f(x), log(eltype(x)(2)))
Bijectors.with_logabsdet_jacobian(f::TrilFromVec, x) = (f(x), -eltype(x)(log(2)))

# Dummy example.
struct MyMatrixDistribution <: ContinuousMatrixDistribution
    dim::Int
end

Base.size(d::MyMatrixDistribution) = (d.dim, d.dim)
function Distributions._rand!(
    rng::AbstractRNG, d::MyMatrixDistribution, x::AbstractMatrix{<:Real}
)
    return randn!(rng, x)
end
function Distributions._logpdf(::MyMatrixDistribution, x::AbstractMatrix{<:Real})
    return - sum(abs2, LowerTriangular(x)) / 2
end

# Skip reconstruction in the inverse-map since it's no longer needed.
DynamicPPL.reconstruct(::TrilFromVec, ::MyMatrixDistribution, x::AbstractVector{<:Real}) = x

# Specify the link-transform to use.
Bijectors.bijector(dist::MyMatrixDistribution) = TrilToVec((dist.dim, dist.dim))
function Bijectors.logpdf_with_trans(dist::MyMatrixDistribution, x, istrans::Bool)
    lp = logpdf(dist, x)
    if istrans
        lp = lp - logabsdetjac(bijector(dist), x)
    end

    return lp
end

@testset "Linking" begin
    # Just making sure the transformations are okay.
    x = randn(3, 3)
    f = TrilToVec((3, 3))
    f_inv = inverse(f)
    y = f(x)
    @test y isa AbstractVector
    @test f_inv(f(x)) == LowerTriangular(x)

    # Within a model.
    dist = MyMatrixDistribution(3)
    @model demo() = m ~ dist
    model = demo()

    vis = DynamicPPL.TestUtils.setup_varinfos(model, rand(model), (@varname(m),))
    @testset "$(short_varinfo_name(vi))" for vi in vis
        # Evaluate once to ensure we have `logp` value.
        vi = last(DynamicPPL.evaluate!!(model, vi, DefaultContext()))
        vi_linked = DynamicPPL.link!!(deepcopy(vi), model)
        # Difference should just be the log-absdet-jacobian "correction".
        @test DynamicPPL.getlogp(vi) - DynamicPPL.getlogp(vi_linked) ≈ log(2)
        @test vi_linked[@varname(m), dist] == LowerTriangular(vi[@varname(m), dist])
        # Linked one should be working with a lower-dimensional representation.
        @test length(vi_linked[:]) < length(vi[:])
        @test length(vi_linked[:]) == 3
    end
end
