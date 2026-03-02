module DynamicPPLUtilsTests

using Bijectors: Bijectors
using Distributions
using DynamicPPL
using LinearAlgebra: LinearAlgebra
using Test

isapprox_nested(a::Number, b::Number; kwargs...) = isapprox(a, b; kwargs...)
isapprox_nested(a::AbstractArray, b::AbstractArray; kwargs...) = isapprox(a, b; kwargs...)
function isapprox_nested(a::LinearAlgebra.Cholesky, b::LinearAlgebra.Cholesky; kwargs...)
    return isapprox(a.U, b.U; kwargs...) && isapprox(a.L, b.L; kwargs...)
end
function isapprox_nested(a::NamedTuple, b::NamedTuple; kwargs...)
    keys(a) == keys(b) || return false
    return all(k -> isapprox_nested(a[k], b[k]; kwargs...), keys(a))
end

@testset "utils.jl" begin
    @testset "addlogprob!" begin
        @model function testmodel()
            global lp_before = getlogjoint(__varinfo__)
            @addlogprob!(42)
            return global lp_after = getlogjoint(__varinfo__)
        end

        varinfo = VarInfo(testmodel())
        @test iszero(lp_before)
        @test getlogjoint(varinfo) == lp_after == 42
        @test getloglikelihood(varinfo) == 42

        @model function testmodel_nt()
            global lp_before = getlogjoint(__varinfo__)
            @addlogprob! (; logprior=(pi + 1), loglikelihood=42)
            return global lp_after = getlogjoint(__varinfo__)
        end

        varinfo = VarInfo(testmodel_nt())
        @test iszero(lp_before)
        @test getlogjoint(varinfo) == lp_after == 42 + 1 + pi
        @test getloglikelihood(varinfo) == 42
        @test getlogprior(varinfo) == pi + 1

        @model function testmodel_nt2()
            global lp_before = getlogjoint(__varinfo__)
            llh_nt = (; loglikelihood=42)
            @addlogprob! llh_nt
            return global lp_after = getlogjoint(__varinfo__)
        end
        varinfo = VarInfo(testmodel_nt2())
        @test iszero(lp_before)
        @test getlogjoint(varinfo) == lp_after == 42
        @test getloglikelihood(varinfo) == 42
        @test iszero(getlogprior(varinfo))
    end

    @testset "transformations" begin
        function test_transformation(dist::Distribution)
            # Create a model and check that we can evaluate it with both unlinked and linked
            # VarInfo. This relies on the transformations working correctly so is more of an
            # 'end to end' test
            @model test() = x ~ dist
            model = test()
            vi_unlinked = VarInfo(model)
            vi_linked = DynamicPPL.link!!(VarInfo(model), model)
            @test (DynamicPPL.evaluate_nowarn!!(model, vi_unlinked); true)
            @test (DynamicPPL.evaluate_nowarn!!(model, vi_linked); true)

            model_init = DynamicPPL.setleafcontext(
                model,
                DynamicPPL.InitContext(DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()),
            )
            @test (DynamicPPL.evaluate_nowarn!!(model_init, vi_unlinked); true)
            model_init = DynamicPPL.setleafcontext(
                model,
                DynamicPPL.InitContext(DynamicPPL.InitFromPrior(), DynamicPPL.LinkAll()),
            )
            @test (DynamicPPL.evaluate_nowarn!!(model_init, vi_linked); true)
        end

        # Unconstrained univariate
        test_transformation(Normal())
        # Constrained univariate
        test_transformation(LogNormal())
        test_transformation(truncated(Normal(); lower=0))
        test_transformation(Exponential(1.0))
        test_transformation(Uniform(-2, 2))
        test_transformation(Beta(2, 2))
        test_transformation(InverseGamma(2, 3))
        # Discrete univariate
        test_transformation(Poisson(3))
        test_transformation(Binomial(10, 0.5))
        # Multivariate
        test_transformation(MvNormal(zeros(3), LinearAlgebra.I))
        test_transformation(
            product_distribution([Normal(), LogNormal()]);
            test_bijector_type_stability=false,
        )
        test_transformation(product_distribution([LogNormal(), LogNormal()]))
        # Matrixvariate
        test_transformation(LKJ(3, 0.5))
        test_transformation(Wishart(7, [1.0 0.0; 0.0 1.0]))
        # This is a pathological example: the linked representation is a matrix
        test_transformation(product_distribution(fill(Dirichlet(ones(4)), 2, 3)))
        # Cholesky
        test_transformation(LKJCholesky(3, 0.5))
        # ProductNamedTupleDistribution
        d = product_distribution((a=Normal(), b=LogNormal()))
        test_transformation(d)
        d_nested = product_distribution((x=LKJCholesky(2, 0.5), y=d))
        test_transformation(d_nested)
    end

    @testset "getargs_dottilde" begin
        # Some things that are not expressions.
        @test DynamicPPL.getargs_dottilde(:x) === nothing
        @test DynamicPPL.getargs_dottilde(1.0) === nothing
        @test DynamicPPL.getargs_dottilde([1.0, 2.0, 4.0]) === nothing

        # Some expressions.
        @test DynamicPPL.getargs_dottilde(:(x ~ Normal(μ, σ))) === nothing
        @test DynamicPPL.getargs_dottilde(:((.~)(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
        @test DynamicPPL.getargs_dottilde(:((~).(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
        @test DynamicPPL.getargs_dottilde(:(x .~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
        @test DynamicPPL.getargs_dottilde(:(@. x ~ Normal(μ, σ))) === nothing
        @test DynamicPPL.getargs_dottilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) ===
            nothing
        @test DynamicPPL.getargs_dottilde(:(@~ Normal.(μ, σ))) === nothing
    end

    @testset "getargs_tilde" begin
        # Some things that are not expressions.
        @test DynamicPPL.getargs_tilde(:x) === nothing
        @test DynamicPPL.getargs_tilde(1.0) === nothing
        @test DynamicPPL.getargs_tilde([1.0, 2.0, 4.0]) === nothing

        # Some expressions.
        @test DynamicPPL.getargs_tilde(:(x ~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
        @test DynamicPPL.getargs_tilde(:((.~)(x, Normal(μ, σ)))) === nothing
        @test DynamicPPL.getargs_tilde(:((~).(x, Normal(μ, σ)))) === nothing
        @test DynamicPPL.getargs_tilde(:(@. x ~ Normal(μ, σ))) === nothing
        @test DynamicPPL.getargs_tilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) ===
            nothing
        @test DynamicPPL.getargs_tilde(:(@~ Normal.(μ, σ))) === nothing
    end
end

end
