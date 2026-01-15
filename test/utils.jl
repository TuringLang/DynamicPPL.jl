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
    end

    @testset "transformations" begin
        function test_transformation(
            dist::Distribution; test_bijector_type_stability::Bool=true
        )
            unlinked = rand(dist)
            unlinked_vec = DynamicPPL.tovec(unlinked)
            @test unlinked_vec isa AbstractVector

            from_vec_trfm = DynamicPPL.from_vec_transform(dist)
            unlinked_again, logjac = Bijectors.with_logabsdet_jacobian(
                from_vec_trfm, unlinked_vec
            )
            @test isapprox_nested(unlinked, unlinked_again)
            @test iszero(logjac)
            # Type stability
            @inferred DynamicPPL.from_vec_transform(dist)
            @inferred Bijectors.with_logabsdet_jacobian(from_vec_trfm, unlinked_vec)

            # Typically the same as `bijector(dist)`, but technically a different
            # function
            b = DynamicPPL.link_transform(dist)
            @test (b(unlinked); true)
            linked, logjac = Bijectors.with_logabsdet_jacobian(b, unlinked)
            @test logjac isa Real

            binv = DynamicPPL.invlink_transform(dist)
            unlinked_again, logjac_inv = Bijectors.with_logabsdet_jacobian(binv, linked)
            @test isapprox_nested(unlinked, unlinked_again)
            @test isapprox(logjac, -logjac_inv)

            linked_vec = DynamicPPL.tovec(linked)
            @test linked_vec isa AbstractVector
            from_linked_vec_trfm = DynamicPPL.from_linked_vec_transform(dist)
            unlinked_again_again = from_linked_vec_trfm(linked_vec)
            @test isapprox_nested(unlinked, unlinked_again_again)

            # Sometimes the bijector itself is not type stable. In this case there is not
            # much we can do in DynamicPPL except skip these tests (it has to be fixed
            # upstream in Bijectors.)
            if test_bijector_type_stability
                @inferred DynamicPPL.from_linked_vec_transform(dist)
                @inferred Bijectors.with_logabsdet_jacobian(
                    from_linked_vec_trfm, linked_vec
                )
            end

            # Create a model and check that we can evaluate it with both unlinked and linked
            # VarInfo. This relies on the transformations working correctly so is more of an
            # 'end to end' test
            @model test() = x ~ dist
            model = test()
            vi_unlinked = VarInfo(model)
            vi_linked = DynamicPPL.link!!(VarInfo(model), model)
            @test (DynamicPPL.evaluate!!(model, vi_unlinked); true)
            @test (DynamicPPL.evaluate!!(model, vi_linked); true)
            model_init = DynamicPPL.setleafcontext(model, DynamicPPL.InitContext())
            @test (DynamicPPL.evaluate!!(model_init, vi_unlinked); true)
            @test (DynamicPPL.evaluate!!(model_init, vi_linked); true)
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

    @testset "tovec" begin
        dist = LKJCholesky(2, 1)
        x = rand(dist)
        @test DynamicPPL.tovec(x) == vec(x.UL)

        nt = (a=[1, 2], b=3.0)
        @test DynamicPPL.tovec(nt) == [1, 2, 3.0]

        t = (2.0, [3.0, 4.0])
        @test DynamicPPL.tovec(t) == [2.0, 3.0, 4.0]
    end
end

end
