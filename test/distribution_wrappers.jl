@testset "distribution_wrappers.jl" begin
    @testset "univariate" begin
        d = Normal()
        nd = DynamicPPL.NoDist(d)

        # Smoke test
        rand(nd)

        # Actual tests
        @test minimum(nd) == -Inf
        @test maximum(nd) == Inf
        @test logpdf(nd, 15.0) == 0
        @test Bijectors.logpdf_with_trans(nd, 30, true) == 0
        @test Bijectors.bijector(nd) == Bijectors.bijector(d)
    end

    @testset "multivariate" begin
        d = Product([Normal(), Uniform()])
        nd = DynamicPPL.NoDist(d)

        # Smoke test
        @test length(rand(nd)) == 2

        # Actual tests
        @test length(nd) == 2
        @test size(nd) == (2,)
        @test minimum(nd) == [-Inf, 0.0]
        @test maximum(nd) == [Inf, 1.0]
        @test logpdf(nd, [15.0, 0.5]) == 0
        @test Bijectors.logpdf_with_trans(nd, [0, 1]) == 0
        @test Bijectors.bijector(nd) == Bijectors.bijector(d)
    end
end
