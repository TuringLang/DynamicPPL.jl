@testset "distribution_wrappers.jl" begin
    d = Normal()
    nd = DynamicPPL.NoDist(d)

    # Smoke test
    rand(nd)

    # Actual tests
    @test minimum(nd) == -Inf
    @test maximum(nd) == Inf
    @test logpdf(nd, 15.0) == 0
    @test Bijectors.Bijectors.logpdf_with_trans(nd, 30, true) == 0
end
