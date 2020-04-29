using DynamicPPL
using Distributions

@testset "distribution_wrappers.jl" begin
    d = Normal()
    nd = DynamicPPL.NoDist(d)

    # Smoke test
    rand(nd)

    # Actual tests
    @test minimum(nd) == -Inf
    @test maximum(nd) == Inf
    @test logpdf(ndf, 15.0) == 0
    @test logpdf_with_trans(nd, 0)
end