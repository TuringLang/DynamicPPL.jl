module DynamicPPLDistributionWrappersTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL: DynamicPPL, @model
using Test: @testset, @test, @test_logs
using Distributions: Normal, Product, logpdf, product_distribution
using Bijectors: Bijectors
using StableRNGs: StableRNG

@testset "distribution_wrappers.jl" begin
    d = Normal()
    nd = DynamicPPL.NoDist(d)

    # Smoke test
    rand(nd)

    # Actual tests
    @test minimum(nd) == -Inf
    @test maximum(nd) == Inf
    @test logpdf(nd, 15.0) == 0
    @test Bijectors.logpdf_with_trans(nd, 30, true) == 0
end

@testset "arraydist on a vector of univariates" begin
    rng = StableRNG(468)
    dists = Normal.(randn(rng, 3))
    x = randn(rng, 3)
    # `Product(v)` calls `Base.depwarn`, which walks a backtrace on every call, so this has
    # to go through the inner constructor instead.
    @test_logs DynamicPPL.arraydist(dists)
    @test DynamicPPL.arraydist(dists) isa Product
    @test logpdf(DynamicPPL.arraydist(dists), x) == logpdf(product_distribution(dists), x)
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
