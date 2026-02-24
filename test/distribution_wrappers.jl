module DynamicPPLDistributionWrappersTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL: DynamicPPL, @model
using Test: @testset, @test
using Distributions: Normal, logpdf
using Bijectors: Bijectors

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

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
