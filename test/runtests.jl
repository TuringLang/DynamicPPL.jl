using Distributions
using DynamicPPL

Random.seed!(100)

@testset verbose = true "submodel tests" begin
    @test 1 == 1
end
