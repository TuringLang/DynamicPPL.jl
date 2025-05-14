
@testset "bijector.jl" begin
    @testset "bijector" begin
        @model function test()
            m ~ Normal()
            s ~ InverseGamma(3, 3)
            return c ~ Dirichlet([1.0, 1.0])
        end

        m = test()
        b = bijector(m)

        # m ∈ ℝ, s ∈ ℝ+, c ∈ 2-simplex 
        # check dimensionalities and ranges
        @test b.length_in == 4
        @test b.length_out == 3
        @test b.ranges_in == [1:1, 2:2, 3:4]
        @test b.ranges_out == [1:1, 2:2, 3:3]
        @test b.ranges_out == [1:1, 2:2, 3:3]

        # check support of mapped variables
        binv = inverse(b)
        zs = mapslices(binv, randn(b.length_out, 10000); dims=1)

        @test all(zs[2, :] .≥ 0)
        @test all(sum(zs[3:4, :]; dims=1) .≈ 1.0)
    end
end
