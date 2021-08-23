using DynamicPPL.Experimental

@testset "Experimental" begin
    @testset "evaluatortype" begin
        f(x) = false

        @model demo() = x ~ Normal()
        f(::modeltype(demo)) = true
        @test f(demo())

        # Leads to re-definition of `demo` with new body.
        @model demo() = x ~ Normal()
        @test !f(demo())

        # Ensure we can specialize on number of arguments.
        @model demo(x) = x ~ Normal()
        f(::modeltype(demo, 1)) = true
        @test f(demo(1.0))
        @test !f(demo()) # should still be `false`

        # Set it to `true` again.
        f(::modeltype(demo)) = true
        @test f(demo())
    end
end
