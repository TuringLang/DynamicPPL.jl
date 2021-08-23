using DynamicPPL.Experimental

@testset "Experimental" begin
    @testset "evaluatortype" begin
        @model demo() = x ~ Normal()

        f(::Model{evaluatortype(demo)}) = true
        f(x) = false

        @test f(demo())

        # Leads to re-definition of `demo` with new body.
        @model demo() = x ~ Normal()
        @test !f(demo())

        # Ensure we can specialize on number of arguments.
        @model demo(x) = x ~ Normal()
        @test f(demo(1.0))
    end
end
