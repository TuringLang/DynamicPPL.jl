module DynamicPPLPobserveMacroTests

using DynamicPPL, Distributions, Test

@testset verbose = true "pobserve_macro.jl" begin
    @testset "loglikelihood is correctly accumulated" begin
        @model function f(x)
            @pobserve for i in eachindex(x)
                x[i] ~ Normal()
            end
        end
        x = randn(3)
        expected_loglike = loglikelihood(Normal(), x)
        vi = VarInfo(f(x))
        @test isapprox(DynamicPPL.getloglikelihood(vi), expected_loglike)
    end

    @testset "doesn't error when varinfo has no likelihood acc" begin
        @model function f(x)
            @pobserve for i in eachindex(x)
                x[i] ~ Normal()
            end
        end
        x = randn(3)
        vi = VarInfo()
        vi = DynamicPPL.setaccs!!(vi, (DynamicPPL.LogPriorAccumulator(),))
        @test DynamicPPL.evaluate!!(f(x), vi) isa Any
    end

    @testset "return values are correct" begin
        @testset "single expression at the end" begin
            @model function f(x)
                xplusone = @pobserve for i in eachindex(x)
                    x[i] ~ Normal()
                    x[i] + 1
                end
                return xplusone
            end
            x = randn(3)
            @test f(x)() == x .+ 1

            @testset "calculations are not repeated" begin
                # Make sure that the final expression inside pobserve is not evaluated
                # multiple times.
                counter = 0
                increment_and_return(y) = (counter += 1; y)
                @model function g(x)
                    xs = @pobserve for i in eachindex(x)
                        x[i] ~ Normal()
                        increment_and_return(x[i])
                    end
                    return xs
                end
                x = randn(3)
                @test g(x)() == x
                @test counter == length(x)
            end
        end

        @testset "tilde expression at the end" begin
            @model function f(x)
                xs = @pobserve for i in eachindex(x)
                    # This should behave as if it returns `x[i]`
                    x[i] ~ Normal()
                end
                return xs
            end
            x = randn(3)
            @test f(x)() == x
        end
    end
end

end
