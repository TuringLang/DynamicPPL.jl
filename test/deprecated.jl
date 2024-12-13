@testset "deprecated" begin
    @testset "@submodel" begin
        @testset "is deprecated" begin
            @model inner() = x ~ Normal()
            @model outer() = @submodel x = inner()
            @test_logs(
                (
                    :warn,
                    "`@submodel model` and `@submodel prefix=... model` are deprecated; see `to_submodel` for the up-to-date syntax.",
                ),
                outer()()
            )

            @model outer_with_prefix() = @submodel prefix = "sub" x = inner()
            @test_logs(
                (
                    :warn,
                    "`@submodel model` and `@submodel prefix=... model` are deprecated; see `to_submodel` for the up-to-date syntax.",
                ),
                outer_with_prefix()()
            )
        end

        @testset "prefixing still works correctly" begin
            @model inner() = x ~ Normal()
            @model function outer()
                a = @submodel inner()
                b = @submodel prefix = "sub" inner()
                return a, b
            end
            @test outer()() isa Tuple{Float64,Float64}
            vi = VarInfo(outer())
            @test @varname(x) in keys(vi)
            @test @varname(var"sub.x") in keys(vi)
        end

        @testset "logp is still accumulated properly" begin
            @model inner_assume() = x ~ Normal()
            @model inner_observe(x, y) = y ~ Normal(x)
            @model function outer(b)
                a = @submodel inner_assume()
                @submodel inner_observe(a, b)
            end
            y_val = 1.0
            model = outer(y_val)
            @test model() == y_val

            x_val = 1.5
            vi = VarInfo(outer(y_val))
            DynamicPPL.setindex!!(vi, x_val, @varname(x))
            @test logprior(model, vi) ≈ logpdf(Normal(), x_val)
            @test loglikelihood(model, vi) ≈ logpdf(Normal(x_val), y_val)
            @test logjoint(model, vi) ≈ logpdf(Normal(), x_val) + logpdf(Normal(x_val), y_val)
        end
    end
end
