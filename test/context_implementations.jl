@testset "context_implementations.jl" begin
    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(fill(0, 2), 2.0)
            z = Vector{Int}(undef, length(x))
            # `z .~ Categorical.(ps)` cannot be parsed by Julia 1.0
            (.~)(z, Categorical.(fill([0.5, 0.5], length(x))))
            for i in 1:length(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo(), SampleFromPrior(), LikelihoodContext())
    end

    # https://github.com/TuringLang/DynamicPPL.jl/issues/28#issuecomment-829223577
    @testset "arrays of distributions" begin
        @model function test(x, y)
            return y .~ Normal.(x)
        end

        for ysize in ((2,), (2, 3), (2, 3, 4))
            # drop trailing dimensions
            for xsize in ntuple(i -> ysize[1:i], length(ysize))
                x = randn(xsize)
                y = randn(ysize)
                z = logjoint(test(x, y), VarInfo())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end

            # singleton dimensions
            for xsize in
                ntuple(i -> (ysize[1:(i - 1)]..., 1, ysize[(i + 1):end]...), length(ysize))
                x = randn(xsize)
                y = randn(ysize)
                z = logjoint(test(x, y), VarInfo())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end
        end
    end
end
