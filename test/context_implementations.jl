@testset "context_implementations.jl" begin
    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(zeros(2), 4 * I)
            z = Vector{Int}(undef, length(x))
            z ~ product_distribution(Categorical.(fill([0.5, 0.5], length(x))))
            for i in 1:length(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo(), SampleFromPrior(), LikelihoodContext())
    end

    @testset "dot tilde with varying sizes" begin
        @testset "assume" begin
            @model function test(x, size)
                y = Array{Float64,length(size)}(undef, size...)
                y .~ Normal(x)
                return y, getlogp(__varinfo__)
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                x = randn()
                model = test(x, ysize)
                y, lp = model()
                @test lp ≈ sum(logpdf.(Normal.(x), y))

                ys = [first(model()) for _ in 1:10_000]
                @test norm(mean(ys) .- x, Inf) < 0.1
                @test norm(std(ys) .- 1, Inf) < 0.1
            end
        end

        @testset "observe" begin
            @model function test(x, y)
                return y .~ Normal(x)
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                x = randn()
                y = randn(ysize)
                z = logjoint(test(x, y), VarInfo())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end
        end
    end
end
