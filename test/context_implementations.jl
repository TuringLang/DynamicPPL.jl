@testset "context_implementations.jl" begin
    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(zeros(2), 4 * I)
            z = Vector{Int}(undef, length(x))
            z .~ Categorical.(fill([0.5, 0.5], length(x)))
            for i in 1:length(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo(), SampleFromPrior(), LikelihoodContext())
    end

    # https://github.com/TuringLang/DynamicPPL.jl/issues/28#issuecomment-829223577
    @testset "dot tilde: arrays of distributions" begin
        @testset "assume" begin
            @model function test(x, size)
                y = Array{Float64,length(size)}(undef, size...)
                y .~ Normal.(x)
                return y, getlogp(__varinfo__)
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                for x in (
                    # scalar
                    randn(),
                    # drop trailing dimensions
                    ntuple(i -> randn(ysize[1:i]), length(ysize))...,
                    # singleton dimensions
                    ntuple(
                        i -> randn(ysize[1:(i - 1)]..., 1, ysize[(i + 1):end]...),
                        length(ysize),
                    )...,
                )
                    model = test(x, ysize)
                    y, lp = model()
                    @test lp ≈ sum(logpdf.(Normal.(x), y))

                    ys = [first(model()) for _ in 1:10_000]
                    @test norm(mean(ys) .- x, Inf) < 0.1
                    @test norm(std(ys) .- 1, Inf) < 0.1
                end
            end
        end

        @testset "observe" begin
            @model function test(x, y)
                return y .~ Normal.(x)
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                for x in (
                    # scalar
                    randn(),
                    # drop trailing dimensions
                    ntuple(i -> randn(ysize[1:i]), length(ysize))...,
                    # singleton dimensions
                    ntuple(
                        i -> randn(ysize[1:(i - 1)]..., 1, ysize[(i + 1):end]...),
                        length(ysize),
                    )...,
                )
                    y = randn(ysize)
                    z = logjoint(test(x, y), VarInfo())
                    @test z ≈ sum(logpdf.(Normal.(x), y))
                end
            end
        end
    end
end
