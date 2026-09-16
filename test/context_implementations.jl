module DynamicPPLContextImplementationTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra: I, norm
using Test

struct ObservationContext <: AbstractContext end
function DynamicPPL.tilde_observe!!(
    ::ObservationContext,
    dist::Distribution,
    value,
    vn::Union{VarName,Nothing},
    template,
    vi::AbstractVarInfo,
)
    return value, DynamicPPL.accumulate_observe!!(vi, dist, value, vn, template)
end

@testset "context_implementations.jl" begin
    @testset "leaf contexts without parameter outputs" begin
        @model function observed_only(x)
            x ~ Normal()
            0.0 ~ Normal()
            return x
        end
        for context in (ObservationContext(), DefaultContext()), threaded in (false, true)
            model = contextualize(setthreadsafe(observed_only(1.0), threaded), context)
            value, vi = DynamicPPL.evaluate_nowarn!!(model, VarInfo())
            @test value == 1.0
            @test getloglikelihood(vi) == logpdf(Normal(), 1.0) + logpdf(Normal(), 0.0)
        end
        @model latent() = x ~ Normal()
        @test_throws "No value was provided" DynamicPPL.evaluate_nowarn!!(
            latent(), VarInfo()
        )
    end

    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(zeros(2), 4 * I)
            z = Vector{Int}(undef, length(x))
            z ~ product_distribution(Categorical.(fill([0.5, 0.5], length(x))))
            for i in eachindex(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo())
    end

    @testset "dot tilde with varying sizes" begin
        @testset "assume" begin
            @model function test(x, size)
                y = Array{Float64,length(size)}(undef, size...)
                y .~ Normal(x)
                return y
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                x = randn()
                model = test(x, ysize)
                y = model()
                lp = logjoint(model, (; y=y))
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
                z = logjoint(test(x, y), VarNamedTuple())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
