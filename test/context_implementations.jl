module DynamicPPLContextImplementationTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra: I, norm
using Test
using Random: Xoshiro

struct ObservationContext <: AbstractContext end

struct ObserveHookDistribution <: ContinuousUnivariateDistribution
    seen::Vector{Union{VarName,Nothing}}
    prefixes::Vector{Any}
end
Distributions.logpdf(::ObserveHookDistribution, x::Real) = logpdf(Normal(), x)
function DynamicPPL.tilde_observe!!(
    prefix, dist::ObserveHookDistribution, left, vn, template, vi
)
    push!(dist.seen, vn)
    push!(dist.prefixes, prefix)
    return tilde_observe!!(prefix, Normal(), left, vn, template, vi)
end

@testset "public observation hook" begin
    @model observations(x, ys, dist) = (x ~ dist; z ~ dist; 0.0 ~ dist; ys .~ dist)
    @model parent(child) = b ~ to_submodel(child)
    seen = Union{VarName,Nothing}[]
    model = condition(
        observations(1.0, [2.0, 3.0], ObserveHookDistribution(seen, Any[])); z=4.0
    )
    templated = prefix(model, @varname(a[2, 3]); template=zeros(2, 3))
    nested = parent(condition(templated, conditioned(templated)))
    for m in (model, prefix(model, @varname(p)), templated, nested)
        empty!(seen)
        prefixes = model.args.dist.prefixes
        empty!(prefixes)
        acc = DynamicPPL.VNTAccumulator{DynamicPPL.POINTWISE_ACCNAME}(
            DynamicPPL.PointwiseLogProb{false,true}()
        )
        _, vi = DynamicPPL.evaluate_nowarn!!(
            m, VarInfo((DynamicPPL.default_accumulators()..., acc))
        )
        @test seen == [@varname(x), @varname(z), nothing, @varname(ys[1]), @varname(ys[2])]
        if m === nested
            @test all(p -> p isa DynamicPPL.PrefixTemplate, prefixes)
            @test all(p -> DynamicPPL._getprefix(p) == @varname(b.a[2, 3]), prefixes)
        else
            @test all(p -> p === m.prefix, prefixes)
        end
        @test getloglikelihood(vi) ≈ sum(logpdf.(Normal(), 0.0:4.0))
        pointwise = DynamicPPL.get_pointwise_logprobs(vi)
        observation_prefix = DynamicPPL._getprefix(first(prefixes))
        for (vn, value) in ((@varname(x), 1.0), (@varname(z), 4.0), (@varname(ys[1]), 2.0))
            name = DynamicPPL.maybe_prefix(vn, observation_prefix)
            @test pointwise[name] ≈ logpdf(Normal(), value)
        end
        if m === templated
            @test size(pointwise.data.a.data) == (2, 3)
        elseif m === nested
            @test size(pointwise.data.b.data.a.data) == (2, 3)
        end
    end
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

    @testset "observations do not dispatch on context" begin
        for T in (Float32, Float64, BigFloat)
            dist = Normal(zero(T), one(T))
            for value in (one(T), T[1, 2])
                for vn in (@varname(x), nothing)
                    result, vi = @inferred tilde_observe!!(
                        nothing, dist, value, vn, NoTemplate(), VarInfo()
                    )
                    @test result === value
                    @test getloglikelihood(vi) ≈ loglikelihood(dist, value)
                    @test iszero(getlogprior(vi))
                end
            end
        end
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
