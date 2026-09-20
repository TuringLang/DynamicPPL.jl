module DynamicPPLSamplingOutputTests
using AbstractMCMC, DynamicPPL, Distributions, Random, Test

@model function model(y)
    x ~ Normal()
    y ~ Normal(x, 1)
    return x + y
end

@testset "SamplingOutput model evaluation" begin
    params = [VarNamedTuple(; x=i + j / 10) for i in 1:2, j in 1:2]
    for draws in (params, map(p -> DynamicPPL.ParamsWithStats(p, (;)), params))
        chain = SamplingOutput(draws; iterations=3:2:5)
        @test returned(model(1), chain) == map(p -> p[@varname(x)] + 1, params)
        for f in (logjoint, logprior, loglikelihood)
            @test f(model(1), chain) ≈ map(p -> f(model(1), p), params)
        end
        predictions = predict(Xoshiro(1), model(missing), chain)
        @test size(predictions) == size(chain)
        @test predictions.iterations == chain.iterations
        @test all(ismissing, predictions.sampler_states)
        @test map(p -> p.params[@varname(x)], predictions.samples) ==
            map(p -> p[@varname(x)], params)
        again = predict(Xoshiro(1), model(missing), chain; include_all=false)
        @test all(p -> !haskey(p.params, @varname(x)), again.samples)
        @test map(p -> p.params[@varname(y)], again.samples) ==
            map(p -> p.params[@varname(y)], predictions.samples)
        @test map(p -> p.stats, again.samples) == map(p -> p.stats, predictions.samples)
        @test predict(model(missing), chain) isa SamplingOutput
        @test_throws ErrorException returned(model(missing), chain)
    end
    @model function indexed_model()
        x = zeros(2)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
    end
    params = DynamicPPL.templated_setindex!!(VarNamedTuple(), 4.0, @varname(x[1]), zeros(2))
    chain = SamplingOutput(fill(params, 1, 1))
    prediction = predict(Xoshiro(1), indexed_model(), chain; include_all=false)[1, 1]
    @test !haskey(prediction.params, @varname(x[1]))
    @test haskey(prediction.params, @varname(x[2]))
end
end
