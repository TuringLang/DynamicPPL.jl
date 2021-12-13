@testset "model.jl" begin
    @testset "convenience functions" begin
        model = gdemo_default

        # sample from model and extract variables
        vi = VarInfo(model)
        s = vi[@varname(s)]
        m = vi[@varname(m)]

        # extract log pdf of variable object
        lp = getlogp(vi)

        # log prior probability
        lprior = logprior(model, vi)
        @test lprior ≈ logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)

        # log likelihood
        llikelihood = loglikelihood(model, vi)
        @test llikelihood ≈ loglikelihood(Normal(m, sqrt(s)), [1.5, 2.0])

        # log joint probability
        ljoint = logjoint(model, vi)
        @test ljoint ≈ lprior + llikelihood
        @test ljoint ≈ lp
    end

    @testset "rng" begin
        model = gdemo_default

        for sampler in (SampleFromPrior(), SampleFromUniform())
            for i in 1:10
                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.GLOBAL_RNG, vi, sampler)
                vals = DynamicPPL.getall(vi)

                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.GLOBAL_RNG, vi, sampler)
                @test DynamicPPL.getall(vi) == vals
            end
        end
    end

    @testset "defaults without VarInfo, Sampler, and Context" begin
        model = gdemo_default

        Random.seed!(100)
        s, m = model()

        Random.seed!(100)
        @test model(Random.GLOBAL_RNG) == (s, m)
    end

    @testset "nameof" begin
        @model function test1(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end
        @model test2(x) = begin
            m ~ Normal(0, 1)
            x ~ Normal(m, 1)
        end

        @test nameof(test1(rand())) == :test1
        @test nameof(test2(rand())) == :test2
    end

    @testset "Internal methods" begin
        model = gdemo_default

        # sample from model and extract variables
        vi = VarInfo(model)

        # Second component of return-value of `evaluate!!` should
        # be a `DynamicPPL.AbstractVarInfo`.
        evaluate_retval = DynamicPPL.evaluate!!(model, vi, DefaultContext())
        @test evaluate_retval[2] isa DynamicPPL.AbstractVarInfo

        # Should not return `AbstractVarInfo` when we call the model.
        call_retval = model()
        @test !any(map(x -> x isa DynamicPPL.AbstractVarInfo, call_retval))
    end
end
