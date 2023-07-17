# some functors (#367)
struct MyModel
    a::Int
end
@model function (f::MyModel)(x)
    m ~ Normal(f.a, 1)
    return x ~ Normal(m, 1)
end
struct MyZeroModel end
@model function (::MyZeroModel)(x)
    m ~ Normal(0, 1)
    return x ~ Normal(m, 1)
end

innermost_distribution_type(d::Distribution) = typeof(d)
function innermost_distribution_type(d::Distributions.ReshapedDistribution)
    return innermost_distribution_type(d.dist)
end
function innermost_distribution_type(d::Distributions.Product)
    dists = map(innermost_distribution_type, d.v)
    if any(!=(dists[1]), dists)
        error("Cannot extract innermost distribution type from $d")
    end

    return dists[1]
end

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
                model(Random.default_rng(), vi, sampler)
                vals = DynamicPPL.getall(vi)

                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.default_rng(), vi, sampler)
                @test DynamicPPL.getall(vi) == vals
            end
        end
    end

    @testset "defaults without VarInfo, Sampler, and Context" begin
        model = gdemo_default

        Random.seed!(100)
        s, m = model()

        Random.seed!(100)
        @test model(Random.default_rng()) == (s, m)
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
        function test3 end
        @model function (::typeof(test3))(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end
        function test4 end
        @model function (a::typeof(test4))(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end

        @test nameof(test1(rand())) == :test1
        @test nameof(test2(rand())) == :test2
        @test nameof(test3(rand())) == :test3
        @test nameof(test4(rand())) == :test4

        # callables
        @test nameof(MyModel(3)(rand())) == Symbol("MyModel(3)")
        @test nameof(MyZeroModel()(rand())) == Symbol("MyZeroModel()")
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

    @testset "Dynamic constraints" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        vi = VarInfo(model)
        spl = SampleFromPrior()
        link!!(vi, spl, model)

        for i in 1:10
            # Sample with large variations.
            r_raw = randn(length(vi[spl])) * 10
            vi[spl] = r_raw
            @test vi[@varname(m)] == r_raw[1]
            @test vi[@varname(x)] != r_raw[2]
            model(vi)
        end
    end

    @testset "rand" begin
        model = gdemo_default

        Random.seed!(1776)
        s, m = model()
        sample_namedtuple = (; s=s, m=m)
        sample_dict = Dict(@varname(s) => s, @varname(m) => m)

        # With explicit RNG
        @test rand(Random.seed!(1776), model) == sample_namedtuple
        @test rand(Random.seed!(1776), NamedTuple, model) == sample_namedtuple
        @test rand(Random.seed!(1776), Dict, model) == sample_dict

        # Without explicit RNG
        Random.seed!(1776)
        @test rand(model) == sample_namedtuple
        Random.seed!(1776)
        @test rand(NamedTuple, model) == sample_namedtuple
        Random.seed!(1776)
        @test rand(Dict, model) == sample_dict
    end

    @testset "default arguments" begin
        @model test_defaults(x, n=length(x)) = x ~ MvNormal(zeros(n), I)
        @test length(test_defaults(missing, 2)()) == 2
    end

    @testset "extract priors" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            priors = extract_priors(model)

            # We know that any variable starting with `s` should have `InverseGamma`
            # and any variable starting with `m` should have `Normal`.
            for (vn, prior) in priors
                if DynamicPPL.getsym(vn) == :s
                    @test innermost_distribution_type(prior) <: InverseGamma
                elseif DynamicPPL.getsym(vn) == :m
                    @test innermost_distribution_type(prior) <: Union{Normal,MvNormal}
                else
                    error("Unexpected variable name: $vn")
                end
            end
        end
    end

    @testset "TestUtils" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            x = rand(model)
            # Ensure log-probability computations are implemented.
            @test logprior(model, x) ≈ DynamicPPL.TestUtils.logprior_true(model, x...)
            @test loglikelihood(model, x) ≈ DynamicPPL.TestUtils.loglikelihood_true(model, x...)
            @test logjoint(model, x) ≈ DynamicPPL.TestUtils.logjoint_true(model, x...)
            @test logjoint(model, x) != DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(model, x...)
            # Ensure `varnames` is implemented.
            vi = last(DynamicPPL.evaluate!!(model, SimpleVarInfo(OrderedDict()), SamplingContext()))
            @test all(collect(keys(vi)) .== DynamicPPL.TestUtils.varnames(model))
            # Ensure `posterior_mean` is implemented.
            @test DynamicPPL.TestUtils.posterior_mean(model) isa typeof(x)
        end
    end
end
