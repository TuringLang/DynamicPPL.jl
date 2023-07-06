@testset "sampler.jl" begin
    @testset "SampleFromPrior and SampleUniform" begin
        @model function gdemo(x, y)
            s ~ InverseGamma(2, 3)
            m ~ Normal(2.0, sqrt(s))
            x ~ Normal(m, sqrt(s))
            return y ~ Normal(m, sqrt(s))
        end

        model = gdemo(1.0, 2.0)
        N = 1_000

        chains = sample(model, SampleFromPrior(), N; progress=false)
        @test chains isa Vector{<:VarInfo}
        @test length(chains) == N

        # Expected value of ``X`` where ``X ~ N(2, ...)`` is 2.
        @test mean(vi[@varname(m)] for vi in chains) ≈ 2 atol = 0.15

        # Expected value of ``X`` where ``X ~ IG(2, 3)`` is 3.
        @test mean(vi[@varname(s)] for vi in chains) ≈ 3 atol = 0.2

        chains = sample(model, SampleFromUniform(), N; progress=false)
        @test chains isa Vector{<:VarInfo}
        @test length(chains) == N

        # `m` is Gaussian, i.e. no transformation is used, so it
        # should have a mean equal to its prior, i.e. 2.
        @test mean(vi[@varname(m)] for vi in chains) ≈ 2 atol = 0.1

        # Expected value of ``exp(X)`` where ``X ~ U[-2, 2]`` is ≈ 1.8.
        @test mean(vi[@varname(s)] for vi in chains) ≈ 1.8 atol = 0.1
    end

    @testset "init" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
                N = 1000
                chain_init = sample(model, SampleFromUniform(), N; progress=false)

                for vn in keys(first(chain_init))
                    if AbstractPPL.subsumes(@varname(s), vn)
                        # `s ~ InverseGamma(2, 3)` and its unconstrained value will be sampled from Unif[-2,2].
                        dist = InverseGamma(2, 3)
                        b = DynamicPPL.link_transform(dist)
                        @test mean(mean(b(vi[vn])) for vi in chain_init) ≈ 0 atol = 0.11
                    elseif AbstractPPL.subsumes(@varname(m), vn)
                        # `m ~ Normal(0, sqrt(s))` and its constrained value is the same.
                        @test mean(mean(vi[vn]) for vi in chain_init) ≈ 0 atol = 0.11
                    else
                        error("Unknown variable name: $vn")
                    end
                end
            end
        end
    end

    @testset "Initial parameters" begin
        # dummy algorithm that just returns initial value and does not perform any sampling
        abstract type OnlyInitAlg end
        struct OnlyInitAlgDefault <: OnlyInitAlg end
        struct OnlyInitAlgUniform <: OnlyInitAlg end
        function DynamicPPL.initialstep(
            rng::Random.AbstractRNG,
            model::Model,
            ::Sampler{<:OnlyInitAlg},
            vi::AbstractVarInfo;
            kwargs...,
        )
            return vi, nothing
        end
        DynamicPPL.getspace(::Sampler{<:OnlyInitAlg}) = ()

        # initial samplers
        DynamicPPL.initialsampler(::Sampler{OnlyInitAlgUniform}) = SampleFromUniform()
        @test DynamicPPL.initialsampler(Sampler(OnlyInitAlgDefault())) == SampleFromPrior()

        for alg in (OnlyInitAlgDefault(), OnlyInitAlgUniform())
            # model with one variable: initialization p = 0.2
            @model function coinflip()
                p ~ Beta(1, 1)
                return 10 ~ Binomial(25, p)
            end
            model = coinflip()
            sampler = Sampler(alg)
            lptrue = logpdf(Binomial(25, 0.2), 10)
            chain = sample(model, sampler, 1; init_params=0.2, progress=false)
            @test chain[1].metadata.p.vals == [0.2]
            @test getlogp(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                init_params=fill(0.2, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.p.vals == [0.2]
                @test getlogp(c[1]) == lptrue
            end

            # model with two variables: initialization s = 4, m = -1
            @model function twovars()
                s ~ InverseGamma(2, 3)
                return m ~ Normal(0, sqrt(s))
            end
            model = twovars()
            lptrue = logpdf(InverseGamma(2, 3), 4) + logpdf(Normal(0, 2), -1)
            chain = sample(model, sampler, 1; init_params=[4, -1], progress=false)
            @test chain[1].metadata.s.vals == [4]
            @test chain[1].metadata.m.vals == [-1]
            @test getlogp(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                init_params=fill([4, -1], 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.s.vals == [4]
                @test c[1].metadata.m.vals == [-1]
                @test getlogp(c[1]) == lptrue
            end

            # set only m = -1
            chain = sample(model, sampler, 1; init_params=[missing, -1], progress=false)
            @test !ismissing(chain[1].metadata.s.vals[1])
            @test chain[1].metadata.m.vals == [-1]

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                init_params=fill([missing, -1], 10),
                progress=false,
            )
            for c in chains
                @test !ismissing(c[1].metadata.s.vals[1])
                @test c[1].metadata.m.vals == [-1]
            end

            # specify `init_params=nothing`
            Random.seed!(1234)
            chain1 = sample(model, sampler, 1; progress=false)
            Random.seed!(1234)
            chain2 = sample(model, sampler, 1; init_params=nothing, progress=false)
            @test chain1[1].metadata.m.vals == chain2[1].metadata.m.vals
            @test chain1[1].metadata.s.vals == chain2[1].metadata.s.vals

            # parallel sampling
            Random.seed!(1234)
            chains1 = sample(model, sampler, MCMCThreads(), 1, 10; progress=false)
            Random.seed!(1234)
            chains2 = sample(
                model, sampler, MCMCThreads(), 1, 10; init_params=nothing, progress=false
            )
            for (c1, c2) in zip(chains1, chains2)
                @test c1[1].metadata.m.vals == c2[1].metadata.m.vals
                @test c1[1].metadata.s.vals == c2[1].metadata.s.vals
            end
        end
    end
end
