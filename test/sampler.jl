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
        chains_svi = sample(
            model, SampleFromPrior(), N; progress=false, tracetype=SimpleVarInfo
        )
        @test chains isa Vector{<:VarInfo}
        @test length(chains) == N
        @test chains_svi isa Vector{<:SimpleVarInfo}
        @test length(chains_svi) == N

        # Expected value of ``X`` where ``X ~ N(2, ...)`` is 2.
        @test mean(vi[@varname(m)] for vi in chains) ≈ 2 atol = 0.15
        @test mean(vi[@varname(m)] for vi in chains_svi) ≈ 2 atol = 0.15

        # Expected value of ``X`` where ``X ~ IG(2, 3)`` is 3.
        @test mean(vi[@varname(s)] for vi in chains) ≈ 3 atol = 0.2
        @test mean(vi[@varname(s)] for vi in chains_svi) ≈ 3 atol = 0.2

        chains = sample(model, SampleFromUniform(), N; progress=false)
        chains_svi = sample(
            model, SampleFromUniform(), N; progress=false, tracetype=SimpleVarInfo
        )
        @test chains isa Vector{<:VarInfo}
        @test length(chains) == N
        @test chains_svi isa Vector{<:SimpleVarInfo}
        @test length(chains_svi) == N

        # `m` is Gaussian, i.e. no transformation is used, so it
        # should have a mean equal to its prior, i.e. 2.
        @test mean(vi[@varname(m)] for vi in chains) ≈ 2 atol = 0.1
        @test mean(vi[@varname(m)] for vi in chains_svi) ≈ 2 atol = 0.1

        # Expected value of ``exp(X)`` where ``X ~ U[-2, 2]`` is ≈ 1.8.
        @test mean(vi[@varname(s)] for vi in chains) ≈ 1.8 atol = 0.1
        @test mean(vi[@varname(s)] for vi in chains_svi) ≈ 1.8 atol = 0.1
    end

    @testset "init" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
                N = 1000
                chain_init = sample(model, SampleFromUniform(), N; progress=false)
                chain_init_svi = sample(
                    model, SampleFromUniform(), N; progress=false, tracetype=SimpleVarInfo
                )

                for chain in (chain_init, chain_init_svi)
                    for vn in keys(first(chain))
                        if AbstractPPL.subsumes(@varname(s), vn)
                            # `s ~ InverseGamma(2, 3)` and its unconstrained value will be sampled from Unif[-2,2].
                            dist = InverseGamma(2, 3)
                            b = DynamicPPL.link_transform(dist)
                            @test mean(mean(b(vi[vn])) for vi in chain) ≈ 0 atol = 0.11
                        elseif AbstractPPL.subsumes(@varname(m), vn)
                            # `m ~ Normal(0, sqrt(s))` and its constrained value is the same.
                            @test mean(mean(vi[vn]) for vi in chain) ≈ 0 atol = 0.11
                        else
                            error("Unknown variable name: $vn")
                        end
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
            chain = sample(model, sampler, 1; initial_params=0.2, progress=false)
            chain_svi = sample(
                model,
                sampler,
                1;
                initial_params=0.2,
                progress=false,
                tracetype=SimpleVarInfo,
            )
            @test chain[1].metadata.p.vals == [0.2]
            @test getlogp(chain[1]) == lptrue
            @test chain_svi[1][@varname(p)] == 0.2
            @test getlogp(chain_svi[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(0.2, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.p.vals == [0.2]
                @test getlogp(c[1]) == lptrue
            end

            chains_svi = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(0.2, 10),
                progress=false,
                tracetype=SimpleVarInfo,
            )
            for c in chains_svi
                @test c[1][@varname(p)] == 0.2
                @test getlogp(c[1]) == lptrue
            end

            # model with two variables: initialization s = 4, m = -1
            @model function twovars()
                s ~ InverseGamma(2, 3)
                return m ~ Normal(0, sqrt(s))
            end
            model = twovars()
            lptrue = logpdf(InverseGamma(2, 3), 4) + logpdf(Normal(0, 2), -1)
            chain = sample(model, sampler, 1; initial_params=[4, -1], progress=false)
            @test chain[1].metadata.s.vals == [4]
            @test chain[1].metadata.m.vals == [-1]
            @test getlogp(chain[1]) == lptrue
            chain_svi = sample(model, sampler, 1; initial_params=[4, -1], progress=false, tracetype=SimpleVarInfo)
            @test chain_svi[1][@varname(s)] == 4
            @test chain_svi[1][@varname(m)] == -1
            @test getlogp(chain_svi[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill([4, -1], 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.s.vals == [4]
                @test c[1].metadata.m.vals == [-1]
                @test getlogp(c[1]) == lptrue
            end

            chains_svi = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill([4, -1], 10),
                progress=false,
                tracetype=SimpleVarInfo,
            )
            for c in chains_svi
                @test c[1][@varname(s)] == 4
                @test c[1][@varname(m)] == -1
                @test getlogp(c[1]) == lptrue
            end

            # set only m = -1
            chain = sample(model, sampler, 1; initial_params=[missing, -1], progress=false)
            @test !ismissing(chain[1].metadata.s.vals[1])
            @test chain[1].metadata.m.vals == [-1]
            chain_svi = sample(
                model,
                sampler,
                1;
                initial_params=[missing, -1],
                progress=false,
                tracetype=SimpleVarInfo,
            )
            @test !ismissing(chain_svi[1][@varname(s)])
            @test chain_svi[1][@varname(m)] == -1

            # parallel sampling
            chains = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill([missing, -1], 10),
                progress=false,
            )
            for c in chains
                @test !ismissing(c[1].metadata.s.vals[1])
                @test c[1].metadata.m.vals == [-1]
            end
            chains_svi = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=fill([missing, -1], 10),
                progress=false,
                tracetype=SimpleVarInfo,
            )
            for c in chains_svi
                @test !ismissing(c[1][@varname(s)])
                @test c[1][@varname(m)] == -1
            end

            # specify `initial_params=nothing`
            Random.seed!(1234)
            chain1 = sample(model, sampler, 1; progress=false)
            chain1_svi = sample(model, sampler, 1; progress=false, tracetype=SimpleVarInfo)
            Random.seed!(1234)
            chain2 = sample(model, sampler, 1; initial_params=nothing, progress=false)
            chain2_svi = sample(
                model,
                sampler,
                1;
                initial_params=nothing,
                progress=false,
                tracetype=SimpleVarInfo,
            )
            @test chain1[1].metadata.m.vals == chain2[1].metadata.m.vals
            @test chain1[1].metadata.s.vals == chain2[1].metadata.s.vals
            @test chain1_svi[1][@varname(m)] == chain2_svi[1][@varname(m)]
            @test chain1_svi[1][@varname(s)] == chain2_svi[1][@varname(s)]

            # parallel sampling
            Random.seed!(1234)
            chains1 = sample(model, sampler, MCMCThreads(), 1, 10; progress=false)
            chains1_svi = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                progress=false,
                tracetype=SimpleVarInfo,
            )
            Random.seed!(1234)
            chains2 = sample(
                model, sampler, MCMCThreads(), 1, 10; initial_params=nothing, progress=false
            )
            chains2_svi = sample(
                model,
                sampler,
                MCMCThreads(),
                1,
                10;
                initial_params=nothing,
                progress=false,
                tracetype=SimpleVarInfo,
            )
            for (c1, c2) in zip(chains1, chains2)
                @test c1[1].metadata.m.vals == c2[1].metadata.m.vals
                @test c1[1].metadata.s.vals == c2[1].metadata.s.vals
            end
            for (c1, c2) in zip(chains1_svi, chains2_svi)
                @test c1[1][@varname(m)] == c2[1][@varname(m)]
                @test c1[1][@varname(s)] == c2[1][@varname(s)]
            end
        end
    end
end
