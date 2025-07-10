@testset "sampler.jl" begin
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

        # initial samplers
        DynamicPPL.init_strategy(::Sampler{OnlyInitAlgUniform}) = UniformInit()
        @test DynamicPPL.init_strategy(Sampler(OnlyInitAlgDefault())) == PriorInit()

        for alg in (OnlyInitAlgDefault(), OnlyInitAlgUniform())
            # model with one variable: initialization p = 0.2
            @model function coinflip()
                p ~ Beta(1, 1)
                return 10 ~ Binomial(25, p)
            end
            model = coinflip()
            sampler = Sampler(alg)
            lptrue = logpdf(Binomial(25, 0.2), 10)
            let inits = ParamsInit((; p=0.2))
                chain = sample(model, sampler, 1; initial_params=inits, progress=false)
                @test chain[1].metadata.p.vals == [0.2]
                @test getlogjoint(chain[1]) == lptrue

                # parallel sampling
                chains = sample(
                    model,
                    sampler,
                    MCMCThreads(),
                    1,
                    10;
                    initial_params=fill(inits, 10),
                    progress=false,
                )
                for c in chains
                    @test c[1].metadata.p.vals == [0.2]
                    @test getlogjoint(c[1]) == lptrue
                end
            end

            # model with two variables: initialization s = 4, m = -1
            @model function twovars()
                s ~ InverseGamma(2, 3)
                return m ~ Normal(0, sqrt(s))
            end
            model = twovars()
            lptrue = logpdf(InverseGamma(2, 3), 4) + logpdf(Normal(0, 2), -1)
            let inits = ParamsInit((; s=4, m=-1))
                chain = sample(model, sampler, 1; initial_params=inits, progress=false)
                @test chain[1].metadata.s.vals == [4]
                @test chain[1].metadata.m.vals == [-1]
                @test getlogjoint(chain[1]) == lptrue

                # parallel sampling
                chains = sample(
                    model,
                    sampler,
                    MCMCThreads(),
                    1,
                    10;
                    initial_params=fill(inits, 10),
                    progress=false,
                )
                for c in chains
                    @test c[1].metadata.s.vals == [4]
                    @test c[1].metadata.m.vals == [-1]
                    @test getlogjoint(c[1]) == lptrue
                end
            end

            # set only m = -1
            for inits in (ParamsInit((; s=missing, m=-1)), ParamsInit((; m=-1)))
                chain = sample(model, sampler, 1; initial_params=inits, progress=false)
                @test !ismissing(chain[1].metadata.s.vals[1])
                @test chain[1].metadata.m.vals == [-1]

                # parallel sampling
                chains = sample(
                    model,
                    sampler,
                    MCMCThreads(),
                    1,
                    10;
                    initial_params=fill(inits, 10),
                    progress=false,
                )
                for c in chains
                    @test !ismissing(c[1].metadata.s.vals[1])
                    @test c[1].metadata.m.vals == [-1]
                end
            end
        end
    end
end
