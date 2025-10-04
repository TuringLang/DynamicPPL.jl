@testset "sampler.jl" begin
    @testset "varnames with same symbol but different type" begin
        struct S <: AbstractMCMC.AbstractSampler end
        DynamicPPL.initialstep(rng, model, ::DynamicPPL.Sampler{S}, vi; kwargs...) = vi
        @model function g()
            y = (; a=1, b=2)
            y.a ~ Normal()
            return y.b ~ Normal()
        end
        model = g()
        spl = DynamicPPL.Sampler(S())
        @test AbstractMCMC.step(Xoshiro(468), g(), spl) isa Any
    end

    @testset "initial_state" begin
        # Model is unused, but has to be a DynamicPPL.Model otherwise we won't hit our
        # overloaded method.
        @model f() = x ~ Normal()
        model = f()
        # This sampler just returns the state it was given as its 'sample'
        struct S <: AbstractMCMC.AbstractSampler end
        function AbstractMCMC.step(
            rng::Random.AbstractRNG,
            model::Model,
            sampler::Sampler{<:S},
            state=nothing;
            kwargs...,
        )
            if state === nothing
                s = rand()
                return s, s
            else
                return state, state
            end
        end
        spl = Sampler(S())

        function AbstractMCMC.bundle_samples(
            samples::Vector{Float64},
            model::Model,
            sampler::Sampler{<:S},
            state,
            chain_type::Type{MCMCChains.Chains};
            kwargs...,
        )
            return MCMCChains.Chains(samples, [:x]; info=(samplerstate=state,))
        end

        N_iters, N_chains = 10, 3

        @testset "single-chain sampling" begin
            chn = sample(model, spl, N_iters; progress=false, chain_type=MCMCChains.Chains)
            initial_value = chn[:x][1]
            @test all(chn[:x] .== initial_value) # sanity check
            chn2 = sample(
                model,
                spl,
                N_iters;
                progress=false,
                initial_state=DynamicPPL.loadstate(chn),
                chain_type=MCMCChains.Chains,
            )
            @test all(chn2[:x] .== initial_value)
        end

        @testset "multiple-chain sampling" begin
            chn = sample(
                model,
                spl,
                MCMCThreads(),
                N_iters,
                N_chains;
                progress=false,
                chain_type=MCMCChains.Chains,
            )
            initial_value = chn[:x][1, :]
            @test all(i -> chn[:x][i, :] == initial_value, 1:N_iters) # sanity check
            chn2 = sample(
                model,
                spl,
                MCMCThreads(),
                N_iters,
                N_chains;
                progress=false,
                initial_state=DynamicPPL.loadstate(chn),
                chain_type=MCMCChains.Chains,
            )
            @test all(i -> chn2[:x][i, :] == initial_value, 1:N_iters)
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

        # initial samplers
        DynamicPPL.init_strategy(::Sampler{OnlyInitAlgUniform}) = InitFromUniform()
        @test DynamicPPL.init_strategy(Sampler(OnlyInitAlgDefault())) == InitFromPrior()

        for alg in (OnlyInitAlgDefault(), OnlyInitAlgUniform())
            # model with one variable: initialization p = 0.2
            @model function coinflip()
                p ~ Beta(1, 1)
                return 10 ~ Binomial(25, p)
            end
            model = coinflip()
            sampler = Sampler(alg)
            lptrue = logpdf(Binomial(25, 0.2), 10)
            let inits = InitFromParams((; p=0.2))
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
            for inits in (
                InitFromParams((s=4, m=-1)),
                (s=4, m=-1),
                InitFromParams(Dict(@varname(s) => 4, @varname(m) => -1)),
                Dict(@varname(s) => 4, @varname(m) => -1),
            )
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
            for inits in (
                InitFromParams((; s=missing, m=-1)),
                InitFromParams(Dict(@varname(s) => missing, @varname(m) => -1)),
                (; s=missing, m=-1),
                Dict(@varname(s) => missing, @varname(m) => -1),
                InitFromParams((; m=-1)),
                InitFromParams(Dict(@varname(m) => -1)),
                (; m=-1)Dict(@varname(m) => -1),
            )
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
