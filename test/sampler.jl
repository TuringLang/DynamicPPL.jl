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

    @testset "initial_state and resume_from kwargs" begin
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
            # using `initial_state`
            chn2 = sample(
                model,
                spl,
                N_iters;
                progress=false,
                initial_state=chn.info.samplerstate,
                chain_type=MCMCChains.Chains,
            )
            @test all(chn2[:x] .== initial_value)
            # using `resume_from`
            chn3 = sample(
                model,
                spl,
                N_iters;
                progress=false,
                resume_from=chn,
                chain_type=MCMCChains.Chains,
            )
            @test all(chn3[:x] .== initial_value)
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
            # using `initial_state`
            chn2 = sample(
                model,
                spl,
                MCMCThreads(),
                N_iters,
                N_chains;
                progress=false,
                initial_state=chn.info.samplerstate,
                chain_type=MCMCChains.Chains,
            )
            @test all(i -> chn2[:x][i, :] == initial_value, 1:N_iters)
            # using `resume_from`
            chn3 = sample(
                model,
                spl,
                MCMCThreads(),
                N_iters,
                N_chains;
                progress=false,
                resume_from=chn,
                chain_type=MCMCChains.Chains,
            )
            @test all(i -> chn3[:x][i, :] == initial_value, 1:N_iters)
        end
    end

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
        # will be drawn from U[-2, 2] and its mean should be 0.
        @test mean(vi[@varname(m)] for vi in chains) ≈ 0.0 atol = 0.1

        # Expected value of ``exp(X)`` where ``X ~ U[-2, 2]`` is ≈ 1.8.
        @test mean(vi[@varname(s)] for vi in chains) ≈ 1.8 atol = 0.1
    end

    @testset "init" begin
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
            let inits = InitFromParams((; s=4, m=-1))
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
            for inits in (InitFromParams((; s=missing, m=-1)), InitFromParams((; m=-1)))
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
