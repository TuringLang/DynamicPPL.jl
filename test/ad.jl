@testset "AD: ForwardDiff, ReverseDiff, and Mooncake" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
        vns = DynamicPPL.TestUtils.varnames(m)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            f = DynamicPPL.LogDensityFunction(m, varinfo)
            # convert to `Vector{Float64}` to avoid `ReverseDiff` initializing the gradients to Integer 0
            # reference: https://github.com/TuringLang/DynamicPPL.jl/pull/571#issuecomment-1924304489
            θ = convert(Vector{Float64}, varinfo[:])
            # Calculate reference logp + gradient of logp using ForwardDiff
            default_adtype = ADTypes.AutoForwardDiff()
            ref_logp, ref_grad = LogDensityProblems.logdensity_and_gradient(
                f, θ, default_adtype
            )

            @testset "$adtype" for adtype in [
                ADTypes.AutoReverseDiff(; compile=false),
                ADTypes.AutoReverseDiff(; compile=true),
                ADTypes.AutoMooncake(; config=nothing),
            ]
                # Mooncake can't currently handle something that is going on in
                # SimpleVarInfo{<:VarNamedVector}. Disable all SimpleVarInfo tests for now.
                if adtype isa ADTypes.AutoMooncake && varinfo isa DynamicPPL.SimpleVarInfo
                    @test_broken 1 == 0
                else
                    logp, grad = LogDensityProblems.logdensity_and_gradient(f, θ, adtype)
                    @test grad ≈ ref_grad
                    @test logp ≈ ref_logp
                end
            end
        end
    end

    @testset "Turing#2151: ReverseDiff compilation & eltype(vi, spl)" begin
        # Failing model
        t = 1:0.05:8
        σ = 0.3
        y = @. rand(sin(t) + Normal(0, σ))
        @model function state_space(y, TT, ::Type{T}=Float64) where {T}
            # Priors 
            α ~ Normal(y[1], 0.001)
            τ ~ Exponential(1)
            η ~ filldist(Normal(0, 1), TT - 1)
            σ ~ Exponential(1)
            # create latent variable
            x = Vector{T}(undef, TT)
            x[1] = α
            for t in 2:TT
                x[t] = x[t - 1] + η[t - 1] * τ
            end
            # measurement model
            y ~ MvNormal(x, σ^2 * I)
            return x
        end
        model = state_space(y, length(t))

        # Dummy sampling algorithm for testing. The test case can only be replicated
        # with a custom sampler, it doesn't work with SampleFromPrior(). We need to
        # overload assume so that model evaluation doesn't fail due to a lack
        # of implementation
        struct MyEmptyAlg end
        DynamicPPL.getspace(::DynamicPPL.Sampler{MyEmptyAlg}) = ()
        DynamicPPL.assume(rng, ::DynamicPPL.Sampler{MyEmptyAlg}, dist, vn, vi) =
            DynamicPPL.assume(dist, vn, vi)

        # Compiling the ReverseDiff tape used to fail here
        spl = Sampler(MyEmptyAlg())
        vi = VarInfo(model)
        ldf = DynamicPPL.LogDensityFunction(vi, model, SamplingContext(spl))
        @test LogDensityProblems.logdensity_and_gradient(
            ldf, vi[:], AutoReverseDiff(; compile=true)
        ) isa Any
    end
end
