TESTED_ADTYPES = [
    ADTypes.AutoReverseDiff(; compile=false),
    ADTypes.AutoReverseDiff(; compile=true),
    ADTypes.AutoMooncake(; config=nothing),
]

@testset "AD correctness" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        f = DynamicPPL.LogDensityFunction(m)
        rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
        vns = DynamicPPL.TestUtils.varnames(m)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # convert to `Vector{Float64}` to avoid `ReverseDiff` initializing the gradients to Integer 0
            # reference: https://github.com/TuringLang/DynamicPPL.jl/pull/571#issuecomment-1924304489
            params = convert(Vector{Float64}, varinfo[:])
            # Use ForwardDiff as reference AD backend
            ref_logp, ref_grad = DynamicPPL.TestUtils.AD.ad_ldp(
                m, params, ADTypes.AutoForwardDiff(), varinfo
            )

            @testset "$adtype" for adtype in TESTED_ADTYPES
                @info "Testing AD for $(m.f) - $(short_varinfo_name(varinfo)) - $adtype"
                logp, grad = DynamicPPL.TestUtils.AD.ad_ldp(m, params, adtype, varinfo)
                @test logp ≈ ref_logp
                @test grad ≈ ref_grad
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
                x[t] = x[t-1] + η[t-1] * τ
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
        @test LogDensityProblemsAD.ADgradient(AutoReverseDiff(; compile=true), ldf) isa Any
    end
end
