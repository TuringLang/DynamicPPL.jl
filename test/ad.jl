using DynamicPPL: LogDensityFunction, LogDensityFunctionWithGrad

@testset "AD: ForwardDiff, ReverseDiff, and Mooncake" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
        vns = DynamicPPL.TestUtils.varnames(m)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            f = LogDensityFunction(m, varinfo)
            x = DynamicPPL.getparams(f)
            # Calculate reference logp + gradient of logp using ForwardDiff
            default_adtype = ADTypes.AutoForwardDiff()
            ldf_with_grad = LogDensityFunctionWithGrad(f, default_adtype)
            ref_logp, ref_grad = LogDensityProblems.logdensity_and_gradient(
                ldf_with_grad, x
            )

            @testset "$adtype" for adtype in [
                AutoReverseDiff(; compile=false),
                AutoReverseDiff(; compile=true),
                AutoMooncake(; config=nothing),
            ]
                @info "Testing AD on: $(m.f) - $(short_varinfo_name(varinfo)) - $adtype"

                # Put predicates here to avoid long lines
                is_mooncake = adtype isa AutoMooncake
                is_1_10 = v"1.10" <= VERSION < v"1.11"
                is_1_11 = v"1.11" <= VERSION < v"1.12"
                is_svi_vnv = varinfo isa SimpleVarInfo{<:DynamicPPL.VarNamedVector}
                is_svi_od = varinfo isa SimpleVarInfo{<:OrderedDict}

                # Mooncake doesn't work with several combinations of SimpleVarInfo.
                if is_mooncake && is_1_11 && is_svi_vnv
                    # https://github.com/compintell/Mooncake.jl/issues/470
                    @test_throws ArgumentError LogDensityFunctionWithGrad(f, adtype)
                elseif is_mooncake && is_1_10 && is_svi_vnv
                    # TODO: report upstream
                    @test_throws UndefRefError LogDensityFunctionWithGrad(f, adtype)
                elseif is_mooncake && is_1_10 && is_svi_od
                    # TODO: report upstream
                    @test_throws Mooncake.MooncakeRuleCompilationError LogDensityFunctionWithGrad(
                        f, adtype
                    )
                else
                    ldf_with_grad = LogDensityFunctionWithGrad(f, adtype)
                    logp, grad = LogDensityProblems.logdensity_and_gradient(
                        ldf_with_grad, x
                    )
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
        DynamicPPL.assume(
            ::Random.AbstractRNG, ::DynamicPPL.Sampler{MyEmptyAlg}, dist, vn, vi
        ) = DynamicPPL.assume(dist, vn, vi)

        # Compiling the ReverseDiff tape used to fail here
        spl = Sampler(MyEmptyAlg())
        vi = VarInfo(model)
        ldf = LogDensityFunction(vi, model, SamplingContext(spl))
        ldf_grad = LogDensityFunctionWithGrad(ldf, AutoReverseDiff(; compile=true))
        @test LogDensityProblems.logdensity_and_gradient(ldf_grad, vi[:]) isa Any
    end
end
