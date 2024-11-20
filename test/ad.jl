@testset "AD: ForwardDiff and ReverseDiff" begin
    @testset "$(m.f)" for m in TU.DEMO_MODELS
        f = DynamicPPL.LogDensityFunction(m)
        rand_param_values = TU.rand_prior_true(m)
        vns = TU.varnames(m)
        varinfos = TU.setup_varinfos(m, rand_param_values, vns)

        @testset "$(TU.short_varinfo_name(varinfo))" for varinfo in varinfos
            f = DynamicPPL.LogDensityFunction(m, varinfo)

            # use ForwardDiff result as reference
            ad_forwarddiff_f = LogDensityProblemsAD.ADgradient(
                ADTypes.AutoForwardDiff(; chunksize=0), f
            )
            # convert to `Vector{Float64}` to avoid `ReverseDiff` initializing the gradients to Integer 0
            # reference: https://github.com/TuringLang/DynamicPPL.jl/pull/571#issuecomment-1924304489
            θ = convert(Vector{Float64}, varinfo[:])
            logp, ref_grad = LogDensityProblems.logdensity_and_gradient(ad_forwarddiff_f, θ)

            @testset "ReverseDiff with compile=$compile" for compile in (false, true)
                adtype = ADTypes.AutoReverseDiff(; compile=compile)
                ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
                _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
                @test grad ≈ ref_grad
            end
        end
    end
end
