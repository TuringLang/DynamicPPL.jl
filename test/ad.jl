@testset "Testing AD by comparing gradient using ForwardDiff and ReverseDiff for model $(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
    f = DynamicPPL.LogDensityFunction(m)
    rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
    vns = DynamicPPL.TestUtils.varnames(m)
    varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

    @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
        f = DynamicPPL.LogDensityFunction(m, varinfo)

        # use ForwardDiff result as reference
        ad_forwarddiff_f = LogDensityProblemsAD.ADgradient(
            ADTypes.AutoForwardDiff(; chunksize=0), f
        )
        θ = identity.(varinfo[:])
        logp, ref_grad = LogDensityProblems.logdensity_and_gradient(ad_forwarddiff_f, θ)

        @testset "ReverseDiff with compile=$compile" for compile in (false, true)
            adtype = ADTypes.AutoReverseDiff(; compile=compile)
            ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
            _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
            @test grad ≈ ref_grad
        end
    end
end
