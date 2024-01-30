@testset "AD test for model $(repr(m.f))" for m in DynamicPPL.TestUtils.DEMO_MODELS
    f = DynamicPPL.LogDensityFunction(m)
    rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
    vns = DynamicPPL.TestUtils.varnames(m)
    varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

    @testset "$varinfo" for varinfo in varinfos
        varinfo = varinfos[2]
        f = DynamicPPL.LogDensityFunction(m, varinfo)

        # use ForwardDiff result as reference
        ad_forwarddiff_f = LogDensityProblemsAD.ADgradient(
            ADTypes.AutoForwardDiff(; chunksize=0), f
        )
        θ = varinfo[:]
        logp, ref_grad = LogDensityProblems.logdensity_and_gradient(ad_forwarddiff_f, θ)

        @testset "with ADType $adtype" for adtype in (
            ADTypes.AutoReverseDiff(false), ADTypes.AutoReverseDiff(true)
        )
            ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
            _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
            @test grad ≈ ref_grad
        end

        if m.f ∉ (
            DynamicPPL.TestUtils.demo_dot_assume_dot_observe,
            DynamicPPL.TestUtils.demo_assume_index_observe,
            DynamicPPL.TestUtils.demo_dot_assume_observe_index,
            DynamicPPL.TestUtils.demo_dot_assume_observe_index_literal,
            DynamicPPL.TestUtils.demo_assume_submodel_observe_index_literal,
            DynamicPPL.TestUtils.demo_dot_assume_observe_submodel,
            DynamicPPL.TestUtils.demo_dot_assume_dot_observe_matrix,
            DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix,
            DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix,
        )
            adtype = ADTypes.AutoZygote()
            ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
            _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
            @test grad ≈ ref_grad
        end
    end
end
