@testset "AD test for model $(repr(m.f))" for m in DynamicPPL.TestUtils.DEMO_MODELS
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

        @testset "with ADType $adtype" for adtype in (
            ADTypes.AutoReverseDiff(false), ADTypes.AutoReverseDiff(true)
        )
            ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
            _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
            @test grad ≈ ref_grad
        end

        if m.f ∈ (
            DynamicPPL.TestUtils.demo_assume_multivariate_observe,
            DynamicPPL.TestUtils.demo_assume_dot_observe,
            DynamicPPL.TestUtils.demo_assume_observe_literal,
            DynamicPPL.TestUtils.demo_assume_literal_dot_observe,
        ) &&
            varinfo isa Union{DynamicPPL.TypedVarInfo,DynamicPPL.SimpleVarInfo{<:NamedTuple}}
            adtype = ADTypes.AutoZygote()
            ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
            _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
            @test grad ≈ ref_grad
        end
    end
end
