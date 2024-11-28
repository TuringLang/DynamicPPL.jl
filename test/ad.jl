@testset "AD: ForwardDiff, ReverseDiff, and Mooncake" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
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
            # convert to `Vector{Float64}` to avoid `ReverseDiff` initializing the gradients to Integer 0
            # reference: https://github.com/TuringLang/DynamicPPL.jl/pull/571#issuecomment-1924304489
            θ = convert(Vector{Float64}, varinfo[:])
            logp, ref_grad = LogDensityProblems.logdensity_and_gradient(ad_forwarddiff_f, θ)

            @testset "$adtype" for adtype in [
                ADTypes.AutoReverseDiff(; compile=false),
                ADTypes.AutoReverseDiff(; compile=true),
                ADTypes.AutoMooncake(; config=nothing),
            ]
                # Mooncake can't currently handle something that is going on in
                # SimpleVarInfo{<:VarNamedVector}. Disable tests for now.
                if adtype isa ADTypes.AutoMooncake &&
                    varinfo isa DynamicPPL.SimpleVarInfo{<:DynamicPPL.VarNamedVector}
                    @test_broken 1 == 0
                else
                    ad_f = LogDensityProblemsAD.ADgradient(adtype, f)
                    _, grad = LogDensityProblems.logdensity_and_gradient(ad_f, θ)
                    @test grad ≈ ref_grad
                end
            end
        end
    end
end
