using JET: JET

@testset "DynamicPPLJETExt.jl" begin
    @testset "determine_varinfo" begin
        @model function demo1()
            x ~ Bernoulli()
            if x
                y ~ Normal()
            else
                z ~ Normal()
            end
        end
        model = demo1()
        @test DynamicPPL.determine_varinfo(model) isa DynamicPPL.UntypedVarInfo

        @model demo2() = x ~ Normal()
        @test DynamicPPL.determine_varinfo(demo2()) isa DynamicPPL.TypedVarInfo

        @model function demo3()
            # Just making sure that nothing strange happens when type inference fails.
            x = Vector(undef, 1)
            x[1] ~ Bernoulli()
            if x[1]
                y ~ Normal()
            else
                z ~ Normal()
            end
        end
        @test DynamicPPL.determine_varinfo(demo3()) isa DynamicPPL.UntypedVarInfo

        # Evaluation works (and it would even do so in practice), but sampling
        # fill fail due to storing `Cauchy{Float64}` in `Vector{Normal{Float64}}`.
        @model function demo4()
            x ~ Bernoulli()
            if x
                y ~ Normal()
            else
                y ~ Cauchy() # different distibution, but same transformation => should work
            end
        end
        @test DynamicPPL.determine_varinfo(demo4()) isa DynamicPPL.UntypedVarInfo
    end

    # @testset "demo models" begin
    #     @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
    #         varinfo = DynamicPPL.DynamicPPL.determine_varinfo(model)
    #         # They should all result in typed.
    #         @test varinfo isa DynamicPPL.TypedVarInfo
    #         # But let's also make sure that they're not lying.
    #         f_eval, argtypes_eval = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
    #             model, varinfo
    #         )
    #         JET.test_call(f_eval, argtypes_eval)

    #         f_sample, argtypes_sample = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
    #             model, varinfo, DynamicPPL.SamplingContext()
    #         )
    #         JET.test_call(f_sample, argtypes_sample)
    #     end
    # end
end
