@testset "DynamicPPLJETExt.jl" begin
    @testset "determine_suitable_varinfo" begin
        @model function demo1()
            x ~ Bernoulli()
            if x
                y ~ Normal()
            else
                z ~ Normal()
            end
        end
        model = demo1()
        @test DynamicPPL.Experimental.determine_suitable_varinfo(model) isa
            DynamicPPL.UntypedVarInfo

        @model demo2() = x ~ Normal()
        @test DynamicPPL.Experimental.determine_suitable_varinfo(demo2()) isa
            DynamicPPL.NTVarInfo

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
        @test DynamicPPL.Experimental.determine_suitable_varinfo(demo3()) isa
            DynamicPPL.UntypedVarInfo

        # Evaluation works (and it would even do so in practice), but sampling
        # fill fail due to storing `Cauchy{Float64}` in `Vector{Normal{Float64}}`.
        @model function demo4()
            x ~ Bernoulli()
            if x
                y ~ Normal()
            else
                y ~ Cauchy() # different distibution, but same transformation
            end
        end
        @test DynamicPPL.Experimental.determine_suitable_varinfo(demo4()) isa
            DynamicPPL.UntypedVarInfo

        # In this model, the type error occurs in the user code rather than in DynamicPPL.
        @model function demo5()
            x ~ Normal()
            xs = Any[]
            push!(xs, x)
            # `sum(::Vector{Any})` can potentially error unless the dynamic manages to resolve the
            # correct `zero` method. As a result, this code will run, but JET will raise this is an issue.
            return sum(xs)
        end
        # Should pass if we're only checking the tilde statements.
        @test DynamicPPL.Experimental.determine_suitable_varinfo(demo5()) isa
            DynamicPPL.NTVarInfo
        # Should fail if we're including errors in the model body.
        @test DynamicPPL.Experimental.determine_suitable_varinfo(
            demo5(); only_ddpl=false
        ) isa DynamicPPL.UntypedVarInfo
    end

    @testset "demo models" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            # Use debug logging below.
            varinfo = DynamicPPL.Experimental.determine_suitable_varinfo(model)
            # Check that the inferred varinfo is indeed suitable for evaluation and sampling
            f_eval, argtypes_eval = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                model, varinfo
            )
            JET.test_call(f_eval, argtypes_eval)

            f_sample, argtypes_sample = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                model, varinfo, DynamicPPL.SamplingContext()
            )
            JET.test_call(f_sample, argtypes_sample)
            # For our demo models, they should all result in typed.
            is_typed = varinfo isa DynamicPPL.NTVarInfo
            @test is_typed
            # If the test failed, check why it didn't infer a typed varinfo
            if !is_typed
                typed_vi = TypedVarInfo(model)
                f_eval, argtypes_eval = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                    model, typed_vi
                )
                JET.test_call(f_eval, argtypes_eval)
                f_sample, argtypes_sample = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                    model, typed_vi, DynamicPPL.SamplingContext()
                )
                JET.test_call(f_sample, argtypes_sample)
            end
        end
    end
end
