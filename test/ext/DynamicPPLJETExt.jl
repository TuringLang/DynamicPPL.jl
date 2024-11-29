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
        @test DynamicPPL.determine_suitable_varinfo(model; verbose=true) isa
            DynamicPPL.UntypedVarInfo

        @model demo2() = x ~ Normal()
        @test DynamicPPL.determine_suitable_varinfo(demo2()) isa DynamicPPL.TypedVarInfo

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
        @test DynamicPPL.determine_suitable_varinfo(demo3(); verbose=true) isa
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
        @test DynamicPPL.determine_suitable_varinfo(demo4(); verbose=true) isa
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
        @test DynamicPPL.determine_suitable_varinfo(demo5(); verbose=true) isa
            DynamicPPL.TypedVarInfo
        # Should fail if we're including errors in the model body.
        @test DynamicPPL.determine_suitable_varinfo(
            demo5(); verbose=true, only_tilde=false
        ) isa DynamicPPL.UntypedVarInfo
    end

    @testset "demo models" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            varinfo = DynamicPPL.DynamicPPL.determine_suitable_varinfo(model)
            # They should all result in typed.
            @test varinfo isa DynamicPPL.TypedVarInfo
            # But let's also make sure that they're not lying.
            f_eval, argtypes_eval = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                model, varinfo
            )
            JET.test_call(f_eval, argtypes_eval)

            f_sample, argtypes_sample = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
                model, varinfo, DynamicPPL.SamplingContext()
            )
            JET.test_call(f_sample, argtypes_sample)
        end
    end
end
