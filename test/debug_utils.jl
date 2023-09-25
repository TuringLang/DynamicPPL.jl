@testset "check_model" begin
    @testset "context interface" begin
        # HACK: Require a model to instantiate it, so let's just grab one.
        model = first(DynamicPPL.TestUtils.DEMO_MODELS)
        context = DynamicPPL.DebugContext(model)
        DynamicPPL.TestUtils.test_context_interface(context)
    end

    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        issuccess, (trace, varnames_seen) = check_model(model)
        # These models should all work.
        @test issuccess

        # Check that the trace contains all the variables in the model.
        assume_stmts = filter(Base.Fix2(hasproperty, :varname), trace)
        vns_iter = mapreduce(vcat, assume_stmts) do record
            vec([record.varname;])
        end
        varnames_in_trace = Set(vns_iter)
        for vn in DynamicPPL.TestUtils.varnames(model)
            @test haskey(varnames_seen, vn)
            @test varnames_seen[vn] == 1
            @test vn in varnames_in_trace
        end

        # Quick checks for `show` of trace.
        @test occursin("assume: ", string(trace))
        @test occursin("observe: ", string(trace))
    end

    @testset "multiple usage of same variable" begin
        @testset "simple" begin
            @model function buggy_demo_model()
                x ~ Normal()
                x ~ Normal()
                return y ~ Normal()
            end
            buggy_model = buggy_demo_model()

            @test_logs (:warn,) (:warn,) check_model(buggy_model)
            issuccess, (trace, varnames_seen) = check_model(
                buggy_model; context=SamplingContext(), record_varinfo=false
            )
            @test !issuccess
            @test_throws ErrorException check_model(buggy_model; error_on_failure=true)
        end

        @testset "submodel" begin
            @model ModelInner() = x ~ Normal()
            @model function ModelOuterBroken()
                @submodel z = ModelInner()
                return x ~ Normal()
            end
            model = ModelOuterBroken()
            @test_throws ErrorException check_model(model; error_on_failure=true)
            
            @model function ModelOuterWorking()
                @submodel prefix=true z = ModelInner()
                x ~ Normal()
                return z
            end
            model = ModelOuterWorking()
            @test first(check_model(model; error_on_failure=true))
        end
    end

    @testset "incorrect use of condition" begin
        @testset "missing in multivariate" begin
            @model function demo_missing_in_multivariate(x)
                x ~ MvNormal(zeros(length(x)), I)
            end
            model = demo_missing_in_multivariate([1.0, missing])
            @test_throws ErrorException check_model(model)
        end

        @testset "condition both in args and context" begin
            @model function demo_condition_both_in_args_and_context(x)
                x ~ Normal()
            end
            model = demo_condition_both_in_args_and_context(1.0)
            for vals in [
                (x = 2.0,),
                OrderedDict(@varname(x) => 2.0,),
                OrderedDict(@varname(x[1]) => 2.0,)
            ]
                conditioned_model = DynamicPPL.condition(model, vals)
                @test_throws ErrorException check_model(conditioned_model; error_on_failure=true)
            end
        end
    end

    @testset "printing statements" begin
        @testset "assume" begin
            @model demo_assume() = x ~ Normal()
            isuccess, (trace, varnames_seen) = check_model(demo_assume())
            @test isuccess
            @test startswith(string(trace), " assume: x ~ Normal")
        end

        @testset "observe" begin
            @model demo_observe(x) = x ~ Normal()
            isuccess, (trace, varnames_seen) = check_model(demo_observe(1.0))
            @test isuccess
            @test occursin(r"observe: \d+\.\d+ ~ Normal", string(trace))
        end
    end
end
