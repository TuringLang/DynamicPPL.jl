@testset "check_model" begin
    @testset "context interface" begin
        # HACK: Require a model to instantiate it, so let's just grab one.
        model = first(DynamicPPL.TestUtils.DEMO_MODELS)
        context = DynamicPPL.DebugUtils.DebugContext(model)
        DynamicPPL.TestUtils.test_context_interface(context)
    end

    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        issuccess, trace = check_model_and_trace(model)
        # These models should all work.
        @test issuccess

        # Check that the trace contains all the variables in the model.
        varnames_in_trace = DynamicPPL.DebugUtils.varnames_in_trace(trace)
        for vn in DynamicPPL.TestUtils.varnames(model)
            @test vn in varnames_in_trace
        end

        # Quick checks for `show` of trace.
        @test occursin("assume: ", string(trace))
        @test occursin("observe: ", string(trace))

        # All these models should have static constraints.
        @test DynamicPPL.has_static_constraints(model)
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
            issuccess = check_model(
                buggy_model; context=SamplingContext(), record_varinfo=false
            )
            @test !issuccess
            @test_throws ErrorException check_model(buggy_model; error_on_failure=true)
        end

        @testset "submodel" begin
            @model ModelInner() = x ~ Normal()
            @model function ModelOuterBroken()
                z ~ to_submodel(ModelInner())
                return x ~ Normal()
            end
            model = ModelOuterBroken()
            @test_throws ErrorException check_model(model; error_on_failure=true)

            @model function ModelOuterWorking()
                z = to_submodel(prefix(ModelInner(), "z"))
                x ~ Normal()
                return z
            end
            model = ModelOuterWorking()
            @test check_model(model; error_on_failure=true)
        end

        @testset "subsumes (x then x[1])" begin
            @model function buggy_subsumes_demo_model()
                x = Vector{Float64}(undef, 2)
                x ~ MvNormal(zeros(2), I)
                x[1] ~ Normal()
                return nothing
            end
            buggy_model = buggy_subsumes_demo_model()

            @test_logs (:warn,) (:warn,) check_model(buggy_model)
            issuccess = check_model(
                buggy_model; context=SamplingContext(), record_varinfo=false
            )
            @test !issuccess
            @test_throws ErrorException check_model(buggy_model; error_on_failure=true)
        end

        @testset "subsumes (x[1] then x)" begin
            @model function buggy_subsumes_demo_model()
                x = Vector{Float64}(undef, 2)
                x[1] ~ Normal()
                x ~ MvNormal(zeros(2), I)
                return nothing
            end
            buggy_model = buggy_subsumes_demo_model()

            @test_logs (:warn,) (:warn,) check_model(buggy_model)
            issuccess = check_model(
                buggy_model; context=SamplingContext(), record_varinfo=false
            )
            @test !issuccess
            @test_throws ErrorException check_model(buggy_model; error_on_failure=true)
        end

        @testset "subsumes (x.a then x)" begin
            @model function buggy_subsumes_demo_model()
                x = (a=nothing,)
                x.a ~ Normal()
                x ~ Normal()
                return nothing
            end
            buggy_model = buggy_subsumes_demo_model()

            @test_logs (:warn,) (:warn,) check_model(buggy_model)
            issuccess = check_model(
                buggy_model; context=SamplingContext(), record_varinfo=false
            )
            @test !issuccess
            @test_throws ErrorException check_model(buggy_model; error_on_failure=true)
        end
    end

    @testset "incorrect use of condition" begin
        @testset "missing in multivariate" begin
            @model function demo_missing_in_multivariate(x)
                return x ~ MvNormal(zeros(length(x)), I)
            end
            model = demo_missing_in_multivariate([1.0, missing])
            @test_throws ErrorException check_model(model)
        end

        @testset "condition both in args and context" begin
            @model function demo_condition_both_in_args_and_context(x)
                return x ~ Normal()
            end
            model = demo_condition_both_in_args_and_context(1.0)
            for vals in [
                (x=2.0,),
                OrderedDict(@varname(x) => 2.0),
                OrderedDict(@varname(x[1]) => 2.0),
            ]
                conditioned_model = DynamicPPL.condition(model, vals)
                @test_throws ErrorException check_model(
                    conditioned_model; error_on_failure=true
                )
            end
        end
    end

    @testset "printing statements" begin
        @testset "assume" begin
            @model demo_assume() = x ~ Normal()
            isuccess, trace = check_model_and_trace(demo_assume())
            @test isuccess
            @test startswith(string(trace), " assume: x ~ Normal")
        end

        @testset "observe" begin
            @model demo_observe(x) = x ~ Normal()
            isuccess, trace = check_model_and_trace(demo_observe(1.0))
            @test isuccess
            @test occursin(r"observe: \d+\.\d+ ~ Normal", string(trace))
        end
    end

    @testset "comparing multiple traces" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        issuccess_1, trace_1 = check_model_and_trace(model)
        issuccess_2, trace_2 = check_model_and_trace(model)
        @test issuccess_1 && issuccess_2

        # Should have the same varnames present.
        varnames_1 = DynamicPPL.DebugUtils.varnames_in_trace(trace_1)
        varnames_2 = DynamicPPL.DebugUtils.varnames_in_trace(trace_2)
        @info varnames_1 == varnames_2

        # But will have different distributions.
        dists_1 = DynamicPPL.DebugUtils.distributions_in_trace(trace_1)
        dists_2 = DynamicPPL.DebugUtils.distributions_in_trace(trace_2)
        @test dists_1[1] == dists_2[1]
        @test dists_1[2] != dists_2[2]

        @test !DynamicPPL.has_static_constraints(model)
    end

    @testset "vector with `undef`" begin
        # Source: https://github.com/TuringLang/Turing.jl/pull/2218
        @model function demo_undef(ns...)
            x = Array{Real}(undef, ns...)
            @. x ~ Normal(0, 2)
        end
        for ns in [(2,), (2, 2), (2, 2, 2)]
            model = demo_undef(ns...)
            @test check_model(model; error_on_failure=true)
        end
    end

    @testset "model_warntype & model_codetyped" begin
        @model demo_without_kwargs(x) = y ~ Normal(x, 1)
        @model demo_with_kwargs(x; z=1) = y ~ Normal(x, z)

        for model in [demo_without_kwargs(1.0), demo_with_kwargs(1.0)]
            codeinfo, retype = DynamicPPL.DebugUtils.model_typed(model)
            @test codeinfo isa Core.CodeInfo
            @test retype <: Tuple

            # Just make sure the following is runnable.
            @test (DynamicPPL.DebugUtils.model_warntype(model); true)
        end
    end
end
