@testset "check_model" begin
    @testset "context interface" begin
        context = DynamicPPL.DebugContext()
        DynamicPPL.TestUtils.test_context_interface(context)
    end

    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        issuccess, (trace, varnames_seen) = DynamicPPL.check_model(model)
        # These models should all work.
        @test issuccess

        # Check that the trace contains all the variables in the model.
        vns_iter = mapreduce(vcat, trace) do record
            vec([record.varname;])
        end
        varnames_in_trace = Set(vns_iter)
        for vn in DynamicPPL.TestUtils.varnames(model)
            @test haskey(varnames_seen, vn)
            @test varnames_seen[vn] == 1
            @test vn in varnames_in_trace
        end
    end

    @testset "buggy model" begin
        @model function buggy_demo_model()
            x ~ Normal()
            x ~ Normal()
            return y ~ Normal()
        end
        buggy_model = buggy_demo_model()

        @test_logs (:warn,) (:warn,) DynamicPPL.check_model(buggy_model)
        issuccess, (trace, varnames_seen) = DynamicPPL.check_model(
            buggy_model; context=SamplingContext(), record_varinfo=false
        )
        @test !issuccess
        @test_throws ErrorException DynamicPPL.check_model(
            buggy_model; error_on_failure=true
        )
    end
end
