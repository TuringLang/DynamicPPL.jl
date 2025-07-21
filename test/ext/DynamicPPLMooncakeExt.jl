using DynamicPPL
using Distributions
using Random
using Test
using StableRNGs
using Mooncake: Mooncake, NoCache, set_to_zero!!, set_to_zero_internal!!, zero_tangent
using DynamicPPL.TestUtils.AD: @be, median

# Define models globally to avoid closure issues
@model function test_model1(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    return x .~ Normal(m, sqrt(s))
end

@model function test_model2(x, y)
    τ ~ Gamma(1, 1)
    σ ~ InverseGamma(2, 3)
    μ ~ Normal(0, τ)
    x .~ Normal(μ, σ)
    return y .~ Normal(μ, σ)
end

@testset "DynamicPPLMooncakeExt" begin
    @testset "istrans rule" begin
        Mooncake.TestUtils.test_rule(
            StableRNG(123456), istrans, VarInfo(); unsafe_perturb=true, interface_only=true
        )
    end

    @testset "set_to_zero!! correctness" begin
        # Test that set_to_zero!! works correctly for DynamicPPL types
        model = test_model1([1.0, 2.0, 3.0])
        vi = VarInfo(Random.default_rng(), model)
        ldf = LogDensityFunction(model, vi, DefaultContext())
        tangent = zero_tangent(ldf)

        # Modify some values
        if hasfield(typeof(tangent.fields.model.fields), :args) &&
            hasfield(typeof(tangent.fields.model.fields.args), :x)
            x_tangent = tangent.fields.model.fields.args.x
            if !isempty(x_tangent)
                x_tangent[1] = 5.0
            end
        end

        # Call set_to_zero!! and verify it works
        result = set_to_zero!!(tangent)
        @test result isa typeof(tangent)

        # Check that values are zeroed
        if hasfield(typeof(tangent.fields.model.fields), :args) &&
            hasfield(typeof(tangent.fields.model.fields.args), :x)
            x_tangent = tangent.fields.model.fields.args.x
            if !isempty(x_tangent)
                @test x_tangent[1] == 0.0
            end
        end
    end

    @testset "Performance improvement" begin
        model = DynamicPPL.TestUtils.DEMO_MODELS[1]
        vi = VarInfo(Random.default_rng(), model)
        ldf = LogDensityFunction(model, vi, DefaultContext())
        tangent = zero_tangent(ldf)

        # Run benchmarks
        result_iddict = @be begin
            cache = IdDict{Any,Bool}()
            set_to_zero_internal!!(cache, tangent)
        end

        result_nocache = @be set_to_zero!!(tangent)

        # Extract median times
        time_iddict = median(result_iddict).time
        time_nocache = median(result_nocache).time

        # We expect NoCache to be faster
        speedup = time_iddict / time_nocache
        @test speedup > 1.5  # Conservative expectation - should be ~4x

        # Sanity check
        @info "Performance improvement" speedup time_iddict_μs = time_iddict / 1000 time_nocache_μs =
            time_nocache / 1000
    end

    @testset "Aliasing safety" begin
        # Test with aliased data
        shared_data = [1.0, 2.0, 3.0]
        model = test_model2(shared_data, shared_data)  # x and y are the same array
        vi = VarInfo(Random.default_rng(), model)
        ldf = LogDensityFunction(model, vi, DefaultContext())
        tangent = zero_tangent(ldf)

        # Check that aliasing is preserved in tangent
        if hasfield(typeof(tangent.fields.model.fields), :args)
            args = tangent.fields.model.fields.args
            if hasfield(typeof(args), :x) && hasfield(typeof(args), :y)
                @test args.x === args.y  # Aliasing should be preserved

                # Modify via x
                if !isempty(args.x)
                    args.x[1] = 10.0
                    @test args.y[1] == 10.0  # Should also change y
                end

                # Zero and check both are zeroed
                # Since x and y are aliased, zeroing one zeros both
                set_to_zero!!(tangent)
                if !isempty(args.x)
                    @test args.x[1] == 0.0
                    @test args.y[1] == 0.0
                end
            end
        end
    end

    @testset "Closure handling" begin
        # Test that closure models are correctly handled

        # Create closure model (captures environment, has circular references)
        function create_closure_model()
            local_var = 42
            @model function closure_model(x)
                s ~ InverseGamma(2, 3)
                m ~ Normal(0, sqrt(s))
                return x .~ Normal(m, sqrt(s))
            end
            return closure_model
        end

        closure_fn = create_closure_model()
        model_closure = closure_fn([1.0, 2.0, 3.0])
        vi_closure = VarInfo(Random.default_rng(), model_closure)
        ldf_closure = LogDensityFunction(model_closure, vi_closure, DefaultContext())
        tangent_closure = zero_tangent(ldf_closure)

        # Test that it works without stack overflow
        @test_nowarn set_to_zero!!(deepcopy(tangent_closure))

        # Compare with global model (no closure)
        model_global = test_model1([1.0, 2.0, 3.0])
        vi_global = VarInfo(Random.default_rng(), model_global)
        ldf_global = LogDensityFunction(model_global, vi_global, DefaultContext())
        tangent_global = zero_tangent(ldf_global)

        # Verify model.f tangent types differ
        f_tangent_closure = tangent_closure.fields.model.fields.f
        f_tangent_global = tangent_global.fields.model.fields.f

        @test f_tangent_global isa Mooncake.NoTangent  # Global function
        @test f_tangent_closure isa Mooncake.Tangent   # Closure function

        # Performance comparison
        time_global = @elapsed for _ in 1:100
            set_to_zero!!(tangent_global)
        end

        time_closure = @elapsed for _ in 1:100
            set_to_zero!!(tangent_closure)
        end

        # Global should be faster (uses NoCache)
        @test time_global < time_closure
    end

    @testset "Struct field assumptions" begin
        # Test that our assumptions about DynamicPPL struct fields are correct
        # These tests will fail if DynamicPPL changes its internal structure

        @testset "LogDensityFunction tangent structure" begin
            model = test_model1([1.0, 2.0, 3.0])
            vi = VarInfo(Random.default_rng(), model)
            ldf = LogDensityFunction(model, vi, DefaultContext())
            tangent = zero_tangent(ldf)

            # Test expected fields exist
            @test hasfield(typeof(tangent), :fields)
            @test hasfield(typeof(tangent.fields), :model)
            @test hasfield(typeof(tangent.fields), :varinfo)
            @test hasfield(typeof(tangent.fields), :context)
            @test hasfield(typeof(tangent.fields), :adtype)
            @test hasfield(typeof(tangent.fields), :prep)

            # Test exact field names match
            @test propertynames(tangent.fields) ==
                (:model, :varinfo, :context, :adtype, :prep)
        end

        @testset "VarInfo tangent structure" begin
            model = test_model1([1.0, 2.0, 3.0])
            vi = VarInfo(Random.default_rng(), model)
            tangent_vi = zero_tangent(vi)

            # Test expected fields exist
            @test hasfield(typeof(tangent_vi), :fields)
            @test hasfield(typeof(tangent_vi.fields), :metadata)
            @test hasfield(typeof(tangent_vi.fields), :logp)
            @test hasfield(typeof(tangent_vi.fields), :num_produce)

            # Test exact field names match
            @test propertynames(tangent_vi.fields) == (:metadata, :logp, :num_produce)
        end

        @testset "Model tangent structure" begin
            model = test_model1([1.0, 2.0, 3.0])
            tangent_model = zero_tangent(model)

            # Test expected fields exist
            @test hasfield(typeof(tangent_model), :fields)
            @test hasfield(typeof(tangent_model.fields), :f)
            @test hasfield(typeof(tangent_model.fields), :args)
            @test hasfield(typeof(tangent_model.fields), :defaults)
            @test hasfield(typeof(tangent_model.fields), :context)

            # Test exact field names match
            @test propertynames(tangent_model.fields) == (:f, :args, :defaults, :context)
        end

        @testset "Metadata tangent structure" begin
            model = test_model1([1.0, 2.0, 3.0])
            vi = VarInfo(Random.default_rng(), model)
            tangent_vi = zero_tangent(vi)
            metadata = tangent_vi.fields.metadata

            # Metadata is a NamedTuple with variable names as keys
            @test metadata isa NamedTuple

            # Each variable's metadata should be a Tangent with the expected fields
            for (varname, var_metadata) in pairs(metadata)
                @test var_metadata isa Mooncake.Tangent
                @test hasfield(typeof(var_metadata), :fields)

                # Test expected fields exist
                @test hasfield(typeof(var_metadata.fields), :idcs)
                @test hasfield(typeof(var_metadata.fields), :vns)
                @test hasfield(typeof(var_metadata.fields), :ranges)
                @test hasfield(typeof(var_metadata.fields), :vals)
                @test hasfield(typeof(var_metadata.fields), :dists)
                @test hasfield(typeof(var_metadata.fields), :orders)
                @test hasfield(typeof(var_metadata.fields), :flags)

                # Test exact field names match
                @test propertynames(var_metadata.fields) ==
                    (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags)
            end
        end
    end
end
