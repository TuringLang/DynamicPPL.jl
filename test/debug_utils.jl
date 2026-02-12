module DynamicPPLDebugUtilsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL, Distributions, Test
using LinearAlgebra: I
using Random: Xoshiro

@testset "check_model" begin
    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        @test check_model(model)
        @test DynamicPPL.has_static_constraints(model)
    end

    @testset "multiple usage of same variable" begin
        function test_model_can_run_but_fails_check(model)
            # Check that it can actually run
            @test VarInfo(model) isa VarInfo
            # but if you call check_model it should fail
            issuccess = check_model(model)
            @test !issuccess
            @test_throws ErrorException check_model(model; error_on_failure=true)
        end

        @testset "simple" begin
            @model function buggy_demo_model()
                x ~ Normal()
                x ~ Normal()
                return y ~ Normal()
            end
            test_model_can_run_but_fails_check(buggy_demo_model())
        end

        @testset "submodel" begin
            @model ModelInner() = x ~ Normal()
            @model function ModelOuterBroken()
                # Without automatic prefixing => `x` s used twice.
                z ~ to_submodel(ModelInner(), false)
                return x ~ Normal()
            end
            test_model_can_run_but_fails_check(ModelOuterBroken())

            @model function ModelOuterWorking()
                # With automatic prefixing => `x` is not duplicated.
                z ~ to_submodel(ModelInner())
                x ~ Normal()
                return z
            end
            model = ModelOuterWorking()
            @test check_model(model)

            # With manual prefixing, https://github.com/TuringLang/DynamicPPL.jl/issues/785
            @model function ModelOuterWorking2()
                x1 ~ to_submodel(DynamicPPL.prefix(ModelInner(), :a), false)
                x2 ~ to_submodel(DynamicPPL.prefix(ModelInner(), :b), false)
                return (x1, x2)
            end
            model = ModelOuterWorking2()
            @test check_model(model)
        end
    end

    @testset "NaN in data" begin
        @model function demo_nan_in_data(x)
            a ~ Normal()
            for i in eachindex(x)
                x[i] ~ Normal(a)
            end
        end
        m = demo_nan_in_data([1.0, NaN])
        @test_throws ErrorException check_model(m; error_on_failure=true)
        # Test NamedTuples with nested arrays, see #898
        @model function demo_nan_complicated(nt)
            nt ~ product_distribution((x=Normal(), y=Dirichlet([2, 4])))
            return x ~ Normal()
        end
        m = demo_nan_complicated((x=1.0, y=[NaN, 0.5]))
        @test_throws ErrorException check_model(m; error_on_failure=true)
    end

    @testset "incorrect use of condition" begin
        @testset "missing in multivariate" begin
            @model function demo_missing_in_multivariate(x)
                return x ~ MvNormal(zeros(length(x)), I)
            end
            model = demo_missing_in_multivariate([1.0, missing])
            @test_throws ErrorException check_model(model; error_on_failure=true)
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

    @testset "discrete distribution check" begin
        @testset "univariate discrete" begin
            @model function demo_discrete()
                x ~ Poisson(3)
                return y ~ Normal()
            end
            model = demo_discrete()
            # Without fail_if_discrete, the model should pass.
            @test check_model(model; error_on_failure=true)
            # With fail_if_discrete, it should fail.
            @test_throws ErrorException check_model(
                model; error_on_failure=true, fail_if_discrete=true
            )
            # Without error_on_failure, it should warn but issuccess is
            # currently not affected by the discrete check (only by varname
            # checks). TODO: issuccess should ideally be false here.
            @test_broken !check_model(model; fail_if_discrete=true)
        end

        @testset "multivariate discrete" begin
            @model function demo_mv_discrete()
                x ~ product_distribution(fill(Poisson(3), 3))
                return y ~ Normal()
            end
            model = demo_mv_discrete()
            @test check_model(model; error_on_failure=true)
            @test_throws ErrorException check_model(
                model; error_on_failure=true, fail_if_discrete=true
            )
        end

        @testset "all continuous should pass" begin
            @model function demo_all_continuous()
                x ~ Normal()
                return y ~ Gamma(2, 1)
            end
            model = demo_all_continuous()
            @test check_model(model; error_on_failure=true, fail_if_discrete=true)
        end
    end

    @testset "with dynamic constraints" begin
        # Run the same model but with different VarInfos.
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        @test check_model(Xoshiro(1), model) && check_model(Xoshiro(2), model)
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
            @test DynamicPPL.DebugUtils.model_warntype(model) isa Any
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
