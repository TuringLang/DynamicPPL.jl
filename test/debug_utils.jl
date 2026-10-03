module DynamicPPLDebugUtilsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL, Distributions, Test
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using Random: Xoshiro

function test_model_fails_check(model)
    issuccess = check_model(model)
    @test !issuccess
    @test_throws ErrorException check_model(model; error_on_failure=true)
end
function test_model_can_run_but_fails_check(model)
    # Check that it can actually run
    @test VarInfo(model) isa VarInfo
    # but if you call check_model it should fail
    return test_model_fails_check(model)
end

@testset "check_model" begin
    @testset "bindings without a reached LHS name" begin
        @model function binding_child(run=true)
            if run
                z ~ Normal()
            end
        end
        @model function binding_parent(run=true, child=binding_child())
            x ~ Normal()
            if run
                a ~ to_submodel(child, false)
            end
        end
        warning = r"Binding `zz` has no LHS top symbol"
        for bind in (condition, fix)
            @test_logs (:warn, warning) @test check_model(
                Xoshiro(1),
                bind(binding_parent(), DynamicPPL.Recursive(); zz=1.0);
                error_on_failure=true,
            )
            @test_logs @test check_model(
                Xoshiro(1), bind(binding_parent(), DynamicPPL.Recursive(); z=1.0)
            )
            # Metadata includes untaken LHS branches in a reached child.
            @test_logs @test check_model(
                Xoshiro(1),
                bind(
                    binding_parent(true, binding_child(false)),
                    DynamicPPL.Recursive();
                    z=1.0,
                ),
            )
            # An unreached child can own a valid name, so this must only warn.
            @test_logs (:warn, r"Binding `z` has no LHS top symbol") @test check_model(
                Xoshiro(1),
                bind(binding_parent(false), DynamicPPL.Recursive(); z=1.0);
                error_on_failure=true,
            )
            @test_throws ArgumentError bind(binding_child(); zz=1.0)
        end
        @model binding_middle() = b ~ to_submodel(binding_child(false), false)
        @test_logs @test check_model(
            Xoshiro(1),
            condition(
                binding_parent(true, binding_middle()), DynamicPPL.Recursive(); z=1.0
            ),
        )
        @test_logs @test check_model(
            Xoshiro(1),
            prefix(condition(binding_parent(), DynamicPPL.Recursive(); z=1.0), @varname(p)),
        )
        @model binding_prefixed() = b ~ to_submodel(binding_child(false))
        # A prefixed descendant's names cannot justify a binding in the parent namespace.
        for child in (binding_prefixed(), prefix(binding_middle(), @varname(p)))
            @test_logs (:warn, r"Binding `z` has no LHS top symbol") @test check_model(
                Xoshiro(1),
                condition(binding_parent(true, child), DynamicPPL.Recursive(); z=1.0),
            )
        end
        @test_logs @test check_model(
            Xoshiro(1),
            condition(
                binding_parent(true, prefix(binding_child(), @varname(p))),
                DynamicPPL.Recursive(),
                @varname(p.z) => 1.0,
            ),
        )
    end

    @testset "provenance binding wrappers" begin
        @model regression(y) = (μ ~ Normal(); y ~ Normal(μ))
        @test check_model(prefix(regression(1.0), @varname(a)))
        @model checked_child(y, checking=false) = begin
            if !checking
                child = DynamicPPL.Model{false}(
                    __model__.f,
                    (; y, checking=true),
                    __model__.defaults,
                    __model__.context,
                    __model__.values;
                    args_on_lhs=DynamicPPL._args_on_lhs(__model__),
                )
                @test check_model(child)
            end
            μ ~ Normal()
            y ~ Normal(μ)
        end
        @model checked_parent() = a ~ to_submodel(checked_child(1.0))
        @test VarInfo(checked_parent()) isa VarInfo
    end

    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        @test check_model(model)
        @test DynamicPPL.has_static_constraints(model)
    end

    @testset "multiple usage of same variable" begin
        @testset "simple" begin
            @model function buggy_demo_model()
                x ~ Normal()
                x ~ Normal()
                return y ~ Normal()
            end
            test_model_can_run_but_fails_check(buggy_demo_model())
        end

        @testset "multithreaded variables overwrite (check_model)" begin
            @model function f_threaded()
                Threads.@threads for i in 1:2
                    x ~ Normal()
                end
            end
            model = setthreadsafe(f_threaded(), true)
            # This should catch the error across threads
            @test_throws ErrorException check_model(model; error_on_failure=true)
        end

        @testset "different sub-indices of the same slice" begin
            # https://github.com/TuringLang/DynamicPPL.jl/issues/1321
            @model function demo_slice_subindices()
                x = Vector{Float64}(undef, 2)
                x[1:2] .~ Normal()
                return x
            end
            @test check_model(demo_slice_subindices(); error_on_failure=true)

            # Same sub-index twice should still fail
            @model function buggy_slice_subindices()
                x = Vector{Float64}(undef, 2)
                x[1:2][1] ~ Normal()
                x[1:2][1] ~ Normal()
                return x
            end
            test_model_can_run_but_fails_check(buggy_slice_subindices())

            # Slices of slices
            @model function buggy_slice_subindices2()
                x = Vector{Float64}(undef, 3)
                x[1:3][1:2] ~ MvNormal(zeros(2), I)
                x[1:3][2:3] ~ MvNormal(zeros(2), I)
                return x
            end
            test_model_can_run_but_fails_check(buggy_slice_subindices2())
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

    @testset "conditioning overrides argument values" begin
        @model function demo_conditioned_argument(x)
            return x ~ Normal()
        end
        model = demo_conditioned_argument(1.0)
        conditioned_model = DynamicPPL.condition(model, (x=2.0,))
        @test check_model(conditioned_model; error_on_failure=true)
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
            @test !check_model(model; fail_if_discrete=true)
            @test_throws ErrorException check_model(
                model; error_on_failure=true, fail_if_discrete=true
            )
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
            @test check_model(model; fail_if_discrete=true)
        end
    end

    @testset "with dynamic constraints" begin
        # Run the same model but with different VarInfos.
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        @test check_model(Xoshiro(1), model) && check_model(Xoshiro(2), model)
        @test !DynamicPPL.has_static_constraints(model)
    end

    @testset "Do not error when vector has uninitialised data" begin
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

            context = InitContext(Xoshiro(1), InitFromParams((; y=2.0)), UnlinkAll())
            _, retype = DynamicPPL.DebugUtils.model_typed(model, VarInfo(); context)
            @test retype <: Tuple{Float64,VarInfo}

            # Just make sure the following is runnable.
            @test DynamicPPL.DebugUtils.model_warntype(model) isa Any

            for vi in (VarInfo(), VarInfo(VectorValueAccumulator()))
                _, argtypes = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(model, vi)
                @test argtypes <: Tuple
                codeinfo, retype = DynamicPPL.DebugUtils.model_typed(model, vi)
                @test codeinfo isa Core.CodeInfo
                @test retype <: Tuple{Float64,VarInfo}
                @test redirect_stdout(devnull) do
                    DynamicPPL.DebugUtils.model_warntype(model, vi) === nothing
                end
            end
        end
    end

    @testset "model body with an argument LHS variable" begin
        @model function unstable_observation(y::T, extra...; center=0.0, kw...) where {T}
            m ~ Normal(center)
            body_scale = m > 0 ? 1.0 : 1
            return y ~ Normal(m, body_scale)
        end
        @model function unstable_keyword(; y::T=1.0) where {T}
            m ~ Normal()
            body_scale = m > 0 ? 1.0 : 1
            return y ~ Normal(m, body_scale)
        end
        @model function unstable_observation(y::Int)
            unwrapped_body = y
            return unwrapped_body
        end
        unwrapped = unstable_observation(1)
        unwrapped_code, _ = DynamicPPL.DebugUtils.model_typed(
            unwrapped, VarInfo(Xoshiro(1), unwrapped), false
        )
        @test :unwrapped_body in unwrapped_code.slotnames
        for original in (
                unstable_observation(1.0),
                unstable_observation(1.0, 2; center=3.0, other=4),
                unstable_keyword(),
            ),
            model in (original, condition(original; y=1.0f0))

            vi = VarInfo(Xoshiro(1), model)
            codeinfo, retype = DynamicPPL.DebugUtils.model_typed(model, vi, false)
            @test :body_scale in codeinfo.slotnames
            @test retype <: Tuple{typeof(conditioned(model)[@varname(y)]),VarInfo}
            mktemp() do _, io
                redirect_stdout(io) do
                    DynamicPPL.DebugUtils.model_warntype(model, vi)
                end
                seekstart(io)
                @test occursin("body_scale", read(io, String))
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
