module DynamicPPLDebugUtilsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL, Distributions, Test
using ForwardDiff: ForwardDiff
using AbstractPPL: of, @of
using LinearAlgebra: I
using Random: Xoshiro

@testset "unused binding template entries" begin
    @model function template_child(x=zeros(2))
        y = zeros(3)
        for i in eachindex(y)
            y[i] ~ Normal()
        end
        x[1] ~ Normal()
        return y
    end
    @model function template_parent(run=true)
        z ~ Normal()
        if run
            a ~ to_submodel(template_child())
        end
    end
    @model template_shared() = a ~ to_submodel(template_child(), false)
    @model template_outer(child) = b ~ to_submodel(child)
    @model function template_fields()
        z = (a=zeros(2),)
        z.a[1] ~ Normal()
        return z
    end
    @model template_field_child() = a ~ to_submodel(template_fields())
    @model template_field_shared() = a ~ to_submodel(template_fields(), false)
    for bind in (condition, fix)
        for (m, valid, invalid, warning) in (
            (
                template_fields(),
                @of(z = @of(a = of(Array, 2))),
                @of(z = @of(typo = of(Array, 2))),
                r"Binding template entry `z.typo`",
            ),
            (
                template_field_shared(),
                @of(z = @of(a = of(Array, 2))),
                @of(z = @of(typo = of(Array, 2))),
                r"Binding template entry `z.typo`",
            ),
            (
                template_field_child(),
                @of(a = @of(z = @of(a = of(Array, 2)))),
                @of(a = @of(z = @of(typo = of(Array, 2)))),
                r"Binding template entry `a.z.typo`",
            ),
            (
                prefix(template_fields(), @varname(p)),
                @of(p = @of(z = @of(a = of(Array, 2)))),
                @of(p = @of(z = @of(typo = of(Array, 2)))),
                r"Binding template entry `p.z.typo`",
            ),
        )
            @test_logs @test check_model(Xoshiro(1), bind(m, valid))
            @test_logs (:warn, warning) @test check_model(Xoshiro(1), bind(m, invalid))
        end
        m = bind(template_child(), @of(y = of(Array, 3), x = of(Array, 100)))
        @test_logs @test check_model(Xoshiro(1), m)
        typo = bind(template_child(), @of(typo = of(Array, 3)))
        @test_logs (:warn, r"Binding template entry `typo` has no LHS variable") @test check_model(
            Xoshiro(1), typo
        )
        @test_logs (:warn, r"Binding template entry `p.typo`") @test check_model(
            Xoshiro(1), prefix(typo, @varname(p))
        )
        @test_logs (:warn, r"Binding template entry `b.typo`") @test check_model(
            Xoshiro(1), template_outer(typo)
        )
        pm = prefix(template_child(), @varname(p))
        @test_logs @test check_model(Xoshiro(1), bind(pm, @of(p = @of(y = of(Array, 3)))))
        @test_logs (:warn, r"Binding template entry `p.typo`") @test check_model(
            Xoshiro(1), bind(pm, @of(p = @of(typo = of(Int))))
        )
        for run in (true, false)
            nested = bind(template_parent(run), @of(a = @of(y = of(Array, 3))))
            if run
                @test_logs @test check_model(Xoshiro(1), nested)
            else
                @test_logs (:warn, r"Binding template entry `a.y`") @test check_model(
                    Xoshiro(1), nested
                )
            end
        end
        nested = bind(template_parent(), @of(a = @of(typo = of(Int))))
        @test_logs (:warn, r"Binding template entry `a.typo`") @test check_model(
            Xoshiro(1), nested
        )
        shared = bind(template_shared(), @varname(y[3]) => 0.5, @of(y = of(Array, 3)))
        @test_logs @test check_model(Xoshiro(1), shared)
        @test_logs (:warn, r"Binding template entry `typo`") @test check_model(
            Xoshiro(1), bind(template_shared(), @of(typo = of(Int)))
        )
        @test_logs (:warn, r"Binding template entry `typo`") @test check_model(
            Xoshiro(1), bind(typo, @of(typo = of(Int)))
        )
    end
end

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
            x ~ truncated(Normal(); lower=0)
            w ~ filldist(Normal(), 2)
            if run
                a ~ to_submodel(child, false)
            end
        end
        warning = r"Binding `zz` has no LHS top symbol"
        for bind in (condition, fix)
            @test_logs (:warn, warning) @test check_model(
                Xoshiro(1), bind(binding_parent(); zz=1.0); error_on_failure=true
            )
            @test_logs @test check_model(Xoshiro(1), bind(binding_parent(); z=1.0))
            # Metadata includes untaken LHS branches in a reached child.
            @test_logs @test check_model(
                Xoshiro(1), bind(binding_parent(true, binding_child(false)); z=1.0)
            )
            # An unreached child can own a valid name, so this must only warn.
            @test_logs (:warn, r"Binding `z` has no LHS top symbol") @test check_model(
                Xoshiro(1), bind(binding_parent(false); z=1.0); error_on_failure=true
            )
            @test_throws ArgumentError bind(binding_child(); zz=1.0)
        end
        @model binding_middle() = b ~ to_submodel(binding_child(false), false)
        @test_logs @test check_model(
            Xoshiro(1), condition(binding_parent(true, binding_middle()); z=1.0)
        )
        @test_logs @test check_model(
            Xoshiro(1), prefix(condition(binding_parent(); z=1.0), @varname(p))
        )
        @model binding_prefixed() = b ~ to_submodel(binding_child(false))
        # A prefixed descendant's names cannot justify a binding in the parent namespace.
        for child in (binding_prefixed(), prefix(binding_middle(), @varname(p)))
            @test_logs (:warn, r"Binding `z` has no LHS top symbol") @test check_model(
                Xoshiro(1), condition(binding_parent(true, child); z=1.0)
            )
        end
        @test_logs @test check_model(
            Xoshiro(1),
            condition(
                binding_parent(true, prefix(binding_child(), @varname(p))),
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
                    __model__.prefix,
                    __model__.values,
                    __model__.context;
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

@testset "unused recursive removals" begin
    rec = DynamicPPL.Recursive()
    @model leaf(x) = x ~ Normal()
    @model function branch(run)
        z ~ Normal()
        if run
            a ~ to_submodel(leaf(1.0))
        end
        return nothing
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        parent = condition(branch(true), @varname(a.x) => 2.0)
        @test_logs (:warn, r"Recursive removal.*a.typo.*unused") check_model(
            Xoshiro(1), remove(parent, rec, @varname(a.typo))
        )
        @test_logs check_model(
            Xoshiro(1), remove(bind(branch(true), @varname(a.x) => 2.0), rec, @varname(a.x))
        )
    end
    unused = decondition(branch(false), rec, @varname(a.x))
    @test_logs (:warn, r"Recursive removal.*a.x.*unused") check_model(Xoshiro(1), unused)
    @test_logs check_model(Xoshiro(1), decondition(branch(true), rec, @varname(a.x)))
    @test_logs (:warn, r"Recursive removal.*p.a.x.*unused") check_model(
        Xoshiro(1), prefix(unused, @varname(p))
    )
    @model outer(m) = b ~ to_submodel(m)
    @test_logs (:warn, r"Recursive removal.*a.x.*unused") check_model(
        Xoshiro(1), outer(unused)
    )
    @test_logs check_model(Xoshiro(1), outer(decondition(branch(true), rec, @varname(a.x))))
end

end # module
