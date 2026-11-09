module DynamicPPLDebugUtilsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL, Distributions, Test
using AbstractPPL: of, @of
using LinearAlgebra: I
using OrderedCollections: OrderedDict
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
                    args_on_lhs=DynamicPPL._lhs_arguments(__model__),
                )
                @test check_model(child)
            end
            μ ~ Normal()
            y ~ Normal(μ)
        end
        @model checked_parent() = a ~ to_submodel(condition(checked_child(missing); y=1.0))
        @test VarInfo(checked_parent()) isa VarInfo
    end

    @testset "binding warnings" begin
        @model warning_leaf() = x ~ Normal()
        @model function warning_parent(child, run=true)
            z ~ Normal()
            if run
                a ~ to_submodel(child)
            end
        end
        @model function warning_indices(run=true)
            z ~ Normal()
            x = zeros(2)
            if run
                x[1] ~ Normal()
            end
            return x
        end

        @testset "$bind" for (bind, remove) in ((condition, decondition), (fix, unfix))
            @testset "unused removals" begin
                unused = remove(warning_parent(warning_leaf()), @varname(a.x))
                used = remove(warning_parent(bind(warning_leaf(); x=1.0)), @varname(a.x))
                dormant = remove(
                    warning_parent(bind(warning_leaf(); x=1.0), false), @varname(a.x)
                )
                local_removal = remove(
                    bind(warning_parent(warning_leaf()), @varname(a.x) => 1.0),
                    @varname(a.x)
                )
                for (unused, used) in (
                    (unused, used),
                    (warning_parent(unused), warning_parent(used)),
                    (dormant, local_removal),
                )
                    @test_logs (:warn, r"Removal.*`a\.x`.*unused") match_mode = :any @test check_model(
                        Xoshiro(1), unused; error_on_failure=true
                    )
                    @test_logs @test check_model(Xoshiro(1), used; error_on_failure=true)
                end
            end

            @testset "bindings in reached and dormant submodels" begin
                used = bind(warning_parent(warning_leaf()), @varname(a.x) => 1.0)
                dormant = bind(warning_parent(warning_leaf(), false), @varname(a.x) => 1.0)
                @test_logs @test check_model(Xoshiro(1), used; error_on_failure=true)
                @test_logs (:warn, r"Partial binding `a\.x`.*not read") match_mode = :any @test check_model(
                    Xoshiro(1), dormant; error_on_failure=true
                )
            end

            @testset "unread partial bindings" begin
                for (model, vn, warning) in (
                    (
                        warning_indices(),
                        @varname(x[2]),
                        r"Partial binding `x\[2\]`.*not read",
                    ),
                    (
                        warning_indices(false),
                        @varname(x[1]),
                        r"Partial binding `x\[1\]`.*not read",
                    ),
                )
                    unused = bind(model, vn => 1.0, @of(x = of(Array, 2)))
                    @test_logs (:warn, warning) match_mode = :any @test check_model(
                        Xoshiro(1), unused; error_on_failure=true
                    )
                end
                used = bind(warning_indices(), @varname(x[1]) => 1.0, @of(x = of(Array, 2)))
                @test_logs @test check_model(Xoshiro(1), used; error_on_failure=true)
            end

            @testset "unused template entries" begin
                unused = bind(warning_indices(), @of(typo = of(Array, 2)))
                @test_logs (:warn, r"Binding template entry `typo`.*no LHS") match_mode =
                    :any @test check_model(Xoshiro(1), unused; error_on_failure=true)
                used = bind(warning_indices(), @of(x = of(Array, 2)))
                @test_logs @test check_model(Xoshiro(1), used; error_on_failure=true)
            end
        end
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
                # Reusing one child address duplicates its latent variable.
                z ~ to_submodel(ModelInner())
                return z ~ to_submodel(ModelInner())
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
                x1 ~ to_submodel(DynamicPPL.prefix(ModelInner(), :a))
                x2 ~ to_submodel(DynamicPPL.prefix(ModelInner(), :b))
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

    @testset "missing in multivariate" begin
        @model multivariate_missing(x) = x ~ MvNormal(zeros(length(x)), I)
        @test_throws ArgumentError check_model(multivariate_missing([1.0, missing]))
        @model missing_slice(y) = y[1:1] ~ MvNormal(zeros(1), I)
        model = missing_slice([missing])
        @test_throws ArgumentError check_model(model; error_on_failure=true)
        @test_throws ArgumentError model(Xoshiro(1))
        @test_throws ArgumentError rand(Xoshiro(1), model)

        @model function slice(x=missing)
            if ismissing(x)
                x = Matrix{Union{Missing,Int}}(missing, 2, 2)
            end
            x[1, :] ~ product_distribution([Poisson(3.0), Poisson(3.0)])
            return x
        end
        model = slice()
        @test isequal(model(Xoshiro(1)), [2 0; missing missing])
        @test rand(Xoshiro(1), model)[@varname(x[1, :])] == [2, 0]
        @test check_model(Xoshiro(1), model; error_on_failure=true)
    end

    @testset "unused templates on placeholder slices" begin
        @model function slice2(x=missing)
            y ~ Normal()
            if x isa Union{Missing,Nothing}
                x = Matrix{Union{Missing,Int}}(missing, 2, 2)
            end
            x[1, :] ~ product_distribution([Poisson(3.0), Poisson(3.0)])
            return x
        end
        @model parent(child) = a ~ to_submodel(child)
        for placeholder in (missing, nothing), bind in (condition, fix)
            child = slice2(placeholder)
            expected = bind(child; y=1.0)(Xoshiro(1))
            templated = bind(child, (; y=1.0), @of(x = of(Array, Int, 2, 2)))
            @test isequal(templated(Xoshiro(1)), expected)
            nested = bind(
                parent(child), @varname(a.y) => 1.0, @of(a = @of(x = of(Array, Int, 2, 2)))
            )
            @test isequal(nested(Xoshiro(1)), expected)
        end
    end

    @testset "conditioning overrides argument values" begin
        @model function demo_conditioned_argument(x)
            return x ~ Normal()
        end
        model = demo_conditioned_argument(1.0)
        @test check_model(condition(model; x=2.0); error_on_failure=true)
        for values in (OrderedDict(@varname(x) => 2.0), OrderedDict(@varname(x[1]) => 2.0))
            @test_throws ArgumentError condition(model, values)
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

    @testset "model_typed exposes the body with an argument on the LHS" begin
        @model function demo_argument_lhs(x)
            x ~ Normal()
            inner_body_value = sin(x)
            return inner_body_value
        end
        model = demo_argument_lhs(1.0)
        codeinfo, _ = DynamicPPL.DebugUtils.model_typed(model, VarInfo(model), false)
        @test :inner_body_value in codeinfo.slotnames
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

@testset "unused templates after a metadata-free model" begin
    @model unused_template() = x ~ Normal()
    model = condition(unused_template(), (x=1.0,), @of(typo = of(Array, Float64, 2)))
    manual = Model{false}((args...) -> nothing, (;), (;))
    ctx = DynamicPPL.DebugUtils.BindingCheckContext(DefaultContext())
    push!(ctx.models, (manual, VarName[]), (model, VarName[]))
    @test_logs (:warn, r"Binding template entry `typo`") DynamicPPL.DebugUtils._warn_unused_templates(
        ctx
    )
end

end # module
