using DynamicPPL: LogDensityFunction
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult, NoTest

@testset "Automatic differentiation" begin
    # Used as the ground truth that others are compared against.
    ref_adtype = AutoForwardDiff()

    test_adtypes = [
        AutoReverseDiff(; compile=false),
        AutoReverseDiff(; compile=true),
        AutoMooncake(; config=nothing),
    ]

    @testset "Unsupported backends" begin
        @model demo() = x ~ Normal()
        @test_logs (:warn, r"not officially supported") LogDensityFunction(
            demo(); adtype=AutoZygote()
        )
    end

    @testset "Correctness" begin
        @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
            rand_param_values = DynamicPPL.TestUtils.rand_prior_true(m)
            vns = DynamicPPL.TestUtils.varnames(m)
            varinfos = DynamicPPL.TestUtils.setup_varinfos(m, rand_param_values, vns)

            @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                linked_varinfo = DynamicPPL.link(varinfo, m)
                f = LogDensityFunction(m, getlogjoint_internal, linked_varinfo)
                x = DynamicPPL.getparams(f)

                # Calculate reference logp + gradient of logp using ForwardDiff
                ref_ad_result = run_ad(m, ref_adtype; varinfo=linked_varinfo, test=NoTest())
                ref_logp, ref_grad = ref_ad_result.value_actual, ref_ad_result.grad_actual

                @testset "$adtype" for adtype in test_adtypes
                    @info "Testing AD on: $(m.f) - $(short_varinfo_name(linked_varinfo)) - $adtype"

                    # Put predicates here to avoid long lines
                    is_mooncake = adtype isa AutoMooncake
                    is_1_10 = v"1.10" <= VERSION < v"1.11"
                    is_1_11_or_1_12 = v"1.11" <= VERSION < v"1.13"
                    is_svi_vnv =
                        linked_varinfo isa SimpleVarInfo{<:DynamicPPL.VarNamedVector}
                    is_svi_od = linked_varinfo isa SimpleVarInfo{<:OrderedDict}

                    # Mooncake doesn't work with several combinations of SimpleVarInfo.
                    if is_mooncake && is_1_11_or_1_12 && is_svi_vnv
                        # https://github.com/compintell/Mooncake.jl/issues/470
                        @test_throws ArgumentError DynamicPPL.LogDensityFunction(
                            m, getlogjoint_internal, linked_varinfo; adtype=adtype
                        )
                    elseif is_mooncake && is_1_10 && is_svi_vnv
                        # TODO: report upstream
                        @test_throws UndefRefError DynamicPPL.LogDensityFunction(
                            m, getlogjoint_internal, linked_varinfo; adtype=adtype
                        )
                    elseif is_mooncake && is_1_10 && is_svi_od
                        # TODO: report upstream
                        @test_throws Mooncake.MooncakeRuleCompilationError DynamicPPL.LogDensityFunction(
                            m, getlogjoint_internal, linked_varinfo; adtype=adtype
                        )
                    else
                        @test run_ad(
                            m,
                            adtype;
                            varinfo=linked_varinfo,
                            test=WithExpectedResult(ref_logp, ref_grad),
                        ) isa Any
                    end
                end
            end
        end
    end

    # Test that various different ways of specifying array types as arguments work with all
    # ADTypes.
    @testset "Array argument types" begin
        test_m = randn(2, 3)

        function eval_logp_and_grad(model, m, adtype)
            ldf = LogDensityFunction(model(); adtype=adtype)
            return LogDensityProblems.logdensity_and_gradient(ldf, m[:])
        end

        @model function scalar_matrix_model(::Type{T}=Float64) where {T<:Real}
            m = Matrix{T}(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        scalar_matrix_model_reference = eval_logp_and_grad(
            scalar_matrix_model, test_m, ref_adtype
        )

        @model function matrix_model(::Type{T}=Matrix{Float64}) where {T}
            m = T(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        matrix_model_reference = eval_logp_and_grad(matrix_model, test_m, ref_adtype)

        @model function scalar_array_model(::Type{T}=Float64) where {T<:Real}
            m = Array{T}(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        scalar_array_model_reference = eval_logp_and_grad(
            scalar_array_model, test_m, ref_adtype
        )

        @model function array_model(::Type{T}=Array{Float64}) where {T}
            m = T(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        array_model_reference = eval_logp_and_grad(array_model, test_m, ref_adtype)

        @testset "$adtype" for adtype in test_adtypes
            scalar_matrix_model_logp_and_grad = eval_logp_and_grad(
                scalar_matrix_model, test_m, adtype
            )
            @test scalar_matrix_model_logp_and_grad[1] ≈ scalar_matrix_model_reference[1]
            @test scalar_matrix_model_logp_and_grad[2] ≈ scalar_matrix_model_reference[2]
            matrix_model_logp_and_grad = eval_logp_and_grad(matrix_model, test_m, adtype)
            @test matrix_model_logp_and_grad[1] ≈ matrix_model_reference[1]
            @test matrix_model_logp_and_grad[2] ≈ matrix_model_reference[2]
            scalar_array_model_logp_and_grad = eval_logp_and_grad(
                scalar_array_model, test_m, adtype
            )
            @test scalar_array_model_logp_and_grad[1] ≈ scalar_array_model_reference[1]
            @test scalar_array_model_logp_and_grad[2] ≈ scalar_array_model_reference[2]
            array_model_logp_and_grad = eval_logp_and_grad(array_model, test_m, adtype)
            @test array_model_logp_and_grad[1] ≈ array_model_reference[1]
            @test array_model_logp_and_grad[2] ≈ array_model_reference[2]
        end
    end
end
