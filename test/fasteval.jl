module DynamicPPLFastEvalTests

using AbstractPPL: AbstractPPL
using Chairmarks
using DynamicPPL
using Distributions
using DistributionsAD: filldist
using ADTypes
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult, NoTest
using LinearAlgebra: I
using Test
using LogDensityProblems: LogDensityProblems

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

@testset "LogDensityFunction: Correctness" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        @testset "$varinfo_func" for varinfo_func in [
            DynamicPPL.untyped_varinfo,
            DynamicPPL.typed_varinfo,
            DynamicPPL.untyped_vector_varinfo,
            DynamicPPL.typed_vector_varinfo,
        ]
            unlinked_vi = varinfo_func(m)
            @testset "$islinked" for islinked in (false, true)
                vi = if islinked
                    DynamicPPL.link!!(unlinked_vi, m)
                else
                    unlinked_vi
                end
                nt_ranges, dict_ranges = DynamicPPL.get_ranges_and_linked(vi)
                params = [x for x in vi[:]]
                # Iterate over all variables
                for vn in keys(vi)
                    # Check that `getindex_internal` returns the same thing as using the ranges
                    # directly
                    range_with_linked = if AbstractPPL.getoptic(vn) === identity
                        nt_ranges[AbstractPPL.getsym(vn)]
                    else
                        dict_ranges[vn]
                    end
                    @test params[range_with_linked.range] ==
                        DynamicPPL.getindex_internal(vi, vn)
                    # Check that the link status is correct
                    @test range_with_linked.is_linked == islinked
                end
            end
        end
    end

    @testset "Threaded observe" begin
        if Threads.nthreads() > 1
            @model function threaded(y)
                x ~ Normal()
                Threads.@threads for i in eachindex(y)
                    y[i] ~ Normal(x)
                end
            end
            N = 100
            model = threaded(zeros(N))
            ldf = DynamicPPL.LogDensityFunction(model)

            xs = [1.0]
            @test LogDensityProblems.logdensity(ldf, xs) ≈
                logpdf(Normal(), xs[1]) + N * logpdf(Normal(xs[1]), 0.0)
        end
    end
end

@testset "LogDensityFunction: interface" begin
    # miscellaneous parts of the LogDensityProblems interface
    @testset "dimensions" begin
        @model function m1()
            x ~ Normal()
            y ~ Normal()
            return nothing
        end
        model = m1()
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.dimension(ldf) == 2

        @model function m2()
            x ~ Dirichlet(ones(4))
            y ~ Categorical(x)
            return nothing
        end
        model = m2()
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.dimension(ldf) == 5
        linked_vi = DynamicPPL.link!!(VarInfo(model), model)
        ldf = DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vi)
        @test LogDensityProblems.dimension(ldf) == 4
    end

    @testset "capabilities" begin
        @model f() = x ~ Normal()
        model = f()
        # No adtype
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{0}()
        # With adtype
        ldf = DynamicPPL.LogDensityFunction(model; adtype=AutoForwardDiff())
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{1}()
    end
end

@testset "LogDensityFunction: performance" begin
    if Threads.nthreads() == 1
        # Evaluating these three models should not lead to any allocations (but only when
        # not using TSVI).
        @model function f()
            x ~ Normal()
            return 1.0 ~ Normal(x)
        end
        @model function submodel_inner()
            m ~ Normal(0, 1)
            s ~ Exponential()
            return (m=m, s=s)
        end
        # Note that for the allocation tests to work on this one, `inner` has
        # to be passed as an argument to `submodel_outer`, instead of just
        # being called inside the model function itself
        @model function submodel_outer(inner)
            params ~ to_submodel(inner)
            y ~ Normal(params.m, params.s)
            return 1.0 ~ Normal(y)
        end
        @testset for model in
                     (f(), submodel_inner() | (; s=0.0), submodel_outer(submodel_inner()))
            vi = VarInfo(model)
            ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi)
            x = vi[:]
            bench = median(@be LogDensityProblems.logdensity(ldf, x))
            @test iszero(bench.allocs)
        end
    end
end

@testset "AD with LogDensityFunction" begin
    # Used as the ground truth that others are compared against.
    ref_adtype = AutoForwardDiff()

    test_adtypes = [
        AutoReverseDiff(; compile=false),
        AutoReverseDiff(; compile=true),
        AutoMooncake(; config=nothing),
    ]

    @testset "Correctness" begin
        @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
            varinfo = VarInfo(m)
            linked_varinfo = DynamicPPL.link(varinfo, m)
            f = LogDensityFunction(m, getlogjoint_internal, linked_varinfo)
            x = [p for p in linked_varinfo[:]]

            # Calculate reference logp + gradient of logp using ForwardDiff
            ref_ad_result = run_ad(m, ref_adtype; varinfo=linked_varinfo, test=NoTest())
            ref_logp, ref_grad = ref_ad_result.value_actual, ref_ad_result.grad_actual

            @testset "$adtype" for adtype in test_adtypes
                @info "Testing AD on: $(m.f) - $adtype"

                @test run_ad(
                    m,
                    adtype;
                    varinfo=linked_varinfo,
                    test=WithExpectedResult(ref_logp, ref_grad),
                ) isa Any
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

end
