module DynamicPPLFastLDFTests

using AbstractPPL: AbstractPPL
using DynamicPPL
using Distributions
using DistributionsAD: filldist
using ADTypes
using DynamicPPL.Experimental: FastLDF
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult, NoTest
using LinearAlgebra: I
using Test
using LogDensityProblems: LogDensityProblems

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
# Need to include this block here in case we run this test file standalone
@static if VERSION < v"1.12"
    using Pkg
    Pkg.add("Mooncake")
    using Mooncake: Mooncake
end

@testset "get_ranges_and_linked" begin
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
                nt_ranges, dict_ranges = DynamicPPL.Experimental.get_ranges_and_linked(vi)
                params = vi[:]
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
end

@testset "AD with FastLDF" begin
    # Used as the ground truth that others are compared against.
    ref_adtype = AutoForwardDiff()

    test_adtypes = @static if VERSION < v"1.12"
        [
            AutoReverseDiff(; compile=false),
            AutoReverseDiff(; compile=true),
            AutoMooncake(; config=nothing),
        ]
    else
        [AutoReverseDiff(; compile=false), AutoReverseDiff(; compile=true)]
    end

    @testset "Correctness" begin
        @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
            varinfo = VarInfo(m)
            linked_varinfo = DynamicPPL.link(varinfo, m)
            f = FastLDF(m, getlogjoint_internal, linked_varinfo)
            x = linked_varinfo[:]

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
            ldf = FastLDF(model(); adtype=adtype)
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
