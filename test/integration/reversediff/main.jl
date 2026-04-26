using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DynamicPPL
using DynamicPPL.TestUtils: ALL_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using Distributions: Normal
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test
using LogDensityProblems: LogDensityProblems
using ReverseDiff: ReverseDiff
using Test: @test, @testset

ADTYPES = (
    ("ReverseDiff", AutoReverseDiff(; compile=false)),
    ("ReverseDiffCompiled", AutoReverseDiff(; compile=true)),
)

@testset "$ad_key" for (ad_key, ad_type) in ADTYPES
    @testset "$(model.f)" for model in ALL_MODELS
        @test run_ad(model, ad_type) isa Any
    end
end

@testset "ReverseDiff compiled prep reduces repeated-call allocations" begin
    @model f() = x ~ Normal()
    ldf_compiled = LogDensityFunction(
        f(), getlogjoint_internal, LinkAll(); adtype=AutoReverseDiff(; compile=true)
    )
    ldf_uncompiled = LogDensityFunction(
        f(), getlogjoint_internal, LinkAll(); adtype=AutoReverseDiff(; compile=false)
    )
    params = rand(ldf_compiled)

    LogDensityProblems.logdensity_and_gradient(ldf_compiled, params)
    LogDensityProblems.logdensity_and_gradient(ldf_uncompiled, params)

    function repeated_call_allocs(ldf, params)
        GC.gc()
        before = Base.gc_num()
        for _ in 1:100
            LogDensityProblems.logdensity_and_gradient(ldf, params)
        end
        after = Base.gc_num()
        return Base.GC_Diff(after, before).allocd
    end

    allocs_compiled = repeated_call_allocs(ldf_compiled, params)
    allocs_uncompiled = repeated_call_allocs(ldf_uncompiled, params)

    @test allocs_compiled < allocs_uncompiled
end
