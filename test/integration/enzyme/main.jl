using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using ADTypes: AutoEnzyme
using Test: @test, @testset
import Enzyme: set_runtime_activity, Forward, Reverse
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test

ADTYPES = Dict(
    "EnzymeForward" => AutoEnzyme(; mode=set_runtime_activity(Forward)),
    "EnzymeReverse" => AutoEnzyme(; mode=set_runtime_activity(Reverse)),
)

@testset "$ad_key" for (ad_key, ad_type) in ADTYPES
    @testset "$(model.f)" for model in DEMO_MODELS
        @test run_ad(model, ad_type) isa Any
    end
end
