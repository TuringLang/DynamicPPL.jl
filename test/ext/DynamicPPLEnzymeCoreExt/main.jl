using DynamicPPL: @model
using DynamicPPL.TestUtils: ALL_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using ADTypes: AutoEnzyme
using Distributions: Normal
using DifferentiationInterface: DifferentiationInterface
using Test: @test, @testset
import Enzyme: set_runtime_activity, Forward, Reverse, Const
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test

ADTYPES = (
    ("EnzymeForward", AutoEnzyme(; mode=set_runtime_activity(Forward))),
    ("EnzymeReverse", AutoEnzyme(; mode=set_runtime_activity(Reverse))),
)

# Tuple element storage, which no `ALL_MODELS` model exercises.
@model function tuple_local()
    x = (0.0, 0.0)
    x[1] ~ Normal()
    x[2] ~ Normal(x[1])
    return x
end

@testset "$ad_key" for (ad_key, ad_type) in ADTYPES
    @testset "$(model.f)" for model in (ALL_MODELS..., tuple_local())
        @test run_ad(model, ad_type) isa Any
    end
end
