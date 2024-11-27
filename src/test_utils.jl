module TestUtils

using DynamicPPL
using LinearAlgebra
using Distributions

using Random: Random
using Bijectors: Bijectors

include("test_utils/model_interface.jl")
include("test_utils/models.jl")

##############################################################
# The remainder of this file contains skeleton implementations for
# DynamicPPLTestExt
##############################################################

function test_context_interface end

"""
Context that multiplies each log-prior by mod
used to test whether varwise_logpriors respects child-context.
"""
struct TestLogModifyingChildContext{T,Ctx} <: DynamicPPL.AbstractContext
    mod::T
    context::Ctx
end

function marginal_mean_of_samples end
function test_sampler end
function test_sampler_on_demo_models end
function test_sampler_continuous end
function test_values end
function setup_varinfos end

end
