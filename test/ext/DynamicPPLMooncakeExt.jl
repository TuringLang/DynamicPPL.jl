@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(StableRNG(123456), istrans, VarInfo(); unsafe_perturb=true, interface_only=true)
end
