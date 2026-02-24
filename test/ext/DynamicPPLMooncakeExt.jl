module DynamicPPLMooncakeExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Mooncake: Mooncake
using StableRNGs: StableRNG
using DynamicPPL: is_transformed, VarInfo
using Test: @testset

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(
        StableRNG(123456),
        is_transformed,
        VarInfo();
        unsafe_perturb=true,
        interface_only=true,
    )
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
