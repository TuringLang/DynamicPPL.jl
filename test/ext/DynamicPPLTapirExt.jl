@testset "DynamicPPLTapirExt" begin
    Tapir.TestUtils.test_rule(
        Xoshiro(123), istrans, VarInfo();
        perf_flag=:none,
        interface_only=true,
        is_primitive=true,
        interp=Tapir.TapirInterpreter(),
    )
end
