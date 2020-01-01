using Test, DynamicPPL
dir = splitdir(splitdir(pathof(DynamicPPL))[1])[1]
include(dir*"/test/Turing/Turing.jl")
using .Turing

@testset "DynamicPPL.jl" begin
    #include("compiler2.jl")
    #include("varinfo.jl")
    #include("prob_macro.jl")
end
