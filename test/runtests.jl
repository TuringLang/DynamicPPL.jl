using Test
include("./Turing/Turing.jl")
using .Turing

@testset "DynamicPPL.jl" begin
    include("compiler.jl")
    include("varinfo.jl")
    include("prob_macro.jl")
end
