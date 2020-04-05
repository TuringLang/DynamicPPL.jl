using Test, DynamicPPL
dir = splitdir(splitdir(pathof(DynamicPPL))[1])[1]
include(dir*"/test/Turing/Turing.jl")
using .Turing

turnprogress(false)

@testset "DynamicPPL.jl" begin
    include("utils.jl")
    include("compiler.jl")
    include("varinfo.jl")
    include("prob_macro.jl")
    include("independence.jl")
end
