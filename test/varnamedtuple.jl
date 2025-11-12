module VarNamedTupleTests

using Test: @testset, @test
using DynamicPPL: @varname, VarNamedTuple
using BangBang: setindex!!

@testset "Basic sets and gets" begin
    vnt = VarNamedTuple()
    vnt = setindex!!(vnt, 32.0, @varname(a))
    @test getindex(vnt, @varname(a)) == 32.0

    vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
    @test getindex(vnt, @varname(b)) == [1, 2, 3]

    vnt = setindex!!(vnt, 64.0, @varname(a))
    @test getindex(vnt, @varname(a)) == 64.0

    vnt = setindex!!(vnt, 15, @varname(b[2]))
    @test getindex(vnt, @varname(b)) == [1, 15, 3]
    @test getindex(vnt, @varname(b[2])) == 15

    vnt = setindex!!(vnt, [10], @varname(c.x.y))
    @test getindex(vnt, @varname(c.x.y)) == [10]

    vnt = setindex!!(vnt, 11, @varname(c.x.y[1]))
    @test getindex(vnt, @varname(c.x.y)) == [11]
    @test getindex(vnt, @varname(c.x.y[1])) == 11

    vnt = setindex!!(vnt, -1.0, @varname(d[4]))
    @test getindex(vnt, @varname(d[4])) == -1.0

    vnt = setindex!!(vnt, -2.0, @varname(d[4]))
    @test getindex(vnt, @varname(d[4])) == -2.0

    vnt = setindex!!(vnt, -3.0, @varname(d[5]))
    @test getindex(vnt, @varname(d[5])) == -3.0

    vnt = setindex!!(vnt, 1.0, @varname(e.f[3].g.h[2].i))
    @test getindex(vnt, @varname(e.f[3].g.h[2].i)) == 1.0
end

end
