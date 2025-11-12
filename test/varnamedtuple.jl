module VarNamedTupleTests

using Test: @inferred, @testset, @test
using DynamicPPL: @varname, VarNamedTuple
using BangBang: setindex!!

@testset "Basic sets and gets" begin
    vnt = VarNamedTuple()
    vnt = @inferred(setindex!!(vnt, 32.0, @varname(a)))
    @test @inferred(getindex(vnt, @varname(a))) == 32.0

    vnt = @inferred(setindex!!(vnt, [1, 2, 3], @varname(b)))
    @test @inferred(getindex(vnt, @varname(b))) == [1, 2, 3]

    vnt = @inferred(setindex!!(vnt, 64.0, @varname(a)))
    @test @inferred(getindex(vnt, @varname(a))) == 64.0

    vnt = @inferred(setindex!!(vnt, 15, @varname(b[2])))
    @test @inferred(getindex(vnt, @varname(b))) == [1, 15, 3]
    @test @inferred(getindex(vnt, @varname(b[2]))) == 15

    vnt = @inferred(setindex!!(vnt, [10], @varname(c.x.y)))
    @test @inferred(getindex(vnt, @varname(c.x.y))) == [10]

    vnt = @inferred(setindex!!(vnt, 11, @varname(c.x.y[1])))
    @test @inferred(getindex(vnt, @varname(c.x.y))) == [11]
    @test @inferred(getindex(vnt, @varname(c.x.y[1]))) == 11

    vnt = @inferred(setindex!!(vnt, -1.0, @varname(d[4])))
    @test @inferred(getindex(vnt, @varname(d[4]))) == -1.0

    vnt = @inferred(setindex!!(vnt, -2.0, @varname(d[4])))
    @test @inferred(getindex(vnt, @varname(d[4]))) == -2.0

    # These can't be @inferred because `d` now has an abstract element type. Note that this
    # does not ruin type stability for other varnames that don't involve `d`.
    vnt = setindex!!(vnt, "a", @varname(d[5]))
    @test getindex(vnt, @varname(d[5])) == "a"

    vnt = @inferred(setindex!!(vnt, 1.0, @varname(e.f[3].g.h[2].i)))
    @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 1.0

    vnt = @inferred(setindex!!(vnt, 2.0, @varname(e.f[3].g.h[2].i)))
    @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 2.0
end

end
