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

    vec = fill(1.0, 4)
    vnt = @inferred(setindex!!(vnt, vec, @varname(j[1:4])))
    @test @inferred(getindex(vnt, @varname(j[1:4]))) == vec

    vec = fill(2.0, 4)
    vnt = @inferred(setindex!!(vnt, vec, @varname(j[2:5])))
    @test @inferred(getindex(vnt, @varname(j[1]))) == 1.0
    @test @inferred(getindex(vnt, @varname(j[2:5]))) == vec

    arr = fill(2.0, (4, 2))
    vn = @varname(k.l[2:5, 3, 1:2, 10])
    vnt = @inferred(setindex!!(vnt, arr, vn))
    @test @inferred(getindex(vnt, vn)) == arr
    @test @inferred(getindex(vnt, @varname(k.l[2, 3, 1:2, 10]))) == fill(2.0, 2)

    vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[2])))
    vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[3])))
    @test @inferred(getindex(vnt, @varname(m[2:3]))) == [1.0, 1.0]
    @test !haskey(vnt, @varname(m[1]))
end

end
