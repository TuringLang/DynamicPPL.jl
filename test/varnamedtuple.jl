module VarNamedTupleTests

using Combinatorics: Combinatorics
using Test: @inferred, @test, @test_throws, @testset
using Distributions: Dirichlet
using DynamicPPL: DynamicPPL, @varname, VarNamedTuple
using DynamicPPL.VarNamedTuples: PartialArray
using AbstractPPL: VarName, prefix
using BangBang: setindex!!

"""
    test_invariants(vnt::VarNamedTuple)

Test properties that should hold for all VarNamedTuples.

Uses @test for all the tests. Intended to be called inside a @testset.
"""
function test_invariants(vnt::VarNamedTuple)
    # Check that for all keys in vnt, haskey is true, and resetting the value is a no-op.
    for k in keys(vnt)
        @test haskey(vnt, k)
        v = getindex(vnt, k)
        vnt2 = setindex!!(copy(vnt), v, k)
        @test vnt == vnt2
        @test isequal(vnt, vnt2)
        @test hash(vnt) == hash(vnt2)
    end
    # Check that the printed representation can be parsed back to an equal VarNamedTuple.
    vnt3 = eval(Meta.parse(repr(vnt)))
    @test vnt == vnt3
    @test isequal(vnt, vnt3)
    @test hash(vnt) == hash(vnt3)
    # Check that merge with an empty VarNamedTuple is a no-op.
    @test merge(vnt, VarNamedTuple()) == vnt
    @test merge(VarNamedTuple(), vnt) == vnt
end

@testset "VarNamedTuple" begin
    @testset "Construction" begin
        vnt1 = VarNamedTuple()
        test_invariants(vnt1)
        vnt1 = setindex!!(vnt1, 1.0, @varname(a))
        vnt1 = setindex!!(vnt1, [1, 2, 3], @varname(b))
        vnt1 = setindex!!(vnt1, "a", @varname(c.d.e))
        test_invariants(vnt1)
        vnt2 = VarNamedTuple(;
            a=1.0, b=[1, 2, 3], c=VarNamedTuple(; d=VarNamedTuple(; e="a"))
        )
        test_invariants(vnt2)
        @test vnt1 == vnt2

        pa1 = PartialArray{Float64,1}()
        pa1 = setindex!!(pa1, 1.0, 16)
        pa2 = PartialArray{Float64,1}(; min_size=(16,))
        pa2 = setindex!!(pa2, 1.0, 16)
        pa3 = PartialArray{Float64,1}(16 => 1.0)
        pa4 = PartialArray{Float64,1}((16,) => 1.0)
        @test pa1 == pa2
        @test pa1 == pa3
        @test pa1 == pa4

        pa1 = PartialArray{String,3}()
        pa1 = setindex!!(pa1, "a", 2, 3, 4)
        pa1 = setindex!!(pa1, "b", 1, 2, 4)
        pa2 = PartialArray{String,3}(; min_size=(16, 16, 16))
        pa2 = setindex!!(pa2, "a", 2, 3, 4)
        pa2 = setindex!!(pa2, "b", 1, 2, 4)
        pa3 = PartialArray{String,3}((2, 3, 4) => "a", (1, 2, 4) => "b")
        @test pa1 == pa2
        @test pa1 == pa3

        @test_throws BoundsError PartialArray{Int,1}((0,) => 1)
        @test_throws BoundsError PartialArray{Int,1}((1, 2) => 1)
        @test_throws MethodError PartialArray{Int,1}((1,) => "a")
        @test_throws MethodError PartialArray{Int,1}((1,) => 1; min_size=(2, 2))
    end

    @testset "Basic sets and gets" begin
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 32.0, @varname(a)))
        @test @inferred(getindex(vnt, @varname(a))) == 32.0
        @test haskey(vnt, @varname(a))
        @test !haskey(vnt, @varname(b))
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, [1, 2, 3], @varname(b)))
        @test @inferred(getindex(vnt, @varname(b))) == [1, 2, 3]
        @test @inferred(getindex(vnt, @varname(b[2]))) == 2
        @test haskey(vnt, @varname(b))
        @test haskey(vnt, @varname(b[1]))
        @test haskey(vnt, @varname(b[1:3]))
        @test !haskey(vnt, @varname(b[4]))
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 64.0, @varname(a)))
        @test @inferred(getindex(vnt, @varname(a))) == 64.0
        @test @inferred(getindex(vnt, @varname(b))) == [1, 2, 3]
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 15, @varname(b[2])))
        @test @inferred(getindex(vnt, @varname(b))) == [1, 15, 3]
        @test @inferred(getindex(vnt, @varname(b[2]))) == 15
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, [10], @varname(c.x.y)))
        @test @inferred(getindex(vnt, @varname(c.x.y))) == [10]
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 11, @varname(c.x.y[1])))
        @test @inferred(getindex(vnt, @varname(c.x.y))) == [11]
        @test @inferred(getindex(vnt, @varname(c.x.y[1]))) == 11
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, -1.0, @varname(d[4])))
        @test @inferred(getindex(vnt, @varname(d[4]))) == -1.0
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, -2.0, @varname(d[4])))
        @test @inferred(getindex(vnt, @varname(d[4]))) == -2.0
        test_invariants(vnt)

        # These can't be @inferred because `d` now has an abstract element type. Note that this
        # does not ruin type stability for other varnames that don't involve `d`.
        vnt = setindex!!(vnt, "a", @varname(d[5]))
        @test getindex(vnt, @varname(d[5])) == "a"
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 1.0, @varname(e.f[3].g.h[2].i)))
        @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 1.0
        @test haskey(vnt, @varname(e.f[3].g.h[2].i))
        @test !haskey(vnt, @varname(e.f[2].g.h[2].i))
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 2.0, @varname(e.f[3].g.h[2].i)))
        @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 2.0
        test_invariants(vnt)

        vec = fill(1.0, 4)
        vnt = @inferred(setindex!!(vnt, vec, @varname(j[1:4])))
        @test @inferred(getindex(vnt, @varname(j[1:4]))) == vec
        @test @inferred(getindex(vnt, @varname(j[2]))) == vec[2]
        @test haskey(vnt, @varname(j[4]))
        @test !haskey(vnt, @varname(j[5]))
        @test_throws BoundsError getindex(vnt, @varname(j[5]))
        test_invariants(vnt)

        vec = fill(2.0, 4)
        vnt = @inferred(setindex!!(vnt, vec, @varname(j[2:5])))
        @test @inferred(getindex(vnt, @varname(j[1]))) == 1.0
        @test @inferred(getindex(vnt, @varname(j[2:5]))) == vec
        @test haskey(vnt, @varname(j[5]))
        test_invariants(vnt)

        arr = fill(2.0, (4, 2))
        vn = @varname(k.l[2:5, 3, 1:2, 2])
        vnt = @inferred(setindex!!(vnt, arr, vn))
        @test @inferred(getindex(vnt, vn)) == arr
        # A subset of the elements set just now.
        @test @inferred(getindex(vnt, @varname(k.l[2, 3, 1:2, 2]))) == fill(2.0, 2)
        test_invariants(vnt)

        # Not enough, or too many, indices.
        @test_throws BoundsError setindex!!(vnt, 0.0, @varname(k.l[1, 2, 3]))
        @test_throws BoundsError setindex!!(vnt, 0.0, @varname(k.l[1, 2, 3, 4, 5]))

        arr = fill(3.0, (3, 3))
        vn = @varname(k.l[1, 1:3, 1:3, 1])
        vnt = @inferred(setindex!!(vnt, arr, vn))
        @test @inferred(getindex(vnt, vn)) == arr
        # A subset of the elements set just now.
        @test @inferred(getindex(vnt, @varname(k.l[1, 1:2, 1:2, 1]))) == fill(3.0, 2, 2)
        # A subset of the elements set previously.
        @test @inferred(getindex(vnt, @varname(k.l[2, 3, 1:2, 2]))) == fill(2.0, 2)
        @test !haskey(vnt, @varname(k.l[2, 3, 3, 2]))
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[2])))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[3])))
        @test @inferred(getindex(vnt, @varname(m[2:3]))) == [1.0, 1.0]
        @test !haskey(vnt, @varname(m[1]))
        test_invariants(vnt)

        # The below tests are mostly significant for the type stability aspect. For the last
        # test to pass, PartialArray needs to actively tighten its eltype when possible.
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(n[1].a)))
        @test @inferred(getindex(vnt, @varname(n[1].a))) == 1.0
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(n[2].a)))
        @test @inferred(getindex(vnt, @varname(n[2].a))) == 1.0
        # This can't be type stable, because n[1] has inhomogeneous types.
        vnt = setindex!!(vnt, 1.0, @varname(n[1].b))
        @test getindex(vnt, @varname(n[1].b)) == 1.0
        # The setindex!! call can't be type stable either, but it should return a
        # VarNamedTuple with a concrete element type, and hence getindex can be inferred.
        vnt = setindex!!(vnt, 1.0, @varname(n[2].b))
        @test @inferred(getindex(vnt, @varname(n[2].b))) == 1.0
        test_invariants(vnt)

        # Some funky Symbols in VarNames
        # TODO(mhauru) This still isn't as robust as it should be, for instance Symbol(":")
        # fails the eval(Meta.parse(print(vnt))) == vnt test because NamedTuple show doesn't
        # respect the eval-property.
        vn1 = VarName{Symbol("a b c")}()
        vnt = @inferred(setindex!!(vnt, 2, vn1))
        @test @inferred(getindex(vnt, vn1)) == 2
        test_invariants(vnt)
        vn2 = VarName{Symbol("1")}()
        vnt = @inferred(setindex!!(vnt, 3, vn2))
        @test @inferred(getindex(vnt, vn2)) == 3
        test_invariants(vnt)
        vn3 = VarName{Symbol("?!")}()
        vnt = @inferred(setindex!!(vnt, 4, vn3))
        @test @inferred(getindex(vnt, vn3)) == 4
        test_invariants(vnt)
        vnt = VarNamedTuple()
        vn4 = prefix(prefix(vn1, vn2), vn3)
        vnt = @inferred(setindex!!(vnt, 5, vn4))
        @test @inferred(getindex(vnt, vn4)) == 5
        test_invariants(vnt)
        vn5 = prefix(prefix(vn3, vn2), vn1)
        vnt = @inferred(setindex!!(vnt, 6, vn5))
        @test @inferred(getindex(vnt, vn5)) == 6
        test_invariants(vnt)
    end

    @testset "equality and hash" begin
        # Test all combinations of having or not having the below values set, and having
        # them set to any of the possible_values, and check that isequal and == return the
        # expected value.
        # NOTE: Be very careful adding new values to these sets. The below test has three
        # nested loops over Combinatorics.combinations, the run time can explode very, very
        # quickly.
        varnames = (@varname(b[1]), @varname(b[3]), @varname(c.d[2].e))
        possible_values = (missing, 1, -0.0, 0.0)
        for vn_set in Combinatorics.combinations(varnames)
            valuesets1 = Combinatorics.with_replacement_combinations(
                possible_values, length(vn_set)
            )
            valuesets2 = Combinatorics.with_replacement_combinations(
                possible_values, length(vn_set)
            )
            for vset1 in valuesets1, vset2 in valuesets2
                vnt1 = VarNamedTuple()
                vnt2 = VarNamedTuple()
                expected_isequal = true
                expected_doubleequal = true
                for (vn, v1, v2) in zip(vn_set, vset1, vset2)
                    vnt1 = setindex!!(vnt1, v1, vn)
                    vnt2 = setindex!!(vnt2, v2, vn)
                    expected_isequal = expected_isequal & isequal(v1, v2)
                    expected_doubleequal = expected_doubleequal & (v1 == v2)
                end
                @test isequal(vnt1, vnt2) == expected_isequal
                @test (vnt1 == vnt2) === expected_doubleequal
                if expected_isequal
                    @test hash(vnt1) == hash(vnt2)
                end
            end
        end
    end

    @testset "merge" begin
        vnt1 = VarNamedTuple()
        vnt2 = VarNamedTuple()
        expected_merge = VarNamedTuple()
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        vnt1 = setindex!!(vnt1, 1.0, @varname(a))
        vnt2 = setindex!!(vnt2, 2.0, @varname(b))
        vnt1 = setindex!!(vnt1, 1, @varname(c))
        vnt2 = setindex!!(vnt2, 2, @varname(c))
        expected_merge = setindex!!(expected_merge, 1.0, @varname(a))
        expected_merge = setindex!!(expected_merge, 2, @varname(c))
        expected_merge = setindex!!(expected_merge, 2.0, @varname(b))
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        vnt1 = VarNamedTuple()
        vnt2 = VarNamedTuple()
        expected_merge = VarNamedTuple()
        vnt1 = setindex!!(vnt1, [1], @varname(d.a))
        vnt2 = setindex!!(vnt2, [2, 2], @varname(d.b))
        vnt1 = setindex!!(vnt1, [1], @varname(d.c))
        vnt2 = setindex!!(vnt2, [2, 2], @varname(d.c))
        expected_merge = setindex!!(expected_merge, [1], @varname(d.a))
        expected_merge = setindex!!(expected_merge, [2, 2], @varname(d.c))
        expected_merge = setindex!!(expected_merge, [2, 2], @varname(d.b))
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        vnt1 = setindex!!(vnt1, 1, @varname(e.a[1]))
        vnt2 = setindex!!(vnt2, 2, @varname(e.a[2]))
        expected_merge = setindex!!(expected_merge, 1, @varname(e.a[1]))
        expected_merge = setindex!!(expected_merge, 2, @varname(e.a[2]))
        vnt1 = setindex!!(vnt1, 1, @varname(e.a[3]))
        vnt2 = setindex!!(vnt2, 2, @varname(e.a[3]))
        expected_merge = setindex!!(expected_merge, 2, @varname(e.a[3]))
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        vnt1 = setindex!!(vnt1, fill(1, 4), @varname(e.a[7:10]))
        vnt2 = setindex!!(vnt2, fill(2, 4), @varname(e.a[8:11]))
        expected_merge = setindex!!(expected_merge, 1, @varname(e.a[7]))
        expected_merge = setindex!!(expected_merge, fill(2, 4), @varname(e.a[8:11]))
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        vnt1 = setindex!!(vnt1, ["1", "1"], @varname(f.a[1].b.c[2, 2].d[1, 3:4]))
        vnt2 = setindex!!(vnt2, ["2", "2"], @varname(f.a[1].b.c[2, 2].d[1, 3:4]))
        expected_merge = setindex!!(
            expected_merge, ["2", "2"], @varname(f.a[1].b.c[2, 2].d[1, 3:4])
        )
        vnt1 = setindex!!(vnt1, :1, @varname(f.a[1].b.c[3, 2].d[1, 1]))
        vnt2 = setindex!!(vnt2, :2, @varname(f.a[1].b.c[4, 2].d[1, 1]))
        expected_merge = setindex!!(expected_merge, :1, @varname(f.a[1].b.c[3, 2].d[1, 1]))
        expected_merge = setindex!!(expected_merge, :2, @varname(f.a[1].b.c[4, 2].d[1, 1]))
        @test merge(vnt1, vnt2) == expected_merge

        # PartialArrays with different sizes.
        vnt1 = VarNamedTuple()
        vnt2 = VarNamedTuple()
        vnt1 = setindex!!(vnt1, 1, @varname(a[1]))
        vnt1 = setindex!!(vnt1, 1, @varname(a[257]))
        vnt2 = setindex!!(vnt2, 2, @varname(a[1]))
        vnt2 = setindex!!(vnt2, 2, @varname(a[2]))
        expected_merge_12 = VarNamedTuple()
        expected_merge_12 = setindex!!(expected_merge_12, 1, @varname(a[257]))
        expected_merge_12 = setindex!!(expected_merge_12, 2, @varname(a[1]))
        expected_merge_12 = setindex!!(expected_merge_12, 2, @varname(a[2]))
        @test @inferred(merge(vnt1, vnt2)) == expected_merge_12
        expected_merge_21 = setindex!!(expected_merge_12, 1, @varname(a[1]))
        @test @inferred(merge(vnt2, vnt1)) == expected_merge_21

        vnt1 = VarNamedTuple()
        vnt2 = VarNamedTuple()
        vnt1 = setindex!!(vnt1, 1, @varname(a[1, 1]))
        vnt1 = setindex!!(vnt1, 1, @varname(a[257, 1]))
        vnt2 = setindex!!(vnt2, :2, @varname(a[1, 1]))
        vnt2 = setindex!!(vnt2, :2, @varname(a[1, 257]))
        expected_merge_12 = VarNamedTuple()
        expected_merge_12 = setindex!!(expected_merge_12, :2, @varname(a[1, 1]))
        expected_merge_12 = setindex!!(expected_merge_12, 1, @varname(a[257, 1]))
        expected_merge_12 = setindex!!(expected_merge_12, :2, @varname(a[1, 257]))
        @test merge(vnt1, vnt2) == expected_merge_12
        expected_merge_21 = setindex!!(expected_merge_12, 1, @varname(a[1, 1]))
        @test merge(vnt2, vnt1) == expected_merge_21
    end

    @testset "keys" begin
        vnt = VarNamedTuple()
        @test @inferred(keys(vnt)) == ()

        vnt = setindex!!(vnt, 1.0, @varname(a))
        # TODO(mhauru) that the below passes @inferred, but any of the later ones don't.
        # We should improve type stability of keys().
        @test @inferred(keys(vnt)) == (@varname(a),)

        vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
        @test keys(vnt) == (@varname(a), @varname(b))

        vnt = setindex!!(vnt, 15, @varname(b[2]))
        @test keys(vnt) == (@varname(a), @varname(b))

        vnt = setindex!!(vnt, [10], @varname(c.x.y))
        @test keys(vnt) == (@varname(a), @varname(b), @varname(c.x.y))

        vnt = setindex!!(vnt, -1.0, @varname(d[4]))
        @test keys(vnt) == (@varname(a), @varname(b), @varname(c.x.y), @varname(d[4]))

        vnt = setindex!!(vnt, 2.0, @varname(e.f[3, 3].g.h[2, 4, 1].i))
        @test keys(vnt) == (
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
        )

        vnt = setindex!!(vnt, fill(1.0, 4), @varname(j[1:4]))
        @test keys(vnt) == (
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
            @varname(j[1]),
            @varname(j[2]),
            @varname(j[3]),
            @varname(j[4]),
        )

        vnt = setindex!!(vnt, 1.0, @varname(j[6]))
        @test keys(vnt) == (
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
            @varname(j[1]),
            @varname(j[2]),
            @varname(j[3]),
            @varname(j[4]),
            @varname(j[6]),
        )

        vnt = setindex!!(vnt, 1.0, @varname(n[2].a))
        @test keys(vnt) == (
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
            @varname(j[1]),
            @varname(j[2]),
            @varname(j[3]),
            @varname(j[4]),
            @varname(j[6]),
            @varname(n[2].a),
        )
    end

    @testset "printing" begin
        vnt = VarNamedTuple()
        io = IOBuffer()
        show(io, vnt)
        output = String(take!(io))
        @test output == "VarNamedTuple()"

        vnt = setindex!!(vnt, "s", @varname(a))
        io = IOBuffer()
        show(io, vnt)
        output = String(take!(io))
        @test output == """VarNamedTuple(a = "s",)"""

        vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
        io = IOBuffer()
        show(io, vnt)
        output = String(take!(io))
        @test output == """VarNamedTuple(a = "s", b = [1, 2, 3])"""

        vnt = setindex!!(vnt, :dada, @varname(c[2]))
        io = IOBuffer()
        show(io, vnt)
        output = String(take!(io))
        @test output == """
            VarNamedTuple(a = "s", b = [1, 2, 3], \
            c = PartialArray{Symbol,1}((2,) => :dada))"""

        vnt = setindex!!(vnt, [16.0, 17.0], @varname(d.e[3].f.g[1:2]))
        io = IOBuffer()
        show(io, vnt)
        output = String(take!(io))
        # Depending on what's in scope, and maybe sometimes even the Julia version,
        # sometimes types in the output are fully qualified, sometimes not. To avoid
        # brittle tests, we normalise the output:
        output = replace(output, "DynamicPPL." => "", "VarNamedTuples." => "")
        @test output == """
            VarNamedTuple(a = "s", b = [1, 2, 3], \
            c = PartialArray{Symbol,1}((2,) => :dada), \
            d = VarNamedTuple(\
            e = PartialArray{VarNamedTuple{(:f,), \
            Tuple{VarNamedTuple{(:g,), \
            Tuple{PartialArray{Float64, 1}}}}},1}((3,) => \
            VarNamedTuple(f = VarNamedTuple(g = PartialArray{Float64,1}((1,) => 16.0, \
            (2,) => 17.0),),)),))"""
    end

    @testset "block variables" begin
        # Tests for setting and getting block variables, i.e. variables that have a non-zero
        # size in a PartialArray, but are not Arrays themselves.
        expected_err = ArgumentError("""
            A non-Array value set with a range of indices must be retrieved with the same
            range of indices.
            """)
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, Dirichlet(3, 1.0), @varname(x[2:4])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(x[2:4]))
        @test @inferred(getindex(vnt, @varname(x[2:4]))) == Dirichlet(3, 1.0)
        @test !haskey(vnt, @varname(x[2:3]))
        @test_throws expected_err getindex(vnt, @varname(x[2:3]))
        @test !haskey(vnt, @varname(x[3]))
        @test_throws expected_err getindex(vnt, @varname(x[3]))
        @test !haskey(vnt, @varname(x[1]))
        @test !haskey(vnt, @varname(x[5]))
        vnt = setindex!!(vnt, 1.0, @varname(x[1]))
        vnt = setindex!!(vnt, 1.0, @varname(x[5]))
        test_invariants(vnt)
        @test haskey(vnt, @varname(x[1]))
        @test haskey(vnt, @varname(x[5]))
        @test_throws expected_err getindex(vnt, @varname(x[1:4]))
        @test_throws expected_err getindex(vnt, @varname(x[2:5]))

        # Setting any of these indices should remove the block variable x[2:4].
        @testset "index = $index" for index in (2, 3, 4, 2:3, 3:5)
            # Test setting different types of values.
            vals = if index isa Int
                (2.0,)
            else
                (fill(2.0, length(index)), Dirichlet(length(index), 2.0))
            end
            @testset "val = $val" for val in vals
                vn = @varname(x[index])
                vnt2 = copy(vnt)
                vnt2 = setindex!!(vnt2, val, vn)
                test_invariants(vnt)
                @test !haskey(vnt2, @varname(x[2:4]))
                @test_throws BoundsError getindex(vnt2, @varname(x[2:4]))
                other_index = index in (2, 2:3) ? 4 : 2
                @test !haskey(vnt2, @varname(x[other_index]))
                @test_throws BoundsError getindex(vnt2, @varname(x[other_index]))
                @test haskey(vnt2, vn)
                @test getindex(vnt2, vn) == val
                @test haskey(vnt2, @varname(x[1]))
                @test_throws BoundsError getindex(vnt2, @varname(x[1:4]))
            end
        end

        # Extra checks, mostly for type stability and to confirm that multidimensional
        # blocks work too.
        struct TwoByTwoBlock end
        Base.size(::TwoByTwoBlock) = (2, 2)
        val = TwoByTwoBlock()
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[1:2, 1:2])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[1:2, 1:2]))
        @test @inferred(getindex(vnt, @varname(y.z[1:2, 1:2]))) == val
        @test !haskey(vnt, @varname(y.z[1, 1]))
        @test_throws expected_err getindex(vnt, @varname(y.z[1, 1]))

        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[2:3, 2:3])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[2:3, 2:3]))
        @test @inferred(getindex(vnt, @varname(y.z[2:3, 2:3]))) == val
        @test !haskey(vnt, @varname(y.z[1:2, 1:2]))
        @test_throws BoundsError getindex(vnt, @varname(y.z[1:2, 1:2]))

        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[4:5, 2:3])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[2:3, 2:3]))
        @test @inferred(getindex(vnt, @varname(y.z[2:3, 2:3]))) == val
        @test haskey(vnt, @varname(y.z[4:5, 2:3]))
        @test @inferred(getindex(vnt, @varname(y.z[4:5, 2:3]))) == val
    end
end

end
