module VarNamedTupleTests

using Combinatorics: Combinatorics
using OrderedCollections: OrderedDict
using Test: @inferred, @test, @test_throws, @testset, @test_broken
using DynamicPPL: DynamicPPL, @varname, VarNamedTuple, subset
using DynamicPPL.VarNamedTuples:
    PartialArray,
    ArrayLikeBlock,
    map_pairs!!,
    map_values!!,
    apply!!,
    templated_setindex!!,
    GrowableArray,
    grow_to_indices!!
using AbstractPPL: AbstractPPL, VarName, concretize, prefix, @opticof
using BangBang: setindex!!, empty!!
using DimensionalData: DimensionalData as DD
using InvertedIndices: InvertedIndices as II
using OffsetArrays: OffsetArrays as OA

struct GetSetTestCase
    # The VarName being set.
    vn
    # Its value.
    val
    # The structure of the top-level symbol in `vn`. Its values are unused.
    template
    # Sub-optics of `vn` that can be accessed once it has been set in a VNT.
    suboptics
end

function test_get_set(
    c::GetSetTestCase; templated_unstable=false, unstable=false, skip_setindex=false
)
    @testset "Templated setindex $(c.vn)" begin
        vnt = VarNamedTuple()
        vnt = if templated_unstable
            @test_broken false
            DynamicPPL.templated_setindex!!(vnt, c.val, c.vn, c.template)
        else
            @inferred(DynamicPPL.templated_setindex!!(vnt, c.val, c.vn, c.template))
        end
        @test @inferred(DynamicPPL.getindex(vnt, c.vn)) == c.val
        for optic in c.suboptics
            new_vn = AbstractPPL.append_optic(c.vn, optic)
            new_val = optic(c.val)
            @test @inferred(DynamicPPL.getindex(vnt, new_vn)) == new_val
        end
        test_invariants(vnt)
    end
    if !skip_setindex
        @testset "setindex $(c.vn)" begin
            vnt = VarNamedTuple()
            vnt = if unstable
                @test_broken false
                DynamicPPL.setindex!!(vnt, c.val, c.vn)
            else
                @inferred(DynamicPPL.setindex!!(vnt, c.val, c.vn))
            end
            @test @inferred(DynamicPPL.getindex(vnt, c.vn)) == c.val
            for optic in c.suboptics
                new_vn = AbstractPPL.append_optic(c.vn, optic)
                new_val = optic(c.val)
                @test @inferred(DynamicPPL.getindex(vnt, new_vn)) == new_val
            end
            test_invariants(vnt)
        end
    end
end

"""
    test_invariants(vnt::VarNamedTuple; skip=())

Test properties that should hold for all VarNamedTuples.

Uses @test for all the tests. Intended to be called inside a @testset.

`skip` is a tuple of symbols indicating which tests are to be skipped.
"""
function test_invariants(vnt::VarNamedTuple)
    # These will be needed repeatedly.
    vnt_keys = keys(vnt)
    vnt_values = values(vnt)

    # Check that for all keys in vnt, haskey is true, and resetting the value is a no-op.
    for k in vnt_keys
        @test haskey(vnt, k)
        v = getindex(vnt, k)
        # ArrayLikeBlocks and PartialArrays are implementation details, and should not be
        # exposed through getindex.
        @test !(v isa ArrayLikeBlock)
        @test !(v isa PartialArray)
        vnt2 = setindex!!(copy(vnt), v, k)
        equality = (vnt == vnt2)
        # The value may be `missing` if vnt itself has values that are missing.
        @test equality === true || equality === missing
        @test isequal(vnt, vnt2)
        @test hash(vnt) == hash(vnt2)
    end

    # Check that merge with an empty VarNamedTuple is a no-op.
    @test isequal(merge(vnt, VarNamedTuple()), vnt)
    @test isequal(merge(VarNamedTuple(), vnt), vnt)

    # Check that the VNT can be constructed back from its keys and values.
    vnt4 = VarNamedTuple()
    for (k, v) in zip(vnt_keys, vnt_values)
        vnt4 = templated_setindex!!(vnt4, v, k, vnt.data[AbstractPPL.getsym(k)])
    end
    @test isequal(vnt, vnt4)

    # Check that vnt isempty only if it has no keys
    was_empty = isempty(vnt)
    @test isequal(was_empty, isempty(vnt_keys))
    @test isequal(was_empty, isempty(vnt_values))

    # Check that vnt can be emptied
    @test empty(vnt) === VarNamedTuple()
    emptied_vnt = empty!!(copy(vnt))
    @test isempty(emptied_vnt)
    @test isempty(keys(emptied_vnt))
    @test isempty(values(emptied_vnt))

    # Check that the copy protected the original vnt from being modified.
    @test isempty(vnt) == was_empty

    # Check that map is a no-op when using identity functions.
    @test isequal(map_pairs!!(pair -> pair.second, copy(vnt)), vnt)
    @test isequal(map_values!!(identity, copy(vnt)), vnt)

    # Check that subsetting works as expected.
    @test isequal(subset(vnt, vnt_keys), vnt)
    @test isequal(subset(vnt, VarName[]), VarNamedTuple())
end

""" A type that has a size but is not an Array. Used in ArrayLikeBlock tests."""
struct SizedThing{T<:Tuple}
    size::T
end
Base.size(st::SizedThing) = st.size

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

        vnt3 = VarNamedTuple((;
            a=1.0, b=[1, 2, 3], c=VarNamedTuple((; d=VarNamedTuple((; e="a"))))
        ))
        test_invariants(vnt3)
        @test vnt1 == vnt3

        vnt4 = VarNamedTuple(
            OrderedDict(
                @varname(a) => 1.0, @varname(b) => [1, 2, 3], @varname(c.d.e) => "a"
            ),
        )
        test_invariants(vnt4)
        @test vnt1 == vnt4

        vnt5 = VarNamedTuple((
            (@varname(a), 1.0), (@varname(b), [1, 2, 3]), (@varname(c.d.e), "a")
        ))
        test_invariants(vnt5)
        @test vnt1 == vnt5
    end

    @testset "individual get/set" begin
        @testset "top-level" begin
            test_get_set(GetSetTestCase(@varname(a), 0.0, 0.0, []))
            test_get_set(
                GetSetTestCase(
                    @varname(b),
                    [1, 2, 3],
                    zeros(3),
                    [
                        @opticof(_[1]),
                        @opticof(_[2]),
                        @opticof(_[3]),
                        @opticof(_[1:3]),
                        @opticof(_[:]),
                        @opticof(_[1:2])
                    ],
                ),
            )
        end

        @testset "Array indices" begin
            test_get_set(GetSetTestCase(@varname(c[2]), 0.42, zeros(3), []))
            # Should still be type stable even though the eltype of the template is different, since
            # the eltype is taken from the value.
            test_get_set(GetSetTestCase(@varname(c[2]), "a", zeros(3), []))
            test_get_set(
                GetSetTestCase(
                    @varname(c[2:3]),
                    [0.42, 1.42],
                    zeros(3),
                    [@opticof(_[1]), @opticof(_[2])],
                ),
            )
        end

        @testset "Properties" begin
            test_get_set(GetSetTestCase(@varname(d.a), 0.42, (; a=1.0), []))
            test_get_set(GetSetTestCase(@varname(d.a.b), 0.42, (; a=(; b=1.0)), []))
            # Same as above, should still be type stable.
            test_get_set(GetSetTestCase(@varname(d.a.b), "a", (; a=(; b=1.0)), []))
            test_get_set(GetSetTestCase(@varname(d.a[2]), 0.42, (; a=zeros(3)), []))
            test_get_set(
                GetSetTestCase(@varname(d.a[2].e), 0.42, (; a=fill((; e=3.0), 3)), [])
            )
            test_get_set(
                GetSetTestCase(
                    @varname(e.a.b),
                    [1.0, 2.0],
                    (; a=(; b=1.0)),
                    [@opticof(_[1]), @opticof(_[2]), @opticof(_[:]), @opticof(_[1:2])],
                ),
            )
        end

        @testset "Wrong templates" begin
            # We want to check that even if the template provided is wrong, we still can do
            # something meaningful. Essentially, this checks the fallback behaviour in
            # make_leaf: if the template does not have the right property, we can
            # just fall back to NoTemplate.
            @testset "Properties" begin
                test_get_set(GetSetTestCase(@varname(x.a), 1.0, nothing, []))
                test_get_set(GetSetTestCase(@varname(x.a.b), 1.0, nothing, []))
                test_get_set(GetSetTestCase(@varname(x.a.b.c), 1.0, nothing, []))
                test_get_set(GetSetTestCase(@varname(x.a.b.c.d), 1.0, nothing, []))
            end

            @testset "Indices" begin
                # Note that all of these would actually error if you tried to use them in a
                # model. That's why I don't care about the type stability of
                # templated_setindex!! below.
                test_get_set(GetSetTestCase(@varname(x[1]), 1.0, nothing, []))
                test_get_set(GetSetTestCase(@varname(x[1, 1]), 1.0, nothing, []))
                test_get_set(
                    GetSetTestCase(@varname(x[1, 1:3]), rand(3), nothing, []);
                    templated_unstable=true,
                )
            end
        end

        @testset "Heavily nested optics" begin
            # TODO(penelopeysm): This is the only combination of things I can find for which
            # templated_setindex!! is not type stable. It needs to at least have property ->
            # index -> property -> index.
            vnt = VarNamedTuple()
            a = (; b=[nothing, (; y=zeros(2))])
            test_get_set(
                GetSetTestCase(@varname(a.b[2].y[1]), 1.0, a, []); templated_unstable=true
            )
            # Apart from the fact that it's 4 levels deep, to trigger the issue, the array
            # template provided (in this case, the element `b`) must also have an abstract
            # eltype. That's why we have to put `nothing` as the first element, rather than
            # just using fill((; y=zeros(2)), 2) as the template, because that does pass.
            # See for yourself:
            a2 = (; b=fill((; y=zeros(2)), 2))
            test_get_set(GetSetTestCase(@varname(a.b[2].y[1]), 1.0, a2, []))
        end

        @testset "Matrices" begin
            test_get_set(GetSetTestCase(@varname(f[1]), 1.0, zeros(2, 2), []))
            # Without a template this will fail because we don't know how big the second dim is,
            # hence skipping the untemplated setindex.
            test_get_set(
                GetSetTestCase(@varname(f[1, :]), [1.0, 2.0], zeros(2, 2), []);
                skip_setindex=true,
            )
            test_get_set(GetSetTestCase(@varname(x[1, 1:3]), rand(3), rand(3, 3), []))
            test_get_set(GetSetTestCase(@varname(f[1, 1]), 1.0, zeros(2, 2), []))
            test_get_set(GetSetTestCase(@varname(f[1, 1]), "a", zeros(2, 2), []))
            test_get_set(
                GetSetTestCase(@varname(f[2, 2].b), 1.0, fill((; b=2.0), 2, 2), [])
            )
        end

        @testset "Nested single-index" begin
            test_get_set(GetSetTestCase(@varname(g[2][2]), 1.0, fill(zeros(2), 2), []))
            gelem = (; a=fill(zeros(2), 2))
            g = fill(fill(gelem, 2), 2)
            test_get_set(GetSetTestCase(@varname(g[2][2].a[1][1]), 1.0, g, []))
        end

        @testset "Nested multi-index" begin
            test_get_set(GetSetTestCase(@varname(g[1:2][2]), 1.0, zeros(2), []))
            test_get_set(
                GetSetTestCase(@varname(g[1:2][2].a), 1.0, fill((; a=1.0), 2), []);
            )
        end

        @testset "OffsetArray" begin
            oa = OA.OffsetArray(zeros(5), 11:15)
            test_get_set(GetSetTestCase(@varname(oa[11]), 1.0, oa, []); skip_setindex=true)
            test_get_set(
                GetSetTestCase(@varname(x.oa[11]), 1.0, (; oa=oa), []); skip_setindex=true
            )
            oa2 = OA.OffsetArray(fill((; g="a"), 5, 4), 11:15, -2:1)
            test_get_set(
                GetSetTestCase(@varname(oa2[11, -1].g), 1.0, oa2, []); skip_setindex=true
            )
            test_get_set(
                GetSetTestCase(@varname(oa2[11, -1]), 1.0, oa2, []); skip_setindex=true
            )
        end

        @testset "InvertedIndices" begin
            # TODO(penelopeysm): Templated setindex fails for II.Not(). I really don't know
            # why but there is some failure in constant propagation when setting the mask
            # using Not(3). Here is a minimiser that illustrates the difference between
            # Not(3) and 1:2:
            #
            #     pa = PartialArray(randn(3), fill(false, 3))
            #     @descend BangBang.setindex!!(pa, [1.0, 2.0], Not(3))
            #     @descend BangBang.setindex!!(pa, [1.0, 2.0], 1:2)
            test_get_set(
                GetSetTestCase(@varname(x[II.Not(3)]), randn(2), zeros(3), []);
                templated_unstable=true,
                skip_setindex=true,
            )
        end

        @testset "DimensionalData" begin
            da = DD.DimArray(randn(3), (DD.X))
            test_get_set(
                GetSetTestCase(@varname(da[DD.X(2)]), 1.0, da, []); skip_setindex=true
            )
            test_get_set(
                GetSetTestCase(@varname(x.da[DD.X(2)]), 1.0, (; da=da), []);
                skip_setindex=true,
            )
            da2 = DD.DimArray(fill((; g="a"), 3, 5), (DD.X, DD.Y))
            test_get_set(
                GetSetTestCase(@varname(da2[DD.X(2), DD.Y(1)]), 1.0, da2, []);
                skip_setindex=true,
            )
            test_get_set(
                GetSetTestCase(@varname(da2[DD.X(2), DD.Y(1)].g), "b", da2, []);
                skip_setindex=true,
            )
        end
    end

    @testset "Setting to same variable multiple times" begin
        # TODO(penelopeysm) write these
    end

    @testset "Chain of sets and gets" begin
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

        vnt = @inferred(templated_setindex!!(vnt, -1.0, @varname(d[4]), randn(4)))
        @test @inferred(getindex(vnt, @varname(d[4]))) == -1.0
        test_invariants(vnt)

        vnt = @inferred(setindex!!(vnt, -2.0, @varname(d[4])))
        @test @inferred(getindex(vnt, @varname(d[4]))) == -2.0
        test_invariants(vnt)

        e = (; f=fill((; g=(; h=fill((; i=0.0), 2))), 3))
        vnt = @inferred(templated_setindex!!(vnt, 1.0, @varname(e.f[3].g.h[2].i), e))
        @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 1.0
        @test haskey(vnt, @varname(e.f[3].g.h[2].i))
        @test !haskey(vnt, @varname(e.f[2].g.h[2].i))
        test_invariants(vnt)

        # TODO(penelopeysm) This one fails type stability.
        # https://github.com/TuringLang/DynamicPPL.jl/issues/1207
        vnt = setindex!!(vnt, 2.0, @varname(e.f[3].g.h[2].i))
        @test @inferred(getindex(vnt, @varname(e.f[3].g.h[2].i))) == 2.0
        test_invariants(vnt)

        jval = fill(1.0, 4)
        vnt = @inferred(templated_setindex!!(vnt, jval, @varname(j[1:4]), zeros(4)))
        @test @inferred(getindex(vnt, @varname(j[1:4]))) == jval
        @test @inferred(getindex(vnt, @varname(j[2]))) == jval[2]
        @test haskey(vnt, @varname(j[4]))
        @test !haskey(vnt, @varname(j[5]))
        @test_throws BoundsError getindex(vnt, @varname(j[5]))
        test_invariants(vnt)

        j2val = fill(2.0, 4)
        vnt = @inferred(templated_setindex!!(vnt, j2val, @varname(j2[2:5]), zeros(5)))
        @test_throws BoundsError getindex(vnt, @varname(j2[1]))
        @test @inferred(getindex(vnt, @varname(j2[2:5]))) == j2val
        @test haskey(vnt, @varname(j2[5]))
        test_invariants(vnt)

        arr = fill(2.0, (4, 2))
        vn = @varname(k.l[2:5, 3, 1:2, 2])
        vnt = @inferred(templated_setindex!!(vnt, arr, vn, (; l=zeros(5, 3, 2, 2))))
        @test @inferred(getindex(vnt, vn)) == arr
        # A subset of the elements set just now.
        @test @inferred(getindex(vnt, @varname(k.l[2, 3, 1:2, 2]))) == fill(2.0, 2)
        test_invariants(vnt)

        # Not enough, or too many, indices.
        @test_throws BoundsError setindex!!(vnt, 0.0, @varname(k.l[1, 2, 3]))
        @test_throws BoundsError setindex!!(vnt, 0.0, @varname(k.l[1, 2, 3, 4, 5]))

        vnt = @inferred(templated_setindex!!(vnt, 1.0, @varname(m[2]), randn(3)))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[3])))
        @test @inferred(getindex(vnt, @varname(m[2:3]))) == [1.0, 1.0]
        @test !haskey(vnt, @varname(m[1]))
        test_invariants(vnt)
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(m[1])))
        @test @inferred(getindex(vnt, @varname(m[1:3]))) == ones(3)
        @test @inferred(getindex(vnt, @varname(m))) == ones(3)
        @test @inferred(getindex(vnt, @varname(m[:]))) == ones(3)
        test_invariants(vnt)

        # The below tests are mostly significant for the type stability aspect. For the last
        # test to pass, PartialArray needs to actively tighten its eltype when possible.
        vnt = VarNamedTuple()
        vnt = @inferred(templated_setindex!!(vnt, 1.0, @varname(n[1].a), [0.0, 0.0]))
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

        # Some funky Symbols in VarNames.
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

        vnt = VarNamedTuple()
        x = [1, 2, 3]
        vn = @varname(x[:])
        vnt = @inferred(templated_setindex!!(vnt, x, vn, x))
        @test haskey(vnt, vn)
        @test @inferred(getindex(vnt, vn)) == x
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vn = @varname(y[:])
        vnt = @inferred(templated_setindex!!(vnt, SizedThing((3,)), vn, randn(3)))
        @test haskey(vnt, vn)
        @test vn in keys(vnt)
        @test @inferred(getindex(vnt, vn)) == SizedThing((3,))
        test_invariants(vnt)

        vnt = VarNamedTuple()
        x = fill("a", (3, 4))
        a = (; b=fill((; c=(; d=zeros(1, 5, 2, 4, 1))), 3))
        vn = @varname(a.b[3].c.d[1, 3:5, 2, :, 1])
        vnt = @inferred(templated_setindex!!(vnt, x, vn, a))
        @test haskey(vnt, vn)
        @test @inferred(getindex(vnt, vn)) == x
        test_invariants(vnt)

        # Indices on indices
        vnt = VarNamedTuple()
        vnt = @inferred(templated_setindex!!(vnt, 1, @varname(a[1][1]), [[randn()]]))
        @test @inferred(getindex(vnt, @varname(a[1][1]))) == 1
        vnt = @inferred(templated_setindex!!(vnt, 1, @varname(ab[1:2][1]), randn(2)))
        @test @inferred(getindex(vnt, @varname(ab[1]))) == 1
        @test @inferred(getindex(vnt, @varname(ab[1:2][1]))) == 1
        @test @inferred(getindex(vnt, @varname(ab[:][1]))) == 1
        @test_throws BoundsError getindex(vnt, @varname(ab[2]))
        vnt = @inferred(
            templated_setindex!!(vnt, [1], @varname(b[1].c[1]), [(; c=randn(1))])
        )
        @test @inferred(getindex(vnt, @varname(b[1].c[1]))) == [1]
        # TODO(penelopeysm) These all have to be changed to templated_setindex!!, but I
        # can't be bothered right now.
        # vnt = @inferred(setindex!!(vnt, [1], @varname(e[3, 2].f[2, 2][10, 10])))
        # @test @inferred(getindex(vnt, @varname(e[3, 2].f[2, 2][10, 10]))) == [1]
        # vnt = @inferred(setindex!!(vnt, [1], @varname(g[3, 2][10, 10].h[2, 2])))
        # @test @inferred(getindex(vnt, @varname(g[3, 2][10, 10].h[2, 2]))) == [1]
    end

    @testset "GrowableArray behaviour" begin
        @testset "grow_to_indices!! on PartialArray->GrowableArray" begin
            pa = PartialArray(GrowableArray(ones(2)), GrowableArray([false, true]))
            @test_throws BoundsError pa[1]
            @test pa[2] == 1.0
            pa = grow_to_indices!!(pa, 1:3)
            @test size(pa.data) == (3,)
            @test_throws BoundsError pa[1]
            @test pa[2] == 1.0
            # pa[3] should still error...
            @test_throws BoundsError pa[3]
            # ... but the underlying data should be there
            @test pa.mask[3] == false
        end

        @testset "grow_to_indices!! can't change dimensionality" begin
            pa = PartialArray(GrowableArray(ones(2)), GrowableArray([false, true]))
            @test_throws ArgumentError grow_to_indices!!(pa, 1, 2)
            pa = PartialArray(GrowableArray(randn(2, 2)), GrowableArray(fill(false, 2, 2)))
            @test_throws ArgumentError grow_to_indices!!(pa, 1)
        end

        @testset "Data is correctly copied when expanding" begin
            x = randn(2, 3)
            vnt = VarNamedTuple()
            for c in CartesianIndices((2, 3))
                vnt = setindex!!(vnt, x[c], @varname(x[c.I...]))
            end
            @test vnt[@varname(x)] == x
        end
    end

    @testset "multiindices" begin
        # These tests essentially check that:
        #   (1) setting a slice works as expected
        #   (2) setting an index of a slice works as expected
        #   (3) setting a slice of a slice works as expected
        #
        # There used to be a litany of bugs around this. See DynamicPPL #1205.
        @testset for (index_chains, template_length) in [
            ((1:2, 1), 3),
            ((2:3, 1), 3),
            ((1:2, 2), 3),
            ((2:3, 2), 3),
            ((1:3, 1:2), 4),
            ((1:3, 2:3), 4),
            ((2:4, 1:2), 4),
            ((2:4, 2:3), 4),
            ((1:3, 1:2, 1), 4),
            ((1:3, 1:2, 2), 4),
            ((2:4, 2:3, 1), 4),
            ((2:4, 2:3, 2), 4),
            ((1:3, 2:3, 1), 4),
            ((1:3, 2:3, 2), 4),
            ((2:4, 1:2, 1), 4),
            ((2:4, 1:2, 2), 4),
            ((1:3, :), 4),
            ((2:4, :), 4),
            ((:, 1:3), 4),
            ((:, 2:4), 4),
            ((1:3, :, 1:2), 4),
            ((2:4, :, 1:2), 4),
            ((1:3, :, 2:3), 4),
            ((2:4, :, 2:3), 4),
            ((1:2, :, 1), 3),
            ((1:2, :, 2), 3),
            ((2:3, :, 1), 3),
            ((2:3, :, 2), 3),
        ]
            # I couldn't figure out a better way to do this. `@varname(x[index_chains...])`
            # works, but it gives the wrong thing: it gives you `x[1:2, 1]` instead of
            # `x[1:2][1]`.
            vn = if length(index_chains) == 2
                @varname(x[index_chains[1]][index_chains[2]])
            elseif length(index_chains) == 3
                @varname(x[index_chains[1]][index_chains[2]][index_chains[3]])
            else
                error("unsupported length")
            end
            # This gets the final position where the values should end up --
            # get_collapsed_indices((1:3, 2:3, 1), nothing) = 2
            function get_collapsed_indices(chain, init)
                if isnothing(init) || init isa Colon
                    return get_collapsed_indices(Base.tail(chain), first(chain))
                end
                isempty(chain) && return init
                inds = first(chain)
                return if inds isa Colon
                    get_collapsed_indices(Base.tail(chain), init)
                else
                    get_collapsed_indices(Base.tail(chain), init[inds])
                end
            end
            final_indices = get_collapsed_indices(index_chains, nothing)

            val = if final_indices isa Integer
                1.0
            else
                fill(1.0, length(final_indices))
            end

            for (func!!, args) in
                ((templated_setindex!!, (zeros(template_length),)), (setindex!!, ()))
                vnt = @inferred(
                    templated_setindex!!(VarNamedTuple(), val, vn, zeros(template_length))
                )
                @test @inferred(getindex(vnt, vn)) == val
                normalized_vn = @varname(x[final_indices])
                @test @inferred(getindex(vnt, normalized_vn)) == val

                # Check that other indices are not set
                for i in 1:template_length
                    if !(i in final_indices)
                        @test_throws BoundsError getindex(vnt, @varname(x[i]))
                    end
                end
            end
        end
    end

    @testset "chained multiindices [1:2][1] -> [1:2][2]" begin
        # These are particularly finicky to deal with. See
        # https://github.com/TuringLang/DynamicPPL.jl/issues/1205.
        #
        # The aim of these tests is to make sure that when you set [1:2][2], you don't
        # lose [1:2][1]. This could happen if you're not careful because when you
        # set [1:2][2], you get back a length-2 PartialArray with only the second element
        # set, and you could overwrite the entire [1:2] entry in the target PartialArray,

        x = zeros(2)
        for (func!!, args) in ((templated_setindex!!, (x,)), (setindex!!, ()))
            vnt = VarNamedTuple()

            vnt = @inferred(func!!(vnt, 1.0, @varname(x[1:2][1]), args...))
            @test @inferred(getindex(vnt, @varname(x[1:2][1]))) == 1.0
            @test @inferred(getindex(vnt, @varname(x[1]))) == 1.0
            @test_throws BoundsError getindex(vnt, @varname(x[2]))
            test_invariants(vnt)

            # Now set the second index
            vnt = @inferred(func!!(vnt, 2.0, @varname(x[1:2][2]), args...))
            @test @inferred(getindex(vnt, @varname(x[1:2][2]))) == 2.0
            @test @inferred(getindex(vnt, @varname(x[2]))) == 2.0
            test_invariants(vnt)

            # Check that the first index is still correct
            @test @inferred(getindex(vnt, @varname(x[1]))) == 1.0
            # Check that we can get both indices
            @test @inferred(getindex(vnt, @varname(x[1:2]))) == [1.0, 2.0]
            @test @inferred(getindex(vnt, @varname(x[:]))) == [1.0, 2.0]
            @test @inferred(getindex(vnt, @varname(x))) == [1.0, 2.0]

            @test keys(vnt) == [@varname(x[1]), @varname(x[2])]
            @test values(vnt) == [1.0, 2.0]
        end
    end

    @testset "chained multiindices [1:2][1] -> [end:end][1]" begin
        # See https://github.com/TuringLang/DynamicPPL.jl/issues/1205.
        # Note that this doesn't work with untemplated setindex!! because `end` will be
        # wrong.
        vnt = VarNamedTuple()
        x = zeros(3)

        vnt = @inferred(templated_setindex!!(vnt, 1.0, @varname(x[1:2][1]), x))
        @test @inferred(getindex(vnt, @varname(x[1:2][1]))) == 1.0
        @test @inferred(getindex(vnt, @varname(x[1]))) == 1.0
        @test_throws BoundsError getindex(vnt, @varname(x[2]))
        @test_throws BoundsError getindex(vnt, @varname(x[3]))
        test_invariants(vnt)

        # Now set the second index
        vnt = @inferred(templated_setindex!!(vnt, 2.0, @varname(x[end:end][1]), x))
        @test @inferred(getindex(vnt, @varname(x[end:end][1]))) == 2.0
        @test @inferred(getindex(vnt, @varname(x[3]))) == 2.0
        test_invariants(vnt)

        # Check that the first index is still correct
        @test @inferred(getindex(vnt, @varname(x[1]))) == 1.0
        # Check that we can get both indices
        @test @inferred(getindex(vnt, @varname(x[[1, 3]]))) == [1.0, 2.0]
        # Check that we can't get all three indices
        @test_throws BoundsError getindex(vnt, @varname(x[1:3]))
        @test_throws BoundsError getindex(vnt, @varname(x[:]))
        @test_throws ArgumentError getindex(vnt, @varname(x))

        @test keys(vnt) == [@varname(x[1]), @varname(x[3])]
        @test values(vnt) == [1.0, 2.0]
    end

    @testset "equality and hash" begin
        # Test all combinations of having or not having the below values set, and having
        # them set to any of the possible_values, and check that isequal and == return the
        # expected value.
        # NOTE: Be very careful adding new values to these sets. The below test has three
        # nested loops over Combinatorics.combinations, the run time can explode very, very
        # quickly.
        b = rand(3)
        c = (; d=[nothing, (; e=rand()), nothing])
        varnames_and_templates = (
            (@varname(b[1]), b), (@varname(b[3]), b), (@varname(c.d[2].e), c)
        )
        possible_values = (missing, 1, -0.0, 0.0)
        for vn_template_set in Combinatorics.combinations(varnames_and_templates)
            valuesets1 = Combinatorics.with_replacement_combinations(
                possible_values, length(vn_template_set)
            )
            valuesets2 = Combinatorics.with_replacement_combinations(
                possible_values, length(vn_template_set)
            )
            for vset1 in valuesets1, vset2 in valuesets2
                vnt1 = VarNamedTuple()
                vnt2 = VarNamedTuple()
                expected_isequal = true
                expected_doubleequal = true
                for ((vn, template), v1, v2) in zip(vn_template_set, vset1, vset2)
                    vnt1 = templated_setindex!!(vnt1, v1, vn, template)
                    vnt2 = templated_setindex!!(vnt2, v2, vn, template)
                    expected_isequal = expected_isequal & isequal(v1, v2)
                    expected_doubleequal = expected_doubleequal & (v1 == v2)
                end
                test_invariants(vnt1)
                test_invariants(vnt2)
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
        test_invariants(vnt1)
        test_invariants(vnt2)

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

        e = (; a=rand(11), b=[rand(20) for _ in 1:4])
        vnt1 = templated_setindex!!(vnt1, 1, @varname(e.a[1]), e)
        vnt2 = templated_setindex!!(vnt2, 2, @varname(e.a[2]), e)
        expected_merge = templated_setindex!!(expected_merge, 1, @varname(e.a[1]), e)
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

        vnt1 = templated_setindex!!(vnt1, 1, @varname(e.b[1][13]), e)
        vnt2 = templated_setindex!!(vnt2, 2, @varname(e.b[2][13]), e)
        expected_merge = templated_setindex!!(expected_merge, 1, @varname(e.b[1][13]), e)
        expected_merge = templated_setindex!!(expected_merge, 2, @varname(e.b[2][13]), e)
        vnt1 = templated_setindex!!(vnt1, 1, @varname(e.b[3][13]), e)
        vnt2 = templated_setindex!!(vnt2, 2, @varname(e.b[3][13]), e)
        expected_merge = templated_setindex!!(expected_merge, 2, @varname(e.b[3][13]), e)
        @test @inferred(merge(vnt1, vnt2)) == expected_merge
        vnt1 = templated_setindex!!(vnt1, 1, @varname(e.b[4][13]), e)
        vnt2 = templated_setindex!!(vnt2, 2, @varname(e.b[4][14]), e)
        expected_merge = templated_setindex!!(expected_merge, 1, @varname(e.b[4][13]), e)
        expected_merge = templated_setindex!!(expected_merge, 2, @varname(e.b[4][14]), e)
        @test @inferred(merge(vnt1, vnt2)) == expected_merge

        struct A
            v # Just need something to disambiguate between two A's.
        end
        Base.size(::A) = (2,)

        @testset "with sized things: same indices" begin
            # When things are stored as ArrayLikeBlocks, need to be careful that merging
            # works correctly. There have been weird bugs in the past around this where
            # part of an ArrayLikeBlock was set, and then the merged result had ALBs from
            # both sides of the merge.
            for (vn, template) in
                [(@varname(x[1:2]), zeros(2)), (@varname(x[1:5][1:2]), zeros(5))]
                # Regardless of how we set it above, the VNT should always store it as
                # x[1:2].
                normalized_vn = @varname(x[1:2])

                # Without templating.
                vnt1 = @inferred(setindex!!(VarNamedTuple(), A(1.0), vn))
                vnt2 = @inferred(setindex!!(VarNamedTuple(), A(2.0), vn))
                @test vnt1[vn] == A(1.0)
                @test vnt2[vn] == A(2.0)
                merged = @inferred(merge(vnt1, vnt2))
                expected_merge = setindex!!(VarNamedTuple(), A(2.0), vn)
                @test merged == expected_merge
                @test only(keys(merged)) == normalized_vn
                @test merged[vn] == A(2.0)
                @test merged[normalized_vn] == A(2.0)

                # With templating.
                vnt1 = @inferred(
                    templated_setindex!!(VarNamedTuple(), A(1.0), vn, template)
                )
                @test vnt1[vn] == A(1.0)
                vnt2 = @inferred(
                    templated_setindex!!(VarNamedTuple(), A(2.0), vn, template)
                )
                @test vnt2[vn] == A(2.0)
                merged = @inferred(merge(vnt1, vnt2))
                expected_merge = templated_setindex!!(VarNamedTuple(), A(2.0), vn, template)
                @test merged == expected_merge
                @test only(keys(merged)) == normalized_vn
                @test merged[normalized_vn] == A(2.0)
            end
        end

        @testset "with sized things: different but overlapping indices" begin
            vn1 = @varname(x[1:2])
            vn2 = @varname(x[2:3])

            # Without templating.
            # TODO(penelopeysm): This is currently broken: we need to let the merge function
            # handle GrowableArrays.
            #=
            vnt1 = @inferred(setindex!!(VarNamedTuple(), A(1.0), vn1))
            @test vnt1[vn1] == A(1.0)
            vnt2 = @inferred(setindex!!(VarNamedTuple(), A(2.0), vn2))
            @test vnt2[vn2] == A(2.0)
            expected_merge = setindex!!(VarNamedTuple(), A(2.0), vn2)
            @test @inferred(merge(vnt1, vnt2)) == expected_merge
            @test merge(vnt1, vnt2)[vn] == A(2.0)
            =#

            # With templating.
            vnt1 = @inferred(templated_setindex!!(VarNamedTuple(), A(1.0), vn1, zeros(3)))
            @test vnt1[vn1] == A(1.0)
            vnt2 = @inferred(templated_setindex!!(VarNamedTuple(), A(2.0), vn2, zeros(3)))
            @test vnt2[vn2] == A(2.0)
            merged = @inferred(merge(vnt1, vnt2))
            expected_merge = templated_setindex!!(VarNamedTuple(), A(2.0), vn2, zeros(3))
            @test merged == expected_merge
            @test only(keys(merged)) == vn2
            @test merge(vnt1, vnt2)[vn2] == A(2.0)
        end

        # TODO(penelopeysm): This set of tests fails. The reason is because later on we
        # have a VarName that looks like d[1, 1][14, 13], i.e., d must be a matrix of
        # matrices. So we have to provide that as the structure for d (via `f`). However,
        # when you try to set `d[1, 3:4]` to `["1", "1"]` BangBang.setindex!! errors.
        # https://github.com/JuliaFolds2/BangBang.jl/issues/44
        #
        # I honestly think this is so pathological that it's not worth worrying about too
        # much for now in DPPL. If someone complains about it, we can probably just honestly
        # say that it's an upstream issue.
        #
        # Of course, I would like to fix it in BangBang, but time and energy, blah blah.
        #= 
        d = (; d=fill(randn(20, 20), 4, 4))
        f = (; a=[(; b=(; c=fill(d, 4, 2)))])
        vnt1 = VarNamedTuple()
        vnt1 = templated_setindex!!(vnt1, ["1", "1"], @varname(f.a[1].b.c[2, 2].d[1, 3:4]), f)
        vnt2 = templated_setindex!!(vnt2, ["2", "2"], @varname(f.a[1].b.c[2, 2].d[1, 3:4]), f)
        expected_merge = setindex!!(
            expected_merge, ["2", "2"], @varname(f.a[1].b.c[2, 2].d[1, 3:4])
        )
        vnt1 = setindex!!(vnt1, :1, @varname(f.a[1].b.c[3, 2].d[1, 1][14, 13]))
        vnt2 = setindex!!(vnt2, :2, @varname(f.a[1].b.c[4, 2].d[1, 1][14, 13]))
        expected_merge = setindex!!(
            expected_merge, :1, @varname(f.a[1].b.c[3, 2].d[1, 1][14, 13])
        )
        expected_merge = setindex!!(
            expected_merge, :2, @varname(f.a[1].b.c[4, 2].d[1, 1][14, 13])
        )
        @test merge(vnt1, vnt2) == expected_merge
        test_invariants(vnt1)
        test_invariants(vnt2)
        =#

        # TODO(penelopeysm): Test that merging partial arrays of different types/sizes
        # fails.
    end

    @testset "subset" begin
        vnt = VarNamedTuple()
        vnt = setindex!!(vnt, 1.0, @varname(a))
        vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
        vnt = setindex!!(vnt, [10], @varname(c.x.y))
        d = randn(3)
        vnt = templated_setindex!!(vnt, :1, @varname(d[1]), d)
        vnt = setindex!!(vnt, :2, @varname(d[2]))
        vnt = setindex!!(vnt, :3, @varname(d[3]))
        e = (; f=fill((; g=(; h=fill((; i=0.0), 2, 4, 1))), 3, 3))
        vnt = templated_setindex!!(vnt, 2.0, @varname(e.f[3, 3].g.h[2, 4, 1].i), e)
        p = fill(zeros(4, 5, 14), 2, 2)
        vnt = templated_setindex!!(
            vnt, SizedThing((3, 1, 4)), @varname(p[2, 1][2:4, 5:5, 11:14]), p
        )
        test_invariants(vnt)

        # TODO(mhauru) I'm a bit saddened by the lack of type stability for subset: It's
        # return type always infers as VarNamedTuple. Improving this would require a
        # different implementation of subset.
        @test subset(vnt, VarName[]) == VarNamedTuple()
        @test subset(vnt, (@varname(z),)) == VarNamedTuple()
        @test subset(vnt, (@varname(d[4]),)) == VarNamedTuple()
        @test subset(vnt, (@varname(d[1, 1]),)) == VarNamedTuple()
        @test subset(vnt, [@varname(a)]) == VarNamedTuple(; a=1.0)
        begin
            # These are a bit annoying to write out as we need to build the expected output
            # manually
            expected_vnt = setindex!!(VarNamedTuple(), [1, 2, 3], @varname(b))
            expected_vnt = templated_setindex!!(expected_vnt, :1, @varname(d[1]), randn(3))
            @test subset(vnt, [@varname(b), @varname(d[1])]) == expected_vnt
        end
        begin
            expected_vnt = templated_setindex!!(
                VarNamedTuple(), :2, @varname(d[2]), randn(3)
            )
            expected_vnt = setindex!!(expected_vnt, :3, @varname(d[3]))
            @test subset(vnt, [@varname(d[2:3])]) == expected_vnt
        end
        # For this one, we can't test exact equality because in the former case we
        # constructed a PartialArray by adding individual elements of d. In the latter
        # case we constructed it directly from a vector. So the internal representation
        # is different, even though in the first case, the PartialArray has all its mask
        # elements as true (and is thus conceptually the same as a full array).
        # We can however test that the underlying values are the same when we try to extract
        # them.
        subsetted_d = subset(vnt, [@varname(d)])
        expected_d = VarNamedTuple((@varname(d) => [:1, :2, :3],))
        for vn in [
            @varname(d),
            @varname(d[1]),
            @varname(d[2]),
            @varname(d[3]),
            @varname(d[:]),
            @varname(d[1:2]),
            @varname(d[2:3])
        ]
            @test getindex(subsetted_d, vn) == getindex(expected_d, vn)
        end

        @test subset(vnt, [@varname(c.x.y)]) == VarNamedTuple((@varname(c.x.y) => [10],))
        @test subset(vnt, [@varname(c)]) == VarNamedTuple((@varname(c.x.y) => [10],))
        begin
            expected_vnt = templated_setindex!!(
                VarNamedTuple(), 2.0, @varname(e.f[3, 3].g.h[2, 4, 1].i), e
            )
            @test subset(vnt, [@varname(e.f[3, 3].g.h[2, 4, 1].i)]) == expected_vnt
        end
        begin
            expected_vnt = templated_setindex!!(
                VarNamedTuple(),
                SizedThing((3, 1, 4)),
                @varname(p[2, 1][2:4, 5:5, 11:14]),
                p,
            )
            @test subset(vnt, [@varname(p[2, 1][2:4, 5:5, 11:14])]) == expected_vnt
        end
        # Cutting the last range a bit short should mean that nothing is returned.
        @test subset(vnt, [@varname(p[2, 1][2:4, 5:5, 11:13])]) == VarNamedTuple()
    end

    @testset "keys and values" begin
        vnt = VarNamedTuple()
        @test @inferred(keys(vnt)) == VarName[]
        @test @inferred(values(vnt)) == Any[]

        vnt = setindex!!(vnt, 1.0, @varname(a))
        # TODO(mhauru) that the below passes @inferred, but any of the later ones don't.
        # We should improve type stability of keys().
        @test @inferred(keys(vnt)) == [@varname(a)]
        @test @inferred(values(vnt)) == [1.0]

        vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
        @test keys(vnt) == [@varname(a), @varname(b)]
        @test values(vnt) == [1.0, [1, 2, 3]]

        vnt = setindex!!(vnt, 15, @varname(b[2]))
        @test keys(vnt) == [@varname(a), @varname(b)]
        @test values(vnt) == [1.0, [1, 15, 3]]

        vnt = setindex!!(vnt, [10], @varname(c.x.y))
        @test keys(vnt) == [@varname(a), @varname(b), @varname(c.x.y)]
        @test values(vnt) == [1.0, [1, 15, 3], [10]]

        vnt = setindex!!(vnt, -1.0, @varname(d[4]))
        @test keys(vnt) == [@varname(a), @varname(b), @varname(c.x.y), @varname(d[4])]
        @test values(vnt) == [1.0, [1, 15, 3], [10], -1.0]

        vnt = setindex!!(vnt, 2.0, @varname(e.f[3, 3].g.h[2, 4, 1].i))
        @test keys(vnt) == [
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
        ]
        @test values(vnt) == [1.0, [1, 15, 3], [10], -1.0, 2.0]

        vnt = setindex!!(vnt, fill(1.0, 4), @varname(j[1:4]))
        @test keys(vnt) == [
            @varname(a),
            @varname(b),
            @varname(c.x.y),
            @varname(d[4]),
            @varname(e.f[3, 3].g.h[2, 4, 1].i),
            @varname(j[1]),
            @varname(j[2]),
            @varname(j[3]),
            @varname(j[4]),
        ]
        @test values(vnt) == [1.0, [1, 15, 3], [10], -1.0, 2.0, fill(1.0, 4)...]

        vnt = setindex!!(vnt, "a", @varname(j[6]))
        @test keys(vnt) == [
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
        ]
        @test values(vnt) == [1.0, [1, 15, 3], [10], -1.0, 2.0, fill(1.0, 4)..., "a"]

        vnt = setindex!!(vnt, 1.0, @varname(n[2].a))
        @test keys(vnt) == [
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
        ]
        @test values(vnt) == [1.0, [1, 15, 3], [10], -1.0, 2.0, fill(1.0, 4)..., "a", 1.0]

        vnt = setindex!!(vnt, SizedThing((3, 1, 4)), @varname(o[2:4, 5:5, 11:14]))
        @test keys(vnt) == [
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
            @varname(o[2:4, 5:5, 11:14]),
        ]
        @test values(vnt) == [
            1.0,
            [1, 15, 3],
            [10],
            -1.0,
            2.0,
            fill(1.0, 4)...,
            "a",
            1.0,
            SizedThing((3, 1, 4)),
        ]

        vnt = setindex!!(vnt, SizedThing((3, 1, 4)), @varname(p[2, 1][2:4, 5:5, 11:14]))
        @test keys(vnt) == [
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
            @varname(o[2:4, 5:5, 11:14]),
            @varname(p[2, 1][2:4, 5:5, 11:14]),
        ]
        @test values(vnt) == [
            1.0,
            [1, 15, 3],
            [10],
            -1.0,
            2.0,
            fill(1.0, 4)...,
            "a",
            1.0,
            SizedThing((3, 1, 4)),
            SizedThing((3, 1, 4)),
        ]
        test_invariants(vnt)
    end

    @testset "length" begin
        # Type inference for length fails in some cases on Julia versions < 1.11
        inference_broken = VERSION < v"1.11"

        vnt = VarNamedTuple()
        @test @inferred(length(vnt)) == 0

        vnt = setindex!!(vnt, 1.0, @varname(a))
        @test @inferred(length(vnt)) == 1

        vnt = setindex!!(vnt, [1, 2, 3], @varname(b))
        @test @inferred(length(vnt)) == 2

        vnt = setindex!!(vnt, 15, @varname(b[2]))
        @test @inferred(length(vnt)) == 2

        vnt = setindex!!(vnt, [10, 11], @varname(c.x.y))
        @test @inferred(length(vnt)) == 3

        vnt = setindex!!(vnt, -1.0, @varname(d[4]))
        @test @inferred(length(vnt)) == 4 broken = inference_broken

        vnt = setindex!!(vnt, ["a", "b"], @varname(d[1:2]))
        @test @inferred(length(vnt)) == 6 broken = inference_broken

        vnt = setindex!!(vnt, 2.0, @varname(e.f[3].g.h[2].i))
        vnt = setindex!!(vnt, 3.0, @varname(e.f[3].g.h[2].j))
        @test @inferred(length(vnt)) == 8 broken = inference_broken

        vnt = setindex!!(vnt, SizedThing((3, 2)), @varname(x[1, 2:4, 2, 1:2, 3]))
        @test @inferred(length(vnt)) == 14 broken = inference_broken

        vnt = setindex!!(vnt, SizedThing((3, 2)), @varname(x[1, 4:6, 2, 1:2, 3]))
        @test @inferred(length(vnt)) == 14 broken = inference_broken

        vnt = setindex!!(vnt, [:a, :b], @varname(y[4][3][2][1:2]))
        @test @inferred(length(vnt)) == 16 broken = inference_broken
        test_invariants(vnt)
    end

    @testset "empty" begin
        # test_invariants already checks that many different kinds of VarNamedTuples can be
        # emptied with empty and empty!!. What remains to check here is that
        # 1) isempty gives the expected results:
        vnt = VarNamedTuple()
        @test @inferred(isempty(vnt)) == true
        vnt = setindex!!(vnt, 1.0, @varname(a))
        @test @inferred(isempty(vnt)) == false
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vnt = setindex!!(vnt, [], @varname(a[1]))
        @test @inferred(isempty(vnt)) == false
        test_invariants(vnt)

        # 2) empty!! keeps PartialArrays in place:
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, [1, 2, 3], @varname(a[1:3])))
        vnt = @inferred(empty!!(vnt))
        @test !haskey(vnt, @varname(a[1]))
        @test !haskey(vnt, @varname(a[1:3]))
        @test haskey(vnt, @varname(a))
        @test_throws BoundsError getindex(vnt, @varname(a[1]))
        @test_throws BoundsError getindex(vnt, @varname(a[1:3]))
        # It's an empty PartialArray now, so attempting to extract anything will fail
        @test_throws ArgumentError getindex(vnt, @varname(a))
        vnt = @inferred(setindex!!(vnt, [1, 2, 3], @varname(a[2:4])))
        @test @inferred(getindex(vnt, @varname(a[2:4]))) == [1, 2, 3]
        @test haskey(vnt, @varname(a[2:4]))
        @test !haskey(vnt, @varname(a[1]))
        test_invariants(vnt)
    end

    @testset "densification" begin
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 1])))
        @test @inferred(getindex(vnt, @varname(a.b[1].c))) == fill(1.0, (1, 1))
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 1])))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 2])))
        @test @inferred(getindex(vnt, @varname(a.b[1].c))) == fill(1.0, (1, 2))
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 1])))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[2, 1])))
        @test @inferred(getindex(vnt, @varname(a.b[1].c))) == fill(1.0, (2, 1))
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 1])))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[1, 2])))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[2, 1])))
        @test_throws ArgumentError @inferred(getindex(vnt, @varname(a.b[1].c)))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[2, 2])))
        @test @inferred(getindex(vnt, @varname(a.b[1].c))) == fill(1.0, (2, 2))
        vnt = @inferred(setindex!!(vnt, 1.0, @varname(a.b[1].c[3, 3])))
        @test_throws ArgumentError @inferred(getindex(vnt, @varname(a.b[1].c)))
        test_invariants(vnt)

        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, SizedThing((2,)), @varname(x[1:2])))
        @test_throws ArgumentError @inferred(getindex(vnt, @varname(x)))
        test_invariants(vnt)
    end

    @testset "block variables" begin
        # Tests for setting and getting block variables, i.e. variables that have a non-zero
        # size in a PartialArray, but are not Arrays themselves.
        expected_err = ArgumentError("""
            A non-Array value set with a range of indices must be retrieved with the same
            range of indices.
            """)
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, SizedThing((3,)), @varname(x[2:4])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(x[2:4]))
        @test @inferred(getindex(vnt, @varname(x[2:4]))) == SizedThing((3,))
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
                (fill(2.0, length(index)), SizedThing((length(index),)))
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
        val = SizedThing((2, 2))
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

        # A lot like above, but with extra indices that are not ranges.
        val = SizedThing((2, 2))
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[2, 1:2, 3, 1:2, 4])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[2, 1:2, 3, 1:2, 4]))
        @test @inferred(getindex(vnt, @varname(y.z[2, 1:2, 3, 1:2, 4]))) == val
        @test !haskey(vnt, @varname(y.z[2, 1, 3, 1, 4]))
        @test_throws expected_err getindex(vnt, @varname(y.z[2, 1, 3, 1, 4]))

        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[2, 2:3, 3, 2:3, 4])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[2, 2:3, 3, 2:3, 4]))
        @test @inferred(getindex(vnt, @varname(y.z[2, 2:3, 3, 2:3, 4]))) == val
        @test !haskey(vnt, @varname(y.z[2, 1:2, 3, 1:2, 4]))
        @test_throws BoundsError getindex(vnt, @varname(y.z[2, 1:2, 3, 1:2, 4]))

        vnt = @inferred(setindex!!(vnt, val, @varname(y.z[3, 2:3, 3, 2:3, 4])))
        test_invariants(vnt)
        @test haskey(vnt, @varname(y.z[2, 2:3, 3, 2:3, 4]))
        @test @inferred(getindex(vnt, @varname(y.z[2, 2:3, 3, 2:3, 4]))) == val
        @test haskey(vnt, @varname(y.z[3, 2:3, 3, 2:3, 4]))
        # Type inference fails on this one for Julia versions < 1.11
        @test @inferred(getindex(vnt, @varname(y.z[3, 2:3, 3, 2:3, 4]))) == val
    end

    @testset "map and friends" begin
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1, @varname(a)))
        vnt = @inferred(setindex!!(vnt, [2, 2], @varname(b[1:2])))
        vnt = @inferred(setindex!!(vnt, [3.0], @varname(c.d)))
        vnt = @inferred(setindex!!(vnt, "a", @varname(e.f[3].g.h[2].i)))
        # The below can't be type stable because the element type of `h` depends on whether
        # we are setting `h[2].j` (which overwrites the earlier `h[2]`) or some other
        # `h[index].j` (which would leave both `h[2].i` and `h[index].j` in the same array).
        vnt = setindex!!(vnt, 5.0, @varname(e.f[3].g.h[2].j))
        vnt = @inferred(
            setindex!!(vnt, SizedThing((2, 2)), @varname(y.z[3, 2:3, 3, 2:3, 4]))
        )
        vnt = @inferred(
            templated_setindex!!(vnt, SizedThing((2,)), @varname(v[:]), randn(2))
        )
        vnt = @inferred(setindex!!(vnt, "", @varname(w[4][3][2, 1])))
        test_invariants(vnt)

        struct AnotherSizedThing{T<:Tuple}
            size::T
        end
        Base.size(st::AnotherSizedThing) = st.size

        call_counter = 0
        function f_val(val)
            call_counter += 1
            if val isa Int
                return val + 10
            elseif val isa AbstractVector{Int}
                return val .+ 10
            elseif val isa Float64
                return val + 1.0
            elseif val isa AbstractVector{Float64}
                return val .- 1.0
            elseif val isa String
                return string(val, "b")
            elseif val isa SizedThing
                return AnotherSizedThing(size(val))
            else
                error("Unexpected value type $(typeof(val))")
            end
        end

        f_pair(pair) = f_val(pair.second)

        val_reduction = mapreduce(pair -> pair.second, vcat, vnt; init=Any[])
        @test val_reduction == vcat(
            Any[], 1, [2, 2], [3.0], "a", 5.0, SizedThing((2, 2)), SizedThing((2,)), ""
        )
        key_reduction = mapreduce(pair -> pair.first, vcat, vnt; init=Any[])
        @test key_reduction == vcat(
            @varname(a),
            @varname(b[1]),
            @varname(b[2]),
            @varname(c.d),
            @varname(e.f[3].g.h[2].i),
            @varname(e.f[3].g.h[2].j),
            @varname(y.z[3, 2:3, 3, 2:3, 4]),
            @varname(v[:]),
            @varname(w[4][3][2, 1]),
        )

        call_counter = 0
        reduction = mapreduce(f_pair, vcat, vnt; init=Any[])
        @test reduction == vcat(
            Any[],
            11,
            [12, 12],
            [2.0],
            "ab",
            6.0,
            AnotherSizedThing((2, 2)),
            AnotherSizedThing((2,)),
            "b",
        )
        # Check that f_pair gets called exactly once per element.
        @test call_counter == length(keys(vnt))

        # TODO(mhauru) This should hopefully be type stable, but fails to be so because of
        # some complex VarNames being too much for constant propagation. See comment in
        # src/varnamedtuple.jl for more.
        call_counter = 0
        vnt_mapped = map_pairs!!(f_pair, copy(vnt))
        # Check that f_pair gets called exactly once per element.
        @test call_counter == length(keys(vnt))
        @test vnt_mapped == map_values!!(f_val, copy(vnt))
        test_invariants(vnt_mapped)
        @test @inferred(getindex(vnt_mapped, @varname(a))) == 11
        @test @inferred(getindex(vnt_mapped, @varname(b[1:2]))) == [12, 12]
        @test @inferred(getindex(vnt_mapped, @varname(c.d))) == [2.0]
        @test @inferred(getindex(vnt_mapped, @varname(e.f[3].g.h[2].i))) == "ab"
        @test @inferred(getindex(vnt_mapped, @varname(e.f[3].g.h[2].j))) == 6.0
        @test @inferred(getindex(vnt_mapped, @varname(y.z[3, 2:3, 3, 2:3, 4]))) ==
            AnotherSizedThing((2, 2))
        @test @inferred(getindex(vnt_mapped, @varname(v[:]))) == AnotherSizedThing((2,))
        @test @inferred(getindex(vnt_mapped, @varname(w[4][3][2, 1]))) == "b"

        call_counter = 0
        vnt_applied = copy(vnt)
        vnt_applied = @inferred(apply!!(f_val, vnt_applied, @varname(a)))
        @test call_counter == 1
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(a))) == 11
        @test @inferred(getindex(vnt_applied, @varname(b[1:2]))) == [2, 2]

        vnt_applied = @inferred(apply!!(f_val, vnt_applied, @varname(b[1:2])))
        # Unlike map_pairs!!, apply!! operates on the whole value at once, rather than
        # element-wise, so this is only one more call.
        @test call_counter == 2
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(a))) == 11
        @test @inferred(getindex(vnt_applied, @varname(b[1:2]))) == [12, 12]

        vnt_applied = @inferred(apply!!(f_val, vnt_applied, @varname(c.d)))
        @test call_counter == 3
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(c.d))) == [2.0]

        vnt_applied = begin
            # The @inferred fails on Julia 1.10.
            @static if VERSION < v"1.11"
                apply!!(f_val, vnt_applied, @varname(e.f[3].g.h[2].i))
            else
                @inferred(apply!!(f_val, vnt_applied, @varname(e.f[3].g.h[2].i)))
            end
        end
        @test call_counter == 4
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(e.f[3].g.h[2].i))) == "ab"
        @test @inferred(getindex(vnt_applied, @varname(e.f[3].g.h[2].j))) == 5.0

        vnt_applied = begin
            # The @inferred fails on Julia 1.10.
            @static if VERSION < v"1.11"
                apply!!(f_val, vnt_applied, @varname(e.f[3].g.h[2].j))
            else
                @inferred(apply!!(f_val, vnt_applied, @varname(e.f[3].g.h[2].j)))
            end
        end
        @test call_counter == 5
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(e.f[3].g.h[2].j))) == 6.0

        # This can't be type stable because y.z might have many elements set, and we can't
        # know at compile time that this sets the only one, thus allowing the element type
        # to be AnotherSizedThing.
        vnt_applied = apply!!(f_val, vnt_applied, @varname(y.z[3, 2:3, 3, 2:3, 4]))
        @test call_counter == 6
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(y.z[3, 2:3, 3, 2:3, 4]))) ==
            AnotherSizedThing((2, 2))

        vnt_applied = apply!!(f_val, vnt_applied, @varname(v[:]))
        @test call_counter == 7
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(v[:]))) == AnotherSizedThing((2,))

        vnt_applied = @inferred(apply!!(f_val, vnt_applied, @varname(w[4][3][2, 1])))
        @test call_counter == 8
        test_invariants(vnt_applied)
        @test @inferred(getindex(vnt_applied, @varname(w[4][3][2, 1]))) == "b"

        # map a function that maps every key => value pair to key => key.
        # For this, use a simpler VarNamedTuple, because block variables don't work with
        # this mapping function. It also allows us to check type stability.
        vnt = VarNamedTuple()
        vnt = @inferred(setindex!!(vnt, 1, @varname(a)))
        vnt = @inferred(setindex!!(vnt, 2, @varname(b[2])))
        vnt = @inferred(setindex!!(vnt, [3.0], @varname(c.d)))
        vnt = @inferred(setindex!!(vnt, :oi, @varname(y.z[3, 2, 3, 2, 4])))
        vnt = @inferred(setindex!!(vnt, "", @varname(w[4][2, 1])))

        get_key(pair) = pair.first
        vnt_key_mapped = @inferred(map_pairs!!(get_key, copy(vnt)))
        vnt_key_mapped_expected = VarNamedTuple()
        for k in keys(vnt)
            vnt_key_mapped_expected = setindex!!(vnt_key_mapped_expected, k, k)
        end
        @test vnt_key_mapped == vnt_key_mapped_expected
    end
end

end
