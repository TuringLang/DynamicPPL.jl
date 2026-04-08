module DynamicPPLTransformedValuesTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
using DynamicPPL
using Distributions
using Random: Xoshiro
using Test

@testset "TransformedValue API" begin
    @testset "get_transform, get_internal_value, set_internal_value" begin
        ft = FixedTransform(nothing)
        transforms = [DynamicLink(), Unlink(), NoTransform(), ft]
        @testset "$tfm" for tfm in transforms
            tv = TransformedValue([1.0, 2.0], tfm)
            @test get_transform(tv) == tfm
            @test get_internal_value(tv) == [1.0, 2.0]

            tv2 = set_internal_value(tv, [3.0, 4.0])
            @test get_internal_value(tv2) == [3.0, 4.0]
            @test get_transform(tv2) == tfm
        end
    end

    @testset "==" begin
        tv1 = TransformedValue([1.0], NoTransform())
        tv2 = TransformedValue([1.0], NoTransform())
        tv3 = TransformedValue([2.0], NoTransform())
        tv4 = TransformedValue([1.0], DynamicLink())
        @test tv1 == tv2
        @test tv1 != tv3
        @test tv1 != tv4
    end

    @testset "get_raw_value" begin
        struct DummyDist <: Distributions.ContinuousMultivariateDistribution end

        @testset "NoTransform" begin
            raw = randn(3)
            tv = TransformedValue(raw, NoTransform())
            @test get_raw_value(tv) == raw
            @test get_raw_value(tv, DummyDist()) == raw
        end

        @testset "FixedTransform" begin
            dist = Beta(2, 5)
            ft = FixedTransform(Bijectors.VectorBijectors.from_linked_vec(dist))
            flink = Bijectors.VectorBijectors.to_linked_vec(dist)
            raw_val = rand(dist)
            linked_val = flink(raw_val)
            tv = TransformedValue(linked_val, ft)
            @test get_raw_value(tv) ≈ raw_val
            @test get_raw_value(tv, dist) ≈ raw_val
        end

        @testset "DynamicLink" begin
            dist = Beta(2, 5)
            flink = Bijectors.VectorBijectors.to_linked_vec(dist)
            raw_val = rand(Xoshiro(468), dist)
            linked_val = flink(raw_val)
            tv = TransformedValue(linked_val, DynamicLink())
            @test_throws ArgumentError get_raw_value(tv)
            @test get_raw_value(tv, dist) ≈ raw_val
        end

        @testset "Unlink" begin
            dist = Dirichlet([1.0, 2.0, 3.0])
            fvec = Bijectors.VectorBijectors.to_vec(dist)
            raw_val = rand(Xoshiro(468), dist)
            vec_val = fvec(raw_val)
            tv = TransformedValue(vec_val, Unlink())
            @test_throws ArgumentError get_raw_value(tv)
            @test_throws "dynamic transforms" get_raw_value(tv)
            @test get_raw_value(tv, dist) ≈ raw_val
        end
    end
end

@testset "get_fixed_transforms" begin
    # Check that `get_fixed_transforms` does indeed return a VNT of the correct transforms.
    xdist = Beta(2, 5)
    ydist = InverseGamma(2, 3)

    @model function two_var_model()
        x ~ xdist
        return y ~ ydist
    end
    model = two_var_model()

    function expected_transform(dist, linked)
        f = if linked
            Bijectors.VectorBijectors.from_linked_vec
        else
            Bijectors.VectorBijectors.from_vec
        end
        return FixedTransform(f(dist))
    end

    @testset "$strategy" for (strategy, x_linked, y_linked) in [
        (LinkAll(), true, true),
        (UnlinkAll(), false, false),
        (LinkSome(Set([@varname(x)]), UnlinkAll()), true, false),
        (LinkSome(Set([@varname(y)]), UnlinkAll()), false, true),
    ]
        transforms = DynamicPPL.get_fixed_transforms(model, strategy)
        expected = @vnt begin
            x := expected_transform(xdist, x_linked)
            y := expected_transform(ydist, y_linked)
        end
        @test transforms == expected
    end
end

@testset "infer_transform_strategy_from_values" begin
    # If all are DynamicLink -> LinkAll
    vnt_linked = @vnt begin
        x := TransformedValue([1.0], DynamicLink())
        y := TransformedValue([2.0, 3.0], DynamicLink())
    end
    @test DynamicPPL.infer_transform_strategy_from_values(vnt_linked) isa LinkAll

    # If all are Unlink -> UnlinkAll
    vnt_unlinked = @vnt begin
        x := TransformedValue([1.0], Unlink())
        y := TransformedValue([2.0, 3.0], Unlink())
    end
    @test DynamicPPL.infer_transform_strategy_from_values(vnt_unlinked) isa UnlinkAll

    # Mixed -> WithTransforms
    vnt_mixed = @vnt begin
        x := TransformedValue([1.0], DynamicLink())
        y := TransformedValue([2.0, 3.0], Unlink())
    end
    strategy = DynamicPPL.infer_transform_strategy_from_values(vnt_mixed)
    @test strategy isa DynamicPPL.WithTransforms
    @test DynamicPPL.target_transform(strategy, @varname(x)) isa DynamicLink
    @test DynamicPPL.target_transform(strategy, @varname(y)) isa Unlink

    # FixedTransform values -> WithTransforms
    ft = FixedTransform(Bijectors.VectorBijectors.from_linked_vec(Beta(2, 5)))
    vnt_fixed = @vnt begin
        x := TransformedValue([0.5], ft)
    end
    strategy = DynamicPPL.infer_transform_strategy_from_values(vnt_fixed)
    @test strategy isa DynamicPPL.WithTransforms
    @test DynamicPPL.target_transform(strategy, @varname(x)) == ft
end

@testset "apply_transform_strategy" begin
    @testset "Dynamic transforms" begin
        vn = @varname(x)
        @testset "$dist" for dist in
                             [Beta(2, 5), InverseGamma(2, 3), Dirichlet([1.0, 2.0, 3.0])]
            raw_val = rand(dist)

            # Build TransformedValues for the same raw_val, but with different input transforms
            flink = Bijectors.VectorBijectors.to_linked_vec(dist)
            fvec = Bijectors.VectorBijectors.to_vec(dist)
            linked_vec, logjac = Bijectors.with_logabsdet_jacobian(flink, raw_val)
            unlinked_vec = fvec(raw_val)

            target_strategies_and_expected_link = [
                (LinkAll(), true),
                (UnlinkAll(), false),
                (LinkSome(Set([vn]), UnlinkAll()), true),
                (UnlinkSome(Set([vn]), LinkAll()), false),
            ]

            @testset "$strategy" for (strategy, expected_link) in
                                     target_strategies_and_expected_link
                @testset "Linked input" begin
                    tv = TransformedValue(linked_vec, DynamicLink())
                    new_raw, new_tv, new_logjac = DynamicPPL.apply_transform_strategy(
                        strategy, tv, vn, dist
                    )
                    @test new_raw ≈ raw_val
                    if expected_link
                        @test new_tv.value == linked_vec
                        @test new_tv.transform == DynamicLink()
                        @test new_logjac ≈ logjac
                    else
                        @test new_tv.value ≈ raw_val
                        @test new_tv.transform == NoTransform()
                        @test iszero(new_logjac)
                    end
                end

                @testset "Vectorised input" begin
                    tv = TransformedValue(unlinked_vec, Unlink())
                    new_raw, new_tv, new_logjac = DynamicPPL.apply_transform_strategy(
                        strategy, tv, vn, dist
                    )
                    @test new_raw ≈ raw_val
                    if expected_link
                        @test new_tv.value ≈ linked_vec
                        @test new_tv.transform == DynamicLink()
                        @test new_logjac ≈ logjac
                    else
                        @test new_tv.value == unlinked_vec
                        @test new_tv.transform == Unlink()
                        @test iszero(new_logjac)
                    end
                end

                @testset "Raw input" begin
                    tv = TransformedValue(raw_val, NoTransform())
                    new_raw, new_tv, new_logjac = DynamicPPL.apply_transform_strategy(
                        strategy, tv, vn, dist
                    )
                    @test new_raw ≈ raw_val
                    if expected_link
                        @test new_tv.value ≈ linked_vec
                        @test new_tv.transform == DynamicLink()
                        @test new_logjac ≈ logjac
                    else
                        @test new_tv.value == raw_val
                        @test new_tv.transform == NoTransform()
                        @test iszero(new_logjac)
                    end
                end
            end
        end
    end

    @testset "FixedTransform" begin
        dist = Beta(2, 2)
        vn = @varname(x)
        ft = FixedTransform(Bijectors.VectorBijectors.from_linked_vec(dist))
        wrong_ft = FixedTransform(
            Bijectors.VectorBijectors.from_linked_vec(InverseGamma(2, 3))
        )

        # Create a TransformedValue with ft
        flink = Bijectors.VectorBijectors.to_linked_vec(dist)
        raw_val = rand(Xoshiro(468), dist)
        linked_val, logjac = Bijectors.with_logabsdet_jacobian(flink, raw_val)
        tv = TransformedValue(linked_val, ft)

        # Matching transform should work
        strategy_ok = DynamicPPL.WithTransforms(VarNamedTuple(; x=ft), UnlinkAll())
        new_raw, new_tv, new_logjac = DynamicPPL.apply_transform_strategy(
            strategy_ok, tv, vn, dist
        )
        @test new_raw ≈ raw_val
        @test new_tv.value == linked_val
        @test new_tv.transform == ft
        @test new_logjac ≈ logjac

        # Mismatched transform should error
        strategy_bad = DynamicPPL.WithTransforms(VarNamedTuple(; x=wrong_ft), UnlinkAll())
        @test_throws ErrorException DynamicPPL.apply_transform_strategy(
            strategy_bad, tv, vn, dist
        )
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
