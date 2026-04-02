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

@testset "apply_transform_strategy: FixedTransform mismatch errors" begin
    dist = Beta(2, 5)
    vn = @varname(x)
    ft = FixedTransform(Bijectors.VectorBijectors.from_linked_vec(dist))
    wrong_ft = FixedTransform(Bijectors.VectorBijectors.from_linked_vec(InverseGamma(2, 3)))

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
    @test new_tv.transform == ft
    @test new_logjac ≈ logjac

    # Mismatched transform should error
    strategy_bad = DynamicPPL.WithTransforms(VarNamedTuple(; x=wrong_ft), UnlinkAll())
    @test_throws ErrorException DynamicPPL.apply_transform_strategy(
        strategy_bad, tv, vn, dist
    )
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
