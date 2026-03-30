module DynamicPPLTransformedValuesTests

using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
using DynamicPPL
using Distributions
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

end
