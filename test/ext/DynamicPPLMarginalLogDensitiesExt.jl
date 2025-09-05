module MarginalLogDensitiesExtTests

using DynamicPPL, Distributions, Test
using MarginalLogDensities
using ADTypes: AutoForwardDiff

@testset "MarginalLogDensities" begin
    # Simple test case.
    @model function demo()
        x ~ MvNormal(zeros(2), [1, 1])
        return y ~ Normal(0, 1)
    end
    model = demo()
    # Marginalize out `x`.

    for vn in [@varname(x), :x]
        for getlogprob in [DynamicPPL.getlogprior, DynamicPPL.getlogjoint]
            marginalized = marginalize(
                model, [vn], getlogprob; hess_adtype=AutoForwardDiff()
            )
            # Compute the marginal log-density of `y = 0.0`.
            @test marginalized([0.0]) ≈ logpdf(Normal(0, 1), 0.0) atol = 1e-5
        end
    end
end

end
