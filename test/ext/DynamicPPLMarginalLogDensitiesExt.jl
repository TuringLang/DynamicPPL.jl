module MarginalLogDensitiesExtTests

using Bijectors: Bijectors
using DynamicPPL, Distributions, Test
using MarginalLogDensities
using ADTypes: AutoForwardDiff

@testset "MarginalLogDensities" begin
    @testset "Basic usage" begin
        @model function demo()
            x ~ MvNormal(zeros(2), [1, 1])
            return y ~ Normal(0, 1)
        end
        model = demo()
        vi = VarInfo(model)
        # Marginalize out `x`.
        for vn in [@varname(x), :x]
            for getlogprob in [DynamicPPL.getlogprior, DynamicPPL.getlogjoint]
                marginalized = marginalize(
                    model, [vn], vi, getlogprob; hess_adtype=AutoForwardDiff()
                )
                for y in range(-5, 5; length=100)
                    @test marginalized([y]) ≈ logpdf(Normal(0, 1), y) atol = 1e-5
                end
            end
        end
    end

    @testset "Respects linked status of VarInfo" begin
        @model function f()
            x ~ Normal()
            return y ~ Beta(2, 2)
        end
        model = f()
        vi_unlinked = VarInfo(model)
        vi_linked = DynamicPPL.link(vi_unlinked, model)

        @testset "unlinked VarInfo" begin
            mx = marginalize(model, [@varname(x)], vi_unlinked)
            for x in range(0.01, 0.99; length=10)
                @test mx([x]) ≈ logpdf(Beta(2, 2), x)
            end
            # generally when marginalising Beta it doesn't go to zero
            my = marginalize(model, [@varname(y)], vi_unlinked)
            diff = my([0.0]) - logpdf(Normal(), 0.0)
            for x in range(-5, 5; length=10)
                @test my([x]) ≈ logpdf(Normal(), x) + diff
            end
        end

        @testset "linked VarInfo" begin
            mx = marginalize(model, [@varname(x)], vi_linked)
            binv = Bijectors.inverse(Bijectors.bijector(Beta(2, 2)))
            for y_linked in range(-5, 5; length=10)
                y_unlinked = binv(y_linked)
                @test mx([y_linked]) ≈ logpdf(Beta(2, 2), y_unlinked)
            end
            # generally when marginalising Beta it doesn't go to zero
            my = marginalize(model, [@varname(y)], vi_linked)
            diff = my([0.0]) - logpdf(Normal(), 0.0)
            for x in range(-5, 5; length=10)
                @test my([x]) ≈ logpdf(Normal(), x) + diff
            end
        end
    end
end

end
