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
        @testset for getlogprob in [DynamicPPL.getlogprior, DynamicPPL.getlogjoint]
            marginalized = marginalize(
                model,
                [@varname(x)];
                varinfo=vi,
                getlogprob=getlogprob,
                hess_adtype=AutoForwardDiff(),
            )
            for y in range(-5, 5; length=100)
                @test marginalized([y]) ≈ logpdf(Normal(0, 1), y) atol = 1e-5
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
            mx = marginalize(model, [@varname(x)]; varinfo=vi_unlinked)
            for x in range(0.01, 0.99; length=10)
                @test mx([x]) ≈ logpdf(Beta(2, 2), x)
            end
            # generally when marginalising Beta it doesn't go to zero
            # https://github.com/TuringLang/DynamicPPL.jl/pull/1036#discussion_r2349388067
            my = marginalize(model, [@varname(y)]; varinfo=vi_unlinked)
            diff = my([0.0]) - logpdf(Normal(), 0.0)
            for x in range(-5, 5; length=10)
                @test my([x]) ≈ logpdf(Normal(), x) + diff
            end
        end

        @testset "linked VarInfo" begin
            mx = marginalize(model, [@varname(x)]; varinfo=vi_linked)
            binv = Bijectors.inverse(Bijectors.bijector(Beta(2, 2)))
            for y_linked in range(-5, 5; length=10)
                y_unlinked = binv(y_linked)
                @test mx([y_linked]) ≈ logpdf(Beta(2, 2), y_unlinked)
            end
            # generally when marginalising Beta it doesn't go to zero
            # https://github.com/TuringLang/DynamicPPL.jl/pull/1036#discussion_r2349388067
            my = marginalize(model, [@varname(y)]; varinfo=vi_linked)
            diff = my([0.0]) - logpdf(Normal(), 0.0)
            for x in range(-5, 5; length=10)
                @test my([x]) ≈ logpdf(Normal(), x) + diff
            end
        end
    end

    @testset "retrieving VarInfo from MLD" begin
        @model function f()
            x ~ Normal()
            return y ~ Beta(2, 2)
        end
        model = f()
        vi_unlinked = VarInfo(model)
        vi_linked = DynamicPPL.link(vi_unlinked, model)

        @testset "unlinked VarInfo" begin
            mx = marginalize(model, [@varname(x)]; varinfo=vi_unlinked)
            mx([0.5]) # evaluate at some point to force calculation of Laplace approx
            vi = VarInfo(mx)
            @test vi[@varname(x)] ≈ mode(Normal())
            vi = VarInfo(mx, [0.5]) # this 0.5 is unlinked
            @test vi[@varname(x)] ≈ mode(Normal())
            @test vi[@varname(y)] ≈ 0.5
        end

        @testset "linked VarInfo" begin
            mx = marginalize(model, [@varname(x)]; varinfo=vi_linked)
            mx([0.5]) # evaluate at some point to force calculation of Laplace approx
            vi = VarInfo(mx)
            @test vi[@varname(x)] ≈ mode(Normal())
            vi = VarInfo(mx, [0.5]) # this 0.5 is linked
            binv = Bijectors.inverse(Bijectors.bijector(Beta(2, 2)))
            @test vi[@varname(x)] ≈ mode(Normal())
            # when using getindex it always returns unlinked values
            @test vi[@varname(y)] ≈ binv(0.5)
        end
    end
end

end
