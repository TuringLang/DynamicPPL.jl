module DynamicPPLChainsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra
using Test

@testset "ParamsWithStats from VarInfo" begin
    @model function f(z)
        x ~ Normal()
        y := x + 1
        return z ~ Normal(y)
    end
    z = 1.0
    model = f(z)

    @testset "with reevaluation" begin
        ps = ParamsWithStats(VarInfo(model), model)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        @test haskey(ps.stats, :logprior)
        @test haskey(ps.stats, :loglikelihood)
        @test haskey(ps.stats, :logjoint)
        @test length(ps.stats) == 3
        @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
        @test ps.params[@varname(y)] ≈ ps.params[@varname(x)] + 1
        @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
        @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(y)]), z)
    end

    @testset "without colon_eq" begin
        ps = ParamsWithStats(VarInfo(model), model; include_colon_eq=false)
        @test haskey(ps.params, @varname(x))
        @test length(ps.params) == 1
        @test haskey(ps.stats, :logprior)
        @test haskey(ps.stats, :loglikelihood)
        @test haskey(ps.stats, :logjoint)
        @test length(ps.stats) == 3
        @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
        @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
        @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(x)] + 1), z)
    end

    @testset "without log probs" begin
        ps = ParamsWithStats(VarInfo(model), model; include_log_probs=false)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        @test isempty(ps.stats)
    end

    @testset "no reevaluation" begin
        # Without VAIM, it should error
        @test_throws ErrorException ParamsWithStats(VarInfo(model))
        # With VAIM, it should work
        vi = DynamicPPL.setaccs!!(VarInfo(model), (DynamicPPL.RawValueAccumulator(true),))
        vi = last(DynamicPPL.evaluate!!(model, vi))
        ps = ParamsWithStats(vi)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        # Because we didn't evaluate with log prob accumulators, there should be no stats
        @test isempty(ps.stats)
    end
end

_safe_length(x) = length(x)
# This actually gives N^2 elements, although there are only really N(N+1)/2 parameters in
# the Cholesky factor. It doesn't really matter because we are comparing like for like i.e.
# both sides of the sum will have the same overcounting.
_safe_length(c::LinearAlgebra.Cholesky) = length(c.UL)

@testset "ParamsWithStats from LogDensityFunction" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        if m.f === DynamicPPL.TestUtils.demo_static_transformation
            # TODO(mhauru) These tests are broken for demo_static_transformation because
            # vi[vn] doesn't know which transform it should apply to the internally stored
            # value. This requires a rethink, either of StaticTransformation or of what the
            # comparison in this test should be.
            @test false broken = true
            continue
        end
        @testset "$islinked" for islinked in (false, true)
            unlinked_vi = VarInfo(m)
            vi = if islinked
                DynamicPPL.link!!(unlinked_vi, m)
            else
                unlinked_vi
            end
            params = [x for x in vi[:]]

            # Get the ParamsWithStats using LogDensityFunction
            ldf = DynamicPPL.LogDensityFunction(m, getlogjoint, vi)
            ps = ParamsWithStats(params, ldf)

            # The keys are not necessarily going to be the same, because `ps.params` was
            # obtained via RawValueAccumulator, which only stores raw values. However, `vi`
            # stores TransformedValue objects. So, if you have something like
            #    x[4:5] ~ MvNormal(zeros(2), I)
            # then `ps.params` will have keys `x[4]` and `x[5]` (since it just contains a
            # PartialArray with those two elements unmasked), whereas `vi` will have the key
            # `x[4:5]` which stores an ArrayLikeBlock with two elements.
            #
            # What we CAN do, though, is to check the size of the thing obtained by
            # indexing into the keys. For `ps.params`, indexing into `x[4]` and `x[5]` will
            # give two floats, each of "length" 1. For `vi`, indexing into `x[4:5]` will
            # give a single object that has length 2. So we can check that the total number
            # of _things_ contained inside is the same.
            #
            # Unfortunately, we need _safe_length to handle Cholesky.
            @test sum(_safe_length(ps.params[vn]) for vn in keys(ps.params)) ==
                sum(_safe_length(vi[vn]) for vn in keys(vi))

            # Iterate over all variables to check that their values match
            for vn in keys(vi)
                @test ps.params[vn] == vi[vn]
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
