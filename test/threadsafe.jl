module DynamicPPLThreadSafeTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Distributions
using DynamicPPL
using Test

@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
const gdemo_default = gdemo_d()

@testset "threadsafe.jl" begin
    @testset "constructor" begin
        vi = VarInfo(gdemo_default)
        threadsafe_vi = @inferred DynamicPPL.ThreadSafeVarInfo(vi)

        @test threadsafe_vi.varinfo === vi
        @test threadsafe_vi.accs_by_thread isa Vector{<:DynamicPPL.AccumulatorTuple}
        @test length(threadsafe_vi.accs_by_thread) == Threads.maxthreadid()
        expected_accs = DynamicPPL.AccumulatorTuple(
            (DynamicPPL.split(acc) for acc in DynamicPPL.getaccs(vi))...
        )
        @test all(accs == expected_accs for accs in threadsafe_vi.accs_by_thread)
    end

    @testset "setthreadsafe" begin
        @model f() = x ~ Normal()
        model = f()
        @test !DynamicPPL.requires_threadsafe(model)
        model = setthreadsafe(model, true)
        @test DynamicPPL.requires_threadsafe(model)
        model = setthreadsafe(model, false)
        @test !DynamicPPL.requires_threadsafe(model)
    end

    # TODO: Add more tests of the public API
    @testset "API" begin
        vi = VarInfo(gdemo_default)
        threadsafe_vi = DynamicPPL.ThreadSafeVarInfo(vi)

        lp = getlogjoint(vi)
        @test getlogjoint(threadsafe_vi) == lp

        threadsafe_vi = DynamicPPL.acclogprior!!(threadsafe_vi, 42)
        @test threadsafe_vi.accs_by_thread[Threads.threadid()][:LogPrior].logp == 42
        @test getlogjoint(vi) == lp
        # float addition might lead to rounding errors so use approx rather than ==
        @test getlogjoint(threadsafe_vi) ≈ lp + 42

        threadsafe_vi = DynamicPPL.resetaccs!!(threadsafe_vi)
        @test iszero(getlogjoint(threadsafe_vi))
        expected_accs = DynamicPPL.AccumulatorTuple(
            (DynamicPPL.split(acc) for acc in DynamicPPL.getaccs(threadsafe_vi.varinfo))...
        )
        @test all(accs == expected_accs for accs in threadsafe_vi.accs_by_thread)

        threadsafe_vi = setlogprior!!(threadsafe_vi, 42)
        @test getlogjoint(threadsafe_vi) == 42
        expected_accs = DynamicPPL.AccumulatorTuple(
            (DynamicPPL.split(acc) for acc in DynamicPPL.getaccs(threadsafe_vi.varinfo))...
        )
        @test all(accs == expected_accs for accs in threadsafe_vi.accs_by_thread)
    end

    @testset "Check that VarInfo is wrapped during model evaluation" begin
        @model function f()
            global vi_ = __varinfo__
            return x ~ Normal(0, 1)
        end
        model = setthreadsafe(f(), true)

        _, vi = DynamicPPL.init!!(model, VarInfo())
        # Inside the model evaluation function, it should be wrapped
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        # But init!! should return the original VarInfo
        @test vi isa DynamicPPL.VarInfo
        # Same with evaluate!!
        _, vi = DynamicPPL.evaluate_nowarn!!(model, vi)
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        @test vi isa DynamicPPL.VarInfo
    end

    @testset "logprob correctness" begin
        x = rand(10_000)

        @model function wthreads(x)
            x[1] ~ Normal(0, 1)
            Threads.@threads for i in 2:length(x)
                x[i] ~ Normal(x[i - 1], 1)
            end
        end
        model = setthreadsafe(wthreads(x), true)

        function correct_lp(x)
            lp = logpdf(Normal(0, 1), x[1])
            for i in 2:length(x)
                lp += logpdf(Normal(x[i - 1], 1), x[i])
            end
            return lp
        end

        _, vi = DynamicPPL.init!!(model, VarInfo())

        # check that logp is correct
        @test getlogjoint(vi) ≈ correct_lp(x)
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
