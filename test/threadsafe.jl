@testset "threadsafe.jl" begin
    @testset "constructor" begin
        vi = VarInfo(TU.gdemo_default)
        threadsafe_vi = @inferred DynamicPPL.ThreadSafeVarInfo(vi)

        @test threadsafe_vi.varinfo === vi
        @test threadsafe_vi.logps isa Vector{typeof(Ref(getlogp(vi)))}
        @test length(threadsafe_vi.logps) == Threads.nthreads()
        @test all(iszero(x[]) for x in threadsafe_vi.logps)
    end

    # TODO: Add more tests of the public API
    @testset "API" begin
        vi = VarInfo(TU.gdemo_default)
        threadsafe_vi = DynamicPPL.ThreadSafeVarInfo(vi)

        lp = getlogp(vi)
        @test getlogp(threadsafe_vi) == lp

        acclogp!!(threadsafe_vi, 42)
        @test threadsafe_vi.logps[Threads.threadid()][] == 42
        @test getlogp(vi) == lp
        @test getlogp(threadsafe_vi) == lp + 42

        resetlogp!!(threadsafe_vi)
        @test iszero(getlogp(vi))
        @test iszero(getlogp(threadsafe_vi))
        @test all(iszero(x[]) for x in threadsafe_vi.logps)

        setlogp!!(threadsafe_vi, 42)
        @test getlogp(vi) == 42
        @test getlogp(threadsafe_vi) == 42
        @test all(iszero(x[]) for x in threadsafe_vi.logps)
    end

    @testset "model" begin
        println("Peforming threading tests with $(Threads.nthreads()) threads")

        x = rand(10_000)

        @model function wthreads(x)
            global vi_ = __varinfo__
            x[1] ~ Normal(0, 1)
            Threads.@threads for i in 2:length(x)
                x[i] ~ Normal(x[i - 1], 1)
            end
        end

        vi = VarInfo()
        wthreads(x)(vi)
        lp_w_threads = getlogp(vi)
        if Threads.nthreads() == 1
            @test vi_ isa VarInfo
        else
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        end

        println("With `@threads`:")
        println("  default:")
        @time wthreads(x)(vi)

        # Ensure that we use `ThreadSafeVarInfo` to handle multithreaded observe statements.
        DynamicPPL.evaluate_threadsafe!!(
            wthreads(x),
            vi,
            SamplingContext(Random.default_rng(), SampleFromPrior(), DefaultContext()),
        )
        @test getlogp(vi) ≈ lp_w_threads
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo

        println("  evaluate_threadsafe!!:")
        @time DynamicPPL.evaluate_threadsafe!!(
            wthreads(x),
            vi,
            SamplingContext(Random.default_rng(), SampleFromPrior(), DefaultContext()),
        )

        @model function wothreads(x)
            global vi_ = __varinfo__
            x[1] ~ Normal(0, 1)
            for i in 2:length(x)
                x[i] ~ Normal(x[i - 1], 1)
            end
        end

        vi = VarInfo()
        wothreads(x)(vi)
        lp_wo_threads = getlogp(vi)
        if Threads.nthreads() == 1
            @test vi_ isa VarInfo
        else
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        end

        println("Without `@threads`:")
        println("  default:")
        @time wothreads(x)(vi)

        @test lp_w_threads ≈ lp_wo_threads

        # Ensure that we use `VarInfo`.
        DynamicPPL.evaluate_threadunsafe!!(
            wothreads(x),
            vi,
            SamplingContext(Random.default_rng(), SampleFromPrior(), DefaultContext()),
        )
        @test getlogp(vi) ≈ lp_w_threads
        @test vi_ isa VarInfo

        println("  evaluate_threadunsafe!!:")
        @time DynamicPPL.evaluate_threadunsafe!!(
            wothreads(x),
            vi,
            SamplingContext(Random.default_rng(), SampleFromPrior(), DefaultContext()),
        )
    end
end
