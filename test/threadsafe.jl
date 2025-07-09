@testset "threadsafe.jl" begin
    @testset "constructor" begin
        vi = VarInfo(gdemo_default)
        threadsafe_vi = @inferred DynamicPPL.ThreadSafeVarInfo(vi)

        @test threadsafe_vi.varinfo === vi
        @test threadsafe_vi.accs_by_thread isa Vector{<:DynamicPPL.AccumulatorTuple}
        @test length(threadsafe_vi.accs_by_thread) == Threads.nthreads() * 2
        expected_accs = DynamicPPL.AccumulatorTuple(
            (DynamicPPL.split(acc) for acc in DynamicPPL.getaccs(vi))...
        )
        @test all(accs == expected_accs for accs in threadsafe_vi.accs_by_thread)
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
        @test getlogjoint(threadsafe_vi) == lp + 42

        threadsafe_vi = resetlogp!!(threadsafe_vi)
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
        model = wthreads(x)

        vi = VarInfo()
        model(vi)
        lp_w_threads = getlogjoint(vi)
        if Threads.nthreads() == 1
            @test vi_ isa VarInfo
        else
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        end

        println("With `@threads`:")
        println("  default:")
        @time model(vi)

        # Ensure that we use `ThreadSafeVarInfo` to handle multithreaded observe statements.
        sampling_model = contextualize(model, SamplingContext(model.context))
        DynamicPPL.evaluate_threadsafe!!(sampling_model, vi)
        @test getlogjoint(vi) ≈ lp_w_threads
        # check that it's wrapped during the model evaluation
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        # ensure that it's unwrapped after evaluation finishes
        @test vi isa VarInfo

        println("  evaluate_threadsafe!!:")
        @time DynamicPPL.evaluate_threadsafe!!(sampling_model, vi)

        @model function wothreads(x)
            global vi_ = __varinfo__
            x[1] ~ Normal(0, 1)
            for i in 2:length(x)
                x[i] ~ Normal(x[i - 1], 1)
            end
        end
        model = wothreads(x)

        vi = VarInfo()
        model(vi)
        lp_wo_threads = getlogjoint(vi)
        if Threads.nthreads() == 1
            @test vi_ isa VarInfo
        else
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        end

        println("Without `@threads`:")
        println("  default:")
        @time model(vi)

        @test lp_w_threads ≈ lp_wo_threads

        # Ensure that we use `VarInfo`.
        sampling_model = contextualize(model, SamplingContext(model.context))
        DynamicPPL.evaluate_threadunsafe!!(sampling_model, vi)
        @test getlogjoint(vi) ≈ lp_w_threads
        @test vi_ isa VarInfo
        @test vi isa VarInfo

        println("  evaluate_threadunsafe!!:")
        @time DynamicPPL.evaluate_threadunsafe!!(sampling_model, vi)
    end
end
