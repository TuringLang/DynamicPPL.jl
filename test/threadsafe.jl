@testset "threadsafe.jl" begin
    @testset "set threadsafe eval" begin
        # A dummy model that lets us see what type of VarInfo is being used for evaluation.
        @model function find_out_varinfo_type()
            x ~ Normal()
            return typeof(__varinfo__)
        end
        model = find_out_varinfo_type()

        # Check the default.
        @test DynamicPPL.USE_THREADSAFE_EVAL[] == (Threads.nthreads() > 1)
        # Disable it.
        DynamicPPL.set_threadsafe_eval!(false)
        @test DynamicPPL.USE_THREADSAFE_EVAL[] == false
        @test !(model() <: DynamicPPL.ThreadSafeVarInfo)
        # Enable it.
        DynamicPPL.set_threadsafe_eval!(true)
        @test DynamicPPL.USE_THREADSAFE_EVAL[] == true
        @test model() <: DynamicPPL.ThreadSafeVarInfo
        # Reset to default to avoid messing with other tests.
        DynamicPPL.set_threadsafe_eval!(Threads.nthreads() > 1)
    end

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

    @testset "model" begin
        println("Peforming threading tests with $(Threads.nthreads()) threads")
        @show DynamicPPL.USE_THREADSAFE_EVAL[]

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
        if DynamicPPL.USE_THREADSAFE_EVAL[]
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        else
            @test vi_ isa VarInfo
        end

        println("With `@threads`:")
        println("  default:")
        @time model(vi)

        # Ensure that we use `ThreadSafeVarInfo` to handle multithreaded observe statements.
        DynamicPPL.evaluate_threadsafe!!(model, vi)
        @test getlogjoint(vi) ≈ lp_w_threads
        # check that it's wrapped during the model evaluation
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        # ensure that it's unwrapped after evaluation finishes
        @test vi isa VarInfo

        println("  evaluate_threadsafe!!:")
        @time DynamicPPL.evaluate_threadsafe!!(model, vi)

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
        if DynamicPPL.USE_THREADSAFE_EVAL[]
            @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        else
            @test vi_ isa VarInfo
        end

        println("Without `@threads`:")
        println("  default:")
        @time model(vi)

        @test lp_w_threads ≈ lp_wo_threads

        # Ensure that we use `VarInfo`.
        DynamicPPL.evaluate_threadunsafe!!(model, vi)
        @test getlogjoint(vi) ≈ lp_w_threads
        @test vi_ isa VarInfo
        @test vi isa VarInfo

        println("  evaluate_threadunsafe!!:")
        @time DynamicPPL.evaluate_threadunsafe!!(model, vi)
    end
end
