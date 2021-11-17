@testset "simple_varinfo.jl" begin
    @testset "constructor & indexing" begin
        @testset "NamedTuple" begin
            svi = SimpleVarInfo(; m=1.0)
            @test getlogp(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test !haskey(svi, @varname(m[1]))

            svi = SimpleVarInfo(; m=[1.0])
            @test getlogp(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m[1]))
            @test !haskey(svi, @varname(m[2]))
            @test svi[@varname(m)][1] == svi[@varname(m[1])]

            svi = SimpleVarInfo{Float32}(; m=1.0)
            @test getlogp(svi) isa Float32

            svi = SimpleVarInfo((m=1.0,), 1.0)
            @test getlogp(svi) == 1.0
        end

        @testset "Dict" begin
            svi = SimpleVarInfo(Dict(@varname(m) => 1.0))
            @test getlogp(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test !haskey(svi, @varname(m[1]))

            svi = SimpleVarInfo(Dict(@varname(m) => [1.0]))
            @test getlogp(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m[1]))
            @test !haskey(svi, @varname(m[2]))
            @test svi[@varname(m)][1] == svi[@varname(m[1])]
        end
    end

    @testset "SimpleVarInfo on $(model.name)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        # We might need to pre-allocate for the variable `m`, so we need
        # to see whether this is the case.
        m = model().m
        svi = if m isa AbstractArray
            SimpleVarInfo((m=similar(m),))
        else
            SimpleVarInfo()
        end

        # Sample a new varinfo!
        _, svi_new = DynamicPPL.evaluate!!(model, svi, SamplingContext())
        # Type of realization for `m` should be unchanged.
        @test typeof(svi_new[@varname(m)]) === typeof(m)
        # Realization for `m` should be different wp. 1.
        @test svi_new[@varname(m)] != m
        # Logjoint should be non-zero wp. 1.
        @test getlogp(svi_new) != 0

        # Evaluation.
        m_eval = if m isa AbstractArray
            randn!(similar(m))
        else
            randn(eltype(m))
        end
        svi_eval = @set svi_new.values.m = m_eval
        svi_eval = DynamicPPL.resetlogp!!(svi_eval)

        logπ = logjoint(model, svi_eval)
        logπ_true = DynamicPPL.TestUtils.logjoint_true(model, svi_eval.values.m)
        @test logπ ≈ logπ_true
    end
end
