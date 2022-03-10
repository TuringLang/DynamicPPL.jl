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

            svi = SimpleVarInfo(; m=(a=[1.0],))
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m.a))
            @test haskey(svi, @varname(m.a[1]))
            @test !haskey(svi, @varname(m.a[2]))
            @test !haskey(svi, @varname(m.a.b))

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

            svi = SimpleVarInfo(Dict(@varname(m) => (a=[1.0],)))
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m.a))
            @test haskey(svi, @varname(m.a[1]))
            @test !haskey(svi, @varname(m.a[2]))
            @test !haskey(svi, @varname(m.a.b))

            svi = SimpleVarInfo(Dict(@varname(m.a) => [1.0]))
            # Now we only have a variable `m.a` which is subsumed by `m`,
            # but we can't guarantee that we have the "entire" `m`.
            @test !haskey(svi, @varname(m))
            @test haskey(svi, @varname(m.a))
            @test haskey(svi, @varname(m.a[1]))
            @test !haskey(svi, @varname(m.a[2]))
            @test !haskey(svi, @varname(m.a.b))
        end
    end

    @testset "SimpleVarInfo on $(nameof(model))" for model in DynamicPPL.TestUtils.DEMO_MODELS
        # We might need to pre-allocate for the variable `m`, so we need
        # to see whether this is the case.
        m = model().m
        svi_nt = if m isa AbstractArray
            SimpleVarInfo((m=similar(m),))
        else
            SimpleVarInfo()
        end
        svi_dict = SimpleVarInfo(VarInfo(model), Dict)

        @testset "$(nameof(typeof(svi.values)))" for svi in (svi_nt, svi_dict)
            # Random seed is set in each `@testset`, so we need to sample
            # a new realization for `m` here.
            m = model().m

            ### Sampling ###
            # Sample a new varinfo!
            _, svi_new = DynamicPPL.evaluate!!(model, svi, SamplingContext())

            # If the `m[1]` varname doesn't exist, this is a univariate model.
            # TODO: Find a better way of dealing with this that is not dependent
            # on knowledge of internals of `model`.
            isunivariate = !haskey(svi_new, @varname(m[1]))

            # Realization for `m` should be different wp. 1.
            if isunivariate
                @test svi_new[@varname(m)] != m
            else
                @test svi_new[@varname(m[1])] != m[1]
                @test svi_new[@varname(m[2])] != m[2]
            end
            # Logjoint should be non-zero wp. 1.
            @test getlogp(svi_new) != 0

            ### Evaluation ###
            # Sample some random testing values.
            m_eval = if m isa AbstractArray
                randn!(similar(m))
            else
                randn(eltype(m))
            end

            # Update the realizations in `svi_new`.
            svi_eval = if isunivariate
                DynamicPPL.setindex!!(svi_new, m_eval, @varname(m))
            else
                DynamicPPL.setindex!!(svi_new, m_eval, [@varname(m[1]), @varname(m[2])])
            end
            # Reset the logp field.
            svi_eval = DynamicPPL.resetlogp!!(svi_eval)

            # Compute `logjoint` using the varinfo.
            logπ = logjoint(model, svi_eval)
            # Extract the parameters from `svi_eval`.
            m_vi = if isunivariate
                svi_eval[@varname(m)]
            else
                svi_eval[[@varname(m[1]), @varname(m[2])]]
            end
            # These should not have changed.
            @test m_vi == m_eval
            # Compute the true `logjoint` and compare.
            logπ_true = DynamicPPL.TestUtils.logjoint_true(model, m_vi)
            @test logπ ≈ logπ_true
        end
    end
end
