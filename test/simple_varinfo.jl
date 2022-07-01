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

    @testset "SimpleVarInfo on $(nameof(model))" for model in
                                                     DynamicPPL.TestUtils.DEMO_MODELS
        # We might need to pre-allocate for the variable `m`, so we need
        # to see whether this is the case.
        svi_nt = SimpleVarInfo(DynamicPPL.TestUtils.example_values(model))
        svi_dict = SimpleVarInfo(VarInfo(model), Dict)

        @testset "$(nameof(typeof(DynamicPPL.values_as(svi))))" for svi in (
            svi_nt,
            svi_dict,
            DynamicPPL.settrans!!(svi_nt, true),
            DynamicPPL.settrans!!(svi_dict, true),
        )
            # Random seed is set in each `@testset`, so we need to sample
            # a new realization for `m` here.
            retval = model()

            ### Sampling ###
            # Sample a new varinfo!
            _, svi_new = DynamicPPL.evaluate!!(model, svi, SamplingContext())

            # Realization for `m` should be different wp. 1.
            for vn in keys(model)
                @test svi_new[vn] != get(retval, vn)
            end

            # Logjoint should be non-zero wp. 1.
            @test getlogp(svi_new) != 0

            ### Evaluation ###
            values_eval_constrained = DynamicPPL.TestUtils.example_values(model)
            if DynamicPPL.istrans(svi)
                values_eval, logπ_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                    model, values_eval_constrained...
                )
            else
                values_eval = values_eval_constrained
                logπ_true = DynamicPPL.TestUtils.logjoint_true(model, values_eval...)
            end

            # Update the realizations in `svi_new`.
            svi_eval = svi_new
            for vn in keys(model)
                svi_eval = DynamicPPL.setindex!!(svi_eval, get(values_eval, vn), vn)
            end

            # Reset the logp field.
            svi_eval = DynamicPPL.resetlogp!!(svi_eval)

            # Compute `logjoint` using the varinfo.
            logπ = logjoint(model, svi_eval)

            # Values should not have changed.
            for vn in keys(model)
                @test svi_eval[vn] == get(values_eval, vn)
            end

            # Compare `logjoint` computations.
            @test logπ ≈ logπ_true
        end
    end

    @testset "Dynamic constraints" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()

        # Initialize.
        svi = DynamicPPL.settrans!!(SimpleVarInfo(), true)
        svi = last(DynamicPPL.evaluate!!(model, svi, SamplingContext()))

        # Sample with large variations in unconstrained space.
        for i in 1:10
            for vn in keys(svi)
                svi = DynamicPPL.setindex!!(svi, 10 * randn(), vn)
            end
            retval, svi = DynamicPPL.evaluate!!(model, svi, DefaultContext())
            @test retval.m == svi[@varname(m)]  # `m` is unconstrained
            @test retval.x ≠ svi[@varname(x)]   # `x` is constrained depending on `m`

            retval_unconstrained, lp_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                model, retval.m, retval.x
            )

            # Realizations from model should all be equal to the unconstrained realization.
            for vn in keys(model)
                @test get(retval_unconstrained, vn) ≈ svi[vn] rtol = 1e-6
            end

            # `getlogp` should be equal to the logjoint with log-absdet-jac correction.
            lp = getlogp(svi)
            @test lp ≈ lp_true
        end
    end
end
