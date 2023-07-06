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

    @testset "link!! & invlink!! on $(nameof(model))" for model in
                                                          DynamicPPL.TestUtils.DEMO_MODELS
        values_constrained = rand(NamedTuple, model)
        @testset "$(typeof(vi))" for vi in (
            SimpleVarInfo(Dict()), SimpleVarInfo(values_constrained), VarInfo(model)
        )
            for vn in DynamicPPL.TestUtils.varnames(model)
                vi = DynamicPPL.setindex!!(vi, get(values_constrained, vn), vn)
            end
            vi = last(DynamicPPL.evaluate!!(model, vi, DefaultContext()))
            lp_orig = getlogp(vi)

            # `link!!`
            vi_linked = link!!(deepcopy(vi), model)
            lp_linked = getlogp(vi_linked)
            values_unconstrained, lp_linked_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                model, values_constrained...
            )
            # Should result in the correct logjoint.
            @test lp_linked ≈ lp_linked_true
            # Should be approx. the same as the "lazy" transformation.
            @test logjoint(model, vi_linked) ≈ lp_linked

            # TODO: Should not `VarInfo` also error here? The current implementation
            # only warns and acts as a no-op.
            if vi isa SimpleVarInfo
                @test_throws AssertionError link!!(vi_linked, model)
            end

            # `invlink!!`
            vi_invlinked = invlink!!(deepcopy(vi_linked), model)
            lp_invlinked = getlogp(vi_invlinked)
            lp_invlinked_true = DynamicPPL.TestUtils.logjoint_true(
                model, values_constrained...
            )
            # Should result in the correct logjoint.
            @test lp_invlinked ≈ lp_invlinked_true
            # Should be approx. the same as the "lazy" transformation.
            @test logjoint(model, vi_invlinked) ≈ lp_invlinked

            # Should result in same values.
            @test all(
                DynamicPPL.getindex_raw(vi_invlinked, vn) ≈ get(values_constrained, vn) for
                vn in DynamicPPL.TestUtils.varnames(model)
            )
        end
    end

    @testset "SimpleVarInfo on $(nameof(model))" for model in
                                                     DynamicPPL.TestUtils.DEMO_MODELS
        model = DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix()

        # We might need to pre-allocate for the variable `m`, so we need
        # to see whether this is the case.
        svi_nt = SimpleVarInfo(rand(NamedTuple, model))
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
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test svi_new[vn] != get(retval, vn)
            end

            # Logjoint should be non-zero wp. 1.
            @test getlogp(svi_new) != 0

            ### Evaluation ###
            values_eval_constrained = rand(NamedTuple, model)
            if DynamicPPL.istrans(svi)
                _values_prior, logpri_true = DynamicPPL.TestUtils.logprior_true_with_logabsdet_jacobian(
                    model, values_eval_constrained...
                )
                values_eval, logπ_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                    model, values_eval_constrained...
                )
                # Make sure that these two computation paths provide the same
                # transformed values.
                @test values_eval == _values_prior
            else
                logpri_true = DynamicPPL.TestUtils.logprior_true(
                    model, values_eval_constrained...
                )
                logπ_true = DynamicPPL.TestUtils.logjoint_true(
                    model, values_eval_constrained...
                )
                values_eval = values_eval_constrained
            end

            # No logabsdet-jacobian correction needed for the likelihood.
            loglik_true = DynamicPPL.TestUtils.loglikelihood_true(
                model, values_eval_constrained...
            )

            # Update the realizations in `svi_new`.
            svi_eval = svi_new
            for vn in DynamicPPL.TestUtils.varnames(model)
                svi_eval = DynamicPPL.setindex!!(svi_eval, get(values_eval, vn), vn)
            end

            # Reset the logp field.
            svi_eval = DynamicPPL.resetlogp!!(svi_eval)

            # Compute `logjoint` using the varinfo.
            logπ = logjoint(model, svi_eval)
            logpri = logprior(model, svi_eval)
            loglik = loglikelihood(model, svi_eval)

            # Values should not have changed.
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test svi_eval[vn] == get(values_eval, vn)
            end

            # Compare log-probability computations.
            @test logpri ≈ logpri_true
            @test loglik ≈ loglik_true
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
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test get(retval_unconstrained, vn) ≈ svi[vn] rtol = 1e-6
            end

            # `getlogp` should be equal to the logjoint with log-absdet-jac correction.
            lp = getlogp(svi)
            @test lp ≈ lp_true
        end
    end

    @testset "Static transformation" begin
        model = DynamicPPL.TestUtils.demo_static_transformation()

        varinfos = DynamicPPL.TestUtils.setup_varinfos(
            model, rand(NamedTuple, model), [@varname(s), @varname(m)]
        )
        @testset "$(short_varinfo_name(vi))" for vi in varinfos
            # Initialize varinfo and link.
            vi_linked = DynamicPPL.link!!(vi, model)

            # Make sure `maybe_invlink_before_eval!!` results in `invlink!!`.
            @test !DynamicPPL.istrans(
                DynamicPPL.maybe_invlink_before_eval!!(
                    deepcopy(vi), SamplingContext(), model
                ),
            )

            # Resulting varinfo should no longer be transformed.
            vi_result = last(DynamicPPL.evaluate!!(model, deepcopy(vi), SamplingContext()))
            @test !DynamicPPL.istrans(vi_result)

            # Set the values to something that is out of domain if we're in constrained space.
            for vn in keys(vi)
                vi_linked = DynamicPPL.setindex!!(vi_linked, -rand(), vn)
            end

            retval, vi_linked_result = DynamicPPL.evaluate!!(
                model, deepcopy(vi_linked), DefaultContext()
            )

            @test DynamicPPL.getindex_raw(vi_linked, @varname(s)) ≠ retval.s  # `s` is unconstrained in original
            @test DynamicPPL.getindex_raw(vi_linked_result, @varname(s)) == retval.s  # `s` is constrained in result

            # `m` should not be transformed.
            @test vi_linked[@varname(m)] == retval.m
            @test vi_linked_result[@varname(m)] == retval.m

            # Compare to truth.
            retval_unconstrained, lp_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                model, retval.s, retval.m
            )

            # Realizations in `vi_linked` should all be equal to the unconstrained realization.
            @test DynamicPPL.getindex_raw(vi_linked, @varname(s)) ≈ retval_unconstrained.s
            @test DynamicPPL.getindex_raw(vi_linked, @varname(m)) ≈ retval_unconstrained.m

            # The resulting varinfo should hold the correct logp.
            lp = getlogp(vi_linked_result)
            @test lp ≈ lp_true
        end
    end
end
