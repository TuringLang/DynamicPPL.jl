@testset "simple_varinfo.jl" begin
    @testset "constructor & indexing" begin
        @testset "NamedTuple" begin
            svi = SimpleVarInfo(; m=1.0)
            @test getlogjoint(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test !haskey(svi, @varname(m[1]))

            svi = SimpleVarInfo(; m=[1.0])
            @test getlogjoint(svi) == 0.0
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
            @test getlogjoint(svi) isa Float32

            svi = SimpleVarInfo((m=1.0,))
            svi = accloglikelihood!!(svi, 1.0)
            @test getlogjoint(svi) == 1.0
        end

        @testset "Dict" begin
            svi = SimpleVarInfo(Dict(@varname(m) => 1.0))
            @test getlogjoint(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test !haskey(svi, @varname(m[1]))

            svi = SimpleVarInfo(Dict(@varname(m) => [1.0]))
            @test getlogjoint(svi) == 0.0
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

        @testset "VarNamedVector" begin
            svi = SimpleVarInfo(push!!(DynamicPPL.VarNamedVector(), @varname(m) => 1.0))
            @test getlogjoint(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test !haskey(svi, @varname(m[1]))

            svi = SimpleVarInfo(push!!(DynamicPPL.VarNamedVector(), @varname(m) => [1.0]))
            @test getlogjoint(svi) == 0.0
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m[1]))
            @test !haskey(svi, @varname(m[2]))
            @test svi[@varname(m)][1] == svi[@varname(m[1])]

            svi = SimpleVarInfo(push!!(DynamicPPL.VarNamedVector(), @varname(m.a) => [1.0]))
            @test haskey(svi, @varname(m))
            @test haskey(svi, @varname(m.a))
            @test haskey(svi, @varname(m.a[1]))
            @test !haskey(svi, @varname(m.a[2]))
            @test !haskey(svi, @varname(m.a.b))
            # The implementation of haskey and getvalue fo VarNamedVector is incomplete, the
            # next test is here to remind of us that.
            svi = SimpleVarInfo(
                push!!(DynamicPPL.VarNamedVector(), @varname(m.a.b) => [1.0])
            )
            @test_broken !haskey(svi, @varname(m.a.b.c.d))
        end
    end

    @testset "link!! & invlink!! on $(nameof(model))" for model in
                                                          DynamicPPL.TestUtils.DEMO_MODELS
        values_constrained = DynamicPPL.TestUtils.rand_prior_true(model)
        @testset "$name" for (name, vi) in (
            ("SVI{Dict}", SimpleVarInfo(Dict{VarName,Any}())),
            ("SVI{NamedTuple}", SimpleVarInfo(values_constrained)),
            ("SVI{VNV}", SimpleVarInfo(DynamicPPL.VarNamedVector())),
            ("TypedVarInfo", DynamicPPL.typed_varinfo(model)),
        )
            for vn in DynamicPPL.TestUtils.varnames(model)
                vi = DynamicPPL.setindex!!(vi, get(values_constrained, vn), vn)
            end
            vi = last(DynamicPPL.evaluate!!(model, vi))

            # Calculate ground truth
            lp_unlinked_true = DynamicPPL.TestUtils.logjoint_true(
                model, values_constrained...
            )
            _, lp_linked_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                model, values_constrained...
            )

            # `link!!`
            vi_linked = link!!(deepcopy(vi), model)
            lp_unlinked = getlogjoint(vi_linked)
            lp_linked = getlogjoint_internal(vi_linked)
            @test lp_linked ≈ lp_linked_true
            @test lp_unlinked ≈ lp_unlinked_true
            @test logjoint(model, vi_linked) ≈ lp_unlinked

            # `invlink!!`
            vi_invlinked = invlink!!(deepcopy(vi_linked), model)
            lp_unlinked = getlogjoint(vi_invlinked)
            also_lp_unlinked = getlogjoint_internal(vi_invlinked)
            @test lp_unlinked ≈ lp_unlinked_true
            @test also_lp_unlinked ≈ lp_unlinked_true
            @test logjoint(model, vi_invlinked) ≈ lp_unlinked

            # Should result in same values.
            @test all(
                DynamicPPL.tovec(DynamicPPL.getindex_internal(vi_invlinked, vn)) ≈
                DynamicPPL.tovec(get(values_constrained, vn)) for
                vn in DynamicPPL.TestUtils.varnames(model)
            )
        end
    end

    @testset "SimpleVarInfo on $(nameof(model))" for model in
                                                     DynamicPPL.TestUtils.DEMO_MODELS
        # We might need to pre-allocate for the variable `m`, so we need
        # to see whether this is the case.
        svi_nt = SimpleVarInfo(DynamicPPL.TestUtils.rand_prior_true(model))
        svi_dict = SimpleVarInfo(VarInfo(model), Dict)
        vnv = DynamicPPL.VarNamedVector()
        for (k, v) in pairs(DynamicPPL.TestUtils.rand_prior_true(model))
            vnv = push!!(vnv, VarName{k}() => v)
        end
        svi_vnv = SimpleVarInfo(vnv)

        @testset "$name" for (name, svi) in (
            ("NamedTuple", svi_nt),
            ("Dict", svi_dict),
            ("VarNamedVector", svi_vnv),
            # TODO(mhauru) Fix linked SimpleVarInfos to work with our test models.
            # DynamicPPL.settrans!!(deepcopy(svi_nt), true),
            # DynamicPPL.settrans!!(deepcopy(svi_dict), true),
            # DynamicPPL.settrans!!(deepcopy(svi_vnv), true),
        )
            # Random seed is set in each `@testset`, so we need to sample
            # a new realization for `m` here.
            retval = model()

            ### Sampling ###
            # Sample a new varinfo!
            _, svi_new = DynamicPPL.evaluate_and_sample!!(model, svi)

            # Realization for `m` should be different wp. 1.
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test svi_new[vn] != get(retval, vn)
            end

            # Logjoint should be non-zero wp. 1.
            @test getlogjoint(svi_new) != 0

            ### Evaluation ###
            values_eval_constrained = DynamicPPL.TestUtils.rand_prior_true(model)
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

            # Reset the logp accumulators.
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
        svi_nt = DynamicPPL.settrans!!(SimpleVarInfo(), true)
        svi_nt = last(DynamicPPL.evaluate_and_sample!!(model, svi_nt))
        svi_vnv = DynamicPPL.settrans!!(SimpleVarInfo(DynamicPPL.VarNamedVector()), true)
        svi_vnv = last(DynamicPPL.evaluate_and_sample!!(model, svi_vnv))

        for svi in (svi_nt, svi_vnv)
            # Sample with large variations in unconstrained space.
            for i in 1:10
                for vn in keys(svi)
                    svi = DynamicPPL.setindex!!(svi, 10 * randn(), vn)
                end
                retval, svi = DynamicPPL.evaluate!!(model, svi)
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
                lp = getlogjoint_internal(svi)
                # needs higher atol because of https://github.com/TuringLang/Bijectors.jl/issues/375
                @test lp ≈ lp_true atol = 1.2e-5
            end
        end
    end

    @testset "Static transformation" begin
        model = DynamicPPL.TestUtils.demo_static_transformation()

        varinfos = DynamicPPL.TestUtils.setup_varinfos(
            model, DynamicPPL.TestUtils.rand_prior_true(model), [@varname(s), @varname(m)]
        )
        @testset "$(short_varinfo_name(vi))" for vi in varinfos
            # Initialize varinfo and link.
            vi_linked = DynamicPPL.link!!(vi, model)

            # Make sure `maybe_invlink_before_eval!!` results in `invlink!!`.
            @test !DynamicPPL.istrans(
                DynamicPPL.maybe_invlink_before_eval!!(deepcopy(vi), model)
            )

            # Resulting varinfo should no longer be transformed.
            vi_result = last(DynamicPPL.evaluate_and_sample!!(model, deepcopy(vi)))
            @test !DynamicPPL.istrans(vi_result)

            # Set the values to something that is out of domain if we're in constrained space.
            for vn in keys(vi)
                vi_linked = DynamicPPL.setindex!!(vi_linked, -rand(), vn)
            end

            # NOTE: Evaluating a linked VarInfo, **specifically when the transformation
            # is static**, will result in an invlinked VarInfo. This is because of
            # `maybe_invlink_before_eval!`, which only invlinks if the transformation
            # is static. (src/abstract_varinfo.jl)
            retval, vi_unlinked_again = DynamicPPL.evaluate!!(model, deepcopy(vi_linked))

            @test DynamicPPL.tovec(DynamicPPL.getindex_internal(vi_linked, @varname(s))) ≠
                DynamicPPL.tovec(retval.s)  # `s` is unconstrained in original
            @test DynamicPPL.tovec(
                DynamicPPL.getindex_internal(vi_unlinked_again, @varname(s))
            ) == DynamicPPL.tovec(retval.s)  # `s` is constrained in result

            # `m` should not be transformed.
            @test vi_linked[@varname(m)] == retval.m
            @test vi_unlinked_again[@varname(m)] == retval.m

            # Get ground truths
            retval_unconstrained, lp_linked_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                model, retval.s, retval.m
            )
            lp_unlinked_true = DynamicPPL.TestUtils.logjoint_true(model, retval.s, retval.m)

            @test DynamicPPL.tovec(DynamicPPL.getindex_internal(vi_linked, @varname(s))) ≈
                DynamicPPL.tovec(retval_unconstrained.s)
            @test DynamicPPL.tovec(DynamicPPL.getindex_internal(vi_linked, @varname(m))) ≈
                DynamicPPL.tovec(retval_unconstrained.m)

            # The unlinked varinfo should hold the unlinked logp.
            lp_unlinked = getlogjoint(vi_unlinked_again)
            @test getlogjoint(vi_unlinked_again) ≈ lp_unlinked_true
        end
    end
end
