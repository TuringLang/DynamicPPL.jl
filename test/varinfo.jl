function check_varinfo_keys(varinfo, vns)
    if varinfo isa DynamicPPL.SimpleOrThreadSafeSimple{<:NamedTuple}
        # NOTE: We can't compare the `keys(varinfo_merged)` directly with `vns`,
        # since `keys(varinfo_merged)` only contains `VarName` with `identity`.
        # So we just check that the original keys are present.
        for vn in vns
            # Should have all the original keys.
            @test haskey(varinfo, vn)
        end
    else
        vns_varinfo = keys(varinfo)
        # Should be equivalent.
        @test union(vns_varinfo, vns) == intersect(vns_varinfo, vns)
    end
end

"""
Return the value of `vn` in `vi`. If one doesn't exist, sample and set it.
"""
function randr(vi::DynamicPPL.VarInfo, vn::VarName, dist::Distribution)
    if !haskey(vi, vn)
        r = rand(dist)
        push!!(vi, vn, r, dist)
        r
    else
        vi[vn]
    end
end

@testset "varinfo.jl" begin
    @testset "VarInfo with NT of Metadata" begin
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2, 3)
            m ~ truncated(Normal(0.0, sqrt(s)), 0.0, 2.0)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        _, vi = DynamicPPL.init!!(model, VarInfo(), InitFromUniform())
        tvi = DynamicPPL.typed_varinfo(vi)

        meta = vi.metadata
        for f in fieldnames(typeof(tvi.metadata))
            fmeta = getfield(tvi.metadata, f)
            for vn in fmeta.vns
                @test tvi[vn] == vi[vn]
                ind = meta.idcs[vn]
                tind = fmeta.idcs[vn]
                @test meta.dists[ind] == fmeta.dists[tind]
                @test meta.is_transformed[ind] == fmeta.is_transformed[tind]
                range = meta.ranges[ind]
                trange = fmeta.ranges[tind]
                @test all(meta.vals[range] .== fmeta.vals[trange])
            end
        end
    end

    @testset "Base" begin
        # Test Base functions:
        #   in, keys, haskey, isempty, push!!, empty!!,
        #   getindex, setindex!, getproperty, setproperty!

        function test_base(vi_original)
            vi = deepcopy(vi_original)
            @test getlogjoint(vi) == 0
            @test isempty(vi[:])

            vn = @varname x
            dist = Normal(0, 1)
            r = rand(dist)

            @test isempty(vi)
            @test !haskey(vi, vn)
            @test !(vn in keys(vi))
            vi = push!!(vi, vn, r, dist)
            @test !isempty(vi)
            @test haskey(vi, vn)
            @test vn in keys(vi)

            @test length(vi[vn]) == 1
            @test vi[vn] == r
            @test vi[:] == [r]
            vi = DynamicPPL.setindex!!(vi, 2 * r, vn)
            @test vi[vn] == 2 * r
            @test vi[:] == [2 * r]

            # TODO(mhauru) Implement these functions for other VarInfo types too.
            if vi isa DynamicPPL.UntypedVectorVarInfo
                delete!(vi, vn)
                @test isempty(vi)
                vi = push!!(vi, vn, r, dist)
            end

            vi = empty!!(vi)
            @test isempty(vi)
            vi = push!!(vi, vn, r, dist)
            @test !isempty(vi)
        end

        test_base(VarInfo())
        test_base(DynamicPPL.typed_varinfo(VarInfo()))
        test_base(SimpleVarInfo())
        test_base(SimpleVarInfo(OrderedDict{VarName,Any}()))
        test_base(SimpleVarInfo(DynamicPPL.VarNamedVector()))
    end

    @testset "get/set/acclogp" begin
        function test_varinfo_logp!(vi)
            @test DynamicPPL.getlogjoint(vi) === 0.0
            vi = DynamicPPL.setlogprior!!(vi, 1.0)
            @test DynamicPPL.getlogprior(vi) === 1.0
            @test DynamicPPL.getloglikelihood(vi) === 0.0
            @test DynamicPPL.getlogjoint(vi) === 1.0
            vi = DynamicPPL.acclogprior!!(vi, 1.0)
            @test DynamicPPL.getlogprior(vi) === 2.0
            @test DynamicPPL.getloglikelihood(vi) === 0.0
            @test DynamicPPL.getlogjoint(vi) === 2.0
            vi = DynamicPPL.setloglikelihood!!(vi, 1.0)
            @test DynamicPPL.getlogprior(vi) === 2.0
            @test DynamicPPL.getloglikelihood(vi) === 1.0
            @test DynamicPPL.getlogjoint(vi) === 3.0
            vi = DynamicPPL.accloglikelihood!!(vi, 1.0)
            @test DynamicPPL.getlogprior(vi) === 2.0
            @test DynamicPPL.getloglikelihood(vi) === 2.0
            @test DynamicPPL.getlogjoint(vi) === 4.0
        end

        vi = VarInfo()
        test_varinfo_logp!(vi)
        test_varinfo_logp!(DynamicPPL.typed_varinfo(vi))
        test_varinfo_logp!(SimpleVarInfo())
        test_varinfo_logp!(SimpleVarInfo(OrderedDict()))
        test_varinfo_logp!(SimpleVarInfo(DynamicPPL.VarNamedVector()))
    end

    @testset "logp accumulators" begin
        @model function demo()
            a ~ Normal()
            b ~ Normal()
            c ~ Normal()
            d ~ Normal()
            return nothing
        end

        values = (; a=1.0, b=2.0, c=3.0, d=4.0)
        lp_a = logpdf(Normal(), values.a)
        lp_b = logpdf(Normal(), values.b)
        lp_c = logpdf(Normal(), values.c)
        lp_d = logpdf(Normal(), values.d)
        m = demo() | (; c=values.c, d=values.d)

        vi = DynamicPPL.unflatten(VarInfo(m), collect(values))

        vi = last(DynamicPPL.evaluate!!(m, deepcopy(vi)))
        @test getlogprior(vi) == lp_a + lp_b
        @test getlogjac(vi) == 0.0
        @test getloglikelihood(vi) == lp_c + lp_d
        @test getlogp(vi) == (; logprior=lp_a + lp_b, logjac=0.0, loglikelihood=lp_c + lp_d)
        @test getlogjoint(vi) == lp_a + lp_b + lp_c + lp_d
        @test begin
            vi = acclogprior!!(vi, 1.0)
            getlogprior(vi) == lp_a + lp_b + 1.0
        end
        @test begin
            vi = accloglikelihood!!(vi, 1.0)
            getloglikelihood(vi) == lp_c + lp_d + 1.0
        end
        @test begin
            vi = setlogprior!!(vi, -1.0)
            getlogprior(vi) == -1.0
        end
        @test begin
            vi = setlogjac!!(vi, -1.0)
            getlogjac(vi) == -1.0
        end
        @test begin
            vi = setloglikelihood!!(vi, -1.0)
            getloglikelihood(vi) == -1.0
        end
        @test begin
            vi = setlogp!!(vi, (logprior=-3.0, logjac=-3.0, loglikelihood=-3.0))
            getlogp(vi) == (; logprior=-3.0, logjac=-3.0, loglikelihood=-3.0)
        end
        @test begin
            vi = acclogp!!(vi, (logprior=1.0, loglikelihood=1.0))
            getlogp(vi) == (; logprior=-2.0, logjac=-3.0, loglikelihood=-2.0)
        end
        @test getlogp(setlogp!!(vi, getlogp(vi))) == getlogp(vi)

        vi = last(
            DynamicPPL.evaluate!!(
                m, DynamicPPL.setaccs!!(deepcopy(vi), (LogPriorAccumulator(),))
            ),
        )
        @test getlogprior(vi) == lp_a + lp_b
        # need regex because 1.11 and 1.12 throw different errors (in 1.12 the
        # missing field is surrounded by backticks)
        @test_throws r"has no field `?LogLikelihood" getloglikelihood(vi)
        @test_throws r"has no field `?LogJacobian" getlogp(vi)
        @test_throws r"has no field `?LogLikelihood" getlogjoint(vi)
        @test begin
            vi = acclogprior!!(vi, 1.0)
            getlogprior(vi) == lp_a + lp_b + 1.0
        end
        @test begin
            vi = setlogprior!!(vi, -1.0)
            getlogprior(vi) == -1.0
        end

        # Test evaluating without any accumulators.
        vi = last(DynamicPPL.evaluate!!(m, DynamicPPL.setaccs!!(deepcopy(vi), ())))
        # need regex because 1.11 and 1.12 throw different errors (in 1.12 the
        # missing field is surrounded by backticks)
        @test_throws r"has no field `?LogPrior" getlogprior(vi)
        @test_throws r"has no field `?LogLikelihood" getloglikelihood(vi)
        @test_throws r"has no field `?LogPrior" getlogp(vi)
        @test_throws r"has no field `?LogPrior" getlogjoint(vi)
    end

    @testset "resetaccs" begin
        # Put in a bunch of accumulators, check that they're all reset either
        # when we call resetaccs!!, empty!!, or evaluate!!.
        @model function demo()
            a ~ Normal()
            return x ~ Normal(a)
        end
        model = demo()
        vi_orig = VarInfo(model)
        # It already has the logp accumulators, so let's add in some more.
        vi_orig = DynamicPPL.setacc!!(vi_orig, DynamicPPL.DebugUtils.DebugAccumulator(true))
        vi_orig = DynamicPPL.setacc!!(vi_orig, DynamicPPL.ValuesAsInModelAccumulator(true))
        vi_orig = DynamicPPL.setacc!!(vi_orig, DynamicPPL.PriorDistributionAccumulator())
        vi_orig = DynamicPPL.setacc!!(
            vi_orig, DynamicPPL.PointwiseLogProbAccumulator{:both}()
        )
        # And evaluate the model once so that they are populated.
        _, vi_orig = DynamicPPL.evaluate!!(model, vi_orig)

        function all_accs_empty(vi::AbstractVarInfo)
            for acc_key in keys(DynamicPPL.getaccs(vi))
                acc = DynamicPPL.getacc(vi, Val(acc_key))
                acc == DynamicPPL.reset(acc) || return false
            end
            return true
        end

        @test !all_accs_empty(vi_orig)

        vi = DynamicPPL.resetaccs!!(deepcopy(vi_orig))
        @test all_accs_empty(vi)
        @test getlogjoint(vi) == 0.0 # for good measure
        @test getlogprior(vi) == 0.0
        @test getloglikelihood(vi) == 0.0

        vi = DynamicPPL.empty!!(deepcopy(vi_orig))
        @test all_accs_empty(vi)
        @test getlogjoint(vi) == 0.0
        @test getlogprior(vi) == 0.0
        @test getloglikelihood(vi) == 0.0

        function all_accs_same(vi1::AbstractVarInfo, vi2::AbstractVarInfo)
            # Check that they have the same accs
            keys1 = Set(keys(DynamicPPL.getaccs(vi1)))
            keys2 = Set(keys(DynamicPPL.getaccs(vi2)))
            keys1 == keys2 || return false
            # Check that they have the same values
            for acc_key in keys1
                acc1 = DynamicPPL.getacc(vi1, Val(acc_key))
                acc2 = DynamicPPL.getacc(vi2, Val(acc_key))
                if acc1 != acc2
                    @show acc1, acc2
                end
                acc1 == acc2 || return false
            end
            return true
        end
        # Hopefully this doesn't matter
        @test all_accs_same(vi_orig, deepcopy(vi_orig))
        # If we re-evaluate, then we expect the accs to be reset prior to evaluation.
        # Thus after re-evaluation, the accs should be exactly the same as before.
        _, vi = DynamicPPL.evaluate!!(model, deepcopy(vi_orig))
        @test all_accs_same(vi, vi_orig)
    end

    @testset "is_transformed flag" begin
        # Test is_transformed and set_transformed!!
        function test_varinfo!(vi)
            vn_x = @varname x
            dist = Normal(0, 1)
            r = rand(dist)

            push!!(vi, vn_x, r, dist)

            # is_transformed is set by default
            @test !is_transformed(vi, vn_x)

            vi = set_transformed!!(vi, true, vn_x)
            @test is_transformed(vi, vn_x)

            vi = set_transformed!!(vi, false, vn_x)
            @test !is_transformed(vi, vn_x)
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!!(DynamicPPL.typed_varinfo(vi)))
    end

    @testset "push!! to VarInfo with NT of Metadata" begin
        vn_x = @varname x
        vn_y = @varname y
        untyped_vi = VarInfo()
        untyped_vi = push!!(untyped_vi, vn_x, 1.0, Normal(0, 1))
        typed_vi = DynamicPPL.typed_varinfo(untyped_vi)
        typed_vi = push!!(typed_vi, vn_y, 2.0, Normal(0, 1))
        @test typed_vi[vn_x] == 1.0
        @test typed_vi[vn_y] == 2.0
    end

    @testset "returned on MCMCChains.Chains" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.ALL_MODELS
            chain = make_chain_from_prior(model, 10)
            # A simple way of checking that the computation is determinstic: run twice and compare.
            res1 = returned(model, MCMCChains.get_sections(chain, :parameters))
            res2 = returned(model, MCMCChains.get_sections(chain, :parameters))
            @test all(res1 .== res2)
        end
    end

    @testset "link!! and invlink!!" begin
        @model gdemo(a, b, ::Type{T}=Float64) where {T} = begin
            s ~ InverseGamma(2, 3)
            m ~ Uniform(0, 2)
            x = Vector{T}(undef, length(a))
            x .~ Normal(m, sqrt(s))
            y = Vector{T}(undef, length(a))
            for i in eachindex(y)
                y[i] ~ Normal(m, sqrt(s))
            end
            a .~ Normal(m, sqrt(s))
            for i in eachindex(b)
                b[i] ~ Normal(x[i] * y[i], sqrt(s))
            end
        end
        model = gdemo([1.0, 1.5], [2.0, 2.5])

        # Check that instantiating the model using InitFromUniform does not
        # perform linking
        # Note (penelopeysm): The purpose of using InitFromUniform specifically in
        # this test is because it samples from the linked distribution i.e. in
        # unconstrained space. However, it does this not by linking the varinfo
        # but by transforming the distributions on the fly. That's why it's
        # worth specifically checking that it can do this without having to
        # change the VarInfo object.
        # TODO(penelopeysm): Move this to InitFromUniform tests rather than here.
        vi = VarInfo()
        meta = vi.metadata
        _, vi = DynamicPPL.init!!(model, vi, InitFromUniform())
        @test all(x -> !is_transformed(vi, x), meta.vns)

        # Check that linking and invlinking set the `is_transformed` flag accordingly
        v = copy(meta.vals)
        vi = link!!(vi, model)
        @test all(x -> is_transformed(vi, x), meta.vns)
        vi = invlink!!(vi, model)
        @test all(x -> !is_transformed(vi, x), meta.vns)
        @test meta.vals ≈ v atol = 1e-10

        # Check that linking and invlinking preserves the values
        vi = DynamicPPL.typed_varinfo(vi)
        meta = vi.metadata
        v_s = copy(meta.s.vals)
        v_m = copy(meta.m.vals)
        v_x = copy(meta.x.vals)
        v_y = copy(meta.y.vals)

        @test all(x -> !is_transformed(vi, x), meta.s.vns)
        @test all(x -> !is_transformed(vi, x), meta.m.vns)
        vi = link!!(vi, model)
        @test all(x -> is_transformed(vi, x), meta.s.vns)
        @test all(x -> is_transformed(vi, x), meta.m.vns)
        vi = invlink!!(vi, model)
        @test all(x -> !is_transformed(vi, x), meta.s.vns)
        @test all(x -> !is_transformed(vi, x), meta.m.vns)
        @test meta.s.vals ≈ v_s atol = 1e-10
        @test meta.m.vals ≈ v_m atol = 1e-10

        # Transform only one variable
        all_vns = vcat(meta.s.vns, meta.m.vns, meta.x.vns, meta.y.vns)
        for vn in [
            @varname(s),
            @varname(m),
            @varname(x),
            @varname(y),
            @varname(x[2]),
            @varname(y[2])
        ]
            target_vns = filter(x -> subsumes(vn, x), all_vns)
            other_vns = filter(x -> !subsumes(vn, x), all_vns)
            @test !isempty(target_vns)
            @test !isempty(other_vns)
            vi = link!!(vi, (vn,), model)
            @test all(x -> is_transformed(vi, x), target_vns)
            @test all(x -> !is_transformed(vi, x), other_vns)
            vi = invlink!!(vi, (vn,), model)
            @test all(x -> !is_transformed(vi, x), all_vns)
            @test meta.s.vals ≈ v_s atol = 1e-10
            @test meta.m.vals ≈ v_m atol = 1e-10
            @test meta.x.vals ≈ v_x atol = 1e-10
            @test meta.y.vals ≈ v_y atol = 1e-10
        end
    end

    @testset "logp evaluation on linked varinfo" begin
        @model demo_constrained() = x ~ truncated(Normal(); lower=0)
        model = demo_constrained()
        vn = @varname(x)
        dist = truncated(Normal(); lower=0)

        function test_linked_varinfo(model, vi)
            # vn and dist are taken from the containing scope
            vi = last(DynamicPPL.init!!(model, vi, InitFromPrior()))
            f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
            x = f(DynamicPPL.getindex_internal(vi, vn))
            @test is_transformed(vi, vn)
            @test getlogjoint_internal(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)
            @test getlogprior_internal(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)
            @test getloglikelihood(vi) == 0.0
            @test getlogjoint(vi) ≈ Bijectors.logpdf_with_trans(dist, x, false)
            @test getlogprior(vi) ≈ Bijectors.logpdf_with_trans(dist, x, false)
        end

        ### `VarInfo`
        # Need to run once since we can't specify that we want to _sample_
        # in the unconstrained space for `VarInfo` without having `vn`
        # present in the `varinfo`.

        ## `untyped_varinfo`
        vi = DynamicPPL.untyped_varinfo(model)
        vi = DynamicPPL.set_transformed!!(vi, true, vn)
        test_linked_varinfo(model, vi)

        ## `typed_varinfo`
        vi = DynamicPPL.typed_varinfo(model)
        vi = DynamicPPL.set_transformed!!(vi, true, vn)
        test_linked_varinfo(model, vi)

        ### `SimpleVarInfo`
        ## `SimpleVarInfo{<:NamedTuple}`
        vi = DynamicPPL.set_transformed!!(SimpleVarInfo(), true)
        test_linked_varinfo(model, vi)

        ## `SimpleVarInfo{<:Dict}`
        vi = DynamicPPL.set_transformed!!(SimpleVarInfo(OrderedDict{VarName,Any}()), true)
        test_linked_varinfo(model, vi)

        ## `SimpleVarInfo{<:VarNamedVector}`
        vi = DynamicPPL.set_transformed!!(SimpleVarInfo(DynamicPPL.VarNamedVector()), true)
        test_linked_varinfo(model, vi)
    end

    @testset "values_as" begin
        @testset "$(nameof(model))" for model in DynamicPPL.TestUtils.ALL_MODELS
            example_values = DynamicPPL.TestUtils.rand_prior_true(model)
            vns = DynamicPPL.TestUtils.varnames(model)

            # Set up the different instances of `AbstractVarInfo` with the desired values.
            varinfos = DynamicPPL.TestUtils.setup_varinfos(
                model, example_values, vns; include_threadsafe=true
            )
            @testset "$(short_varinfo_name(vi))" for vi in varinfos
                # Just making sure.
                DynamicPPL.TestUtils.test_values(vi, example_values, vns)

                @testset "NamedTuple" begin
                    vals = values_as(vi, NamedTuple)
                    for vn in vns
                        if haskey(vals, Symbol(vn))
                            # Assumed to be of form `(var"m[1]" = 1.0, ...)`.
                            @test getindex(vals, Symbol(vn)) == getindex(vi, vn)
                        else
                            # Assumed to be of form `(m = [1.0, ...], ...)`.
                            @test AbstractPPL.getvalue(vals, vn) == getindex(vi, vn)
                        end
                    end
                end

                @testset "OrderedDict" begin
                    vals = values_as(vi, OrderedDict)
                    # All varnames in `vns` should be subsumed by one of `keys(vals)`.
                    @test all(vns) do vn
                        any(DynamicPPL.subsumes(vn_left, vn) for vn_left in keys(vals))
                    end
                    # Iterate over `keys(vals)` because we might have scenarios such as
                    # `vals = OrderedDict(@varname(m) => [1.0])` but `@varname(m[1])` is
                    # the varname present in `vns`, not `@varname(m)`.
                    for vn in keys(vals)
                        @test getindex(vals, vn) == getindex(vi, vn)
                    end
                end
            end
        end
    end

    @testset "unflatten + linking" begin
        @testset "Model: $(model.f)" for model in [
            DynamicPPL.TestUtils.demo_one_variable_multiple_constraints(),
            DynamicPPL.TestUtils.demo_lkjchol(),
        ]
            @testset "mutating=$mutating" for mutating in [false, true]
                value_true = DynamicPPL.TestUtils.rand_prior_true(model)
                varnames = DynamicPPL.TestUtils.varnames(model)
                varinfos = DynamicPPL.TestUtils.setup_varinfos(
                    model, value_true, varnames; include_threadsafe=true
                )
                @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                    if varinfo isa DynamicPPL.SimpleOrThreadSafeSimple{<:NamedTuple}
                        # NOTE: this is broken since we'll end up trying to set
                        #
                        #    varinfo[@varname(x[4:5])] = [x[4],]
                        #
                        # upon linking (since `x[4:5]` will be projected onto a 1-dimensional
                        # space). In the case of `SimpleVarInfo{<:NamedTuple}`, this results in
                        # calling `setindex!!(varinfo.values, [x[4],], @varname(x[4:5]))`, which
                        # in turn attempts to call `setindex!(varinfo.values.x, [x[4],], 4:5)`,
                        # i.e. a vector of length 1 (`[x[4],]`) being assigned to 2 indices (`4:5`).
                        @test_broken false
                        continue
                    end

                    if DynamicPPL.has_varnamedvector(varinfo) && mutating
                        # NOTE: Can't handle mutating `link!` and `invlink!` `VarNamedVector`.
                        @test_broken false
                        continue
                    end

                    # Evaluate the model once to update the logp of the varinfo.
                    varinfo = last(DynamicPPL.evaluate!!(model, varinfo))

                    varinfo_linked = if mutating
                        DynamicPPL.link!!(deepcopy(varinfo), model)
                    else
                        DynamicPPL.link(varinfo, model)
                    end
                    for vn in keys(varinfo)
                        @test DynamicPPL.is_transformed(varinfo_linked, vn)
                    end
                    @test length(varinfo[:]) > length(varinfo_linked[:])
                    varinfo_linked_unflattened = DynamicPPL.unflatten(
                        varinfo_linked, varinfo_linked[:]
                    )
                    @test length(varinfo_linked_unflattened[:]) == length(varinfo_linked[:])

                    lp_true = DynamicPPL.TestUtils.logjoint_true(model, value_true...)
                    value_linked_true, lp_linked_true = DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(
                        model, value_true...
                    )

                    lp = logjoint(model, varinfo)
                    @test lp ≈ lp_true
                    @test getlogjoint(varinfo) ≈ lp_true
                    lp_linked_internal = getlogjoint_internal(varinfo_linked)
                    @test lp_linked_internal ≈ lp_linked_true

                    # TODO: Compare values once we are no longer working with `NamedTuple` for
                    # the true values, e.g. `value_true`.

                    if !mutating
                        # This is also compatible with invlinking of unflattened varinfo.
                        varinfo_invlinked = DynamicPPL.invlink(
                            varinfo_linked_unflattened, model
                        )
                        @test length(varinfo_invlinked[:]) == length(varinfo[:])
                        @test getlogjoint(varinfo_invlinked) ≈ lp_true
                        @test getlogjoint_internal(varinfo_invlinked) ≈ lp_true
                    end
                end
            end
        end
    end

    @testset "unflatten type stability" begin
        @model function demo(y)
            x ~ Normal()
            y ~ Normal(x, 1)
            return nothing
        end

        model = demo(0.0)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(
            model, (; x=1.0), (@varname(x),); include_threadsafe=true
        )
        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # Skip the inconcrete `SimpleVarInfo` types, since checking for type
            # stability for them doesn't make much sense anyway.
            if varinfo isa SimpleVarInfo{<:AbstractDict} ||
                varinfo isa DynamicPPL.ThreadSafeVarInfo{<:SimpleVarInfo{<:AbstractDict}}
                continue
            end
            @inferred DynamicPPL.unflatten(varinfo, varinfo[:])
        end
    end

    @testset "subset" begin
        @model function demo_subsetting_varinfo(::Type{TV}=Vector{Float64}) where {TV}
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            x = TV(undef, 2)
            x[1] ~ Normal(m, sqrt(s))
            x[2] ~ Normal(m, sqrt(s))
            return (; s, m, x)
        end
        model = demo_subsetting_varinfo()
        vns = [@varname(s), @varname(m), @varname(x[1]), @varname(x[2])]

        # `VarInfo` supports, effectively, arbitrary subsetting.
        varinfos = DynamicPPL.TestUtils.setup_varinfos(
            model, model(), vns; include_threadsafe=true
        )
        varinfos_standard = filter(Base.Fix2(isa, VarInfo), varinfos)
        varinfos_simple = filter(
            Base.Fix2(isa, DynamicPPL.SimpleOrThreadSafeSimple), varinfos
        )

        # `VarInfo` supports subsetting using, basically, arbitrary varnames.
        vns_supported_standard = [
            [@varname(s)],
            [@varname(m)],
            [@varname(x[1])],
            [@varname(x[2])],
            [@varname(s), @varname(m)],
            [@varname(s), @varname(x[1])],
            [@varname(s), @varname(x[2])],
            [@varname(m), @varname(x[1])],
            [@varname(m), @varname(x[2])],
            [@varname(x[1]), @varname(x[2])],
            [@varname(s), @varname(m), @varname(x[1])],
            [@varname(s), @varname(m), @varname(x[2])],
            [@varname(s), @varname(x[1]), @varname(x[2])],
            [@varname(m), @varname(x[1]), @varname(x[2])],
        ]

        # Patterns requiring `subsumes`.
        vns_supported_with_subsumes = [
            [@varname(s), @varname(x)] => [@varname(s), @varname(x[1]), @varname(x[2])],
            [@varname(m), @varname(x)] => [@varname(m), @varname(x[1]), @varname(x[2])],
            [@varname(s), @varname(m), @varname(x)] =>
                [@varname(s), @varname(m), @varname(x[1]), @varname(x[2])],
        ]

        # `SimpleVarInfo` only supports subsetting using the varnames as they appear
        # in the model.
        vns_supported_simple = filter(∈(vns), vns_supported_standard)

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # All variables.
            check_varinfo_keys(varinfo, vns)

            # Added a `convert` to make the naming of the testsets a bit more readable.
            # `SimpleVarInfo{<:NamedTuple}` only supports subsetting with "simple" varnames,
            ## i.e. `VarName{sym}()` without any indexing, etc.
            vns_supported =
                if varinfo isa DynamicPPL.SimpleOrThreadSafeSimple &&
                    values_as(varinfo) isa NamedTuple
                    vns_supported_simple
                else
                    vns_supported_standard
                end

            @testset ("$(convert(Vector{VarName}, vns_subset)) empty") for vns_subset in
                                                                           vns_supported
                varinfo_subset = subset(varinfo, VarName[])
                @test isempty(varinfo_subset)
            end

            @testset "$(convert(Vector{VarName}, vns_subset))" for vns_subset in
                                                                   vns_supported
                varinfo_subset = subset(varinfo, vns_subset)
                # Should now only contain the variables in `vns_subset`.
                check_varinfo_keys(varinfo_subset, vns_subset)
                # Values should be the same.
                @test [varinfo_subset[vn] for vn in vns_subset] == [varinfo[vn] for vn in vns_subset]

                # `merge` with the original.
                varinfo_merged = merge(varinfo, varinfo_subset)
                vns_merged = keys(varinfo_merged)
                # Should be equivalent.
                check_varinfo_keys(varinfo_merged, vns)
                # Values should be the same.
                @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
            end

            @testset "$(convert(Vector{VarName}, vns_subset))" for (
                vns_subset, vns_target
            ) in vns_supported_with_subsumes
                varinfo_subset = subset(varinfo, vns_subset)
                # Should now only contain the variables in `vns_subset`.
                check_varinfo_keys(varinfo_subset, vns_target)
                # Values should be the same.
                @test [varinfo_subset[vn] for vn in vns_target] == [varinfo[vn] for vn in vns_target]

                # `merge` with the original.
                varinfo_merged = merge(varinfo, varinfo_subset)
                vns_merged = keys(varinfo_merged)
                # Should be equivalent.
                check_varinfo_keys(varinfo_merged, vns)
                # Values should be the same.
                @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
            end

            @testset "$(convert(Vector{VarName}, vns_subset)) order" for vns_subset in
                                                                         vns_supported
                varinfo_subset = subset(varinfo, vns_subset)
                vns_subset_reversed = reverse(vns_subset)
                varinfo_subset_reversed = subset(varinfo, vns_subset_reversed)
                @test varinfo_subset[:] == varinfo_subset_reversed[:]
                ground_truth = [varinfo[vn] for vn in vns_subset]
                @test varinfo_subset[:] == ground_truth
            end
        end

        # For certain varinfos we should have errors.
        # `SimpleVarInfo{<:NamedTuple}` can only handle varnames with `identity`.
        varinfo = varinfos[findfirst(Base.Fix2(isa, SimpleVarInfo{<:NamedTuple}), varinfos)]
        @testset "$(short_varinfo_name(varinfo)): failure cases" begin
            @test_throws ArgumentError subset(
                varinfo, [@varname(s), @varname(m), @varname(x[1])]
            )
        end
    end

    @testset "merge" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.ALL_MODELS
            vns = DynamicPPL.TestUtils.varnames(model)
            varinfos = DynamicPPL.TestUtils.setup_varinfos(
                model,
                DynamicPPL.TestUtils.rand_prior_true(model),
                vns;
                include_threadsafe=true,
            )
            @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                @testset "with itself" begin
                    # Merging itself should be a no-op.
                    varinfo_merged = merge(varinfo, varinfo)
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
                end

                @testset "with itself (3-argument version)" begin
                    # Merging itself should be a no-op.
                    varinfo_merged = merge(varinfo, varinfo, varinfo)
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
                end

                @testset "with empty" begin
                    # Empty is 1st argument.
                    # Merging with an empty `VarInfo` should be a no-op.
                    varinfo_merged = merge(empty!!(deepcopy(varinfo)), varinfo)
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]

                    # Empty is 2nd argument.
                    # Merging with an empty `VarInfo` should be a no-op.
                    varinfo_merged = merge(varinfo, empty!!(deepcopy(varinfo)))
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
                end

                @testset "with different value" begin
                    x = DynamicPPL.TestUtils.rand_prior_true(model)
                    varinfo_changed = DynamicPPL.TestUtils.update_values!!(
                        deepcopy(varinfo), x, vns
                    )
                    # After `merge`, we should have the same values as `x`.
                    varinfo_merged = merge(varinfo, varinfo_changed)
                    DynamicPPL.TestUtils.test_values(varinfo_merged, x, vns)
                end
            end
        end

        @testset "different models" begin
            @model function demo_merge_different_y()
                x ~ Uniform()
                return y ~ Normal()
            end
            @model function demo_merge_different_z()
                x ~ Normal()
                return z ~ Normal()
            end
            model_left = demo_merge_different_y()
            model_right = demo_merge_different_z()

            varinfo_left = VarInfo(model_left)
            varinfo_right = VarInfo(model_right)
            varinfo_right = DynamicPPL.set_transformed!!(varinfo_right, true, @varname(x))

            varinfo_merged = merge(varinfo_left, varinfo_right)
            vns = [@varname(x), @varname(y), @varname(z)]
            check_varinfo_keys(varinfo_merged, vns)

            # Right has precedence.
            @test varinfo_merged[@varname(x)] == varinfo_right[@varname(x)]
            @test DynamicPPL.is_transformed(varinfo_merged, @varname(x))
        end
    end

    # The below used to error, testing to avoid regression.
    @testset "merge different dimensions" begin
        vn = @varname(x)
        vi_single = VarInfo()
        vi_single = push!!(vi_single, vn, 1.0, Normal())
        vi_double = VarInfo()
        vi_double = push!!(vi_double, vn, [0.5, 0.6], Dirichlet(2, 1.0))
        @test merge(vi_single, vi_double)[vn] == [0.5, 0.6]
        @test merge(vi_double, vi_single)[vn] == 1.0
    end

    @testset "issue #842" begin
        model = DynamicPPL.TestUtils.demo_dot_assume_observe()
        varinfo = VarInfo(model)

        n = length(varinfo[:])
        # `Bool`.
        @test getlogjoint(DynamicPPL.unflatten(varinfo, fill(true, n))) isa typeof(float(1))
        # `Int`.
        @test getlogjoint(DynamicPPL.unflatten(varinfo, fill(1, n))) isa typeof(float(1))
    end
end
