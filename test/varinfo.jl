function check_varinfo_keys(varinfo, vns)
    vns_varinfo = keys(varinfo)
    @test union(vns_varinfo, vns) == intersect(vns_varinfo, vns)
end

function check_metadata_type_equal(v1::VarInfo, v2::VarInfo)
    @test typeof(v1.values) == typeof(v2.values)
end
function check_metadata_type_equal(
    v1::DynamicPPL.ThreadSafeVarInfo{<:AbstractVarInfo},
    v2::DynamicPPL.ThreadSafeVarInfo{<:AbstractVarInfo},
)
    return check_metadata_type_equal(v1.varinfo, v2.varinfo)
end

using Random: Xoshiro

@testset "varinfo.jl" begin
    @testset "Base" begin
        # Test Base functions:
        #   in, keys, haskey, isempty, setindex!!, empty!!,
        #   getindex, setindex!, getproperty, setproperty!

        vi = VarInfo()
        @test getlogjoint(vi) == 0
        @test isempty(vi[:])

        vn = @varname x
        r = rand()

        @test isempty(vi)
        @test !haskey(vi, vn)
        @test !(vn in keys(vi))
        vi = setindex!!(vi, r, vn)
        @test !isempty(vi)
        @test haskey(vi, vn)
        @test vn in keys(vi)

        @test length(vi[vn]) == 1
        @test vi[vn] == r
        @test vi[:] == [r]
        vi = DynamicPPL.setindex!!(vi, 2 * r, vn)
        @test vi[vn] == 2 * r
        @test vi[:] == [2 * r]

        vi = empty!!(vi)
        @test isempty(vi)
        vi = setindex!!(vi, r, vn)
        @test !isempty(vi)
    end

    @testset "get/set/acclogp" begin
        vi = VarInfo()
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

        vi = DynamicPPL.unflatten!!(VarInfo(m), collect(values))

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
        vi = VarInfo()
        vn_x = @varname x
        r = rand()

        vi = setindex!!(vi, r, vn_x)

        # is_transformed is unset by default
        @test !is_transformed(vi, vn_x)

        vi = set_transformed!!(vi, true, vn_x)
        @test is_transformed(vi, vn_x)

        vi = set_transformed!!(vi, false, vn_x)
        @test !is_transformed(vi, vn_x)
    end

    # TODO(mhauru) Move this to a different file.
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
        _, vi = DynamicPPL.init!!(model, vi, InitFromUniform())
        vals = values(vi)

        all_transformed(vi) = mapreduce(
            p -> p.second isa DynamicPPL.LinkedVectorValue, &, vi.values; init=true
        )
        any_transformed(vi) = mapreduce(
            p -> p.second isa DynamicPPL.LinkedVectorValue, |, vi.values; init=false
        )

        @test !any_transformed(vi)

        # Check that linking and invlinking set the `is_transformed` flag accordingly
        vi = link!!(vi, model)
        @test all_transformed(vi)
        vi = invlink!!(vi, model)
        @test !any_transformed(vi)
        @test values(vi) ≈ vals atol = 1e-10

        # Transform only one variable
        all_vns = keys(vi)
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
            @test all_transformed(subset(vi, target_vns))
            @test !any_transformed(subset(vi, other_vns))
            vi = invlink!!(vi, (vn,), model)
            @test !any_transformed(vi)
            @test values(vi) ≈ vals atol = 1e-10
        end
    end

    @testset "instantiation with transform strategy" begin
        @model function f()
            x ~ Beta(2, 2)
            return y ~ LogNormal(0, 1)
        end

        function test_transform_strategy(
            transform_strategy::DynamicPPL.AbstractTransformStrategy,
            model::DynamicPPL.Model,
            expected_linked_vns::Set{<:VarName},
        )
            # Test that the variables are linked according to the transform strategy
            vi = VarInfo(Xoshiro(468), model, transform_strategy)
            for vn in keys(vi)
                if vn in expected_linked_vns
                    @test DynamicPPL.get_transformed_value(vi, vn) isa
                        DynamicPPL.LinkedVectorValue
                else
                    @test DynamicPPL.get_transformed_value(vi, vn) isa
                        DynamicPPL.VectorValue
                end
            end
            # Test that initialising directly is the same as linking later (if rng is the
            # same)
            if transform_strategy isa LinkAll
                vi2 = VarInfo(Xoshiro(468), model)
                vi2 = DynamicPPL.link!!(vi2, model)
                @test vi == vi2
            end
            if transform_strategy isa LinkSome
                vi2 = VarInfo(Xoshiro(468), model)
                vi2 = DynamicPPL.link!!(vi2, transform_strategy.vns, model)
                @test vi == vi2
            end
        end

        model = f()
        test_transform_strategy(LinkAll(), model, Set([@varname(x), @varname(y)]))
        test_transform_strategy(
            LinkSome((@varname(x),), UnlinkAll()), model, Set([@varname(x)])
        )
        test_transform_strategy(
            LinkSome((@varname(y),), UnlinkAll()), model, Set([@varname(y)])
        )
        test_transform_strategy(
            LinkSome((@varname(x), @varname(y)), UnlinkAll()),
            model,
            Set([@varname(x), @varname(y)]),
        )
        test_transform_strategy(UnlinkAll(), model, Set{VarName}())
        test_transform_strategy(
            UnlinkSome((@varname(x),), LinkAll()), model, Set{VarName}()
        )
        test_transform_strategy(
            UnlinkSome((@varname(y),), LinkAll()), model, Set{VarName}()
        )
        test_transform_strategy(
            UnlinkSome((@varname(x), @varname(y)), LinkAll()), model, Set{VarName}()
        )
    end

    @testset "unflatten!! + linking" begin
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
                    varinfo_linked_unflattened = DynamicPPL.unflatten!!(
                        copy(varinfo_linked), varinfo_linked[:]
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

    @testset "unflatten!! type stability" begin
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
            @inferred DynamicPPL.unflatten!!(varinfo, varinfo[:])
        end
    end

    @testset "internal_values_as_vector" begin
        @model function internal_values()
            x ~ Normal()
            y ~ Beta(2, 2)
            return z ~ Dirichlet(ones(3))
        end
        distributions = OrderedDict(
            @varname(x) => Normal(),
            @varname(y) => Beta(2, 2),
            @varname(z) => Dirichlet(ones(3)),
        )
        unlinked_values = OrderedDict(
            @varname(x) => 1.0, @varname(y) => 0.5, @varname(z) => [0.2, 0.3, 0.5]
        )

        model = internal_values()
        @testset for link_strategy in [
            UnlinkAll(),
            LinkAll(),
            LinkSome((@varname(y),), UnlinkAll()),
            LinkSome((@varname(x), @varname(z)), UnlinkAll()),
        ]
            vi = VarInfo(model, link_strategy, InitFromParams(unlinked_values))

            expected_vector_values = Float64[]
            for (vn, dist) in distributions
                target = target_transform(link_strategy, vn)
                vn_vec_val = if target isa DynamicLink
                    DynamicPPL.to_linked_vec_transform(dist)(unlinked_values[vn])
                elseif target isa Unlink
                    DynamicPPL.to_vec_transform(dist)(unlinked_values[vn])
                else
                    error("don't know how to handle transform type $target")
                end
                append!(expected_vector_values, vn_vec_val)
            end
            @test internal_values_as_vector(vi) ≈ expected_vector_values
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

        # `VarInfo` supports subsetting using, basically, arbitrary varnames.
        vns_supported = [
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

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # All variables.
            check_varinfo_keys(varinfo, vns)

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
                    # Metadata types should be exactly the same.
                    check_metadata_type_equal(varinfo_merged, varinfo)
                end

                @testset "with itself (3-argument version)" begin
                    # Merging itself should be a no-op.
                    varinfo_merged = merge(varinfo, varinfo, varinfo)
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
                    # Metadata types should be exactly the same.
                    check_metadata_type_equal(varinfo_merged, varinfo)
                end

                @testset "with empty" begin
                    # Empty is 1st argument.
                    # Merging with an empty `VarInfo` should be a no-op.
                    varinfo_merged = merge(empty!!(deepcopy(varinfo)), varinfo)
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]

                    # Metadata types should be exactly the same.
                    check_metadata_type_equal(varinfo_merged, varinfo)
                    # Empty is 2nd argument.
                    # Merging with an empty `VarInfo` should be a no-op.
                    varinfo_merged = merge(varinfo, empty!!(deepcopy(varinfo)))
                    # Varnames should be unchanged.
                    check_varinfo_keys(varinfo_merged, vns)
                    # Values should be the same.
                    @test [varinfo_merged[vn] for vn in vns] == [varinfo[vn] for vn in vns]
                    # Metadata types should be exactly the same.
                    check_metadata_type_equal(varinfo_merged, varinfo)
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
        vi_single = setindex!!(vi_single, 1.0, vn)
        vi_double = VarInfo()
        vi_double = setindex!!(vi_double, [0.5, 0.6], vn)
        @test merge(vi_single, vi_double)[vn] == [0.5, 0.6]
        @test merge(vi_double, vi_single)[vn] == 1.0
    end

    @testset "issue #842" begin
        model = DynamicPPL.TestUtils.demo_dot_assume_observe()
        varinfo = VarInfo(model)

        n = length(varinfo[:])
        # `Bool`.
        @test getlogjoint(DynamicPPL.unflatten!!(varinfo, fill(true, n))) isa
            typeof(float(1))
        # `Int`.
        @test getlogjoint(DynamicPPL.unflatten!!(varinfo, fill(1, n))) isa typeof(float(1))
    end
end
