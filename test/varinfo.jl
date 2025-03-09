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
    elseif DynamicPPL.is_flagged(vi, vn, "del")
        DynamicPPL.unset_flag!(vi, vn, "del")
        r = rand(dist)
        vi[vn] = DynamicPPL.tovec(r)
        DynamicPPL.setorder!(vi, vn, DynamicPPL.get_num_produce(vi))
        r
    else
        vi[vn]
    end
end

@testset "varinfo.jl" begin
    @testset "TypedVarInfo with Metadata" begin
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2, 3)
            m ~ truncated(Normal(0.0, sqrt(s)), 0.0, 2.0)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        vi = VarInfo(DynamicPPL.Metadata())
        model(vi, SampleFromUniform())
        tvi = TypedVarInfo(vi)

        meta = vi.metadata
        for f in fieldnames(typeof(tvi.metadata))
            fmeta = getfield(tvi.metadata, f)
            for vn in fmeta.vns
                @test tvi[vn] == vi[vn]
                ind = meta.idcs[vn]
                tind = fmeta.idcs[vn]
                @test meta.dists[ind] == fmeta.dists[tind]
                @test meta.orders[ind] == fmeta.orders[tind]
                for flag in keys(meta.flags)
                    @test meta.flags[flag][ind] == fmeta.flags[flag][tind]
                end
                range = meta.ranges[ind]
                trange = fmeta.ranges[tind]
                @test all(meta.vals[range] .== fmeta.vals[trange])
            end
        end
    end

    @testset "Base" begin
        # Test Base functions:
        #   string, Symbol, ==, hash, in, keys, haskey, isempty, push!!, empty!!,
        #   getindex, setindex!, getproperty, setproperty!
        csym = gensym()
        vn1 = @varname x[1][2]
        @test string(vn1) == "x[1][2]"
        @test Symbol(vn1) == Symbol("x[1][2]")

        vn2 = @varname x[1][2]
        @test vn2 == vn1
        @test hash(vn2) == hash(vn1)

        function test_base!!(vi_original)
            vi = empty!!(vi_original)
            @test getlogp(vi) == 0
            @test isempty(vi[:])

            vn = @varname x
            dist = Normal(0, 1)
            r = rand(dist)

            @test isempty(vi)
            @test ~haskey(vi, vn)
            @test !(vn in keys(vi))
            vi = push!!(vi, vn, r, dist)
            @test ~isempty(vi)
            @test haskey(vi, vn)
            @test vn in keys(vi)

            @test length(vi[vn]) == 1
            @test vi[vn] == r
            vi = DynamicPPL.setindex!!(vi, 2 * r, vn)
            @test vi[vn] == 2 * r

            # TODO(mhauru) Implement these functions for other VarInfo types too.
            if vi isa DynamicPPL.VectorVarInfo
                delete!(vi, vn)
                @test isempty(vi)
                vi = push!!(vi, vn, r, dist)
            end

            vi = empty!!(vi)
            @test isempty(vi)
            vi = push!!(vi, vn, r, dist)
            @test ~isempty(vi)
        end

        vi = VarInfo()
        test_base!!(vi)
        test_base!!(TypedVarInfo(vi))
        test_base!!(SimpleVarInfo())
        test_base!!(SimpleVarInfo(Dict()))
        test_base!!(SimpleVarInfo(DynamicPPL.VarNamedVector()))
    end

    @testset "get/set/acc/resetlogp" begin
        function test_varinfo_logp!(vi)
            @test DynamicPPL.getlogp(vi) === 0.0
            vi = DynamicPPL.setlogp!!(vi, 1.0)
            @test DynamicPPL.getlogp(vi) === 1.0
            vi = DynamicPPL.acclogp!!(vi, 1.0)
            @test DynamicPPL.getlogp(vi) === 2.0
            vi = DynamicPPL.resetlogp!!(vi)
            @test DynamicPPL.getlogp(vi) === 0.0
        end

        vi = VarInfo()
        test_varinfo_logp!(vi)
        test_varinfo_logp!(TypedVarInfo(vi))
        test_varinfo_logp!(SimpleVarInfo())
        test_varinfo_logp!(SimpleVarInfo(Dict()))
        test_varinfo_logp!(SimpleVarInfo(DynamicPPL.VarNamedVector()))
    end

    @testset "flags" begin
        # Test flag setting:
        #    is_flagged, set_flag!, unset_flag!
        function test_varinfo!(vi)
            vn_x = @varname x
            dist = Normal(0, 1)
            r = rand(dist)

            push!!(vi, vn_x, r, dist)

            # del is set by default
            @test !is_flagged(vi, vn_x, "del")

            set_flag!(vi, vn_x, "del")
            @test is_flagged(vi, vn_x, "del")

            unset_flag!(vi, vn_x, "del")
            @test !is_flagged(vi, vn_x, "del")
        end
        vi = VarInfo(DynamicPPL.Metadata())
        test_varinfo!(vi)
        test_varinfo!(empty!!(TypedVarInfo(vi)))
    end

    @testset "push!! to TypedVarInfo" begin
        vn_x = @varname x
        vn_y = @varname y
        untyped_vi = VarInfo()
        untyped_vi = push!!(untyped_vi, vn_x, 1.0, Normal(0, 1))
        typed_vi = TypedVarInfo(untyped_vi)
        typed_vi = push!!(typed_vi, vn_y, 2.0, Normal(0, 1))
        @test typed_vi[vn_x] == 1.0
        @test typed_vi[vn_y] == 2.0
    end

    @testset "setval! & setval_and_resample!" begin
        @model function testmodel(x)
            n = length(x)
            s ~ truncated(Normal(), 0, Inf)
            m ~ MvNormal(zeros(n), I)
            return x ~ MvNormal(m, s^2 * I)
        end

        @model function testmodel_univariate(x, ::Type{TV}=Vector{Float64}) where {TV}
            n = length(x)
            s ~ truncated(Normal(), 0, Inf)

            m = TV(undef, n)
            for i in eachindex(m)
                m[i] ~ Normal()
            end

            for i in eachindex(x)
                x[i] ~ Normal(m[i], s)
            end
        end

        x = randn(5)
        model_mv = testmodel(x)
        model_uv = testmodel_univariate(x)

        for model in [model_uv, model_mv]
            m_vns = model == model_uv ? [@varname(m[i]) for i in 1:5] : @varname(m)
            s_vns = @varname(s)

            vi_typed = VarInfo(
                model, SampleFromPrior(), DefaultContext(), DynamicPPL.Metadata()
            )
            vi_untyped = VarInfo(DynamicPPL.Metadata())
            vi_vnv = VarInfo(DynamicPPL.VarNamedVector())
            vi_vnv_typed = VarInfo(
                model, SampleFromPrior(), DefaultContext(), DynamicPPL.VarNamedVector()
            )
            model(vi_untyped, SampleFromPrior())
            model(vi_vnv, SampleFromPrior())

            model_name = model == model_uv ? "univariate" : "multivariate"
            @testset "$(model_name), $(short_varinfo_name(vi))" for vi in [
                vi_untyped, vi_typed, vi_vnv, vi_vnv_typed
            ]
                Random.seed!(23)
                vicopy = deepcopy(vi)

                ### `setval` ###
                # TODO(mhauru) The interface here seems inconsistent between Metadata and
                # VarNamedVector. I'm lazy to fix it though, because I think we need to
                # rework it soon anyway.
                if vi in [vi_vnv, vi_vnv_typed]
                    DynamicPPL.setval!(vicopy, zeros(5), m_vns)
                else
                    DynamicPPL.setval!(vicopy, (m=zeros(5),))
                end
                # Setting `m` fails for univariate due to limitations of `setval!`
                # and `setval_and_resample!`. See docstring of `setval!` for more info.
                if model == model_uv && vi in [vi_untyped, vi_typed]
                    @test_broken vicopy[m_vns] == zeros(5)
                else
                    @test vicopy[m_vns] == zeros(5)
                end
                @test vicopy[s_vns] == vi[s_vns]

                # Ordering is NOT preserved => fails for multivariate model.
                DynamicPPL.setval!(
                    vicopy, (; (Symbol("m[$i]") => i for i in (1, 3, 5, 4, 2))...)
                )
                if model == model_uv
                    @test vicopy[m_vns] == 1:5
                else
                    @test vicopy[m_vns] == [1, 3, 5, 4, 2]
                end
                @test vicopy[s_vns] == vi[s_vns]

                DynamicPPL.setval!(
                    vicopy, (; (Symbol("m[$i]") => i for i in (1, 2, 3, 4, 5))...)
                )
                DynamicPPL.setval!(vicopy, (s=42,))
                @test vicopy[m_vns] == 1:5
                @test vicopy[s_vns] == 42

                ### `setval_and_resample!` ###
                if model == model_mv && vi == vi_untyped
                    # Trying to re-run model with `MvNormal` on `vi_untyped` will call
                    # `MvNormal(μ::Vector{Real}, Σ)` which causes `StackOverflowError`
                    # so we skip this particular case.
                    continue
                end

                if vi in [vi_vnv, vi_vnv_typed]
                    # `setval_and_resample!` works differently for `VarNamedVector`: All
                    # values will be resampled when model(vicopy) is called. Hence the below
                    # tests are not applicable.
                    continue
                end

                vicopy = deepcopy(vi)
                DynamicPPL.setval_and_resample!(vicopy, (m=zeros(5),))
                model(vicopy)
                # Setting `m` fails for univariate due to limitations of `subsumes(::String, ::String)`
                if model == model_uv
                    @test_broken vicopy[m_vns] == zeros(5)
                else
                    @test vicopy[m_vns] == zeros(5)
                end
                @test vicopy[s_vns] != vi[s_vns]

                # Ordering is NOT preserved.
                DynamicPPL.setval_and_resample!(
                    vicopy, (; (Symbol("m[$i]") => i for i in (1, 3, 5, 4, 2))...)
                )
                model(vicopy)
                if model == model_uv
                    @test vicopy[m_vns] == 1:5
                else
                    @test vicopy[m_vns] == [1, 3, 5, 4, 2]
                end
                @test vicopy[s_vns] != vi[s_vns]

                # Correct ordering.
                DynamicPPL.setval_and_resample!(
                    vicopy, (; (Symbol("m[$i]") => i for i in (1, 2, 3, 4, 5))...)
                )
                model(vicopy)
                @test vicopy[m_vns] == 1:5
                @test vicopy[s_vns] != vi[s_vns]

                DynamicPPL.setval_and_resample!(vicopy, (s=42,))
                model(vicopy)
                @test vicopy[m_vns] != 1:5
                @test vicopy[s_vns] == 42
            end
        end

        # https://github.com/TuringLang/DynamicPPL.jl/issues/250
        @model function demo()
            return x ~ filldist(MvNormal([1, 100], I), 2)
        end

        vi = VarInfo(demo())
        vals_prev = vi.metadata.x.vals
        ks = [@varname(x[1, 1]), @varname(x[2, 1]), @varname(x[1, 2]), @varname(x[2, 2])]
        DynamicPPL.setval!(vi, vi.metadata.x.vals, ks)
        @test vals_prev == vi.metadata.x.vals

        DynamicPPL.setval_and_resample!(vi, vi.metadata.x.vals, ks)
        @test vals_prev == vi.metadata.x.vals
    end

    @testset "setval! on chain" begin
        # Define a helper function
        """
            test_setval!(model, chain; sample_idx = 1, chain_idx = 1)

        Test `setval!` on `model` and `chain`.

        Worth noting that this only supports models containing symbols of the forms
        `m`, `m[1]`, `m[1, 2]`, not `m[1][1]`, etc.
        """
        function test_setval!(model, chain; sample_idx=1, chain_idx=1)
            var_info = VarInfo(model)
            θ_old = var_info[:]
            DynamicPPL.setval!(var_info, chain, sample_idx, chain_idx)
            θ_new = var_info[:]
            @test θ_old != θ_new
            vals = DynamicPPL.values_as(var_info, OrderedDict)
            iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
            for (n, v) in mapreduce(collect, vcat, iters)
                n = string(n)
                if Symbol(n) ∉ keys(chain)
                    # Assume it's a group
                    chain_val = vec(
                        MCMCChains.group(chain, Symbol(n)).value[sample_idx, :, chain_idx]
                    )
                    v_true = vec(v)
                else
                    chain_val = chain[sample_idx, n, chain_idx]
                    v_true = v
                end

                @test v_true == chain_val
            end
        end

        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = make_chain_from_prior(model, 10)
            # A simple way of checking that the computation is determinstic: run twice and compare.
            res1 = returned(model, MCMCChains.get_sections(chain, :parameters))
            res2 = returned(model, MCMCChains.get_sections(chain, :parameters))
            @test all(res1 .== res2)
            test_setval!(model, MCMCChains.get_sections(chain, :parameters))
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

        # Check that instantiating the model does not perform linking
        vi = VarInfo()
        meta = vi.metadata
        model(vi, SampleFromUniform())
        @test all(x -> !istrans(vi, x), meta.vns)

        # Check that linking and invlinking set the `trans` flag accordingly
        v = copy(meta.vals)
        vi = link!!(vi, model)
        @test all(x -> istrans(vi, x), meta.vns)
        vi = invlink!!(vi, model)
        @test all(x -> !istrans(vi, x), meta.vns)
        @test meta.vals ≈ v atol = 1e-10

        # Check that linking and invlinking preserves the values
        vi = TypedVarInfo(vi)
        meta = vi.metadata
        v_s = copy(meta.s.vals)
        v_m = copy(meta.m.vals)
        v_x = copy(meta.x.vals)
        v_y = copy(meta.y.vals)

        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> !istrans(vi, x), meta.m.vns)
        vi = link!!(vi, model)
        @test all(x -> istrans(vi, x), meta.s.vns)
        @test all(x -> istrans(vi, x), meta.m.vns)
        vi = invlink!!(vi, model)
        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> !istrans(vi, x), meta.m.vns)
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
            @test all(x -> istrans(vi, x), target_vns)
            @test all(x -> !istrans(vi, x), other_vns)
            vi = invlink!!(vi, (vn,), model)
            @test all(x -> !istrans(vi, x), all_vns)
            @test meta.s.vals ≈ v_s atol = 1e-10
            @test meta.m.vals ≈ v_m atol = 1e-10
            @test meta.x.vals ≈ v_x atol = 1e-10
            @test meta.y.vals ≈ v_y atol = 1e-10
        end
    end

    @testset "istrans" begin
        @model demo_constrained() = x ~ truncated(Normal(), 0, Inf)
        model = demo_constrained()
        vn = @varname(x)
        dist = truncated(Normal(), 0, Inf)

        ### `VarInfo`
        # Need to run once since we can't specify that we want to _sample_
        # in the unconstrained space for `VarInfo` without having `vn`
        # present in the `varinfo`.
        ## `UntypedVarInfo`
        vi = VarInfo()
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        vi = DynamicPPL.settrans!!(vi, true, vn)
        # Sample in unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
        x = f(DynamicPPL.getindex_internal(vi, vn))
        @test getlogp(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)

        ## `TypedVarInfo`
        vi = VarInfo(model)
        vi = DynamicPPL.settrans!!(vi, true, vn)
        # Sample in unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
        x = f(DynamicPPL.getindex_internal(vi, vn))
        @test getlogp(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)

        ### `SimpleVarInfo`
        ## `SimpleVarInfo{<:NamedTuple}`
        vi = DynamicPPL.settrans!!(SimpleVarInfo(), true)
        # Sample in unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
        x = f(DynamicPPL.getindex_internal(vi, vn))
        @test getlogp(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)

        ## `SimpleVarInfo{<:Dict}`
        vi = DynamicPPL.settrans!!(SimpleVarInfo(Dict()), true)
        # Sample in unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
        x = f(DynamicPPL.getindex_internal(vi, vn))
        @test getlogp(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)

        ## `SimpleVarInfo{<:VarNamedVector}`
        vi = DynamicPPL.settrans!!(SimpleVarInfo(DynamicPPL.VarNamedVector()), true)
        # Sample in unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, vi, SamplingContext()))
        f = DynamicPPL.from_linked_internal_transform(vi, vn, dist)
        x = f(DynamicPPL.getindex_internal(vi, vn))
        @test getlogp(vi) ≈ Bijectors.logpdf_with_trans(dist, x, true)
    end

    @testset "values_as" begin
        @testset "$(nameof(model))" for model in DynamicPPL.TestUtils.DEMO_MODELS
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
                            @test get(vals, vn) == getindex(vi, vn)
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
                    varinfo = last(DynamicPPL.evaluate!!(model, varinfo, DefaultContext()))

                    varinfo_linked = if mutating
                        DynamicPPL.link!!(deepcopy(varinfo), model)
                    else
                        DynamicPPL.link(varinfo, model)
                    end
                    for vn in keys(varinfo)
                        @test DynamicPPL.istrans(varinfo_linked, vn)
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
                    @test getlogp(varinfo) ≈ lp_true
                    lp_linked = getlogp(varinfo_linked)
                    @test lp_linked ≈ lp_linked_true

                    # TODO: Compare values once we are no longer working with `NamedTuple` for
                    # the true values, e.g. `value_true`.

                    if !mutating
                        # This is also compatible with invlinking of unflattened varinfo.
                        varinfo_invlinked = DynamicPPL.invlink(
                            varinfo_linked_unflattened, model
                        )
                        @test length(varinfo_invlinked[:]) == length(varinfo[:])
                        @test getlogp(varinfo_invlinked) ≈ lp_true
                    end
                end
            end
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
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
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
            varinfo_right = DynamicPPL.settrans!!(varinfo_right, true, @varname(x))

            varinfo_merged = merge(varinfo_left, varinfo_right)
            vns = [@varname(x), @varname(y), @varname(z)]
            check_varinfo_keys(varinfo_merged, vns)

            # Right has precedence.
            @test varinfo_merged[@varname(x)] == varinfo_right[@varname(x)]
            @test DynamicPPL.istrans(varinfo_merged, @varname(x))
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

    @testset "sampling from linked varinfo" begin
        # `~`
        @model function demo(n=1)
            x = Vector(undef, n)
            for i in eachindex(x)
                x[i] ~ Exponential()
            end
            return x
        end
        model1 = demo(1)
        varinfo1 = DynamicPPL.link!!(VarInfo(model1), model1)
        # Sampling from `model2` should hit the `istrans(vi) == true` branches
        # because all the existing variables are linked.
        model2 = demo(2)
        varinfo2 = last(
            DynamicPPL.evaluate!!(model2, deepcopy(varinfo1), SamplingContext())
        )
        for vn in [@varname(x[1]), @varname(x[2])]
            @test DynamicPPL.istrans(varinfo2, vn)
        end

        # `.~`
        @model function demo_dot(n=1)
            x ~ Exponential()
            if n > 1
                y = Vector(undef, n - 1)
                y .~ Exponential()
            end
            return x
        end
        model1 = demo_dot(1)
        varinfo1 = DynamicPPL.link!!(DynamicPPL.untyped_varinfo(model1), model1)
        # Sampling from `model2` should hit the `istrans(vi) == true` branches
        # because all the existing variables are linked.
        model2 = demo_dot(2)
        varinfo2 = last(
            DynamicPPL.evaluate!!(model2, deepcopy(varinfo1), SamplingContext())
        )
        for vn in [@varname(x), @varname(y[1])]
            @test DynamicPPL.istrans(varinfo2, vn)
        end
    end

    # NOTE: It is not yet clear if this is something we want from all varinfo types.
    # Hence, we only test the `VarInfo` types here.
    @testset "vector_getranges for `VarInfo`" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            vns = DynamicPPL.TestUtils.varnames(model)
            nt = DynamicPPL.TestUtils.rand_prior_true(model)
            varinfos = DynamicPPL.TestUtils.setup_varinfos(
                model, nt, vns; include_threadsafe=true
            )
            # Only keep `VarInfo` types.
            varinfos = filter(
                Base.Fix2(isa, DynamicPPL.VarInfoOrThreadSafeVarInfo), varinfos
            )
            @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                x = values_as(varinfo, Vector)

                # Let's just check all the subsets of `vns`.
                @testset "$(convert(Vector{Any},vns_subset))" for vns_subset in
                                                                  combinations(vns)
                    ranges = DynamicPPL.vector_getranges(varinfo, vns_subset)
                    @test length(ranges) == length(vns_subset)
                    for (r, vn) in zip(ranges, vns_subset)
                        @test x[r] == DynamicPPL.tovec(varinfo[vn])
                    end
                end

                # Let's try some failure cases.
                @test DynamicPPL.vector_getranges(varinfo, VarName[]) == UnitRange{Int}[]
                # Non-existent variables.
                @test_throws KeyError DynamicPPL.vector_getranges(
                    varinfo, [VarName{gensym("vn")}()]
                )
                @test_throws KeyError DynamicPPL.vector_getranges(
                    varinfo, [VarName{gensym("vn")}(), VarName{gensym("vn")}()]
                )
                # Duplicate variables.
                ranges_duplicated = DynamicPPL.vector_getranges(varinfo, repeat(vns, 2))
                @test x[reduce(vcat, ranges_duplicated)] == repeat(x, 2)
            end
        end
    end

    @testset "orders" begin
        @model empty_model() = x = 1

        csym = gensym() # unique per model
        vn_z1 = @varname z[1]
        vn_z2 = @varname z[2]
        vn_z3 = @varname z[3]
        vn_z4 = @varname z[4]
        vn_a1 = @varname a[1]
        vn_a2 = @varname a[2]
        vn_b = @varname b

        vi = DynamicPPL.VarInfo()
        dists = [Categorical([0.7, 0.3]), Normal()]

        # First iteration, variables are added to vi
        # variables samples in order: z1,a1,z2,a2,z3
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1])
        randr(vi, vn_a1, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_b, dists[2])
        randr(vi, vn_z2, dists[1])
        randr(vi, vn_a2, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1])
        @test vi.metadata.orders == [1, 1, 2, 2, 2, 3]
        @test DynamicPPL.get_num_produce(vi) == 3

        DynamicPPL.reset_num_produce!(vi)
        DynamicPPL.set_retained_vns_del!(vi)
        @test DynamicPPL.is_flagged(vi, vn_z1, "del")
        @test DynamicPPL.is_flagged(vi, vn_a1, "del")
        @test DynamicPPL.is_flagged(vi, vn_z2, "del")
        @test DynamicPPL.is_flagged(vi, vn_a2, "del")
        @test DynamicPPL.is_flagged(vi, vn_z3, "del")

        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1])
        randr(vi, vn_a1, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z2, dists[1])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1])
        randr(vi, vn_a2, dists[2])
        @test vi.metadata.orders == [1, 1, 2, 2, 3, 3]
        @test DynamicPPL.get_num_produce(vi) == 3

        vi = empty!!(DynamicPPL.TypedVarInfo(vi))
        # First iteration, variables are added to vi
        # variables samples in order: z1,a1,z2,a2,z3
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1])
        randr(vi, vn_a1, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_b, dists[2])
        randr(vi, vn_z2, dists[1])
        randr(vi, vn_a2, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1])
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 2]
        @test vi.metadata.b.orders == [2]
        @test DynamicPPL.get_num_produce(vi) == 3

        DynamicPPL.reset_num_produce!(vi)
        DynamicPPL.set_retained_vns_del!(vi)
        @test DynamicPPL.is_flagged(vi, vn_z1, "del")
        @test DynamicPPL.is_flagged(vi, vn_a1, "del")
        @test DynamicPPL.is_flagged(vi, vn_z2, "del")
        @test DynamicPPL.is_flagged(vi, vn_a2, "del")
        @test DynamicPPL.is_flagged(vi, vn_z3, "del")

        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1])
        randr(vi, vn_a1, dists[2])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z2, dists[1])
        DynamicPPL.increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1])
        randr(vi, vn_a2, dists[2])
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 3]
        @test vi.metadata.b.orders == [2]
        @test DynamicPPL.get_num_produce(vi) == 3
    end
end
