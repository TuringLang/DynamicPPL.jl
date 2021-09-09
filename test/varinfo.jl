@testset "varinfo.jl" begin
    @testset "TypedVarInfo" begin
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2, 3)
            m ~ truncated(Normal(0.0, sqrt(s)), 0.0, 2.0)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        vi = VarInfo()
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
                @test meta.gids[ind] == fmeta.gids[tind]
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
        @test inspace(vn1, (:x,))

        function test_base!(vi)
            empty!!(vi)
            @test getlogp(vi) == 0
            @test get_num_produce(vi) == 0

            vn = @varname x
            dist = Normal(0, 1)
            r = rand(dist)
            gid = Selector()

            @test isempty(vi)
            @test ~haskey(vi, vn)
            @test !(vn in keys(vi))
            push!!(vi, vn, r, dist, gid)
            @test ~isempty(vi)
            @test haskey(vi, vn)
            @test vn in keys(vi)

            @test length(vi[vn]) == 1
            @test length(vi[SampleFromPrior()]) == 1

            @test vi[vn] == r
            @test vi[SampleFromPrior()][1] == r
            vi[vn] = [2 * r]
            @test vi[vn] == 2 * r
            @test vi[SampleFromPrior()][1] == 2 * r
            vi[SampleFromPrior()] = [3 * r]
            @test vi[vn] == 3 * r
            @test vi[SampleFromPrior()][1] == 3 * r

            empty!!(vi)
            @test isempty(vi)
            push!!(vi, vn, r, dist, gid)

            function test_inspace()
                space = (:x, :y, @varname(z[1]), @varname(M[1:10, :]))

                @test inspace(@varname(x), space)
                @test inspace(@varname(y), space)
                @test inspace(@varname(x[1]), space)
                @test inspace(@varname(z[1][1]), space)
                @test inspace(@varname(z[1][:]), space)
                @test inspace(@varname(z[1][2:3:10]), space)
                @test inspace(@varname(M[[2, 3], 1]), space)
                @test inspace(@varname(M[:, 1:4]), space)
                @test inspace(@varname(M[1, [2, 4, 6]]), space)
                @test !inspace(@varname(z[2]), space)
                @test !inspace(@varname(z), space)
            end
            return test_inspace()
        end
        vi = VarInfo()
        test_base!(vi)
        test_base!(empty!!(TypedVarInfo(vi)))
    end
    @testset "flags" begin
        # Test flag setting:
        #    is_flagged, set_flag!, unset_flag!
        function test_varinfo!(vi)
            vn_x = @varname x
            dist = Normal(0, 1)
            r = rand(dist)
            gid = Selector()

            push!!(vi, vn_x, r, dist, gid)

            # del is set by default
            @test !is_flagged(vi, vn_x, "del")

            set_flag!(vi, vn_x, "del")
            @test is_flagged(vi, vn_x, "del")

            unset_flag!(vi, vn_x, "del")
            @test !is_flagged(vi, vn_x, "del")
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!!(TypedVarInfo(vi)))
    end
    @testset "setgid!" begin
        vi = VarInfo()
        meta = vi.metadata
        vn = @varname x
        dist = Normal(0, 1)
        r = rand(dist)
        gid1 = Selector()
        gid2 = Selector(2, :HMC)

        push!!(vi, vn, r, dist, gid1)
        @test meta.gids[meta.idcs[vn]] == Set([gid1])
        setgid!!(vi, gid2, vn)
        @test meta.gids[meta.idcs[vn]] == Set([gid1, gid2])

        vi = empty!!(TypedVarInfo(vi))
        meta = vi.metadata
        push!!(vi, vn, r, dist, gid1)
        @test meta.x.gids[meta.x.idcs[vn]] == Set([gid1])
        setgid!!(vi, gid2, vn)
        @test meta.x.gids[meta.x.idcs[vn]] == Set([gid1, gid2])
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

            vi_typed = VarInfo(model)
            vi_untyped = VarInfo()
            model(vi_untyped, SampleFromPrior())

            for vi in [vi_untyped, vi_typed]
                vicopy = deepcopy(vi)

                ### `setval` ###
                DynamicPPL.setval!(vicopy, (m=zeros(5),))
                # Setting `m` fails for univariate due to limitations of `setval!`
                # and `setval_and_resample!`. See docstring of `setval!` for more info.
                if model == model_uv
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
end
