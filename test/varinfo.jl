@testset "varinfo.jl" begin
    @testset "TypedVarInfo" begin
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2,3)
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
        #   string, Symbol, ==, hash, in, keys, haskey, isempty, push!, empty!,
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
            empty!(vi)
            @test getlogp(vi) == 0
            @test get_num_produce(vi) == 0

            vn = @varname x
            dist = Normal(0, 1)
            r = rand(dist)
            gid = Selector()

            @test isempty(vi)
            @test ~haskey(vi, vn)
            push!(vi, vn, r, dist, gid)
            @test ~isempty(vi)
            @test haskey(vi, vn)

            @test length(vi[vn]) == 1
            @test length(vi[SampleFromPrior()]) == 1

            @test vi[vn] == r
            @test vi[SampleFromPrior()][1] == r
            vi[vn] = [2*r]
            @test vi[vn] == 2*r
            @test vi[SampleFromPrior()][1] == 2*r
            vi[SampleFromPrior()] = [3*r]
            @test vi[vn] == 3*r
            @test vi[SampleFromPrior()][1] == 3*r

            empty!(vi)
            @test isempty(vi)
            push!(vi, vn, r, dist, gid)

            function test_inspace()
                space = (:x, :y, @varname(z[1]), @varname(M[1:10, :]))

                @test inspace(@varname(x), space)
                @test inspace(@varname(y), space)
                @test inspace(@varname(x[1]), space)
                @test inspace(@varname(z[1][1]), space)
                @test inspace(@varname(z[1][:]), space)
                @test inspace(@varname(z[1][2:3:10]), space)
                @test inspace(@varname(M[[2,3], 1]), space)
                @test inspace(@varname(M[:, 1:4]), space)
                @test inspace(@varname(M[1, [2, 4, 6]]), space)
                @test !inspace(@varname(z[2]), space)
                @test !inspace(@varname(z), space)
            end
            test_inspace()
        end
        vi = VarInfo()
        test_base!(vi)
        test_base!(empty!(TypedVarInfo(vi)))
    end
    @testset "flags" begin
        # Test flag setting:
        #    is_flagged, set_flag!, unset_flag!
        function test_varinfo!(vi)
            vn_x = @varname x
            dist = Normal(0, 1)
            r = rand(dist)
            gid = Selector()

            push!(vi, vn_x, r, dist, gid)

            # del is set by default
            @test !is_flagged(vi, vn_x, "del")

            set_flag!(vi, vn_x, "del")
            @test is_flagged(vi, vn_x, "del")

            unset_flag!(vi, vn_x, "del")
            @test !is_flagged(vi, vn_x, "del")
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!(TypedVarInfo(vi)))
    end
    @testset "setgid!" begin
        vi = VarInfo()
        meta = vi.metadata
        vn = @varname x
        dist = Normal(0, 1)
        r = rand(dist)
        gid1 = Selector()
        gid2 = Selector(2, :HMC)

        push!(vi, vn, r, dist, gid1)
        @test meta.gids[meta.idcs[vn]] == Set([gid1])
        setgid!(vi, gid2, vn)
        @test meta.gids[meta.idcs[vn]] == Set([gid1, gid2])

        vi = empty!(TypedVarInfo(vi))
        meta = vi.metadata
        push!(vi, vn, r, dist, gid1)
        @test meta.x.gids[meta.x.idcs[vn]] == Set([gid1])
        setgid!(vi, gid2, vn)
        @test meta.x.gids[meta.x.idcs[vn]] == Set([gid1, gid2])
    end
    @testset "setval!" begin
        @model function testmodel(x)
            n = length(x)
            s ~ truncated(Normal(), 0, Inf)
            m ~ MvNormal(n, 1.0)
            x ~ MvNormal(m, s)
        end

        x = randn(5)
        model = testmodel(x)

        # UntypedVarInfo
        vi = VarInfo()
        model(vi, SampleFromPrior())

        vicopy = deepcopy(vi)
        DynamicPPL.setval!(vicopy, (m = zeros(5),))
        @test vicopy[@varname(m)] == zeros(5)
        @test vicopy[@varname(s)] == vi[@varname(s)]

        DynamicPPL.setval!(vicopy, (; (Symbol("m[$i]") => i for i in (1, 3, 5, 4, 2))...))
        @test vicopy[@varname(m)] == 1:5
        @test vicopy[@varname(s)] == vi[@varname(s)]

        DynamicPPL.setval!(vicopy, (s = 42,))
        @test vicopy[@varname(m)] == 1:5
        @test vicopy[@varname(s)] == 42

        # TypedVarInfo
        vi = VarInfo(model)

        vicopy = deepcopy(vi)
        DynamicPPL.setval!(vicopy, (m = zeros(5),))
        @test vicopy[@varname(m)] == zeros(5)
        @test vicopy[@varname(s)] == vi[@varname(s)]

        DynamicPPL.setval!(vicopy, (; (Symbol("m[$i]") => i for i in (1, 3, 5, 4, 2))...))
        @test vicopy[@varname(m)] == 1:5
        @test vicopy[@varname(s)] == vi[@varname(s)]

        DynamicPPL.setval!(vicopy, (s = 42,))
        @test vicopy[@varname(m)] == 1:5
        @test vicopy[@varname(s)] == 42
    end
end
