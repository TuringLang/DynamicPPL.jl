@testset "varinfo.jl" begin
    # Declare empty model to make the Sampler constructor work.
    @model empty_model() = begin
        x = 1
    end

    function randr(
        vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count::Bool=false
    )
        if !haskey(vi, vn)
            r = rand(dist)
            push!!(vi, vn, r, dist, spl)
            r
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(dist)
            vi[vn] = vectorize(dist, r)
            setorder!(vi, vn, get_num_produce(vi))
            r
        else
            count && checkindex(vn, vi, spl)
            DynamicPPL.updategid!(vi, vn, spl)
            vi[vn]
        end
    end

    @testset "link!" begin
        # Test linking spl and vi:
        #    link!, invlink!, istrans
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2, 3)
            m ~ Uniform(0, 2)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        vi = VarInfo()
        meta = vi.metadata
        model(vi, SampleFromUniform())
        @test all(x -> !istrans(vi, x), meta.vns)

        alg = HMC(0.1, 5)
        spl = Sampler(alg, model)
        v = copy(meta.vals)
        link!(vi, spl)
        @test all(x -> istrans(vi, x), meta.vns)
        invlink!(vi, spl)
        @test all(x -> !istrans(vi, x), meta.vns)
        @test meta.vals == v

        vi = TypedVarInfo(vi)
        meta = vi.metadata
        alg = HMC(0.1, 5)
        spl = Sampler(alg, model)
        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> !istrans(vi, x), meta.m.vns)
        v_s = copy(meta.s.vals)
        v_m = copy(meta.m.vals)
        link!(vi, spl)
        @test all(x -> istrans(vi, x), meta.s.vns)
        @test all(x -> istrans(vi, x), meta.m.vns)
        invlink!(vi, spl)
        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> !istrans(vi, x), meta.m.vns)
        @test meta.s.vals == v_s
        @test meta.m.vals == v_m

        # Transforming only a subset of the variables
        link!(vi, spl, Val((:m,)))
        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> istrans(vi, x), meta.m.vns)
        invlink!(vi, spl, Val((:m,)))
        @test all(x -> !istrans(vi, x), meta.s.vns)
        @test all(x -> !istrans(vi, x), meta.m.vns)
        @test meta.s.vals == v_s
        @test meta.m.vals == v_m
    end
    @testset "orders" begin
        csym = gensym() # unique per model
        vn_z1 = @varname z[1]
        vn_z2 = @varname z[2]
        vn_z3 = @varname z[3]
        vn_z4 = @varname z[4]
        vn_a1 = @varname a[1]
        vn_a2 = @varname a[2]
        vn_b = @varname b

        vi = VarInfo()
        dists = [Categorical([0.7, 0.3]), Normal()]

        spl1 = Sampler(PG(5), empty_model())
        spl2 = Sampler(PG(5), empty_model())

        # First iteration, variables are added to vi
        # variables samples in order: z1,a1,z2,a2,z3
        increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_b, dists[2], spl2)
        randr(vi, vn_z2, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1], spl1)
        @test vi.metadata.orders == [1, 1, 2, 2, 2, 3]
        @test get_num_produce(vi) == 3

        reset_num_produce!(vi)
        set_retained_vns_del_by_spl!(vi, spl1)
        @test is_flagged(vi, vn_z1, "del")
        @test is_flagged(vi, vn_a1, "del")
        @test is_flagged(vi, vn_z2, "del")
        @test is_flagged(vi, vn_a2, "del")
        @test is_flagged(vi, vn_z3, "del")

        increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z2, dists[1], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        @test vi.metadata.orders == [1, 1, 2, 2, 3, 3]
        @test get_num_produce(vi) == 3

        vi = empty!!(TypedVarInfo(vi))
        # First iteration, variables are added to vi
        # variables samples in order: z1,a1,z2,a2,z3
        increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_b, dists[2], spl2)
        randr(vi, vn_z2, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1], spl1)
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 2]
        @test vi.metadata.b.orders == [2]
        @test get_num_produce(vi) == 3

        reset_num_produce!(vi)
        set_retained_vns_del_by_spl!(vi, spl1)
        @test is_flagged(vi, vn_z1, "del")
        @test is_flagged(vi, vn_a1, "del")
        @test is_flagged(vi, vn_z2, "del")
        @test is_flagged(vi, vn_a2, "del")
        @test is_flagged(vi, vn_z3, "del")

        increment_num_produce!(vi)
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z2, dists[1], spl1)
        increment_num_produce!(vi)
        randr(vi, vn_z3, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 3]
        @test vi.metadata.b.orders == [2]
        @test get_num_produce(vi) == 3
    end
    @testset "replay" begin
        # Generate synthesised data
        xs = rand(Normal(0.5, 1), 100)

        # Define model
        @model function priorsinarray(xs, ::Type{T}=Float64) where {T}
            begin
                priors = Vector{T}(undef, 2)
                priors[1] ~ InverseGamma(2, 3)
                priors[2] ~ Normal(0, sqrt(priors[1]))
                for i in 1:length(xs)
                    xs[i] ~ Normal(priors[2], sqrt(priors[1]))
                end
                priors
            end
        end

        # Sampling
        chain = sample(priorsinarray(xs), HMC(0.01, 10), 10)
    end
    @testset "varname" begin
        @model function mat_name_test()
            p = Array{Any}(undef, 2, 2)
            for i in 1:2, j in 1:2
                p[i, j] ~ Normal(0, 1)
            end
            return p
        end
        chain = sample(mat_name_test(), HMC(0.2, 4), 1000)
        check_numerical(chain, ["p[1,1]"], [0]; atol=0.25)

        @model function marr_name_test()
            p = Array{Array{Any}}(undef, 2)
            p[1] = Array{Any}(undef, 2)
            p[2] = Array{Any}(undef, 2)
            for i in 1:2, j in 1:2
                p[i][j] ~ Normal(0, 1)
            end
            return p
        end

        chain = sample(marr_name_test(), HMC(0.2, 4), 1000)
        check_numerical(chain, ["p[1][1]"], [0]; atol=0.25)
    end
    @testset "varinfo" begin
        dists = [Normal(0, 1), MvNormal(zeros(2), I), Wishart(7, [1 0.5; 0.5 1])]
        function test_varinfo!(vi)
            @test getlogp(vi) === 0.0
            vi = setlogp!!(vi, 1)
            @test getlogp(vi) === 1.0
            vi = acclogp!!(vi, 1)
            @test getlogp(vi) === 2.0
            vi = resetlogp!!(vi)
            @test getlogp(vi) === 0.0

            spl2 = Sampler(PG(5, :w, :u), empty_model())
            vn_w = @varname w
            randr(vi, vn_w, dists[1], spl2, true)

            vn_x = @varname x
            vn_y = @varname y
            vn_z = @varname z
            vns = [vn_x, vn_y, vn_z]

            spl1 = Sampler(PG(5, :x, :y, :z), empty_model())
            for i in 1:3
                r = randr(vi, vns[i], dists[i], spl1, false)
                val = vi[vns[i]]
                @test sum(val - r) <= 1e-9
            end

            idcs = DynamicPPL._getidcs(vi, spl1)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 3
            else
                @test length(idcs) == 3
            end
            @test length(vi[spl1]) == 7

            idcs = DynamicPPL._getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 1
            else
                @test length(idcs) == 1
            end
            @test length(vi[spl2]) == 1

            vn_u = @varname u
            randr(vi, vn_u, dists[1], spl2, true)

            idcs = DynamicPPL._getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 2
            else
                @test length(idcs) == 2
            end
            @test length(vi[spl2]) == 2
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!!(TypedVarInfo(vi)))

        @model igtest() = begin
            x ~ InverseGamma(2, 3)
            y ~ InverseGamma(2, 3)
            z ~ InverseGamma(2, 3)
            w ~ InverseGamma(2, 3)
            u ~ InverseGamma(2, 3)
        end

        # Test the update of group IDs
        g_demo_f = igtest()

        # This test section no longer seems as applicable, considering the
        # user will never end up using an UntypedVarInfo. The `VarInfo`
        # Varible is also not passed around in the same way as it used to be.

        # TODO: Has to be fixed

        #= g = Sampler(Gibbs(PG(10, :x, :y, :z), HMC(0.4, 8, :w, :u)), g_demo_f)
        vi = VarInfo()
        g_demo_f(vi, SampleFromPrior())
        _, state = @inferred AbstractMCMC.step(Random.default_rng(), g_demo_f, g)
        pg, hmc = state.states
        @test pg isa TypedVarInfo
        @test hmc isa Turing.Inference.HMCState
        vi1 = state.vi
        @test mapreduce(x -> x.gids, vcat, vi1.metadata) ==
            [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]), Set{Selector}(), Set{Selector}()]

        @inferred g_demo_f(vi1, hmc)
        @test mapreduce(x -> x.gids, vcat, vi1.metadata) ==
            [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]), Set([hmc.selector]), Set([hmc.selector])]

        g = Sampler(Gibbs(PG(10, :x, :y, :z), HMC(0.4, 8, :w, :u)), g_demo_f)
        pg, hmc = g.state.samplers
        vi = empty!!(TypedVarInfo(vi))
        @inferred g_demo_f(vi, SampleFromPrior())
        pg.state.vi = vi
        step!(Random.default_rng(), g_demo_f, pg, 1)
        vi = pg.state.vi
        @inferred g_demo_f(vi, hmc)
        @test vi.metadata.x.gids[1] == Set([pg.selector])
        @test vi.metadata.y.gids[1] == Set([pg.selector])
        @test vi.metadata.z.gids[1] == Set([pg.selector])
        @test vi.metadata.w.gids[1] == Set([hmc.selector])
        @test vi.metadata.u.gids[1] == Set([hmc.selector]) =#
    end
end
