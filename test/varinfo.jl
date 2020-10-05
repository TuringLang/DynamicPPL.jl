using .Turing, Random
using DynamicPPL: Selector, reconstruct, invlink, CACHERESET,
    SampleFromPrior, Sampler, SampleFromUniform,
    _getidcs, set_retained_vns_del_by_spl!, is_flagged,
    set_flag!, unset_flag!, VarInfo, TypedVarInfo,
    getlogp, setlogp!, resetlogp!, acclogp!, vectorize,
    setorder!, updategid!
using DynamicPPL, LinearAlgebra
using Distributions
using ForwardDiff: Dual

import AbstractMCMC

using Test

dir = splitdir(splitdir(pathof(DynamicPPL))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "RandomVariables.jl" begin
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
    @testset "link!" begin
        # Test linking spl and vi:
        #    link!, invlink!, istrans
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2,3)
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
        @test all(x -> ~istrans(vi, x), meta.s.vns)
        @test all(x -> ~istrans(vi, x), meta.m.vns)
        @test meta.s.vals == v_s
        @test meta.m.vals == v_m
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
    @testset "orders" begin
        function randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler)
            if ~haskey(vi, vn)
                r = rand(dist)
                push!(vi, vn, r, dist, spl)
                r
            elseif is_flagged(vi, vn, "del")
                unset_flag!(vi, vn, "del")
                r = rand(dist)
                vi[vn] = vectorize(dist, r)
                setorder!(vi, vn, get_num_produce(vi))
                r
            else
                updategid!(vi, vn, spl)
                vi[vn]
            end
        end

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

        vi = empty!(TypedVarInfo(vi))
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
        @model priorsinarray(xs, ::Type{T}=Float64) where {T} = begin
            priors = Vector{T}(undef, 2)
            priors[1] ~ InverseGamma(2, 3)
            priors[2] ~ Normal(0, sqrt(priors[1]))
            for i = 1:length(xs)
                xs[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        # Sampling
        chain = sample(priorsinarray(xs), HMC(0.01, 10), 10)
    end
    @testset "varname" begin
        i, j, k = 1, 2, 3

        vn1 = @varname x[1]
        @test vn1 == VarName(:x, ((1,),))

        # Symbol
        v_sym = string(:x)
        @test v_sym == "x"

        # Array
        v_arr = @varname x[i]
        @test v_arr.indexing == ((1,),)

        # Matrix
        v_mat = @varname x[i,j]
        @test v_mat.indexing == ((1, 2),)

        v_mat = @varname x[i,j,k]
        @test v_mat.indexing == ((1,2,3),)

        v_mat = @varname x[1,2][1+5][45][3][i]
        @test v_mat.indexing == ((1,2), (6,), (45,), (3,), (1,))

        @model mat_name_test() = begin
            p = Array{Any}(undef, 2, 2)
            for i in 1:2, j in 1:2
                p[i,j] ~ Normal(0, 1)
            end
            p
        end
        chain = sample(mat_name_test(), HMC(0.2, 4), 1000)
        check_numerical(chain, ["p[1,1]"], [0], atol = 0.25)

        # Multi array
        v_arrarr = @varname x[i][j]
        @test v_arrarr.indexing == ((1,), (2,))

        @model marr_name_test() = begin
            p = Array{Array{Any}}(undef, 2)
            p[1] = Array{Any}(undef, 2)
            p[2] = Array{Any}(undef, 2)
            for i in 1:2, j in 1:2
                p[i][j] ~ Normal(0, 1)
            end
            p
        end

        chain = sample(marr_name_test(), HMC(0.2, 4), 1000)
        check_numerical(chain, ["p[1][1]"], [0], atol = 0.25)
    end
    @testset "varinfo" begin
        dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]
        function test_varinfo!(vi)
            @test getlogp(vi) === 0.0
            setlogp!(vi, 1)
            @test getlogp(vi) === 1.0
            acclogp!(vi, 1)
            @test getlogp(vi) === 2.0
            resetlogp!(vi)
            @test getlogp(vi) === 0.0

            spl2 = Sampler(PG(5, :w, :u), empty_model())
            vn_w = @varname w
            randr(vi, vn_w, dists[1], spl2, true)

            vn_x = @varname x
            vn_y = @varname y
            vn_z = @varname z
            vns = [vn_x, vn_y, vn_z]

            spl1 = Sampler(PG(5, :x, :y, :z), empty_model())
            for i = 1:3
                r = randr(vi, vns[i], dists[i], spl1, false)
                val = vi[vns[i]]
                @test sum(val - r) <= 1e-9
            end

            idcs = _getidcs(vi, spl1)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 3
            else
                @test length(idcs) == 3
            end
            @test length(vi[spl1]) == 7

            idcs = _getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 1
            else
                @test length(idcs) == 1
            end
            @test length(vi[spl2]) == 1

            vn_u = @varname u
            randr(vi, vn_u, dists[1], spl2, true)

            idcs = _getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 2
            else
                @test length(idcs) == 2
            end
            @test length(vi[spl2]) == 2
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!(TypedVarInfo(vi)))

        @model igtest() = begin
            x ~ InverseGamma(2,3)
            y ~ InverseGamma(2,3)
            z ~ InverseGamma(2,3)
            w ~ InverseGamma(2,3)
            u ~ InverseGamma(2,3)
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
        _, state = @inferred AbstractMCMC.step(Random.GLOBAL_RNG, g_demo_f, g)
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
        vi = empty!(TypedVarInfo(vi))
        @inferred g_demo_f(vi, SampleFromPrior())
        pg.state.vi = vi
        step!(Random.GLOBAL_RNG, g_demo_f, pg, 1)
        vi = pg.state.vi
        @inferred g_demo_f(vi, hmc)
        @test vi.metadata.x.gids[1] == Set([pg.selector])
        @test vi.metadata.y.gids[1] == Set([pg.selector])
        @test vi.metadata.z.gids[1] == Set([pg.selector])
        @test vi.metadata.w.gids[1] == Set([hmc.selector])
        @test vi.metadata.u.gids[1] == Set([hmc.selector]) =#
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
