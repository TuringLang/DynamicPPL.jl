# from `ad_utils.jl`
# TODO: not used anymore, remove?
# TODO: FDM is renamed to FiniteDifferences.jl
"""
    test_reverse_mode_ad(forward, f, ȳ, x...; rtol=1e-6, atol=1e-6)

Check that the reverse-mode sensitivities produced by an AD library are correct for `f`
at `x...`, given sensitivity `ȳ` w.r.t. `y = f(x...)` up to `rtol` and `atol`.
"""
function test_reverse_mode_ad(f, ȳ, x...; rtol=1e-6, atol=1e-6)
    # Perform a regular forwards-pass.
    y = f(x...)

    # Use Tracker to compute reverse-mode sensitivities.
    y_tracker, back_tracker = Tracker.forward(f, x...)
    x̄s_tracker = back_tracker(ȳ)

    # Use Zygote to compute reverse-mode sensitivities.
    y_zygote, back_zygote = Zygote.pullback(f, x...)
    x̄s_zygote = back_zygote(ȳ)

    test_rd = length(x) == 1 && y isa Number
    if test_rd
        # Use ReverseDiff to compute reverse-mode sensitivities.
        if x[1] isa Array
            x̄s_rd = similar(x[1])
            tp = ReverseDiff.GradientTape(x -> f(x), x[1])
            ReverseDiff.gradient!(x̄s_rd, tp, x[1])
            x̄s_rd .*= ȳ
            y_rd = ReverseDiff.value(tp.output)
            @assert y_rd isa Number
        else
            x̄s_rd = [x[1]]
            tp = ReverseDiff.GradientTape(x -> f(x[1]), [x[1]])
            ReverseDiff.gradient!(x̄s_rd, tp, [x[1]])
            y_rd = ReverseDiff.value(tp.output)[1]
            x̄s_rd = x̄s_rd[1] * ȳ
            @assert y_rd isa Number
        end
    end

    # Use finite differencing to compute reverse-mode sensitivities.
    x̄s_fdm = FDM.j′vp(central_fdm(5, 1), f, ȳ, x...)

    # Check that Tracker forwards-pass produces the correct answer.
    @test isapprox(y, Tracker.data(y_tracker), atol=atol, rtol=rtol)

    # Check that Zygpte forwards-pass produces the correct answer.
    @test isapprox(y, y_zygote, atol=atol, rtol=rtol)

    if test_rd
        # Check that ReverseDiff forwards-pass produces the correct answer.
        @test isapprox(y, y_rd, atol=atol, rtol=rtol)
    end

    # Check that Tracker reverse-mode sensitivities are correct.
    @test all(zip(x̄s_tracker, x̄s_fdm)) do (x̄_tracker, x̄_fdm)
        isapprox(Tracker.data(x̄_tracker), x̄_fdm; atol=atol, rtol=rtol)
    end

    # Check that Zygote reverse-mode sensitivities are correct.
    @test all(zip(x̄s_zygote, x̄s_fdm)) do (x̄_zygote, x̄_fdm)
        isapprox(x̄_zygote, x̄_fdm; atol=atol, rtol=rtol)
    end

    if test_rd
        # Check that ReverseDiff reverse-mode sensitivities are correct.
        @test isapprox(x̄s_rd, x̄s_zygote[1]; atol=atol, rtol=rtol)
    end
end

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo(model)

    # Collect symbols.
    vnms = Vector(undef, length(syms))
    vnvals = Vector{Float64}()
    for i in 1:length(syms)
        s = syms[i]
        vnms[i] = getfield(vi.metadata, s).vns[1]

        vals = getval(vi, vnms[i])
        for i in eachindex(vals)
            push!(vnvals, vals[i])
        end
    end

    # Compute primal.
    x = vec(vnvals)
    logp = f(x)

    # Call ForwardDiff's AD directly.
    grad_FWAD = sort(ForwardDiff.gradient(f, x))

    # Compare with `logdensity_and_gradient`.
    z = vi[SampleFromPrior()]
    for chunksize in (0, 1, 10), standardtag in (true, false, 0, 3)
        ℓ = LogDensityProblemsAD.ADgradient(
            Turing.AutoForwardDiff(; chunksize=chunksize, tag=standardtag),
            Turing.LogDensityFunction(vi, model, SampleFromPrior(), DynamicPPL.DefaultContext()),
        )
        l, ∇E = LogDensityProblems.logdensity_and_gradient(ℓ, z)

        # Compare result
        @test l ≈ logp
        @test sort(∇E) ≈ grad_FWAD atol = 1e-9
    end
end

@testset "ad.jl" begin
    @model function gdemo_d()
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        1.5 ~ Normal(m, sqrt(s))
        2.0 ~ Normal(m, sqrt(s))
        return s, m
    end
      
    gdemo_default = gdemo_d()

    @testset "adr" begin
        ad_test_f = gdemo_default
        vi = DynamicPPL.VarInfo(ad_test_f)
        ad_test_f(vi, SampleFromPrior())
        svn = vi.metadata.s.vns[1]
        mvn = vi.metadata.m.vns[1]
        _s = getval(vi, svn)[1]
        _m = getval(vi, mvn)[1]

        dist_s = InverseGamma(2, 3)

        # Hand-written logp
        function logp(x::Vector)
            s = x[2]
            # s = invlink(dist_s, s)
            m = x[1]
            lik_dist = Normal(m, sqrt(s))
            lp = logpdf(dist_s, s) + logpdf(Normal(0, sqrt(s)), m)
            lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
            lp
        end

        # Call ForwardDiff's AD
        g = x -> ForwardDiff.gradient(logp, x)
        # _s = link(dist_s, _s)
        _x = [_m, _s]
        grad_FWAD = sort(g(_x))

        ℓ = Turing.LogDensityFunction(vi, ad_test_f, SampleFromPrior(), DynamicPPL.DefaultContext())
        x = map(x -> Float64(x), vi[SampleFromPrior()])

        trackerℓ = LogDensityProblemsAD.ADgradient(Turing.AutoTracker(), ℓ)
        if isdefined(Base, :get_extension)
            @test trackerℓ isa Base.get_extension(LogDensityProblemsAD, :LogDensityProblemsADTrackerExt).TrackerGradientLogDensity
        else
            @test trackerℓ isa LogDensityProblemsAD.LogDensityProblemsADTrackerExt.TrackerGradientLogDensity
        end
        @test trackerℓ.ℓ === ℓ
        ∇E1 = LogDensityProblems.logdensity_and_gradient(trackerℓ, x)[2]
        @test sort(∇E1) ≈ grad_FWAD atol = 1e-9

        zygoteℓ = LogDensityProblemsAD.ADgradient(Turing.AutoZygote(), ℓ)
        if isdefined(Base, :get_extension)
            @test zygoteℓ isa Base.get_extension(LogDensityProblemsAD, :LogDensityProblemsADZygoteExt).ZygoteGradientLogDensity
        else
            @test zygoteℓ isa LogDensityProblemsAD.LogDensityProblemsADZygoteExt.ZygoteGradientLogDensity
        end
        @test zygoteℓ.ℓ === ℓ
        ∇E2 = LogDensityProblems.logdensity_and_gradient(zygoteℓ, x)[2]
        @test sort(∇E2) ≈ grad_FWAD atol = 1e-9
    end

    @testset "general AD tests" begin
        # Tests gdemo gradient.
        function logp1(x::Vector)
            dist_s = InverseGamma(2, 3)
            s = x[2]
            m = x[1]
            lik_dist = Normal(m, sqrt(s))
            lp = Turing.logpdf_with_trans(dist_s, s, false) + Turing.logpdf_with_trans(Normal(0, sqrt(s)), m, false)
            lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
            return lp
        end

        test_model_ad(gdemo_default, logp1, [:m, :s])

        # Test Wishart AD.
        @model function wishart_ad()
            v ~ Wishart(7, [1 0.5; 0.5 1])
            v
        end

        # Hand-written logp
        function logp3(x)
            dist_v = Wishart(7, [1 0.5; 0.5 1])
            v = [x[1] x[3]; x[2] x[4]]
            lp = Turing.logpdf_with_trans(dist_v, v, false)
            return lp
        end

        test_model_ad(wishart_ad(), logp3, [:v])
    end

    # @testset "Simplex Tracker, Zygote and ReverseDiff (with and without caching) AD" begin
    #     @model function dir()
    #         theta ~ Dirichlet(1 ./ fill(4, 4))
    #     end
    #     sample(dir(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
    #     sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
    #     sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(true)), 1000)
    # end

    # @testset "PDMatDistribution AD" begin
    #     @model function wishart()
    #         theta ~ Wishart(4, Matrix{Float64}(I, 4, 4))
    #     end

    #     sample(wishart(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
    #     sample(wishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)

    #     @model function invwishart()
    #         theta ~ InverseWishart(4, Matrix{Float64}(I, 4, 4))
    #     end

    #     sample(invwishart(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
    #     sample(invwishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
    # end

    @testset "Hessian test" begin
        @model function tst(x, ::Type{TV}=Vector{Float64}) where {TV}
            params = TV(undef, 2)
            @. params ~ Normal(0, 1)

            x ~ MvNormal(params, I)
        end

        function make_logjoint(model::DynamicPPL.Model, ctx::DynamicPPL.AbstractContext)
            # setup
            varinfo_init = Turing.VarInfo(model)
            spl = DynamicPPL.SampleFromPrior()
            varinfo_init = DynamicPPL.link!!(varinfo_init, spl, model)

            function logπ(z; unlinked=false)
                varinfo = DynamicPPL.unflatten(varinfo_init, spl, z)

                # TODO(torfjelde): Pretty sure this is a mistake.
                # Why are we not linking `varinfo` rather than `varinfo_init`?
                if unlinked
                    varinfo_init = DynamicPPL.invlink!!(varinfo_init, spl, model)
                end
                varinfo = last(DynamicPPL.evaluate!!(model, varinfo, DynamicPPL.SamplingContext(spl, ctx)))
                if unlinked
                    varinfo_init = DynamicPPL.link!!(varinfo_init, spl, model)
                end

                return -DynamicPPL.getlogp(varinfo)
            end

            return logπ
        end

        data = [0.5, -0.5]
        model = tst(data)

        likelihood = make_logjoint(model, DynamicPPL.LikelihoodContext())
        target(x) = likelihood(x, unlinked=true)

        H_f = ForwardDiff.hessian(target, zeros(2))
        H_r = ReverseDiff.hessian(target, zeros(2))
        @test H_f == [1.0 0.0; 0.0 1.0]
        @test H_f == H_r
    end

    @testset "memoization: issue #1393" begin

        @model function demo(data)
            sigma ~ Uniform(0.0, 20.0)
            data ~ Normal(0, sigma)
        end

        N = 1000
        for i in 1:5
            d = Normal(0.0, i)
            data = rand(d, N)
            chn = sample(demo(data), NUTS(0.65; adtype=AutoReverseDiff(true)), 1000)
            @test mean(Array(chn[:sigma])) ≈ std(data) atol = 0.5
        end

    end

    @testset "tag" begin
        for chunksize in (0, 1, 10)
            ad = Turing.AutoForwardDiff(; chunksize=chunksize)
            @test ad === Turing.AutoForwardDiff(; chunksize=chunksize)
            @test Turing.Essential.standardtag(ad)
            for standardtag in (false, 0, 1)
                @test !Turing.Essential.standardtag(Turing.AutoForwardDiff(; chunksize=chunksize, tag=standardtag))
            end
        end
    end

    @testset "ReverseDiff compiled without linking" begin
        f = DynamicPPL.LogDensityFunction(gdemo_default)
        θ = DynamicPPL.getparams(f)

        f_rd = LogDensityProblemsAD.ADgradient(Turing.AutoReverseDiff(; compile=false), f)
        f_rd_compiled = LogDensityProblemsAD.ADgradient(Turing.AutoReverseDiff(; compile=true), f)

        ℓ, ℓ_grad = LogDensityProblems.logdensity_and_gradient(f_rd, θ)
        ℓ_compiled, ℓ_grad_compiled = LogDensityProblems.logdensity_and_gradient(f_rd_compiled, θ)

        @test ℓ == ℓ_compiled
        @test ℓ_grad == ℓ_grad_compiled
    end
end
