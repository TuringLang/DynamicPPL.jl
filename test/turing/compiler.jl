@testset "compiler.jl" begin
    @testset "assume" begin
        @model function test_assume()
            x ~ Bernoulli(1)
            y ~ Bernoulli(x / 2)
            x, y
        end

        smc = SMC()
        pg = PG(10)

        res1 = sample(test_assume(), smc, 1000)
        res2 = sample(test_assume(), pg, 1000)

        check_numerical(res1, [:y], [0.5], atol=0.1)
        check_numerical(res2, [:y], [0.5], atol=0.1)

        # Check that all xs are 1.
        @test all(isone, res1[:x])
        @test all(isone, res2[:x])
    end
    @testset "beta binomial" begin
        prior = Beta(2,2)
        obs = [0,1,0,1,1,1,1,1,1,1]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
        meanp = exact.α / (exact.α + exact.β)

        @model function testbb(obs)
            p ~ Beta(2,2)
            x ~ Bernoulli(p)
            for i in 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            p, x
        end

        smc = SMC()
        pg = PG(10)
        gibbs = Gibbs(HMC(0.2, 3, :p), PG(10, :x))

        chn_s = sample(testbb(obs), smc, 1000)
        chn_p = sample(testbb(obs), pg, 2000)
        chn_g = sample(testbb(obs), gibbs, 1500)

        check_numerical(chn_s, [:p], [meanp], atol=0.05)
        check_numerical(chn_p, [:x], [meanp], atol=0.1)
        check_numerical(chn_g, [:x], [meanp], atol=0.1)
    end
    @testset "forbid global" begin
        xs = [1.5 2.0]
        # xx = 1

        @model function fggibbstest(xs)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))
            # xx ~ Normal(m, sqrt(s)) # this is illegal

            for i = 1:length(xs)
                xs[i] ~ Normal(m, sqrt(s))
                # for xx in xs
                # xx ~ Normal(m, sqrt(s))
            end
            s, m
        end

        gibbs = Gibbs(PG(10, :s), HMC(0.4, 8, :m))
        chain = sample(fggibbstest(xs), gibbs, 2)
    end
    @testset "new grammar" begin
        x = Float64[1 2]

        @model function gauss(x)
            priors = TArray{Float64}(2)
            priors[1] ~ InverseGamma(2,3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        chain = sample(gauss(x), PG(10), 10)
        chain = sample(gauss(x), SMC(), 10)

        @model function gauss2(::Type{TV} = Vector{Float64}; x) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2,3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        chain = sample(gauss2(x = x), PG(10), 10)
        chain = sample(gauss2(x = x), SMC(), 10)

        chain = sample(gauss2(Vector{Float64}; x = x), PG(10), 10)
        chain = sample(gauss2(Vector{Float64}; x = x), SMC(), 10)
    end
    @testset "new interface" begin
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        @model function newinterface(obs)
            p ~ Beta(2,2)
            for i = 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            p
        end

        chain = sample(
            newinterface(obs),
            HMC{Turing.ForwardDiffAD{2}}(0.75, 3, :p, :x),
            100,
        )
    end
    @testset "no return" begin
        @model function noreturn(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end

        chain = sample(noreturn([1.5 2.0]), HMC(0.15, 6), 1000)
        check_numerical(chain, [:s, :m], [49/24, 7/6])
    end
    @testset "observe" begin
        @model function test()
          z ~ Normal(0,1)
          x ~ Bernoulli(1)
          1 ~ Bernoulli(x / 2)
          0 ~ Bernoulli(x / 2)
          x
        end

        is  = IS()
        smc = SMC()
        pg  = PG(10)

        res_is = sample(test(), is, 10000)
        res_smc = sample(test(), smc, 1000)
        res_pg = sample(test(), pg, 100)

        @test all(isone, res_is[:x])
        @test res_is.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_smc[:x])
        @test res_smc.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_pg[:x])
    end
    @testset "sample" begin
        alg = Gibbs(HMC(0.2, 3, :m), PG(10, :s))
        chn = sample(gdemo_default, alg, 1000);
    end
    @testset "vectorization @." begin
        @model function vdemo1(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        @model function vdemo1b(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, $(sqrt(s)))
            return s, m
        end

        res = sample(vdemo1b(x), alg, 250)

        D = 2
        @model function vdemo2(x)
            μ ~ MvNormal(zeros(D), ones(D))
            @. x ~ $(MvNormal(μ, ones(D)))
        end

        alg = HMC(0.01, 5)
        res = sample(vdemo2(randn(D, 100)), alg, 250)

        # Vector assumptions
        N = 10
        setchunksize(N)
        alg = HMC(0.2, 4)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i = 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
          x = Vector{Real}(undef, N)
          @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 2 * ones(N))

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        N = 3
        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            @. x ~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end
    @testset "vectorization .~" begin
        @model function vdemo1(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            x .~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        D = 2
        @model function vdemo2(x)
            μ ~ MvNormal(zeros(D), ones(D))
            x .~ MvNormal(μ, ones(D))
        end

        alg = HMC(0.01, 5)
        res = sample(vdemo2(randn(D,100)), alg, 250)

        # Vector assumptions
        N = 10
        setchunksize(N)
        alg = HMC(0.2, 4)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i = 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            x .~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 2 * ones(N))

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            x .~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            x .~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end
    @testset "Type parameters" begin
        N = 10
        setchunksize(N)
        alg = HMC(0.01, 5)
        x = randn(1000)
        @model function vdemo1(::Type{T}=Float64) where {T}
            x = Vector{T}(undef, N)
            for i = 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo1(), alg, 250)
        t_loop = @elapsed res = sample(vdemo1(Float64), alg, 250)

        vdemo1kw(; T) = vdemo1(T)
        t_loop = @elapsed res = sample(vdemo1kw(T = Float64), alg, 250)

        @model function vdemo2(::Type{T}=Float64) where {T <: Real}
            x = Vector{T}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo2(), alg, 250)
        t_vec = @elapsed res = sample(vdemo2(Float64), alg, 250)

        vdemo2kw(; T) = vdemo2(T)
        t_vec = @elapsed res = sample(vdemo2kw(T = Float64), alg, 250)

        @model function vdemo3(::Type{TV}=Vector{Float64}) where {TV <: AbstractVector}
            x = TV(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo3(), alg, 250)
        sample(vdemo3(Vector{Float64}), alg, 250)

        vdemo3kw(; T) = vdemo3(T)
        sample(vdemo3kw(T = Vector{Float64}), alg, 250)
    end
end