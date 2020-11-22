using .Turing, Distributions, Test, Random
using DynamicPPL: VarInfo

Random.seed!(129)

dir = splitdir(splitdir(pathof(DynamicPPL))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

Random.seed!(129)

@testset "prob_macro" begin
    @testset "scalar" begin
        @model function demo(x)
            m ~ Normal()
            x ~ Normal(m, 1)
        end

        mval = 3
        xval = 2
        iters = 1000

        logprior = logpdf(Normal(), mval)
        loglike = logpdf(Normal(mval, 1), xval)
        logjoint = logprior + loglike

        model = demo(xval)
        @test logprob"m = mval | model = model" == logprior
        @test logprob"m = mval | x = xval, model = model" == logprior
        @test logprob"x = xval | m = mval, model = model" == loglike
        @test logprob"x = xval, m = mval | model = model" == logjoint

        varinfo = VarInfo(demo(missing))
        @test logprob"x = xval, m = mval | model = model, varinfo = varinfo" == logjoint

        varinfo = VarInfo(demo(xval))
        @test logprob"m = mval | model = model, varinfo = varinfo" == logprior
        @test logprob"m = mval | x = xval, model = model, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = model, varinfo = varinfo" == loglike

        chain = sample(demo(xval), IS(), iters; save_state = true)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())
        lps = logpdf.(Normal.(chain["m"], 1), xval)
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = model" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = model, varinfo = varinfo" == lps

        # multiple chains
        pchain = chainscat(chain, chain)
        pchain2 = chainscat(chain2, chain2)
        plps = repeat(lps, 1, 2)
        @test logprob"x = xval | chain = pchain" == plps
        @test logprob"x = xval | chain = pchain2, model = model" == plps
        @test logprob"x = xval | chain = pchain, varinfo = varinfo" == plps
        @test logprob"x = xval | chain = pchain2, model = model, varinfo = varinfo" == plps
    end

    @testset "vector" begin
        n = 5
        @model function demo(x, n = n)
            m ~ MvNormal(n, 1.0)
            x ~ MvNormal(m, 1.0)
        end
        mval = rand(n)
        xval = rand(n)
        iters = 1000

        logprior = logpdf(MvNormal(n, 1.0), mval)
        loglike = logpdf(MvNormal(mval, 1.0), xval)
        logjoint = logprior + loglike

        model = demo(xval)
        @test logprob"m = mval | model = model" == logprior
        @test logprob"x = xval | m = mval, model = model" == loglike
        @test logprob"x = xval, m = mval | model = model" == logjoint

        varinfo = VarInfo(demo(xval))
        @test logprob"m = mval | model = model, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = model, varinfo = varinfo" == loglike
        # Currently, we cannot easily pre-allocate `VarInfo` for vector data

        chain = sample(demo(xval), HMC(0.5, 1), iters; save_state = true)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())

        names = namesingroup(chain, "m")
        lps = [
            logpdf(MvNormal(chain.value[i, names, j], 1.0), xval)
            for i in 1:size(chain, 1), j in 1:size(chain, 3)
        ]
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = model" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = model, varinfo = varinfo" == lps

        # multiple chains
        pchain = chainscat(chain, chain)
        pchain2 = chainscat(chain2, chain2)
        plps = repeat(lps, 1, 2)
        @test logprob"x = xval | chain = pchain" == plps
        @test logprob"x = xval | chain = pchain2, model = model" == plps
        @test logprob"x = xval | chain = pchain, varinfo = varinfo" == plps
        @test logprob"x = xval | chain = pchain2, model = model, varinfo = varinfo" == plps
    end

    @testset "issue#137" begin
        @model function model1(y, group, n_groups)
            σ ~ truncated(Cauchy(0, 1), 0, Inf)
            α ~ filldist(Normal(0, 10), n_groups)
            μ = α[group]
            y ~ MvNormal(μ, σ)
        end

        y = randn(100)
        group = rand(1:4, 100)
        n_groups = 4

        chain1 = sample(model1(y, group, n_groups), NUTS(0.65), 2_000; save_state=true)
        logprob"y = y[[1]] | group = group[[1]], n_groups = n_groups, chain = chain1"

        @model function model2(y, group, n_groups)
            σ ~ truncated(Cauchy(0, 1), 0, Inf)
            α ~ filldist(Normal(0, 10), n_groups)
            for i in 1:length(y)
                y[i] ~ Normal(α[group[i]], σ)
            end
        end

        chain2 = sample(model2(y, group, n_groups), NUTS(0.65), 2_000; save_state=true)
        logprob"y = y[[1]] | group = group[[1]], n_groups = n_groups, chain = chain2"
    end

    @testset "issue190" begin
        @model function gdemo(x, y)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            x ~ filldist(Normal(m, sqrt(s)), length(y))
            for i in 1:length(y)
                y[i] ~ Normal(x[i], sqrt(s))
            end
        end
        c = Chains(rand(10, 2), [:m, :s])
        model_gdemo = gdemo([1.0, 0.0], [1.5, 0.0])
        r1 = prob"y = [1.5] | chain=c, model = model_gdemo, x = [1.0]"
        r2 = map(c[:s]) do s
            # exp(logpdf(..)) not pdf because this is exactly what the prob"" macro does, so we test r1 == r2
            exp(logpdf(Normal(1, sqrt(s)), 1.5))
        end
        @test r1 == r2
    end
end
