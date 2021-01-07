@testset "prob_macro.jl" begin
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
