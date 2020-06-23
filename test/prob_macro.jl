using .Turing, Distributions, Test, Random
using DynamicPPL: VarInfo

Random.seed!(129)

dir = splitdir(splitdir(pathof(DynamicPPL))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

Random.seed!(129)

@testset "prob_macro" begin
    @testset "scalar" begin
        @model demo(x) = begin
            m ~ Normal()
            x ~ Normal(m, 1)
        end

        mval = 3
        xval = 2
        iters = 1000

        logprior = logpdf(Normal(), mval)
        loglike = logpdf(Normal(mval, 1), xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"m = mval | x = xval, model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"m = mval | x = xval, model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        varinfo = VarInfo(demo(missing))
        @test logprob"x = xval, m = mval | model = demo, varinfo = varinfo" == logjoint

        chain = sample(demo(xval), IS(), iters; save_state = true)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())
        lps = logpdf.(Normal.(vec(chain["m"]), 1), xval)
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = demo" == lps
        varinfo = VarInfo(demo(xval))
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = demo, varinfo = varinfo" == lps
    end

    @testset "vector" begin
        n = 5
        @model demo(x, n = n, ::Type{T} = Float64) where {T} = begin
            m = Vector{T}(undef, n)
            @. m ~ Normal()
            @. x ~ Normal.(m, 1)
        end
        mval = rand(n)
        xval = rand(n)
        iters = 1000

        logprior = sum(logpdf.(Normal(), mval))
        like(m, x) = sum(logpdf.(Normal.(m, 1), x))
        loglike = like(mval, xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        # Currently, we cannot easily pre-allocate `VarInfo` for vector data

        chain = sample(demo(xval), HMC(0.5, 1), iters; save_state = true)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())

        names = namesingroup(chain, "m")
        lps = map(1:iters) do iter
            like([chain[iter, name, 1] for name in names], xval)
        end
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = demo" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = demo, varinfo = varinfo" == lps
    end
end
