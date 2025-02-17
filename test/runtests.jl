using Distributions
using DynamicPPL
using Random
using Test

Random.seed!(100)

@testset verbose = true "submodel tests" begin
    @testset "sanity check with original models" begin
        @model f() = x ~ Normal()
        model = f()
        vi = VarInfo(model)
        # check parent varinfo
        @test Set(keys(vi)) == Set([@varname(x)])
        @test vi[@varname(x)] isa Float64
        # check logp
        @test DynamicPPL.getlogp(vi) ≈ logpdf(Normal(), vi[@varname(x)])
    end

    @testset "submodel - assume - no rhs" begin
        @model function g()
            a ~ Normal()
            return "foo"
        end
        @model function f()
            x ~ Normal()
            lhs ~ g()
            return (__varinfo__, lhs)
        end
        model = f()
        (vi, lhs) = model()
        # Check parent model varinfo
        @test Set(keys(vi)) == Set([@varname(x), @varname(lhs.a)])
        @test vi[@varname(x)] isa Float64
        @test vi[@varname(lhs.a)] isa Float64
        # check the lhs of submodel tilde
        @test lhs isa OrderedDict
        @test lhs[@varname(a)] isa Float64
        @test lhs[@varname(a)] == vi[@varname(lhs.a)]
        # check logp accumulated correctly
        @test DynamicPPL.getlogp(vi) ≈
            logpdf(Normal(), vi[@varname(x)]) + logpdf(Normal(), vi[@varname(lhs.a)])
    end

    @testset "submodel - assume - with rhs" begin
        @model function g()
            a ~ Normal()
            return "foo"
        end
        @model function f()
            x ~ Normal()
            lhs ~ g() --> rhs
            return (__varinfo__, lhs, rhs)
        end
        model = f()
        (vi, lhs, rhs) = model()
        # Check parent model varinfo
        @test Set(keys(vi)) == Set([@varname(x), @varname(lhs.a)])
        @test vi[@varname(x)] isa Float64
        @test vi[@varname(lhs.a)] isa Float64
        # check the lhs of submodel tilde
        @test lhs isa OrderedDict
        @test lhs[@varname(a)] isa Float64
        @test lhs[@varname(a)] == vi[@varname(lhs.a)]
        # check the rhs
        @test rhs == "foo"
        # check logp accumulated correctly
        @test DynamicPPL.getlogp(vi) ≈
            logpdf(Normal(), vi[@varname(x)]) + logpdf(Normal(), vi[@varname(lhs.a)])
    end

    @testset "submodel - assume - nested with rhs" begin
        # OK, this is getting a bit confusing, so I added some annotations.
        @model function h()
            q ~ Normal()
            return "bar"
        end
        @model function g()
            p ~ Normal()
            a ~ h() --> b
            # Here, `a` should be an OrderedDict with a single key, `q`
            # `b` should be "bar"
            return ("foo", a, b)
        end
        @model function f()
            x ~ Normal()
            lhs ~ g() --> rhs
            # Here, `lhs` should be an OrderedDict with two keys, `p` and `a`
            # lhs[`p`] should be a Float64, and lhs[`a`] should itself be an
            # OrderedDict with a single key `q`.
            # `rhs` should be the return value of g, i.e. a 3-tuple
            # ("foo", OrderedDict(`q` -> Float64), "bar")
            return (__varinfo__, lhs, rhs)
        end

        model = f()
        (vi, lhs, rhs) = model()
        # Check parent model varinfo
        @test Set(keys(vi)) == Set([@varname(x), @varname(lhs.p), @varname(lhs.a.q)])
        @test vi[@varname(x)] isa Float64
        @test vi[@varname(lhs.p)] isa Float64
        @test vi[@varname(lhs.a.q)] isa Float64
        # check the lhs of submodel tilde
        @test lhs isa OrderedDict
        @test lhs[@varname(p)] isa Float64
        @test lhs[@varname(p)] == vi[@varname(lhs.p)]
        @test_throws KeyError lhs[@varname(a)][@varname(q)] isa Float64
        @test_throws KeyError lhs[@varname(a)][@varname(q)] == vi[@varname(lhs.a.q)]
        # check the rhs of submodel tilde
        (foo, a, bar) = rhs
        @test foo == "foo"
        @test a isa OrderedDict
        @test_throws KeyError a[@varname(q)] isa Float64
        @test_throws KeyError a[@varname(q)] == vi[@varname(lhs.a.q)]
        @test bar == "bar"
        # check logp accumulated correctly
        @test DynamicPPL.getlogp(vi) ≈
            logpdf(Normal(), vi[@varname(x)]) +
              logpdf(Normal(), vi[@varname(lhs.p)]) +
              logpdf(Normal(), vi[@varname(lhs.a.q)])
    end
end
