macro custom(expr)
    (Meta.isexpr(expr, :call, 3) && expr.args[1] === :~) ||
        error("incorrect macro usage")
    quote
        $(esc(expr.args[2])) = 0.0
    end
end

@testset "compiler.jl" begin
    @testset "model macro" begin
        @model function testmodel_comp(x, y)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))

            return x, y
        end
        testmodel_comp(1.0, 1.2)

        # check if drawing from the prior works
        @model function testmodel01(x = missing)
            x ~ Normal()
            return x
        end
        f0_mm = testmodel01()
        @test mean(f0_mm() for _ in 1:1000) ≈ 0. atol=0.1

        # Test #544
        @model function testmodel02(x = missing)
            if x === missing
                x = Vector{Float64}(undef, 2)
            end
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end
        f0_mm = testmodel02()
        @test all(x -> isapprox(x, 0; atol = 0.1), mean(f0_mm() for _ in 1:1000))

        @model function testmodel03(x = missing)
            x ~ Bernoulli(0.5)
            return x
        end
        f01_mm = testmodel03()
        @test mean(f01_mm() for _ in 1:1000) ≈ 0.5 atol=0.1

        # test if we get the correct return values
        @model function testmodel1(x1, x2)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        f1_mm = testmodel1(1., 10.)
        @test f1_mm() == (1, 10)

        # alternatives with keyword arguments
        testmodel1kw(; x1, x2) = testmodel1(x1, x2)
        f1_mm = testmodel1kw(x1 = 1., x2 = 10.)
        @test f1_mm() == (1, 10)

        @model function testmodel2(; x1, x2)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        f1_mm = testmodel2(x1=1., x2=10.)
        @test f1_mm() == (1, 10)

        @info "Testing the compiler's ability to catch bad models..."

        # Test for assertions in observe statements.
        @model function brokentestmodel_observe1(x1, x2)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_observe1(1., 2.)
        @test_throws ArgumentError btest()

        @model function brokentestmodel_observe2(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x = Vector{Float64}(undef, 2)
            x ~ [Normal(m, sqrt(s)), 2.0]

            return x
        end

        btest = brokentestmodel_observe2([1., 2.])
        @test_throws ArgumentError btest()

        # Test for assertions in assume statements.
        @model function brokentestmodel_assume1()
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_assume1()
        @test_throws ArgumentError btest()

        @model function brokentestmodel_assume2()
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x = Vector{Float64}(undef, 2)
            x ~ [Normal(m, sqrt(s)), 2.0]

            return x
        end

        btest = brokentestmodel_assume2()
        @test_throws ArgumentError btest()

        # Test missing input arguments
        @model function testmodel_missing1(x)
            x ~ Bernoulli(0.5)
            return x
        end
        @test_throws MethodError testmodel_missing1()

        # Test missing initialization for vector observation turned parameter
        @model function testmodel_missing2(x)
            x[1] ~ Bernoulli(0.5)
            return x
        end
        @test_throws MethodError testmodel_missing2(missing)()

        # Test use of internal names
        @model function testmodel_missing3(x)
            x[1] ~ Bernoulli(0.5)
            global varinfo_ = _varinfo
            global sampler_ = _sampler
            global model_ = _model
            global context_ = _context
            global rng_ = _rng
            global lp = getlogp(_varinfo)
            return x
        end
        model = testmodel_missing3([1.0])
        varinfo = VarInfo(model)
        @test getlogp(varinfo) == lp
        @test varinfo_ isa AbstractVarInfo
        @test model_ === model
        @test sampler_ === SampleFromPrior()
        @test context_ === DefaultContext()
        @test rng_ isa Random.AbstractRNG

        # disable warnings
        @model function testmodel_missing4(x)
            x[1] ~ Bernoulli(0.5)
            global varinfo_ = _varinfo
            global sampler_ = _sampler
            global model_ = _model
            global context_ = _context
            global rng_ = _rng
            global lp = getlogp(_varinfo)
            return x
        end false
        lpold = lp
        model = testmodel_missing4([1.0])
        varinfo = VarInfo(model)
        @test getlogp(varinfo) == lp == lpold

        # test DPPL#61
        @model function testmodel_missing5(z)
            m ~ Normal()
            z[1:end] ~ MvNormal(fill(m, length(z)), 1.0)
            return m
        end
        model = testmodel_missing5(rand(10))
        @test all(z -> isapprox(z, 0; atol = 0.2), mean(model() for _ in 1:1000))

        # test Turing#1464
        @model function gdemo(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            for i in eachindex(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end
        x = [1.0, missing]
        VarInfo(gdemo(x))
        @test ismissing(x[2])

        # https://github.com/TuringLang/Turing.jl/issues/1464#issuecomment-731153615
        vi = VarInfo(gdemo(x))
        @test haskey(vi.metadata, :x)
        vi = VarInfo(gdemo(x))
        @test haskey(vi.metadata, :x)
    end
    @testset "nested model" begin
        function makemodel(p)
            @model function testmodel(x)
                x[1] ~ Bernoulli(p)
                global lp = getlogp(_varinfo)
                return x
            end
            return testmodel
        end
        model = makemodel(0.5)([1.0])
        varinfo = VarInfo(model)
        @test getlogp(varinfo) == lp
    end
    @testset "user-defined variable name" begin
        @model f1() = x ~ NamedDist(Normal(), :y)
        @model f2() = x ~ NamedDist(Normal(), @varname(y[2][:,1]))
        @model f3() = x ~ NamedDist(Normal(), @varname(y[1]))
        vi1 = VarInfo(f1())
        vi2 = VarInfo(f2())
        vi3 = VarInfo(f3())
        @test haskey(vi1.metadata, :y)
        @test vi1.metadata.y.vns[1] == VarName(:y)
        @test haskey(vi2.metadata, :y)
        @test vi2.metadata.y.vns[1] == VarName(:y, ((2,), (Colon(), 1)))
        @test haskey(vi3.metadata, :y)
        @test vi3.metadata.y.vns[1] == VarName(:y, ((1,),))
    end
    @testset "custom tilde" begin
        @model demo() = begin
            $(@custom m ~ Normal())
            return m
        end
        model = demo()
        @test all(iszero(model()) for _ in 1:1000)
    end
    @testset "docstring" begin
        "This is a test"
        @model function demo(x)
            m ~ Normal()
            x ~ Normal(m, 1)
        end

        s = @doc(demo)
        @test string(s) == "This is a test\n"

        # Verify that adding docstring didn't completely break execution of model
        m = demo(0.)
        @test m() isa Float64
    end
    @testset "type annotations" begin
        @model function demo_without(x)
            x ~ Normal()
        end
        @test isempty(VarInfo(demo_without(0.0)))

        @model function demo_with(x::Real)
            x ~ Normal()
        end
        @test isempty(VarInfo(demo_with(0.0)))
    end

    @testset "macros within model" begin
        # Macro expansion
        macro mymodel(ex)
            # check if expression was modified by the DynamicPPL "compiler"
            if ex == :(y ~ Uniform())
	            return esc(:(x ~ Normal()))
	        else
	            return esc(:(z ~ Exponential()))
	        end
        end

        @model function demo()
            @mymodel(y ~ Uniform())
        end

        @test haskey(VarInfo(demo()), @varname(x))

        # Interpolation
        macro mymodel()
            return esc(:(return 42))
        end

        @model function demo()
            x ~ Normal()
            $(@mymodel())
        end

        @test demo()() == 42
    end
end
