macro custom(expr)
    (Meta.isexpr(expr, :call, 3) && expr.args[1] === :~) || error("incorrect macro usage")
    quote
        $(esc(expr.args[2])) = 0.0
    end
end

macro mymodel1(ex)
    # check if expression was modified by the DynamicPPL "compiler"
    if ex == :(y ~ Uniform())
        return esc(:(x ~ Normal()))
    else
        return esc(:(z ~ Exponential()))
    end
end

struct MyModelStruct{T}
    x::T
end
Base.:~(x, y::MyModelStruct) = y.x
macro mymodel2(ex)
    # check if expression was modified by the DynamicPPL "compiler"
    if ex == :(y ~ Uniform())
        # Just returns 42
        return :(4 ~ MyModelStruct(42))
    else
        return :(return -1)
    end
end

@testset "compiler.jl" begin
    @testset "model macro" begin
        @model function testmodel_comp(x, y)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))

            return x, y
        end
        testmodel_comp(1.0, 1.2)

        # check if drawing from the prior works
        @model function testmodel01(x=missing)
            x ~ Normal()
            return x
        end
        f0_mm = testmodel01()
        @test mean(f0_mm() for _ in 1:1000) ≈ 0.0 atol = 0.1

        # Test #544
        @model function testmodel02(x=missing)
            if x === missing
                x = Vector{Float64}(undef, 2)
            end
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end
        f0_mm = testmodel02()
        @test all(x -> isapprox(x, 0; atol=0.1), mean(f0_mm() for _ in 1:1000))

        @model function testmodel03(x=missing)
            x ~ Bernoulli(0.5)
            return x
        end
        f01_mm = testmodel03()
        @test mean(f01_mm() for _ in 1:1000) ≈ 0.5 atol = 0.1

        # test if we get the correct return values
        @model function testmodel1(x1, x2)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        f1_mm = testmodel1(1.0, 10.0)
        @test f1_mm() == (1, 10)

        # alternatives with keyword arguments
        testmodel1kw(; x1, x2) = testmodel1(x1, x2)
        f1_mm = testmodel1kw(; x1=1.0, x2=10.0)
        @test f1_mm() == (1, 10)

        @model function testmodel2(; x1, x2)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        f1_mm = testmodel2(; x1=1.0, x2=10.0)
        @test f1_mm() == (1, 10)

        @info "Testing the compiler's ability to catch bad models..."

        # Test for assertions in observe statements.
        @model function brokentestmodel_observe1(x1, x2)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_observe1(1.0, 2.0)
        @test_throws ArgumentError btest()

        @model function brokentestmodel_observe2(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x = Vector{Float64}(undef, 2)
            x ~ [Normal(m, sqrt(s)), 2.0]

            return x
        end

        btest = brokentestmodel_observe2([1.0, 2.0])
        @test_throws ArgumentError btest()

        # Test for assertions in assume statements.
        @model function brokentestmodel_assume1()
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_assume1()
        @test_throws ArgumentError btest()

        @model function brokentestmodel_assume2()
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

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
            global varinfo_ = __varinfo__
            global sampler_ = __context__.sampler
            global model_ = __model__
            global context_ = __context__
            global rng_ = __context__.rng
            global lp = getlogp(__varinfo__)
            return x
        end
        model = testmodel_missing3([1.0])
        varinfo = VarInfo(model)
        @test getlogp(varinfo) == lp
        @test varinfo_ isa AbstractVarInfo
        @test model_ === model
        @test context_ isa SamplingContext
        @test rng_ isa Random.AbstractRNG

        # disable warnings
        @model function testmodel_missing4(x)
            x[1] ~ Bernoulli(0.5)
            global varinfo_ = __varinfo__
            global sampler_ = __context__.sampler
            global model_ = __model__
            global context_ = __context__
            global rng_ = __context__.rng
            global lp = getlogp(__varinfo__)
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
        @test all(z -> isapprox(z, 0; atol=0.2), mean(model() for _ in 1:1000))

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
                x[1] ~ Bernoulli(p)
                global lp = getlogp(__varinfo__)
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
        @model f2() = x ~ NamedDist(Normal(), @varname(y[2][:, 1]))
        @model f3() = x ~ NamedDist(Normal(), @varname(y[1]))
        vi1 = VarInfo(f1())
        vi2 = VarInfo(f2())
        vi3 = VarInfo(f3())
        @test haskey(vi1.metadata, :y)
        @test vi1.metadata.y.vns[1] == @varname(y)
        @test haskey(vi2.metadata, :y)
        @test vi2.metadata.y.vns[1] == @varname(y[2][:, 1])
        @test haskey(vi3.metadata, :y)
        @test vi3.metadata.y.vns[1] == @varname(y[1])
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
            return x ~ Normal(m, 1)
        end

        s = @doc(demo)
        @test string(s) == "This is a test\n"

        # Verify that adding docstring didn't completely break execution of model
        m = demo(0.0)
        @test m() isa Float64
    end
    @testset "type annotations" begin
        @model function demo_without(x)
            return x ~ Normal()
        end
        @test isempty(VarInfo(demo_without(0.0)))

        @model function demo_with(x::Real)
            return x ~ Normal()
        end
        @test isempty(VarInfo(demo_with(0.0)))
    end

    @testset "macros within model" begin
        # Macro expansion
        @model function demo1()
            @mymodel1(y ~ Uniform())
        end

        @test haskey(VarInfo(demo1()), @varname(x))

        # Interpolation
        # Will fail if:
        # 1. Compiler expands `y ~ Uniform()` before expanding the macros
        #    => returns -1.
        # 2. `@mymodel` is expanded before entire `@model` has been
        #    expanded => errors since `MyModelStruct` is not a distribution,
        #    and hence `tilde_observe` errors.
        @model function demo2()
            return $(@mymodel2(y ~ Uniform()))
        end
        @test demo2()() == 42
    end

    @testset "submodel" begin
        # No prefix, 1 level.
        @model function demo1(x)
            return x ~ Normal()
        end
        @model function demo2(x, y)
            @submodel demo1(x)
            return y ~ Uniform()
        end
        # No observation.
        m = demo2(missing, missing)
        vi = VarInfo(m)
        ks = keys(vi)
        @test @varname(x) ∈ ks
        @test @varname(y) ∈ ks

        # Observation in top-level.
        m = demo2(missing, 1.0)
        vi = VarInfo(m)
        ks = keys(vi)
        @test @varname(x) ∈ ks
        @test @varname(y) ∉ ks

        # Observation in nested model.
        m = demo2(1000.0, missing)
        vi = VarInfo(m)
        ks = keys(vi)
        @test @varname(x) ∉ ks
        @test @varname(y) ∈ ks

        # Observe all.
        m = demo2(1000.0, 0.5)
        vi = VarInfo(m)
        ks = keys(vi)
        @test isempty(ks)

        # Check values makes sense.
        @model function demo3(x, y)
            @submodel demo1(x)
            return y ~ Normal(x)
        end
        m = demo3(1000.0, missing)
        # Mean of `y` should be close to 1000.
        @test abs(mean([VarInfo(m)[@varname(y)] for i in 1:10]) - 1000) ≤ 10

        # Prefixed submodels and usage of submodel return values.
        @model function demo_return(x)
            x ~ Normal()
            return x
        end

        @model function demo_useval(x, y)
            x1 = @submodel sub1 demo_return(x)
            x2 = @submodel sub2 demo_return(y)

            return z ~ Normal(x1 + x2 + 100, 1.0)
        end
        m = demo_useval(missing, missing)
        vi = VarInfo(m)
        ks = keys(vi)
        @test VarName{Symbol("sub1.x")}() ∈ ks
        @test VarName{Symbol("sub2.x")}() ∈ ks
        @test @varname(z) ∈ ks
        @test abs(mean([VarInfo(m)[@varname(z)] for i in 1:10]) - 100) ≤ 10

        # AR1 model. Dynamic prefixing.
        @model function AR1(num_steps, α, μ, σ, ::Type{TV}=Vector{Float64}) where {TV}
            η ~ MvNormal(num_steps, 1.0)
            δ = sqrt(1 - α^2)

            x = TV(undef, num_steps)
            x[1] = η[1]
            @inbounds for t in 2:num_steps
                x[t] = @. α * x[t - 1] + δ * η[t]
            end

            return @. μ + σ * x
        end

        @model function demo(y)
            α ~ Uniform()
            μ ~ Normal()
            σ ~ truncated(Normal(), 0, Inf)

            num_steps = length(y[1])
            num_obs = length(y)
            @inbounds for i in 1:num_obs
                x = @submodel $(Symbol("ar1_$i")) AR1(num_steps, α, μ, σ)
                y[i] ~ MvNormal(x, 0.1)
            end
        end

        ys = [randn(10), randn(10)]
        m = demo(ys)
        vi = VarInfo(m)

        for k in [:α, :μ, :σ, Symbol("ar1_1.η"), Symbol("ar1_2.η")]
            @test VarName{k}() ∈ keys(vi)
        end
    end

    @testset "check_tilde_rhs" begin
        @test_throws ArgumentError DynamicPPL.check_tilde_rhs(randn())

        x = Normal()
        @test DynamicPPL.check_tilde_rhs(x) === x

        x = [Laplace(), Normal(), MvNormal(3, 1.0)]
        @test DynamicPPL.check_tilde_rhs(x) === x
    end
    @testset "isliteral" begin
        @test DynamicPPL.isliteral(:([1.0]))
        @test DynamicPPL.isliteral(:([[1.0], 1.0]))
        @test DynamicPPL.isliteral(:((1.0, 1.0)))

        @test !(DynamicPPL.isliteral(:([x])))
        @test !(DynamicPPL.isliteral(:([[x], 1.0])))
        @test !(DynamicPPL.isliteral(:((x, 1.0))))
    end

    @testset "array literals" begin
        # Verify that we indeed can parse this.
        @test @model(function array_literal_model()
            # `assume` and literal `observe`
            m ~ MvNormal(2, 1.0)
            return [10.0, 10.0] ~ MvNormal(m, 0.5 * ones(2))
        end) isa Function

        @model function array_literal_model2()
            # `assume` and literal `observe`
            m ~ MvNormal(2, 1.0)
            return [10.0, 10.0] ~ MvNormal(m, 0.5 * ones(2))
        end

        @test array_literal_model2()() == [10.0, 10.0]
    end

    # https://github.com/TuringLang/DynamicPPL.jl/issues/260
    @testset "anonymous function" begin
        error = ArgumentError("anonymous functions without name are not supported")
        @test_throws LoadError(@__FILE__, (@__LINE__) + 1, error) @macroexpand begin
            @model function (x)
                return x ~ Normal()
            end
        end
        @test_throws LoadError(@__FILE__, (@__LINE__) + 1, error) @macroexpand begin
            model = @model(x -> (x ~ Normal()))
        end
    end
end
