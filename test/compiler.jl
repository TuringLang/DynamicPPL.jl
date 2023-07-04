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

# Used to test sampling of immutable types.
struct MyCoolStruct{T}
    a::T
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
        @test length(methods(testmodel_comp)) == 2
        testmodel_comp(1.0, 1.2)

        # check if drawing from the prior works
        @model function testmodel01(x=missing)
            x ~ Normal()
            return x
        end
        @test length(methods(testmodel01)) == 4
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
        @test length(methods(testmodel02)) == 4
        f0_mm = testmodel02()
        @test all(x -> isapprox(x, 0; atol=0.1), mean(f0_mm() for _ in 1:1000))

        @model function testmodel03(x=missing)
            x ~ Bernoulli(0.5)
            return x
        end
        f01_mm = testmodel03()
        @test length(methods(testmodel03)) == 4
        @test mean(f01_mm() for _ in 1:1000) ≈ 0.5 atol = 0.1

        # test if we get the correct return values
        @model function testmodel1(x1, x2)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        @test length(methods(testmodel1)) == 2
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
        @test length(methods(testmodel2)) == 2
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
            z[1:end] ~ MvNormal(fill(m, length(z)), I)
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

        # Non-array variables
        @model function testmodel_nonarray(x, y)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, √s)
            for i in 1:(length(x.a) - 1)
                x.a[i] ~ Normal(m, √s)
            end

            # Dynamic indexing
            x.a[end] ~ Normal(100.0, 1.0)

            # Immutable set
            y.a ~ Normal()

            # Dotted
            z = Vector{Float64}(undef, 3)
            z[1:2] .~ Normal()
            z[end:end] .~ Normal()

            return (; s=s, m=m, x=x, y=y, z=z)
        end

        m_nonarray = testmodel_nonarray(
            MyCoolStruct([missing, missing]), MyCoolStruct(missing)
        )
        result = m_nonarray()
        @test !any(ismissing, result.x.a)
        @test result.y.a !== missing
        @test result.x.a[end] > 10

        # Ensure that we can work with `Vector{Real}(undef, N)` which is the
        # reason why we're using `BangBang.prefermutation` in `src/compiler.jl`
        # rather than the default from Setfield.jl.
        # Related: https://github.com/jw3126/Setfield.jl/issues/157
        @model function vdemo()
            x = Vector{Real}(undef, 10)
            for i in eachindex(x)
                x[i] ~ Normal(0, sqrt(4))
            end

            return x
        end
        x = vdemo()()
        @test all((isassigned(x, i) for i in eachindex(x)))
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

        # Conditioning
        f1_c = f1() | (y=1,)
        f2_c = f2() | NamedTuple((Symbol(@varname(y[2][:, 1])) => 1,))
        f3_c = f3() | NamedTuple((Symbol(@varname(y[1])) => 1,))
        @test f1_c() == 1
        # TODO(torfjelde): We need conditioning for `Dict`.
        @test_broken f2_c() == 1
        @test_broken f3_c() == 1
        @test_broken getlogp(VarInfo(f1_c)) ==
            getlogp(VarInfo(f2_c)) ==
            getlogp(VarInfo(f3_c))
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
            @submodel prefix = "sub1" x1 = demo_return(x)
            @submodel prefix = "sub2" x2 = demo_return(y)

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
            η ~ MvNormal(zeros(num_steps), I)
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
                @submodel prefix = "ar1_$i" x = AR1(num_steps, α, μ, σ)
                y[i] ~ MvNormal(x, 0.01 * I)
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

        x = [Laplace(), Normal(), MvNormal(zeros(3), I)]
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
            m ~ MvNormal(zeros(2), I)
            return [10.0, 10.0] ~ MvNormal(m, 0.25 * I)
        end) isa Function

        @model function array_literal_model2()
            # `assume` and literal `observe`
            m ~ MvNormal(zeros(2), I)
            return [10.0, 10.0] ~ MvNormal(m, 0.25 * I)
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

    @testset "dispatching with model" begin
        f(x) = false

        @model demo() = x ~ Normal()
        @test !f(demo())
        f(::Model{typeof(demo)}) = true
        @test f(demo())

        # Leads to re-definition of `demo` and trait is not affected.
        @test length(methods(demo)) == 2
        @model demo() = x ~ Normal()
        @test length(methods(demo)) == 2
        @test f(demo())

        # Ensure we can specialize on arguments.
        @model demo(x) = x ~ Normal()
        @test length(methods(demo)) == 4
        @test f(demo(1.0))
        f(::Model{typeof(demo),(:x,)}) = false
        @test !f(demo(1.0))
        @test f(demo()) # should still be `true`

        # Set it to `false` again.
        f(::Model{typeof(demo),()}) = false
        @test !f(demo())
    end

    @testset "return value" begin
        # Make sure that a return-value of `x = 1` isn't combined into
        # an attempt at a `NamedTuple` of the form `(x = 1, __varinfo__)`.
        @model empty_model() = return x = 1
        empty_vi = VarInfo()
        retval_and_vi = DynamicPPL.evaluate!!(empty_model(), empty_vi, SamplingContext())
        @test retval_and_vi isa Tuple{Int,typeof(empty_vi)}

        # Even if the return-value is `AbstractVarInfo`, we should return
        # a `Tuple` with `AbstractVarInfo` in the second component too.
        @model demo() = return __varinfo__
        retval, svi = DynamicPPL.evaluate!!(demo(), SimpleVarInfo(), SamplingContext())
        @test svi == SimpleVarInfo()
        if Threads.nthreads() > 1
            @test retval isa DynamicPPL.ThreadSafeVarInfo{<:SimpleVarInfo}
            @test retval.varinfo == svi
        else
            @test retval == svi
        end

        # We should not be altering return-values other than at top-level.
        @model function demo()
            # If we also replaced this `return` inside of `f`, then the
            # final `return` would be include `__varinfo__`.
            f(x) = return x^2
            return f(1.0)
        end
        retval, svi = DynamicPPL.evaluate!!(demo(), SimpleVarInfo(), SamplingContext())
        @test retval isa Float64

        @model demo() = x ~ Normal()
        retval, svi = DynamicPPL.evaluate!!(demo(), SimpleVarInfo(), SamplingContext())

        # Return-value when using `@submodel`
        @model inner() = x ~ Normal()
        # Without assignment.
        @model outer() = @submodel inner()
        @test outer()() isa Real

        # With assignment.
        @model outer() = @submodel x = inner()
        @test outer()() isa Real
    end

    @testset "issue #368: hasmissing dispatch" begin
        @test !DynamicPPL.hasmissing(typeof(Union{}[]))

        # (nested) arrays with `Missing` eltypes
        @test DynamicPPL.hasmissing(Vector{Union{Missing,Float64}})
        @test DynamicPPL.hasmissing(Matrix{Union{Missing,Real}})
        @test DynamicPPL.hasmissing(Vector{Matrix{Union{Missing,Float32}}})

        # no `Missing`
        @test !DynamicPPL.hasmissing(Vector{Float64})
        @test !DynamicPPL.hasmissing(Matrix{Real})
        @test !DynamicPPL.hasmissing(Vector{Matrix{Float32}})
    end

    @testset "issue #393: anonymous argument with type parameter" begin
        @model f_393(::Val{ispredict}=Val(false)) where {ispredict} = ispredict ? 0 : 1
        @test f_393()() == 1
        @test f_393(Val(true))() == 0
    end

    @testset "splatting of args and kwargs" begin
        @model function f_splat_test_1(x; y::T=1, kwargs...) where {T}
            x ~ Normal(y, 1)
            return x, y, T, NamedTuple(kwargs)
        end

        # Non-empty `kwargs...`.
        res = f_splat_test_1(1; z=2, w=3)()
        @test res == (1, 1, Int, (z=2, w=3))

        # Empty `kwargs...`.
        res = f_splat_test_1(1)()
        @test res == (1, 1, Int, NamedTuple())

        @model function f_splat_test_2(x, args...; y::T=1, kwargs...) where {T}
            x ~ Normal(y, 1)
            return x, args, y, T, NamedTuple(kwargs)
        end

        # Non-empty `args...` and non-empty `kwargs...`.
        res = f_splat_test_2(1, 2, 3; z=2, w=3)()
        @test res == (1, (2, 3), 1, Int, (z=2, w=3))

        # Empty `args...` and empty `kwargs...`.
        res = f_splat_test_2(1)()
        @test res == (1, (), 1, Int, NamedTuple())
    end
end
