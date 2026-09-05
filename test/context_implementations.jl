module DynamicPPLContextImplementationTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra: I, norm
using Random: Xoshiro
using Test

struct UnimplementedContext <: AbstractContext end

struct RecordingContext{C<:AbstractContext} <: AbstractContext
    inner::C
    assumed::Vector{VarName}
end
function RecordingContext(inner)
    return RecordingContext(inner, VarName[])
end
function DynamicPPL.get_param_eltype(ctx::RecordingContext)
    return DynamicPPL.get_param_eltype(ctx.inner)
end
function DynamicPPL.tilde_assume!!(
    ctx::RecordingContext, dist::Distribution, vn::VarName, template, vi::AbstractVarInfo
)
    push!(ctx.assumed, vn)
    return DynamicPPL.tilde_assume!!(ctx.inner, dist, vn, template, vi)
end

@model function child(y=2.0)
    x ~ Normal()
    y ~ Normal(x)
    z := x + y
    0.0 ~ Normal()
    return z
end
@model parent(m, ::Val{auto_prefix}) where {auto_prefix} = a ~ to_submodel(m, auto_prefix)
@model outer(m) = b ~ to_submodel(m)

@testset "context_implementations.jl" begin
    @testset "explicit evaluation interface" begin
        @test_throws MethodError evaluate!!(child(), VarInfo())
        @static if VERSION >= v"1.11"
            @test Base.ispublic(DynamicPPL, :make_evaluate_args_and_kwargs)
            @test Base.ispublic(DynamicPPL, :store_coloneq_value!!)
        end
    end

    @testset "observations do not dispatch on context" begin
        for T in (Float32, Float64, BigFloat)
            dist = Normal(zero(T), one(T))
            for value in (one(T), T[1, 2])
                for vn in (@varname(x), nothing)
                    result, vi = @inferred tilde_observe!!(
                        child(), dist, value, vn, NoTemplate(), OnlyAccsVarInfo()
                    )
                    @test result === value
                    @test getloglikelihood(vi) ≈ loglikelihood(dist, value)
                    @test iszero(getlogprior(vi))
                end
            end
        end
        @test !applicable(
            tilde_observe!!,
            DefaultContext(),
            Normal(),
            1.0,
            nothing,
            NoTemplate(),
            OnlyAccsVarInfo(),
        )
        @test !applicable(
            DynamicPPL.store_coloneq_value!!,
            DefaultContext(),
            @varname(z),
            1.0,
            NoTemplate(),
            OnlyAccsVarInfo(),
        )
    end

    @testset "no context hooks needed without latent sites" begin
        accs = OnlyAccsVarInfo((
            DynamicPPL.default_accumulators()..., RawValueAccumulator(true)
        ))
        result, vi = evaluate!!(fix(child(); x=1.0), UnimplementedContext(), accs)
        @test result == 3.0
        @test get_raw_values(vi)[@varname(z)] == 3.0
        @test iszero(getlogprior(vi))
        @test getlogjoint(vi) ≈ logpdf(Normal(1.0), 2.0) + logpdf(Normal(), 0.0)
    end

    @testset "contexts belong to evaluations" begin
        for auto_prefix in (true, false)
            inner = auto_prefix ? child() : prefix(child(), @varname(a))
            model = outer(parent(inner, Val(auto_prefix)))
            @test !hasproperty(model, :context)
            for x in (1.0, 3.0)
                strategy = InitFromParams(VarNamedTuple((@varname(b.a.x) => x,)))
                ctx = RecordingContext(InitContext(Xoshiro(1), strategy, UnlinkAll()))
                accs = OnlyAccsVarInfo((
                    DynamicPPL.default_accumulators()..., RawValueAccumulator(true)
                ))
                result, vi = evaluate!!(model, ctx, accs)
                @test result == x + 2.0
                @test ctx.assumed == [@varname(b.a.x)]
                @test get_raw_values(vi)[@varname(b.a.z)] == result
                @test getlogjoint(vi) ≈
                    logpdf(Normal(), x) + logpdf(Normal(x), 2.0) + logpdf(Normal(), 0.0)
                @test iszero(getlogjoint(accs))
                @test first(@inferred evaluate!!(model, ctx.inner, OnlyAccsVarInfo())) ==
                    result
            end
        end

        model = child()
        ctx = InitContext(Xoshiro(1), InitFromParams((; x=1.0)), UnlinkAll())
        result, vi = @inferred evaluate!!(model, ctx, VarInfo())
        @test result == 3.0
        @test first(@inferred evaluate!!(model, DefaultContext(get_values(vi)), vi)) ==
            result
        ctx = InitContext(Xoshiro(1), InitFromParams((; x=3.0)), UnlinkAll())
        result, vi = @inferred evaluate!!(model, ctx, vi)
        @test result == 5.0
        @test first(@inferred evaluate!!(model, DefaultContext(get_values(vi)), vi)) ==
            result
    end

    @testset "DefaultContext supplies inputs independently of outputs" begin
        @test_throws ArgumentError DefaultContext(VarNamedTuple(; x=1.0))
        @test_throws ArgumentError DefaultContext(
            VarNamedTuple(; x=TransformedValue([1.0], NoTransform()))
        )
        @test_throws KeyError evaluate!!(child(), DefaultContext(), OnlyAccsVarInfo())
        @test_throws KeyError evaluate!!(
            child(), DefaultContext(), VarInfo(child(), InitFromParams((; x=1.0)))
        )
        nested = DynamicPPL.TestUtils.demo_nested_colons()
        nested_input = VarInfo(Xoshiro(1), nested)
        @test DynamicPPL.get_param_eltype(DefaultContext(get_values(nested_input))) ==
            Float64
        for T in (Float32, Float64, BigFloat), threaded in (false, true)
            model = setthreadsafe(child(T(2)), threaded)
            input = VarInfo(Xoshiro(1), model, InitFromParams((; x=one(T))))
            context = DefaultContext(get_values(input))
            @test DynamicPPL.get_param_eltype(context) == T
            for output in (
                OnlyAccsVarInfo(),
                VarInfo(),
                VarInfo(Xoshiro(2), model, InitFromParams((; x=T(9)))),
            )
                result, output = evaluate!!(model, context, output)
                @test result == T(3)
                @test getlogjoint(output) ≈
                    logpdf(Normal(), one(T)) +
                      logpdf(Normal(one(T)), T(2)) +
                      logpdf(Normal(), 0.0)
                if output isa VarInfo
                    @test get_values(output) == get_values(input)
                end
            end
        end

        @model function dependent_support()
            x ~ Exponential()
            y ~ Uniform(zero(x), x)
            return y
        end
        model = dependent_support()
        input = VarInfo(Xoshiro(1), model, InitFromParams((; x=2.0, y=0.5)), LinkAll())
        context = DefaultContext(get_values(input))
        # The same linked y is 0.5 under Uniform(0, 2), but 1.0 under Uniform(0, 4).
        changed = fix(model; x=4.0)
        result, output = @inferred evaluate!!(changed, context, OnlyAccsVarInfo())
        expected = VarInfo(Xoshiro(1), changed, InitFromParams((; y=1.0)), LinkAll())
        @test result ≈ 1.0
        @test getlogjoint(output) ≈ getlogjoint(expected)
        @test getlogjac(output) ≈ getlogjac(expected)
    end

    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(zeros(2), 4 * I)
            z = Vector{Int}(undef, length(x))
            z ~ product_distribution(Categorical.(fill([0.5, 0.5], length(x))))
            for i in eachindex(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo())
    end

    @testset "dot tilde with varying sizes" begin
        @testset "assume" begin
            @model function test(x, size)
                y = Array{Float64,length(size)}(undef, size...)
                y .~ Normal(x)
                return y
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                x = randn()
                model = test(x, ysize)
                y = model()
                lp = logjoint(model, (; y=y))
                @test lp ≈ sum(logpdf.(Normal.(x), y))

                ys = [first(model()) for _ in 1:10_000]
                @test norm(mean(ys) .- x, Inf) < 0.1
                @test norm(std(ys) .- 1, Inf) < 0.1
            end
        end

        @testset "observe" begin
            @model function test(x, y)
                return y .~ Normal(x)
            end

            for ysize in ((2,), (2, 3), (2, 3, 4))
                x = randn()
                y = randn(ysize)
                z = logjoint(test(x, y), VarInfo())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
