module DynamicPPLContextImplementationTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra: I, norm
using Random: Xoshiro
using Test

struct UnimplementedStrategy <: AbstractInitStrategy end

struct RecordingStrategy{S<:AbstractInitStrategy} <: AbstractInitStrategy
    inner::S
    assumed::Vector{VarName}
end
RecordingStrategy(inner) = RecordingStrategy(inner, VarName[])
function DynamicPPL.get_param_eltype(strategy::RecordingStrategy)
    return DynamicPPL.get_param_eltype(strategy.inner)
end
function DynamicPPL.init(rng, vn::VarName, dist::Distribution, strategy::RecordingStrategy)
    push!(strategy.assumed, vn)
    return DynamicPPL.init(rng, vn, dist, strategy.inner)
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
        @test !hasproperty(child(), :context)
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
                        child(), dist, value, vn, NoTemplate(), VarInfo()
                    )
                    @test result === value
                    @test getloglikelihood(vi) ≈ loglikelihood(dist, value)
                    @test iszero(getlogprior(vi))
                end
            end
        end
        @test !applicable(
            tilde_observe!!,
            Context(InitFromPrior(), UnlinkAll()),
            Normal(),
            1.0,
            nothing,
            NoTemplate(),
            VarInfo(),
        )
        @test !applicable(
            DynamicPPL.store_coloneq_value!!,
            Context(InitFromPrior(), UnlinkAll()),
            @varname(z),
            1.0,
            NoTemplate(),
            VarInfo(),
        )
    end

    @testset "no context hooks needed without latent sites" begin
        accs = VarInfo((DynamicPPL.default_accumulators()..., RawValueAccumulator(true)))
        result, vi = evaluate!!(
            fix(child(); x=1.0), Context(UnimplementedStrategy(), UnlinkAll()), accs
        )
        @test result == 3.0
        @test get_raw_values(vi)[@varname(z)] == 3.0
        @test iszero(getlogprior(vi))
        @test getlogjoint(vi) ≈ logpdf(Normal(1.0), 2.0) + logpdf(Normal(), 0.0)
    end

    @testset "contexts belong to evaluations" begin
        for auto_prefix in (true, false)
            inner = auto_prefix ? child() : prefix(child(), @varname(a))
            model = outer(parent(inner, Val(auto_prefix)))
            for x in (1.0, 3.0)
                strategy = InitFromParams(VarNamedTuple((@varname(b.a.x) => x,)))
                recording = RecordingStrategy(strategy)
                ctx = Context(Xoshiro(1), recording, UnlinkAll())
                accs = VarInfo((
                    DynamicPPL.default_accumulators()..., RawValueAccumulator(true)
                ))
                result, vi = evaluate!!(model, ctx, accs)
                @test result == x + 2.0
                @test recording.assumed == [@varname(b.a.x)]
                @test get_raw_values(vi)[@varname(b.a.z)] == result
                @test getlogjoint(vi) ≈
                    logpdf(Normal(), x) + logpdf(Normal(x), 2.0) + logpdf(Normal(), 0.0)
                @test iszero(getlogjoint(accs))
                @test first(@inferred evaluate!!(model, ctx, VarInfo())) == result
            end
        end

        model = child()
        ctx = Context(Xoshiro(1), InitFromParams((; x=1.0)), UnlinkAll())
        result, vi = @inferred evaluate!!(model, ctx, VarInfo(VectorValueAccumulator()))
        @test result == 3.0
        ctx = Context(Xoshiro(1), InitFromParams((; x=3.0)), UnlinkAll())
        result, vi = @inferred evaluate!!(model, ctx, vi)
        @test result == 5.0
        @test first(
            @inferred evaluate!!(
                model,
                Context(
                    InitFromParams(get_values(vi), nothing),
                    DynamicPPL.infer_transform_strategy_from_values(get_values(vi)),
                ),
                vi,
            )
        ) == result
    end

    @testset "Context supplies inputs independently of outputs" begin
        @test fieldnames(VarInfo) == (:accs,)
        @test !isdefined(DynamicPPL, :OnlyAccsVarInfo)
        @test !isdefined(DynamicPPL, :DefaultContext)
        @test !isdefined(DynamicPPL, :InitContext)
        @test :AbstractContext ∉ names(DynamicPPL)
        @test isconcretetype(typeof(Context(Xoshiro(1), InitFromPrior(), UnlinkAll())))
        empty_context = Context(
            Xoshiro(1), InitFromParams(VarNamedTuple(), nothing), UnlinkAll()
        )
        @test_throws ErrorException evaluate!!(child(), empty_context, VarInfo())
        @test_throws ErrorException evaluate!!(
            child(), empty_context, VarInfo(child(), InitFromParams((; x=1.0)))
        )
        for T in (Float32, Float64, BigFloat), threaded in (false, true)
            model = setthreadsafe(child(T(2)), threaded)
            input = VarInfo(Xoshiro(1), model, InitFromParams((; x=one(T))))
            context = Context(
                Xoshiro(1), InitFromParams(get_vector_values(input), nothing), UnlinkAll()
            )
            @test DynamicPPL.get_param_eltype(context) == T
            for output in (
                VarInfo(),
                VarInfo(VectorValueAccumulator(), DynamicPPL.default_accumulators()...),
                VarInfo(Xoshiro(2), model, InitFromParams((; x=T(9)))),
            )
                result, output = evaluate!!(model, context, output)
                @test result == T(3)
                @test getlogjoint(output) ≈
                    logpdf(Normal(), one(T)) +
                      logpdf(Normal(one(T)), T(2)) +
                      logpdf(Normal(), 0.0)
                if DynamicPPL.hasacc(output, Val(DynamicPPL.VECTORVAL_ACCNAME))
                    @test get_vector_values(output) == get_vector_values(input)
                end
            end
        end

        # Inputs determine transforms even when the reused output recorded linked values.
        @model positive() = x ~ Exponential()
        old = VarInfo(positive(), InitFromParams((; x=2.0)), LinkAll())
        context = Context(Xoshiro(1), InitFromParams((; x=3.0), nothing), UnlinkAll())
        result, output = evaluate!!(positive(), context, old)
        @test result == 3.0
        @test iszero(getlogjac(output))
        @test get_vector_values(output)[@varname(x)].transform isa Unlink
        @test first(init!!(positive(), old, InitFromParams((; x=4.0), nothing))) == 4.0

        @test first(
            evaluate!!(
                positive(), Context(Xoshiro(1), InitFromPrior(), UnlinkAll()), VarInfo()
            ),
        ) == first(
            evaluate!!(
                positive(), Context(Xoshiro(1), InitFromPrior(), UnlinkAll()), VarInfo()
            ),
        )

        @model function optional_site(include_x)
            if include_x
                x ~ Normal()
            end
            return y ~ Normal()
        end
        outputs = VarInfo(VectorValueAccumulator(), RawValueAccumulator(false))
        context = Context(Xoshiro(1), InitFromPrior(), UnlinkAll())
        _, outputs = evaluate!!(optional_site(true), context, outputs)
        inputs = get_vector_values(outputs)
        context = Context(Xoshiro(1), InitFromParams(inputs, nothing), UnlinkAll())
        _, outputs = evaluate!!(optional_site(false), context, outputs)
        @test !haskey(get_vector_values(outputs), @varname(x))
        @test !haskey(get_raw_values(outputs), @varname(x))
        @test haskey(inputs, @varname(x))
        @test outputs == copy(outputs)
        @test isequal(outputs, copy(outputs))
        @test hash(outputs) == hash(copy(outputs))

        @model function dependent_support()
            x ~ Exponential()
            y ~ Uniform(zero(x), x)
            return y
        end
        model = dependent_support()
        input = VarInfo(Xoshiro(1), model, InitFromParams((; x=2.0, y=0.5)), LinkAll())
        context = Context(
            InitFromParams(get_values(input), nothing),
            DynamicPPL.infer_transform_strategy_from_values(get_values(input)),
        )
        # The same linked y is 0.5 under Uniform(0, 2), but 1.0 under Uniform(0, 4).
        changed = fix(model; x=4.0)
        result, output = @inferred evaluate!!(changed, context, VarInfo())
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
                z = logjoint(test(x, y), VarNamedTuple())
                @test z ≈ sum(logpdf.(Normal.(x), y))
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
