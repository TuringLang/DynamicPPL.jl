module DynamicPPLThreadSafeTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Distributions
using DynamicPPL
using Test
using ForwardDiff
using Random: Xoshiro
using LogDensityProblems: logdensity

@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
const gdemo_default = gdemo_d()

@model discrete_parameter() = x ~ Bernoulli(0.3)
@model continuous_parameter() = x ~ Normal()
@model observation_only() = 0.0 ~ Normal()

@testset "threadsafe.jl" begin
    @testset "parameter types permit floating-point accumulation" begin
        for T in (
            Int, Bool, Rational{Int}, Float32, BigFloat, ForwardDiff.Dual{Nothing,Float64,1}
        )
            vi = @inferred DynamicPPL.ThreadSafeVarInfo(
                OnlyAccsVarInfo(LogPriorAccumulator()), T
            )
            @test getlogprior(vi) isa float(T)
        end
        for T in (Any, Union{})
            vi = @inferred DynamicPPL.ThreadSafeVarInfo(
                OnlyAccsVarInfo(LogPriorAccumulator(big"0.0")), T
            )
            @test getlogprior(vi) isa BigFloat
        end

        discrete = setthreadsafe(discrete_parameter(), true)
        @test logjoint(discrete, (; x=1)) ≈ logpdf(Bernoulli(0.3), 1)
        @test logjoint(setthreadsafe(observation_only(), true), (;)) ≈ logpdf(Normal(), 0.0)

        continuous = setthreadsafe(continuous_parameter(), true)
        value, vi = init!!(
            Xoshiro(1), continuous, OnlyAccsVarInfo(), InitFromParams((;)), UnlinkAll()
        )
        @test getlogprior(vi) ≈ logpdf(Normal(), value)
        @test_throws ErrorException init!!(
            continuous, OnlyAccsVarInfo(), InitFromParams((;), nothing), UnlinkAll()
        )
        @test ForwardDiff.derivative(x -> logjoint(continuous, (; x)), 2.0) ≈ -2.0
    end

    @testset "unknown parameter types" begin
        @model observed(y) = y ~ Normal()
        for strategy in (InitFromParams((;)), InitFromPrior())
            function density(y)
                _, vi = init!!(
                    Xoshiro(1),
                    setthreadsafe(observed(y), true),
                    OnlyAccsVarInfo(),
                    strategy,
                    UnlinkAll(),
                )
                return getlogjoint(vi)
            end
            @test density(big"2.0") isa BigFloat
            @test ForwardDiff.derivative(density, 2.0) ≈ -2.0
        end
        @model fallback() = x ~ Normal(big"0.0", big"1.0")
        _, vi = init!!(
            Xoshiro(1),
            setthreadsafe(fallback(), true),
            OnlyAccsVarInfo(),
            InitFromParams((;)),
            UnlinkAll(),
        )
        @test getlogprior(vi) isa BigFloat
    end

    @testset "parameter containers preserve numeric types" begin
        @model function indexed_parameter()
            x = zeros(1)
            return x[1] ~ Normal()
        end
        model = setthreadsafe(indexed_parameter(), true)
        ldf = LogDensityFunction(model)
        for container in (
            identity,
            x -> Real[x...],
            x -> Any[x...],
            x -> Union{Int,eltype(x)}[x...],
            x -> view(Real[x...], :),
        )
            @test ForwardDiff.derivative(x -> logjoint(model, (; x=container([x]))), 2.0) ≈
                -2.0
        end
        @test ForwardDiff.derivative(x -> logdensity(ldf, Real[x]), 2.0) ≈ -2.0
        for T in (Float32, BigFloat)
            @test logjoint(model, (; x=Real[T(2)])) isa T
            @test (@inferred DynamicPPL.get_param_eltype(InitFromParams((; x=T[2])))) === T
        end
        dual = ForwardDiff.Dual(2.0, 1.0)
        wrapped = VarNamedTuple(; x=TransformedValue(Real[dual], Unlink()))
        @test get_param_eltype(DynamicPPL.InitFromParamsUnsafe(wrapped)) === typeof(dual)
        buffer = Vector{Real}(undef, 2)
        buffer[1] = dual
        for value in (buffer, [buffer])
            params = VarNamedTuple(; x=value)
            @test DynamicPPL.get_param_eltype(InitFromParams(params)) === typeof(dual)
        end
        @test DynamicPPL.get_param_eltype(InitFromParams((; x=Real[]))) === Union{}
        @test DynamicPPL.get_param_eltype(InitFromParams((; x=Vector{Real}(undef, 1)))) ===
            Union{}
    end

    @testset "unknown accumulator storage survives reconstruction" begin
        vi = DynamicPPL.ThreadSafeVarInfo(OnlyAccsVarInfo(LogPriorAccumulator()), Any)
        for rebuilt in (copy(vi), DynamicPPL.setacc!!(vi, LogPriorAccumulator()))
            rebuilt = DynamicPPL.acclogprior!!(rebuilt, ForwardDiff.Dual(2.0, 1.0))
            @test ForwardDiff.partials(getlogprior(rebuilt))[1] == 1.0
        end
    end

    @testset "constructor" begin
        vi = VarInfo(gdemo_default)
        threadsafe_vi = @inferred DynamicPPL.ThreadSafeVarInfo(vi)

        @test threadsafe_vi.varinfo === vi
        @test threadsafe_vi.accs_by_task isa IdDict{DynamicPPL.TaskId}
        @test isempty(threadsafe_vi.accs_by_task)

        vnt_acc = DynamicPPL.VNTAccumulator{:Test}(
            (val, _...) -> val, VarNamedTuple(; x=1.0)
        )
        threadsafe_vnt_vi = @inferred DynamicPPL.ThreadSafeVarInfo(OnlyAccsVarInfo(vnt_acc))
        @test_nowarn DynamicPPL.map_accumulators!!(identity, threadsafe_vnt_vi)
    end

    @testset "setthreadsafe" begin
        @model f() = x ~ Normal()
        model = f()
        @test !DynamicPPL.requires_threadsafe(model)
        model = setthreadsafe(model, true)
        @test DynamicPPL.requires_threadsafe(model)
        model = setthreadsafe(model, false)
        @test !DynamicPPL.requires_threadsafe(model)
    end

    # TODO: Add more tests of the public API
    @testset "API" begin
        vi = VarInfo(gdemo_default)
        threadsafe_vi = DynamicPPL.ThreadSafeVarInfo(vi)

        lp = getlogjoint(vi)
        @test getlogjoint(threadsafe_vi) == lp

        threadsafe_vi = DynamicPPL.acclogprior!!(threadsafe_vi, 42)
        @test getlogjoint(vi) == lp
        # float addition might lead to rounding errors so use approx rather than ==
        @test getlogjoint(threadsafe_vi) ≈ lp + 42

        copied_vi = @inferred copy(threadsafe_vi)
        @test isempty(copied_vi.accs_by_task)
        copied_vi = DynamicPPL.acclogprior!!(copied_vi, 1)
        @test getlogjoint(copied_vi) ≈ lp + 43
        @test getlogjoint(threadsafe_vi) ≈ lp + 42

        threadsafe_vi = DynamicPPL.resetaccs!!(threadsafe_vi)
        @test iszero(getlogjoint(threadsafe_vi))
        @test isempty(threadsafe_vi.accs_by_task)

        threadsafe_vi = setlogprior!!(threadsafe_vi, 42)
        @test getlogjoint(threadsafe_vi) == 42
        @test isempty(threadsafe_vi.accs_by_task)
    end

    @testset "tasks own accumulator state" begin
        ntasks = 2
        ready = Threads.Atomic{Int}(0)
        release = Threads.Atomic{Bool}(false)
        vi = DynamicPPL.ThreadSafeVarInfo(
            OnlyAccsVarInfo(DynamicPPL.LogLikelihoodAccumulator())
        )
        tasks = map(1:ntasks) do _
            Threads.@spawn DynamicPPL.map_accumulator!!(vi, Val(:LogLikelihood)) do acc
                Threads.atomic_add!(ready, 1)
                while !release[]
                    yield()
                end
                return DynamicPPL.acclogp(acc, 1.0)
            end
        end
        status = timedwait(() -> ready[] == ntasks, 30; pollint=0.001)
        release[] = true
        @test status === :ok
        fetch.(tasks)

        @test getloglikelihood(vi) == ntasks
        @test length(vi.accs_by_task) == ntasks
    end

    @testset "aggregation preserves mutable accumulator state" begin
        accname = Val(:VectorParamAccumulator)
        main_acc = DynamicPPL.VectorParamAccumulator(
            [1.0, 0.0], [true, false], VarNamedTuple()
        )
        vi = DynamicPPL.ThreadSafeVarInfo(OnlyAccsVarInfo(main_acc))
        vi = DynamicPPL.map_accumulator!!(vi, accname) do acc
            acc.vals[2] = 2.0
            acc.set_indices[2] = true
            acc
        end

        @test DynamicPPL.getacc(vi, accname).vals == [1.0, 2.0]
        @test main_acc.vals == [1.0, 0.0]
        @test main_acc.set_indices == [true, false]
        @test DynamicPPL.getacc(vi, accname).vals == [1.0, 2.0]

        copied_vi = copy(vi)
        @test DynamicPPL.get_vector_params(copied_vi) == [1.0, 2.0]
        @test DynamicPPL.getacc(vi, accname).vals == [1.0, 2.0]
    end

    @testset "colon-eq extraction during threaded evaluation" begin
        @model function colon_eq(n)
            x = collect(1:n)
            Threads.@threads for i in eachindex(x)
                x[i] := i
            end
        end
        model = setthreadsafe(colon_eq(10), true)
        vi = OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(true))
        _, vi = DynamicPPL.init!!(model, vi, InitFromPrior(), UnlinkAll())
        @test length(DynamicPPL.get_raw_values(vi)) == 10
    end

    @testset "Check that VarInfo is wrapped during model evaluation" begin
        @model function f()
            global vi_ = __varinfo__
            return x ~ Normal(0, 1)
        end
        model = setthreadsafe(f(), true)

        _, vi = DynamicPPL.init!!(model, VarInfo())
        # Inside the model evaluation function, it should be wrapped
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        # But init!! should return the original VarInfo
        @test vi isa DynamicPPL.VarInfo
        # Same with evaluate!!
        _, vi = DynamicPPL.evaluate_nowarn!!(model, vi)
        @test vi_ isa DynamicPPL.ThreadSafeVarInfo
        @test vi isa DynamicPPL.VarInfo
    end

    @testset "Type stability of getlogjoint" begin
        # init!!(...) itself is not type stable (unclear exactly why, but it has to do with
        # __varinfo__ being boxed since Threads.@threads creates a closure). It fails to
        # infer the type of AbstractVarInfo returned. However we expect that getlogjoint
        # should be type stable since regardless of what kind of AbstractVarInfo is passed
        # in, it should always return a Float64.
        @model function f(y)
            x ~ Normal()
            Threads.@threads for i in eachindex(y)
                y[i] ~ Normal(x)
            end
            return nothing
        end
        y = fill(1.0, 10)
        model = setthreadsafe(f(y), true)

        @testset for vi in (VarInfo(), VarInfo(model), OnlyAccsVarInfo())
            @inferred getlogjoint(
                last(DynamicPPL.init!!(model, vi, InitFromPrior(), UnlinkAll()))
            )
        end
    end

    @testset "check_model with threadsafe" begin
        # This is a partial test for https://github.com/TuringLang/DynamicPPL.jl/issues/1157
        @model function f()
            Threads.@threads for _ in 1:10
                x ~ Normal()
            end
        end
        model = setthreadsafe(f(), true)
        @test !check_model(model)
    end

    @testset "assumes are threadsafe" begin
        # See https://github.com/TuringLang/DynamicPPL.jl/pull/1284.
        #
        # Note: anything that involves VarInfo is still thread-unsafe. But anything
        # that uses OnlyAccsVarInfo is fine
        @model function threaded_assume()
            x = zeros(10)
            Threads.@threads for i in eachindex(x)
                x[i] ~ Normal()
            end
        end
        model = setthreadsafe(threaded_assume(), true)

        @testset "rand" begin
            vnt = rand(model)
            for i in 1:10
                @test haskey(vnt, @varname(x[i]))
            end
        end
        @testset "logprob" begin
            xfixed = rand(10)
            params = VarNamedTuple(; x=xfixed)
            @test logprior(model, params) ≈ sum(logpdf.(Normal(), xfixed))
            @test iszero(loglikelihood(model, params))
            @test logjoint(model, params) ≈ sum(logpdf.(Normal(), xfixed))
        end
    end

    @testset "logprob correctness" begin
        x = rand(10_000)

        @model function wthreads(x)
            x[1] ~ Normal(0, 1)
            Threads.@threads for i in 2:length(x)
                x[i] ~ Normal(x[i - 1], 1)
            end
        end
        model = setthreadsafe(wthreads(x), true)

        function correct_lp(x)
            lp = logpdf(Normal(0, 1), x[1])
            for i in 2:length(x)
                lp += logpdf(Normal(x[i - 1], 1), x[i])
            end
            return lp
        end

        _, vi = DynamicPPL.init!!(model, VarInfo())

        # check that logp is correct
        @test getlogjoint(vi) ≈ correct_lp(x)
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
