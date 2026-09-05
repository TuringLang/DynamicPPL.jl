module DynamicPPLConditionFixTests

using Dates: now
using ComponentArrays: ComponentVector
using Distributions
using DimensionalData: DimArray, X
using DynamicPPL
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using LogDensityProblems: LogDensityProblems
using Test

@info "Testing $(@__FILE__)..."
__now__ = now()

struct ObservationRecord{A,B}
    a::A
    b::B
end

struct ReplacementRecord{A,B}
    a::A
    b::B
end

@testset "condition and fix" begin
    @model function demo_cond_fix()
        x ~ Normal()
        return y ~ Normal(x)
    end
    model = demo_cond_fix()

    function test_logp_correct(
        op::Union{typeof(condition),typeof(fix)}, transformed::Model, x
    )
        y = 1.0
        values = VarNamedTuple(; y)
        @test logprior(transformed, values) == logpdf(Normal(x), y)
        if op === condition
            @test loglikelihood(transformed, values) == logpdf(Normal(), x)
        else
            @test iszero(loglikelihood(transformed, values))
        end
    end

    @testset "$op input forms" for op in (condition, fix)
        x = 0.5
        transformed_models = (
            op(model; x),
            op(model, VarNamedTuple(; x)),
            op(model, (; x)),
            op(model, Dict(@varname(x) => x)),
            op(model, @varname(x) => x),
        )
        for transformed in transformed_models
            test_logp_correct(op, transformed, x)
        end
        if op === condition
            test_logp_correct(condition, model | VarNamedTuple(; x), x)
            test_logp_correct(condition, model | (; x), x)
            test_logp_correct(condition, model | Dict(@varname(x) => x), x)
            test_logp_correct(condition, model | (@varname(x) => x), x)
        end
    end

    @testset "values are model fields" begin
        conditioned_model = condition(condition(model; x=1.0); x=2.0, y=3.0)
        @test conditioned(conditioned_model)[@varname(x)] == 2.0
        @test conditioned(conditioned_model)[@varname(y)] == 3.0
        @test conditioned_model() == 3.0

        fixed_model = fix(fix(model; x=1.0); x=2.0, y=3.0)
        @test fixed(fixed_model)[@varname(x)] == 2.0
        @test fixed(fixed_model)[@varname(y)] == 3.0
        @test fixed_model() == 3.0

        @model return_x() = x ~ Normal()
        for first_op in (condition, fix), last_op in (condition, fix)
            transformed = last_op(first_op(return_x(); x=1.0); x=2.0)
            @test transformed() == 2.0
            @test logjoint(transformed, VarNamedTuple()) ==
                (last_op === condition ? logpdf(Normal(), 2.0) : 0.0)
            @test isempty(conditioned(transformed)) == (last_op === fix)
            @test isempty(fixed(transformed)) == (last_op === condition)
        end
    end

    @testset "argument buffers can be initialized in the model body" begin
        @model function initialize_buffer(x)
            fill!(x, [1.0])
            x[1] ~ MvNormal([0.0], [1.0;;])
            return x
        end
        for wrap in (identity, x -> view(x, :))
            data = wrap(Vector{Vector{Float64}}(undef, 1))
            buffer_model = initialize_buffer(data)
            @test !isassigned(data, 1)
            @test buffer_model() == [[1.0]]
        end
    end

    @testset "partial overrides preserve siblings" begin
        @model function indexed()
            x = zeros(2, 2)
            x[1] ~ Normal()
            x[2, 2] ~ Normal()
            return x
        end
        @model function properties()
            x = (; a=0.0, b=0.0)
            x.a ~ Normal()
            x.b ~ Normal()
            return x
        end
        for first_op in (condition, fix), last_op in (condition, fix)
            original = first_op(indexed(); x=[1.0 0.0; 0.0 2.0])
            changed = last_op(original, @varname(x[1]) => 3.0)
            @test changed()[1] == 3.0
            @test changed()[2, 2] == 2.0
            @test original()[1] == 1.0
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ==
                (first_op === condition ? logpdf(Normal(), 2.0) : 0.0) +
                  (last_op === condition ? logpdf(Normal(), 3.0) : 0.0)
            original = first_op(properties(); x=(; a=1.0, b=2.0))
            changed = last_op(original, @varname(x.a) => 3.0)
            @test changed() == (; a=3.0, b=2.0)
            @test original() == (; a=1.0, b=2.0)
        end
    end

    @testset "observation arguments retain body computations" begin
        @model scalar_input(x=1.0) = (x += 1; x ~ Normal(); return x)
        @model keyword_input(; x=1.0) = (x += 1; x ~ Normal(); return x)
        @model repeated_input(x) = (x += 1; x ~ Normal(); x += 1; x ~ Normal(); return x)
        @model wrapped_input(model) = a ~ to_submodel(model)
        @model expanded_array(x) = (x = vcat(x, oftype(first(x), 2)); x[2] ~ Normal(); x)
        @model expanded_record(x) = (x = (; x..., b=oftype(x.a, 2)); x.b ~ Normal(); x)
        for T in (Float32, Float64, BigFloat),
            changed in (expanded_array(T[1]), expanded_record((; a=T(1))))

            @test loglikelihood(changed, VarNamedTuple()) ≈ logpdf(Normal(), T(2))
            @test wrapped_input(changed)() == changed()
        end
        @test scalar_input()() == keyword_input()() == 2.0
        for T in (Float32, Float64, BigFloat),
            constructor in (scalar_input, x -> keyword_input(; x))

            original = constructor(T(1))
            for model in (original, condition(decondition(original); x=T(1)))
                @test model() == T(2)
                @test loglikelihood(model, VarNamedTuple()) ≈ logpdf(Normal(), T(2))
            end
            @test fix(original; x=T(3))() == T(3)
            @test condition(original; x=T(3))() == T(4)
            nested = condition(wrapped_input(original), @varname(a.x) => T(3))
            @test nested() == T(4)
            @test loglikelihood(nested, VarNamedTuple()) ≈ logpdf(Normal(), T(4))
        end
        @test repeated_input(1.0)() == 3.0
        @model splatted_input(x...) = (
            x = collect(x) .+ 1; x ~ MvNormal(zeros(length(x)), I); return x
        )
        @test splatted_input(1.0)() == [2.0]
        @test loglikelihood(repeated_input(1.0), VarNamedTuple()) ≈
            logpdf(Normal(), 2.0) + logpdf(Normal(), 3.0)
        @test ForwardDiff.derivative(
            x -> loglikelihood(scalar_input(x), VarNamedTuple()), 1.0
        ) == -2.0

        @model function partial_input(x)
            x = x .+ 1
            before = copy(x)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return (; before, x)
        end
        for data in (Float32[0, 0], BigFloat[0, 0], DimArray([0.0, 0.0], X)), i in 1:2
            changed_model = condition(
                decondition(partial_input(data)), @varname(x[i]) => 3.0
            )
            result, vi = init!!(
                changed_model,
                OnlyAccsVarInfo(),
                InitFromParams((; x=[7.0, 7.0])),
                UnlinkAll(),
            )
            @test result.before[i] == result.x[i] == 4.0
            @test result.before[3 - i] == 1.0
            @test result.x[3 - i] == 7.0
            @test getloglikelihood(vi) ≈ logpdf(Normal(), 4.0)
            @test data == [0, 0]
        end
        partial_loglik =
            p -> loglikelihood(
                condition(decondition(partial_input(zeros(2))), @varname(x[1]) => p),
                (; x=[7.0, 7.0]),
            )
        @test ForwardDiff.derivative(partial_loglik, 3.0) == -4.0
    end

    @testset "successive nested array overrides retain shape" begin
        @model function nested_array(x)
            for i in eachindex(x), j in eachindex(x[i])
                x[i][j] ~ Normal()
            end
            return x
        end
        @model outer_array(model) = a ~ to_submodel(model)
        for first_op in (condition, fix),
            second_op in (condition, fix),
            third_op in (condition, fix),
            T in (Float32, BigFloat)

            data = reshape([[T(i)] for i in 1:4], 2, 2)
            original = nested_array(data)
            first = first_op(original, @varname(x[1][1]) => T(10))
            second = second_op(first, @varname(x[2][1]) => T(20))
            third = third_op(second, @varname(x[1, 1][1]) => T(30))
            expected = reshape([[T(30)], [T(20)], [T(3)], [T(4)]], 2, 2)
            @test third() == expected
            @test third_op(outer_array(second), @varname(a.x[1][1]) => T(30))() == expected
            @test first()[2][1] == T(2)
            @test original() == data
            @test loglikelihood(third, VarNamedTuple()) ≈
                (third_op === condition ? logpdf(Normal(), T(30)) : zero(T)) +
                  (second_op === condition ? logpdf(Normal(), T(20)) : zero(T)) +
                  logpdf(Normal(), T(3)) +
                  logpdf(Normal(), T(4))
        end
        data = reshape([[Float32(i)] for i in 1:4], 2, 2)
        replaced = condition(nested_array([[0.0f0]]); x=data)
        replaced = fix(replaced, @varname(x[1][1]) => 10.0f0)
        replaced = condition(replaced, @varname(x[2][1]) => 20.0f0)
        @test replaced() == reshape([[10.0f0], [20.0f0], [3.0f0], [4.0f0]], 2, 2)

        partial = condition(decondition(nested_array(data)), @varname(x[1][1]) => 10.0f0)
        partial = condition(partial, @varname(x[2][1]) => 20.0f0)
        result, _ = init!!(
            partial, OnlyAccsVarInfo(), InitFromParams((; x=data)), UnlinkAll()
        )
        @test result == reshape([[10.0f0], [20.0f0], [3.0f0], [4.0f0]], 2, 2)

        observations = @vnt begin
            @template x = data
            x[1][1] := 10.0f0
        end
        partial = @test_logs condition(decondition(nested_array(data)), observations)
        @test size(conditioned(partial).data.x) == (2, 2)
    end

    @testset "argument observations can be replaced and removed" begin
        @model argument_model(x) = x ~ Normal()
        for initial in (1.0f0, 1.0, big"1.0")
            original = argument_model(initial)
            @test original() === initial
            @test isempty(keys(VarInfo(original)))
            @test conditioned(original)[@varname(x)] === initial
            @test loglikelihood(original, VarNamedTuple()) == logpdf(Normal(), initial)
            @test conditioned(original) == conditioned(condition(original; x=initial))
            observed = condition(argument_model(initial); x=2.0)
            @test observed() == 2.0
            @test logjoint(observed, VarNamedTuple()) == logpdf(Normal(), 2.0)
            @test keys(VarInfo(decondition(observed))) == [@varname(x)]
            @test condition(decondition(original); x=3.0)() == 3.0
        end

        @model keyword_argument(; x=1.0) = x ~ Normal()
        @test keyword_argument()() == 1.0
        @test keyword_argument(; x=2.0)() == 2.0
        @test keys(VarInfo(decondition(keyword_argument()))) == [@varname(x)]

        @model function array_argument(x; config=nothing)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        for data in ([1.0f0, 2.0f0], BigFloat[1, 2], DimArray([1.0, 2.0], X))
            original = array_argument(data)
            @test keys(conditioned(original)) == [@varname(x)]
            latent = decondition(original, :x)
            result, _ = init!!(
                latent, OnlyAccsVarInfo(), InitFromParams((; x=[3.0, 4.0])), UnlinkAll()
            )
            @test result == [3.0, 4.0]
            @test data == [1.0, 2.0]
            @test original() == [1.0, 2.0]
            @test condition(latent; x=[5.0, 6.0])() == [5.0, 6.0]
            @test data == [1.0, 2.0]
        end
        ldf = LogDensityFunction(decondition(array_argument(zeros(2))))
        logdensity = p -> LogDensityProblems.logdensity(ldf, p)
        @test ForwardDiff.gradient(logdensity, [1.0, 2.0]) ≈ [-1.0, -2.0]

        @model inner(y) = y ~ Normal()
        @model function outer(x)
            a ~ to_submodel(inner(x))
            b ~ to_submodel(inner(x))
            return (; a, b)
        end
        transformed = condition(outer(0.0), @varname(b.y) => 1.0)
        @test isempty(keys(VarInfo(transformed)))
        @test transformed().a == 0.0
        @test transformed().b == 1.0

        @model function reread(x)
            x ~ Normal()
            return x + x
        end
        observed = condition(reread(1.0); x=2.0)
        @test observed() == 4.0
        @test fix(decondition(observed); x=3.0)() == 6.0
    end

    @testset "replacement arguments drive model execution" begin
        @model function indexed_argument(x)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        for op in (condition, fix),
            data in ([1.0f0], BigFloat[1, 2, 3], DimArray([1.0, 2.0, 3.0], X))

            original = indexed_argument(zeros(2))
            changed = op(original; x=data)
            @test changed() === data
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ≈
                (op === condition ? sum(logpdf.(Normal(), data)) : 0.0)
            @test original() == zeros(2)
        end
        loglik =
            x -> loglikelihood(condition(indexed_argument(zeros(2)); x), VarNamedTuple())
        @test ForwardDiff.gradient(loglik, [1.0, 2.0, 3.0]) ≈ [-1.0, -2.0, -3.0]

        @model function read_before_site(; x=1.0)
            y ~ Normal(x)
            x ~ Normal()
            return x
        end
        for op in (condition, fix)
            changed = op(read_before_site(); x=2.0)
            @test logprior(changed, (; y=2.0)) == logpdf(Normal(2.0), 2.0)
        end
        logp = x -> logprior(condition(read_before_site(); x), (; y=2.0))
        @test ForwardDiff.derivative(logp, 1.0) == 1.0
    end

    @testset "partial property overrides preserve struct fields" begin
        @model function record_sites(x)
            x.a ~ Normal()
            x.b[1] ~ Normal()
            x.b[2] ~ Normal()
            return x
        end
        data = ObservationRecord(1.0, [2.0, 3.0])
        for first_op in (condition, fix), last_op in (condition, fix)
            original = first_op(record_sites(data); x=data)
            changed = last_op(original, @varname(x.a) => 4.0, @varname(x.b[1]) => 5.0)
            result = changed()
            @test result isa ObservationRecord
            @test result.a == 4.0
            @test result.b == [5.0, 3.0]
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ≈
                (first_op === condition ? logpdf(Normal(), 3.0) : 0.0) +
                  (last_op === condition ? sum(logpdf.(Normal(), [4.0, 5.0])) : 0.0)
            @test original().a == data.a == 1.0
            @test original().b == data.b == [2.0, 3.0]
        end
        @test_throws ArgumentError condition(record_sites(data), @varname(x.unknown) => 1.0)
    end

    @testset "property overrides retain replacement containers" begin
        @model fields(x) = (x.a ~ Normal(); return x)
        @model nested_fields(m) = child ~ to_submodel(m)
        for first_op in (condition, fix),
            last_op in (condition, fix),
            (original, replacement) in (
                ((; a=0.0f0), (; a=1.0f0, b=2.0f0)),
                (ObservationRecord(0.0f0, 0.0f0), ReplacementRecord(1.0f0, 2.0f0)),
                (ObservationRecord(big"0", big"0"), (; a=big"1", b=big"2")),
            )

            base = first_op(fields(original); x=replacement)
            changed = last_op(base, @varname(x.a) => oftype(replacement.a, 3))
            result = changed()
            @test typeof(result) === typeof(replacement)
            @test result.a == 3
            @test result.b == 2
            @test base().a == replacement.a == 1
            @test loglikelihood(changed, VarNamedTuple()) ≈
                (last_op === condition ? logpdf(Normal(), 3) : 0)
            nested = last_op(
                nested_fields(base), @varname(child.x.a) => oftype(replacement.a, 4)
            )
            @test typeof(nested()) === typeof(replacement)
            @test nested().a == 4
            @test nested().b == 2

            remove = last_op === condition ? decondition : unfix
            latent = remove(changed, @varname(x.a))
            result, _ = init!!(
                latent,
                OnlyAccsVarInfo(),
                InitFromParams((; x=(; a=oftype(replacement.a, 7)))),
                UnlinkAll(),
            )
            @test typeof(result) === typeof(replacement)
            @test result.a == 7
            @test result.b == 2
        end
        parent = condition(
            nested_fields(fields((; a=0.0))), @varname(child.x) => (; a=1.0, b=2.0)
        )
        parent = fix(parent, @varname(child.x.a) => 3.0)
        @test parent() == (; a=3.0, b=2.0)
        result, _ = @inferred evaluate!!(
            parent, DefaultContext(VarNamedTuple()), OnlyAccsVarInfo()
        )
        @test result == (; a=3.0, b=2.0)

        @model array_fields(x) = (x[1].a ~ Normal(); return x)
        parent = condition(
            nested_fields(array_fields([(; a=0.0)])),
            @varname(child.x[1]) => (; a=1.0, b=2.0),
        )
        @test fix(parent, @varname(child.x[1].a) => 3.0)() == [(; a=3.0, b=2.0)]
        partial = condition(
            decondition(array_fields([(; a=0.0)])), @varname(x[1]) => (; a=1.0, b=2.0)
        )
        @test fix(partial, @varname(x[1].a) => 3.0)() == [(; a=3.0, b=2.0)]

        base = condition(fields(ObservationRecord(0.0, 0.0)); x=ReplacementRecord(1.0, 2.0))
        loglik = p -> loglikelihood(condition(base, @varname(x.a) => p), VarNamedTuple())
        @test ForwardDiff.derivative(loglik, 3.0) == -3.0
        selected = conditioned(condition(base, @varname(x.a) => 3.0))
        @test selected[@varname(x)] isa ReplacementRecord
        @test selected == conditioned(condition(base, @varname(x.a) => 3.0))
        @test condition(fields(ObservationRecord(0.0, 0.0)), selected)().a == 3.0
        mixed = fix(base, @varname(x.a) => 3.0)
        supplied = merge(conditioned(mixed), fixed(mixed))
        @test supplied[@varname(x)] isa ReplacementRecord
        @test supplied[@varname(x)].a == 3.0
        @test supplied[@varname(x)].b == 2.0

        @model namespace_child(x, y) = (x ~ Normal(); y ~ Normal(); return (x, y))
        @model namespace_parent() = child ~ to_submodel(namespace_child(0.0, 0.0))
        parent = condition(namespace_parent(); child=(; x=1.0, y=2.0))
        parent = decondition(fix(parent, @varname(child.x) => 3.0), @varname(child.y))
        @test parent() == (3.0, 0.0)
    end

    @testset "indexed tuple bindings preserve tuples and roles" begin
        @model tuple_sites(x) = (x[1] ~ Normal(); x[2] ~ Normal(); return x)
        @model nested_tuple(m) = child ~ to_submodel(m)
        for T in (Float32, BigFloat),
            first_op in (condition, fix),
            last_op in (condition, fix)

            original = tuple_sites((T(1), T(2)))
            first = first_op(original, @varname(x[1]) => T(3))
            changed = last_op(first, @varname(x[2]) => T(4))
            @test changed() == (T(3), T(4))
            @test merge(conditioned(changed), fixed(changed))[@varname(x)] == (T(3), T(4))
            @test original() == (T(1), T(2))
            @test first() == (T(3), T(2))
            @test loglikelihood(changed, VarNamedTuple()) ≈
                (first_op === condition ? logpdf(Normal(), T(3)) : zero(T)) +
                  (last_op === condition ? logpdf(Normal(), T(4)) : zero(T))
            @test last_op(nested_tuple(first), @varname(child.x[2]) => T(4))() ==
                (T(3), T(4))
        end
        @model tuple_with_record(x) = (x[1].a ~ Normal(); return x)
        changed = fix(tuple_with_record(((; a=1.0), 2.0)), @varname(x[1].a) => 3.0)
        @test changed() == ((; a=3.0), 2.0)
        @model tuple_with_indexed(x) = (x[1][1] ~ Normal(); return x)
        @model tuple_with_nested_record(x) = (x[1].a[1] ~ Normal(); return x)
        for (original, vn, expected) in (
            (tuple_with_record(((; a=1.0),)), @varname(x[1].a), ((; a=3.0),)),
            (tuple_with_indexed(((1.0,),)), @varname(x[1][1]), ((3.0,),)),
            (tuple_with_indexed(([1.0],)), @varname(x[1][1]), ([3.0],)),
            (
                tuple_with_nested_record(((; a=(1.0,)),)),
                @varname(x[1].a[1]),
                ((; a=(3.0,)),),
            ),
        )
            fixed_model = fix(original, vn => 3.0)
            @test fixed(fixed_model)[vn] == 3.0
            @test decondition(fixed_model)() == expected
            @test fixed(decondition(fixed_model))[vn] == 3.0
            conditioned_model = condition(fix(original; x=expected), vn => 3.0)
            @test conditioned(conditioned_model)[vn] == 3.0
            @test conditioned(unfix(conditioned_model))[vn] == 3.0
            @test merge(conditioned(fixed_model), fixed(fixed_model))[@varname(x)] ==
                expected
            nested = nested_tuple(fixed_model)
            @test decondition(nested)() == expected
        end
        loglik =
            p -> loglikelihood(
                condition(tuple_sites((0.0, 2.0)), @varname(x[1]) => p), VarNamedTuple()
            )
        @test ForwardDiff.derivative(loglik, 3.0) == -3.0
    end

    @testset "observation role lookup does not copy slices" begin
        @model slices(x) = x[:] ~ Normal()
        function evaluation_bytes(model)
            vi = OnlyAccsVarInfo()
            init!!(model, vi, InitFromPrior(), UnlinkAll())
            return @allocated init!!(model, vi, InitFromPrior(), UnlinkAll())
        end
        small, large = slices(zeros(10_000)), slices(zeros(100_000))
        @test evaluation_bytes(large) - evaluation_bytes(small) < 1_000_000
        @test loglikelihood(large, VarNamedTuple()) ≈ 100_000 * logpdf(Normal(), 0.0)
    end

    @testset "partial updates retain complete argument replacements" begin
        @model function indexed_replacement(x)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        for data in (Float32[1, 2, 3], BigFloat[1, 2, 3], DimArray([1.0, 2.0, 3.0], X))
            original = condition(indexed_replacement(zeros(2)); x=data)
            same = condition(original, @varname(x[1]) => data[1])
            @test same() == original() == data
            @test typeof(same()) === typeof(data)
            @test loglikelihood(same, VarNamedTuple()) ≈
                loglikelihood(original, VarNamedTuple())
            mixed = fix(same, @varname(x[1]) => data[1])
            @test mixed() == data
            @test isempty(keys(VarInfo(mixed)))
            @test loglikelihood(mixed, VarNamedTuple()) ≈ sum(logpdf.(Normal(), data[2:3]))
        end
    end

    @testset "replacement arguments bind evaluator type parameters" begin
        @model function typed_observation(x::AbstractVector{T}) where {T}
            x::AbstractVector{T}
            x[1] ~ Normal()
            return (T, x)
        end
        @model nested_observation(m) = a ~ to_submodel(m)
        for op in (condition, fix), T in (Float32, BigFloat)
            typed_model = op(typed_observation([0.0]); x=T[2])
            @test typed_model() == (T, T[2])
            nested = op(nested_observation(typed_observation([0.0])), @varname(a.x) => T[2])
            @test nested() == (T, T[2])
        end
        @model function ordinary_input(x)
            return y ~ Normal(x)
        end
        @test logprior(condition(ordinary_input(1.0); x=2.0), (; y=1.0)) ==
            logpdf(Normal(), 0.0)
    end

    @testset "component properties and indices share observations" begin
        @model function indexed_components()
            x = ComponentVector(; a=0.0, b=0.0)
            x[1] ~ Normal()
            x.b ~ Normal()
            return x
        end
        @model joint_components(n=2) = x ~ MvNormal(zeros(n), I)
        for op in (condition, fix)
            data = ComponentVector(; a=1.0, b=2.0)
            original = op(indexed_components(); x=data)
            changed = op(original, @varname(x.a) => 3.0)
            changed = op(changed, @varname(x[2]) => 4.0)
            changed = op(changed, @varname(x.b) => 5.0)
            @test changed() == ComponentVector(; a=3.0, b=5.0)
            @test isempty(keys(VarInfo(changed)))
            @test original() == data
            joint = op(op(joint_components(); x=data), @varname(x.a) => 1.0)
            @test joint() == data
            @test joint() isa ComponentVector
            @test logjoint(joint, VarNamedTuple()) ≈
                (op === condition ? logpdf(MvNormal(zeros(2), I), data) : 0.0)
        end
        for (data, vn, value, expected) in (
                (
                    ComponentVector(; a=[1.0, 2.0], b=3.0),
                    @varname(x.a),
                    [4.0, 5.0],
                    ComponentVector(; a=[4.0, 5.0], b=3.0),
                ),
                (
                    ComponentVector(; a=(b=[1.0, 2.0], c=3.0)),
                    @varname(x.a.b[2]),
                    4.0,
                    ComponentVector(; a=(b=[1.0, 4.0], c=3.0)),
                ),
                (
                    ComponentVector(; a=[1.0 2.0; 3.0 4.0]),
                    @varname(x.a[2, 1]),
                    5.0,
                    ComponentVector(; a=[1.0 2.0; 5.0 4.0]),
                ),
            ),
            op in (condition, fix)

            original = op(joint_components(length(data)); x=data)
            changed = op(original, vn => value)
            @test changed() == expected
            @test typeof(changed()) === typeof(data)
            @test original() == data
            @test logjoint(changed, VarNamedTuple()) ≈ (
                op === condition ? logpdf(MvNormal(zeros(length(data)), I), expected) : 0.0
            )
            changed = op(changed, @varname(x[1]) => expected[1])
            @test changed() == expected
        end
    end

    @testset "joint named tuples reconstruct nested arrays" begin
        @model joint_namedtuple() =
            x ~ product_distribution((; a=MvNormal(zeros(2), I), b=Normal()))
        data = (; a=[1.0, 2.0], b=3.0)
        for op in (condition, fix)
            original = op(joint_namedtuple(); x=data)
            changed = op(original, @varname(x.a[1]) => 1.0)
            @test changed() == original() == data
            @test logjoint(changed, VarNamedTuple()) == logjoint(original, VarNamedTuple())
        end
    end

    @testset "decondition and unfix" begin
        conditioned_model = condition(model; x=1.0, y=2.0)
        @test isempty(keys(conditioned(decondition(conditioned_model))))
        @test keys(conditioned(decondition(conditioned_model, :x))) == [@varname(y)]
        @test keys(conditioned(decondition(conditioned_model, @varname(x)))) ==
            [@varname(y)]

        fixed_model = fix(model; x=1.0, y=2.0)
        @test isempty(keys(fixed(unfix(fixed_model))))
        @test keys(fixed(unfix(fixed_model, :x))) == [@varname(y)]
        @test keys(fixed(unfix(fixed_model, @varname(x)))) == [@varname(y)]

        nested_values = VarNamedTuple((@varname(a.x) => 1.0, @varname(b) => 2.0))
        @test keys(
            conditioned(decondition(condition(model, nested_values), @varname(a)))
        ) == [@varname(b)]
        @test keys(fixed(unfix(fix(model, nested_values), @varname(a)))) == [@varname(b)]

        mixed = fix(condition(model; x=1.0); y=2.0)
        @test fixed(decondition(mixed)) == fixed(mixed)
        @test conditioned(unfix(mixed)) == conditioned(mixed)
    end

    @testset "parent model values override submodel values" begin
        @model inner() = x ~ Normal()
        @model function outer(inner_model)
            return a ~ to_submodel(inner_model)
        end

        for inner_op in (condition, fix)
            inner_model = inner_op(inner(); x=1.0)
            @test outer(inner_model)() == 1.0
            for outer_op in (condition, fix)
                transformed = outer_op(outer(inner_model), @varname(a.x) => 2.0)
                @test transformed() == 2.0
                @test logjoint(transformed, VarNamedTuple()) ==
                    (outer_op === condition ? logpdf(Normal(), 2.0) : 0.0)
            end
        end
    end

    @testset "immutable data can be fixed" begin
        @model function ntfix()
            m ~ Normal()
            data = (; x=undef)
            data.x ~ Normal(m, 1.0)
            return data.x
        end
        fixed_model = fix(ntfix(), (; data=(; x=5.0)))
        accs = OnlyAccsVarInfo(RawValueAccumulator(false))
        retval, accs = init!!(fixed_model, accs, InitFromPrior(), UnlinkAll())
        @test retval == 5.0
        @test get_raw_values(accs)[@varname(m)] isa Real
    end

    @testset "multivariate values" begin
        @model function mvnorm()
            x ~ MvNormal(zeros(3), I)
            return x
        end
        templated = @vnt begin
            @template x = zeros(3)
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        untemplated = @vnt begin
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        for op in (condition, fix), values in (templated, untemplated)
            @test op(mvnorm(), values)() == [1.0, 2.0, 3.0]
        end
    end
    @testset "merging growable and templated conditions (#1481)" begin
        @model function partial_array_model()
            x = zeros(2, 2)
            x[1, 1] ~ Normal()
            x[2, 1] ~ Normal()
            return x
        end
        @model function partial_array_parent(child_model)
            return child ~ to_submodel(child_model)
        end
        next_values = @vnt begin
            @template x = zeros(2, 2)
            x[2, 1] := 2.5
        end
        for op in (condition, fix)
            model = partial_array_model()
            nested = op(partial_array_parent(model), Dict(@varname(child.x[1, 1]) => 1.5))
            model = op(model, Dict(@varname(x[1, 1]) => 1.5))
            for (partial_model, values) in
                ((model, next_values), (nested, VarNamedTuple(; child=next_values)))
                @test returned(op(partial_model, values), VarNamedTuple()) ==
                    [1.5 0.0; 2.5 0.0]
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
