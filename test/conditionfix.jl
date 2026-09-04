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
        @test conditioned_model.context === model.context
        @test conditioned(conditioned_model)[@varname(x)] == 2.0
        @test conditioned(conditioned_model)[@varname(y)] == 3.0
        @test conditioned_model() == 3.0

        fixed_model = fix(fix(model; x=1.0); x=2.0, y=3.0)
        @test fixed_model.context === model.context
        @test fixed(fixed_model)[@varname(x)] == 2.0
        @test fixed(fixed_model)[@varname(y)] == 3.0
        @test fixed_model() == 3.0

        @model return_x() = x ~ Normal()
        @test fix(condition(return_x(); x=1.0); x=2.0)() == 2.0

        for first_op in (condition, fix), last_op in (condition, fix)
            transformed = last_op(first_op(return_x(); x=1.0); x=2.0)
            @test transformed() == 2.0
            @test logjoint(transformed, VarNamedTuple()) ==
                (last_op === condition ? logpdf(Normal(), 2.0) : 0.0)
            @test isempty(conditioned(transformed)) == (last_op === fix)
            @test isempty(fixed(transformed)) == (last_op === condition)
        end
    end

    @testset "missing is not a stochastic role" begin
        for op in (condition, fix), value in (missing, [1.0, missing], (; a=missing))
            @test_throws ArgumentError op(model; x=value)
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
        @test_throws ArgumentError argument_model(missing)

        @model keyword_argument(; x=1.0) = x ~ Normal()
        @test keyword_argument()() == 1.0
        @test keyword_argument(; x=2.0)() == 2.0
        @test keys(VarInfo(decondition(keyword_argument()))) == [@varname(x)]

        @model function array_argument(x; config=missing)
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

        conditioned_inner = condition(inner(); x=1.0)
        @test outer(conditioned_inner)() == 1.0
        @test condition(outer(conditioned_inner), @varname(a.x) => 2.0)() == 2.0

        fixed_inner = fix(inner(); x=1.0)
        @test outer(fixed_inner)() == 1.0
        @test fix(outer(fixed_inner), @varname(a.x) => 2.0)() == 2.0

        for inner_op in (condition, fix), outer_op in (condition, fix)
            transformed = outer_op(outer(inner_op(inner(); x=1.0)), @varname(a.x) => 2.0)
            @test transformed() == 2.0
            @test logjoint(transformed, VarNamedTuple()) ==
                (outer_op === condition ? logpdf(Normal(), 2.0) : 0.0)
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
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
