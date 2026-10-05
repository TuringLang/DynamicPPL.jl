module DynamicPPLConditionFixTests

using AbstractPPL: AbstractPPL, of, @of
using Dates: now
using ADTypes: AutoForwardDiff
using ComponentArrays: ComponentVector, getaxes
using Distributions
using DimensionalData: DimArray, X
using DynamicPPL
using ForwardDiff: ForwardDiff
using LinearAlgebra: I, Transpose, Adjoint
using LogDensityProblems: LogDensityProblems
using OffsetArrays: OffsetArray
using Test
using Random: Xoshiro
using StaticArrays: SVector, MVector, SizedArray
using StableRNGs: StableRNG
using Logging: NullLogger, with_logger

@info "Testing $(@__FILE__)..."
__now__ = now()

@model function observed_view(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
@model observed_view_parent(x) = a ~ to_submodel(observed_view(x))
@testset "partial edits check argument storage beneath another layer" begin
    for recursive in (false, true)
        vn = recursive ? @varname(a.x[1]) : @varname(x[1])
        for wrap in (identity, x -> view(x, :))
            m = if recursive
                observed_view_parent(wrap(zeros(2)))
            else
                observed_view(wrap(zeros(2)))
            end
            m = if recursive
                condition(m, @varname(a.x) => [1.0, 2.0, 3.0])
            else
                condition(m; x=[1.0, 2.0, 3.0])
            end
            if wrap === identity
                @test fix(m, vn => 9.0)(Xoshiro(1)) == [9.0, 2.0, 3.0]
            elseif recursive
                @test_throws ArgumentError fix(m, vn => 9.0)(Xoshiro(1))
            else
                @test_throws ArgumentError fix(m, vn => 9.0)
            end
        end
    end
end

@model function aliased_argument(y)
    for i in eachindex(y), j in eachindex(y[i])
        y[i][j] ~ Normal()
    end
    return y
end
@model function aliased_fields(y)
    for j in eachindex(y.u)
        y.u[j] ~ Normal()
        y.v[j] ~ Normal()
    end
    return y
end
@testset "latent arrays own backing storage" begin
    for sibling in (identity, x -> view(x, :)), container in (:array, :namedtuple)
        inner = [1.0, 2.0]
        if container === :array
            m = decondition(aliased_argument([inner, sibling(inner)]), @varname(y[1][1]))
            params = (; y=[[5.0]])
        else
            value = (u=inner, v=sibling(inner))
            m = decondition(aliased_fields(value), @varname(y.u[1]))
            params = (; y=(u=[5.0],))
        end
        @test logjoint(m, params) ≈ sum(logpdf.(Normal(), [5.0, 2.0, 1.0, 2.0]))
        @test inner == [1.0, 2.0]
    end
end

@model leaf_owner() = (x = zeros(1); x[1] ~ Normal(); x)
@model parent_owner() = a ~ to_submodel(leaf_owner())
@model function indexed_owner()
    a = Vector{Any}(undef, 1)
    return a[1] ~ to_submodel(leaf_owner())
end
@model property_owner() = (p = (a=zeros(1),); p.a[1] ~ Normal(); p)
@testset "nested owners validate replacement types" begin
    for bind in (condition, fix)
        m = bind(parent_owner(), @varname(a.x) => Float32[1])
        @test_throws ArgumentError bind(m, @varname(a.x[1]) => 0.1)
        @test_throws ArgumentError bind(m, @varname(a.x) => [1.0])
        indexed = bind(indexed_owner(), @varname(a[1].x) => Float32[1])
        @test_throws ArgumentError bind(indexed, @varname(a[1].x[1]) => 0.1)
        @test_throws ArgumentError bind(indexed, @varname(a[1].x) => [1.0])
        for (owner, address) in ((m, @varname(a.x[1])), (indexed, @varname(a[1].x[1])))
            valid = bind(owner, address => 0.5)
            listed = bind === condition ? conditioned(valid) : fixed(valid)
            @test listed[address] === 0.5f0
            @test valid(Xoshiro(1)) == [0.5f0]
        end
        m = bind(property_owner(), @varname(p.a) => Float32[1])
        @test_throws ArgumentError bind(m, @varname(p.a[1]) => 0.1)
        @test_throws ArgumentError bind(m, @varname(p.a) => [1.0])
        for value in (0.5, 1)
            valid = bind(m, @varname(p.a[1]) => value)
            listed = bind === condition ? conditioned(valid) : fixed(valid)
            @test listed[@varname(p.a[1])] === Float32(value)
        end
        resized = bind(m, @varname(p.a) => Float32[1, 2])
        listed = bind === condition ? conditioned(resized) : fixed(resized)
        @test listed[@varname(p.a)] == Float32[1, 2]
    end
end

@model smaller_fixed(p) = (p.a ~ Normal(); p.b ~ Normal(); p)
@model smaller_fixed_parent() = a ~ to_submodel(smaller_fixed((a=1.0, b=2.0, c=3.0)))
@testset "fixed field owner keeps its fields" begin
    for recursive in (false, true)
        m = if recursive
            fix(smaller_fixed_parent(), @varname(a.p) => (a=4.0, b=5.0))
        else
            fix(smaller_fixed((a=1.0, b=2.0, c=3.0)); p=(a=4.0, b=5.0))
        end
        address = recursive ? @varname(a.p.b) : @varname(p.b)
        partial = unfix(m, address)
        @test partial(Xoshiro(1)) == (a=4.0, b=2.0)
        rebound = fix(m, address => 6.0)
        @test rebound(Xoshiro(1)) == (a=4.0, b=6.0)
        @test unfix(partial)(Xoshiro(1)) == (a=1.0, b=2.0, c=3.0)
        @test logjoint(partial, (;)) ≈ logpdf(Normal(), 2.0)
    end
end

@testset "partial argument preparation inference" begin
    @model array_argument(x) = x[1] ~ Normal()
    @model named_argument(x) = x.a[1] ~ Normal()
    for T in (Float32, Float64, BigFloat, ForwardDiff.Dual{Nothing,Float64,1})
        data = T[1, 2, 3]
        for (original, address, template) in (
            (array_argument(data), @varname(x[1]), data),
            (named_argument((a=data, b=T(4))), @varname(x.a[1]), (a=data, b=T(4))),
        )
            for bind in (condition, fix, decondition)
                model = if bind === decondition
                    bind(original, address)
                else
                    bind(original, address => zero(T))
                end
                actual = @inferred DynamicPPL.prepare_model_argument(
                    model, @varname(x), template
                )
                expected_data = bind === decondition ? data : T[0, 2, 3]
                expected =
                    template isa NamedTuple ? (a=expected_data, b=T(4)) : expected_data
                @test actual == expected
                prepared_data = actual isa NamedTuple ? actual.a : actual
                @test prepared_data !== data
                prepared_data[2] = T(5)
                @test data == T[1, 2, 3]
            end
        end
    end
end

struct ObservationRecord{A,B}
    a::A
    b::B
end

struct ReplacementRecord{A,B}
    a::A
    b::B
end

mutable struct MissingRecord
    a::Union{Missing,Float64}
end

struct MetadataRecord
    a::Float64
    b::Missing
end

struct LatentRecord{X,Y}
    X::X
    y::Y
end
mutable struct MutableLatentRecord{X,Y}
    X::X
    y::Y
end
@model function selective_copy(d)
    m ~ Normal()
    for i in eachindex(d.y)
        d.y[i] ~ Normal(m + d.X[i, 1])
    end
    return d
end

struct VirtualBindingRecord
    a::Float64
end
Base.propertynames(::VirtualBindingRecord) = (:field,)
Base.getproperty(x::VirtualBindingRecord, ::Symbol) = getfield(x, :a)

@testset "partial argument containers are checked at binding time" begin
    @model dictionary_argument(x) = (x[:a] ~ Normal(); x[:b] ~ Normal(); x)
    @model nested_dictionary_argument(x) = (x.d[:a] ~ Normal(); x.d[:b] ~ Normal(); x)
    @model nested_property_argument(x) = (x.s.field ~ Normal(); x)
    @model record_argument(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
    dictionary = Dict(:a => missing, :b => 2.0)
    virtual = VirtualBindingRecord(1.0)
    for (model, address, container) in (
        (dictionary_argument(dictionary), @varname(x[:a]), dictionary),
        (nested_dictionary_argument((d=dictionary,)), @varname(x.d[:a]), dictionary),
        (nested_property_argument((s=virtual,)), @varname(x.s.field), virtual),
    )
        message = [
            "ArgumentError:", "owner", "`$address`", string(typeof(container)), "whole"
        ]
        container isa AbstractDict && push!(message, "NamedTuple")
        matches = err -> all(part -> occursin(part, err), message)
        @test_throws matches decondition(model, address)
        for bind in (condition, fix)
            @test_throws matches bind(model, address => 3.0)
            @test_throws matches bind(decondition(model), address => 3.0)
        end
        @test_throws matches unfix(fix(model; x=model.args.x), address)
    end
    # Whole dictionaries, including those nested in supported containers, remain valid.
    data = Dict(:a => 1.0, :b => 2.0)
    for (model, value) in (
        (dictionary_argument(data), data),
        (nested_dictionary_argument((d=data,)), (d=data,)),
    )
        @test returned(model, (;)) == value
        for bind in (condition, fix)
            @test returned(bind(model; x=value), (;)) == value
        end
        @test returned(decondition(model, @varname(x)), (x=value,)) == value
        @test returned(unfix(fix(model; x=value), @varname(x)), (;)) == value
    end
    source = ObservationRecord(1.0, 2.0)
    for bind in (condition, fix)
        @test_throws ArgumentError bind(record_argument(source), @varname(x.a) => 3.0)
        @test returned(
            bind(record_argument(source); x=ObservationRecord(3.0, 2.0)), (;)
        ) === ObservationRecord(3.0, 2.0)
    end
    @test_throws ArgumentError decondition(record_argument(source), @varname(x.a))
    @test returned(decondition(record_argument(source)), (x=(a=3.0, b=2.0),)) ===
        ObservationRecord(3.0, 2.0)
end

@testset "condition and fix" begin
    @testset "structured arguments supply partial binding storage" begin
        @model fields_storage(p) = (p.a[1] ~ Normal(); p.a[2] ~ Normal(); p.a)
        @model tuple_storage(p) = (p[1] ~ Normal(); p[2] ~ Normal(); p)
        cases = ((fields_storage((a=[1.0, 2.0],)), @varname(p.a)),)
        for (model, root) in cases, bind in (condition, fix), latent in (false, true)
            base = latent ? decondition(model) : model
            for (optic, value, expected) in (
                (@varname(x[end]), 3.0, [1.0, 3.0]),
                (@varname(x[:]), [3.0, 4.0], [3.0, 4.0]),
                (@varname(x[[true, false]]), [3.0], [3.0, 2.0]),
            )
                address = AbstractPPL.append_optic(root, AbstractPPL.getoptic(optic))
                edited = bind(base, address => value)
                supplied = root == @varname(p) ? (p=[1.0, 2.0],) : (p=(a=[1.0, 2.0],),)
                @test collect(returned(edited, supplied)) == expected
            end
        end
    end
    @testset "partial argument bindings need concrete storage" begin
        @model function placeholder_indices(x=missing)
            (ismissing(x) || x === nothing) && (x = zeros(1))
            x = x .+ 1
            before = x[1]
            x[1] ~ Normal()
            return (before, x[1])
        end
        @model function placeholder_fields(p=missing)
            (ismissing(p) || p === nothing) && (p = (a=0.0,))
            p.a ~ Normal()
            return p
        end
        templated = DynamicPPL.@vnt begin
            @template x = zeros(1)
            x[1] := 2.0
        end
        message = r"ArgumentError: .*partial.*`[xp]`.*(nothing|missing).*concrete argument.*whole binding"
        for absent in (nothing, missing), bind in (condition, fix)
            for model in
                (placeholder_indices(absent), decondition(placeholder_indices(absent)))
                @test_throws message bind(model, @varname(x[1]) => 2.0)
                @test_throws message bind(model, (@varname(x[1]) => 2.0,))
                @test bind(model, templated)(Xoshiro(1)) ==
                    (3.0, bind === condition ? 3.0 : 2.0)
                @test_throws message bind(model, of((;)), @varname(x[1]) => 2.0)
                @test_throws message model | (@varname(x[1]) => 2.0)
                @test (model | templated)(Xoshiro(1)) == (3.0, 3.0)
                whole = bind(model; x=[2.0])
                @test whole(Xoshiro(1)) == (3.0, bind === condition ? 3.0 : 2.0)
            end
            latent = decondition(placeholder_indices(absent))
            whole = bind(latent; x=[2.0])
            @test bind(whole, @varname(x[1]) => 4.0)(Xoshiro(1)) ==
                (5.0, bind === condition ? 5.0 : 4.0)
            @test bind(latent, :x => [2.0], @varname(x[1]) => 4.0)(Xoshiro(1)) ==
                (5.0, bind === condition ? 5.0 : 4.0)
            fields = decondition(placeholder_fields(absent))
            @test_throws message bind(fields, @varname(p.a) => 2.0)
            @test bind(fields; p=(a=2.0,))(Xoshiro(1)) == (a=2.0,)
            @test decondition(placeholder_indices(absent))(Xoshiro(1)) ==
                (1.0, rand(Xoshiro(1), Normal()))
            @test fields(Xoshiro(1)) == (a=rand(Xoshiro(1), Normal()),)
        end
        @model function gdemo(x=missing)
            (ismissing(x) || x === nothing) && (x = zeros(2))
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        @model keyword_gdemo(; x=nothing) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        @model typed_gdemo(x::Union{Nothing,Vector{Int}}=nothing) = (
            x[1] ~ Normal(); x[2] ~ Normal(); x
        )
        @model produced_parent(child) = a ~ to_submodel(child)
        draws = rand(Xoshiro(1), decondition(gdemo()))
        incomplete = conditioned(
            decondition(condition(gdemo(zeros(2)); x=[2.0, 3.0]), @varname(x[2]))
        )
        for model in (gdemo(), gdemo(nothing), keyword_gdemo()), bind in (condition, fix)
            bound = bind(model, draws)
            @test bound(Xoshiro(2)) == draws[@varname(x)]
            @test logjoint(bound, (;)) ≈
                (bind === condition ? sum(logpdf.(Normal(), draws[@varname(x)])) : 0.0)
            @test (model | draws)(Xoshiro(2)) == draws[@varname(x)]
            @test_throws message bind(model, incomplete)
            @test_throws message model | incomplete
            @test produced_parent(bound)(Xoshiro(2)) == draws[@varname(x)]
            parent_draws = rand(Xoshiro(1), produced_parent(decondition(gdemo())))
            @test bind(produced_parent(model), parent_draws)(Xoshiro(2)) ==
                parent_draws[@varname(a.x)]
        end
        @model function nested_gdemo(x=nothing)
            x === nothing && (x = [zeros(2)])
            x[1][1] ~ Normal()
            x[1][2] ~ Normal()
            return x
        end
        nested_draws = rand(Xoshiro(1), decondition(nested_gdemo()))
        nested_incomplete = conditioned(
            decondition(nested_gdemo([[2.0, 3.0]]), @varname(x[1][2]))
        )
        for bind in (condition, fix)
            @test bind(nested_gdemo(), nested_draws)(Xoshiro(2)) ==
                [nested_draws[@varname(x[1])]]
            @test_throws message bind(nested_gdemo(), nested_incomplete)
        end
        for bind in (condition, fix)
            @test_throws "declared argument type" bind(typed_gdemo(), draws)
            integers = conditioned(condition(gdemo([2, 3]), @varname(x[1]) => 2))
            @test bind(typed_gdemo(), integers)(Xoshiro(2)) == [2, 3]
        end
        concrete = condition(
            decondition(placeholder_indices(zeros(1))), @varname(x[1]) => 2.0
        )
        @test concrete(Xoshiro(1)) == (3.0, 3.0)
        @test loglikelihood(concrete, (;)) == logpdf(Normal(), 3.0)

        @model function runtime_placeholder(absent, bind)
            m ~ Normal()
            child = bind(decondition(placeholder_indices(absent)), @varname(x[1]) => m)
            a ~ to_submodel(child)
            return a
        end
        for absent in (nothing, missing), bind in (condition, fix)
            model = runtime_placeholder(absent, bind)
            @test_throws message returned(model, (m=2.0,))
            @test_throws message ForwardDiff.derivative(m -> logjoint(model, (; m)), 2.0)
        end
    end

    @testset "unfix restores argument-supplied observations from declared argument LHS variables" begin
        @model scalar_argument(x) = x ~ Normal()
        @model keyword_argument(; x) = x ~ Normal()
        for observed in (scalar_argument(1.0), keyword_argument(; x=1.0))
            direct = DynamicPPL.Model{false}(
                observed.f, observed.args, observed.defaults; args_on_lhs=(:x,)
            )
            for original in (direct, condition(direct; x=4.0), decondition(direct))
                for names in ((), (@varname(x),))
                    restored = unfix(fix(original; x=2.0), names...)
                    @test conditioned(restored) == conditioned(original)
                    @test rand(Xoshiro(1), restored) == rand(Xoshiro(1), original)
                end
            end
            for original in (observed, decondition(observed))
                restored = unfix(fix(original; x=2.0), @varname(x))
                @test conditioned(restored) == conditioned(original)
                @test rand(Xoshiro(1), restored) == rand(Xoshiro(1), original)
            end
        end
    end

    @testset "missing diagnostics follow the LHS variable role" begin
        @model missing_binding() = x ~ Normal()
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            bound = bind(missing_binding(); x=missing)
            @test_throws "LHS variable `x` contains `missing`; make it latent with `$remove`." bound(
                Xoshiro(1)
            )
            @test remove(bound, @varname(x))(Xoshiro(1)) isa Real
        end
    end

    @testset "NamedTuple bindings and LHS variables require field names" begin
        @model integer_lhs(x) = (x[1] ~ Normal(); x)
        @model nested_integer_lhs(x) = (x.p[1] ~ Normal(); x)
        @model array_integer_lhs(x) = (x[1][1] ~ Normal(); x)
        @model local_integer_lhs() = (x = (a=1.0, b=2.0); x[1] ~ Normal(); x)
        for (model, value, address, field) in (
            (integer_lhs, (a=1.0, b=2.0), @varname(x[1]), "x.a"),
            (nested_integer_lhs, (p=(a=1.0, b=2.0),), @varname(x.p[1]), "x.p.a"),
            (array_integer_lhs, [(a=1.0, b=2.0)], @varname(x[1][1]), "x[1].a"),
        )
            message = ArgumentError(
                "Integer indexing into a NamedTuple at `$address` is unsupported; use `$field` instead.",
            )
            for (origin, m) in (
                (identity, model(value)),
                (condition, condition(model(nothing); x=value)),
                (fix, fix(model(nothing); x=value)),
            )
                for bind in (condition, fix)
                    # The other layer still has only the placeholder argument for storage.
                    expected = if origin !== identity && bind !== origin
                        r"ArgumentError: .*partial.*`x`.*nothing.*concrete argument.*whole binding"
                    else
                        message
                    end
                    @test_throws expected bind(m, address => 3.0)
                end
                for remove in (decondition, unfix)
                    if remove === unfix && origin === condition
                        @test conditioned(remove(m, address)) == conditioned(m)
                        continue
                    end
                    expected = if remove === decondition && origin === fix
                        "supplies no template"
                    else
                        "Cannot remove"
                    end
                    @test_throws expected remove(m, address)
                end
                @test_throws message m(Xoshiro(1))
            end
            @test_throws message decondition(model(value))(Xoshiro(1))
        end
        message = "ArgumentError: Integer indexing into a NamedTuple at `x[1]` is unsupported; use `x.a` instead."
        for m in (
            local_integer_lhs(),
            condition(local_integer_lhs(), @varname(x[1]) => 3.0),
            fix(local_integer_lhs(), @varname(x[1]) => 3.0),
        )
            @test_throws message m(Xoshiro(1))
        end
        @test_throws "ArgumentError: Integer indexing into a NamedTuple at `x[1]` is unsupported; use a field name instead." integer_lhs((;))(
            Xoshiro(1)
        )
        @test_throws "ArgumentError: Integer indexing into a NamedTuple at `x[3]` is unsupported; use a field name instead." fix(
            integer_lhs((a=1.0, b=2.0)), @varname(x[3]) => 3.0
        )
        @model nested_integer_binding() = child ~ to_submodel(local_integer_lhs())
        @test_throws message nested_integer_binding()(Xoshiro(1))
        for bind in (condition, fix)
            @test_throws message bind(
                condition(local_integer_lhs(); x=(a=1.0, b=2.0)),
                VarNamedTuple((@varname(x[1]) => 3.0,)),
            )
        end
        @model named_lhs(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
        for bind in (condition, fix)
            m = bind(named_lhs((a=1.0, b=2.0)), @varname(x.a) => 3.0)
            for update in (condition, fix)
                @test_throws message update(m, @varname(x[1]) => 4.0)
            end
            for remove in (decondition, unfix)
                @test_throws "Cannot remove" remove(m, @varname(x[1]))
            end
        end
        for value in ([1.0, 2.0],),
            (bind, remove) in ((condition, decondition), (fix, unfix))

            m = bind(integer_lhs(value), @varname(x[1]) => 3.0)
            @test m(Xoshiro(1))[1] == 3.0
            expected = remove === unfix ? 1.0 : rand(Xoshiro(1), Normal())
            @test remove(m, @varname(x[1]))(Xoshiro(1))[1] == expected
        end
    end

    @testset "whole nothing arguments require decondition" begin
        @model function placeholder_array(x=nothing)
            x === nothing && (x = zeros(2))
            for i in 1:2
                x[i] ~ Normal()
            end
            return x
        end
        @model function placeholder_keyword_array(; x=nothing)
            x === nothing && (x = zeros(2))
            for i in 1:2
                x[i] ~ Normal()
            end
            return x
        end
        @model placeholder_scalar(y=nothing) = y ~ Normal()
        @model placeholder_keyword_scalar(; y=nothing) = y ~ Normal()
        for (models, vn, names, value) in (
                (
                    (placeholder_array(), placeholder_keyword_array()),
                    @varname(x),
                    [@varname(x[1]), @varname(x[2])],
                    [1.0, 2.0],
                ),
                (
                    (placeholder_scalar(), placeholder_keyword_scalar()),
                    @varname(y),
                    [@varname(y)],
                    1.0,
                ),
            ),
            model in models

            @test conditioned(model)[vn] === nothing
            @test_throws r"ArgumentError: .*nothing.*decondition" model(Xoshiro(1))
            latent = decondition(model, vn)
            vi = VarInfo(Xoshiro(1), latent)
            @test keys(vi) == names
            @test getloglikelihood(vi) == 0
            expected = if value isa Vector
                rand(Xoshiro(1), Normal(), 2)
            else
                rand(Xoshiro(1), Normal())
            end
            @test latent(Xoshiro(1)) == expected
            @test decondition(model)(Xoshiro(1)) == expected
            for (bind, remove) in ((condition, decondition), (fix, unfix))
                bound = bind(model, vn => value)
                @test bound(Xoshiro(1)) == value
                @test isempty(keys(VarInfo(Xoshiro(1), bound)))
                restored = remove(bound, vn)
                if remove === decondition
                    @test isempty(conditioned(restored))
                    @test keys(VarInfo(Xoshiro(1), restored)) == names
                else
                    @test conditioned(restored)[vn] === nothing
                    @test_throws r"ArgumentError: .*nothing.*decondition" restored(
                        Xoshiro(1)
                    )
                end
            end
        end
        partially_observed = condition(
            decondition(placeholder_array(zeros(2)), @varname(x)), @varname(x[1]) => 1.0
        )
        @test partially_observed(Xoshiro(1)) == [1.0, rand(Xoshiro(1), Normal())]
        @test keys(VarInfo(Xoshiro(1), partially_observed)) == [@varname(x[2])]
        @test isempty(fixed(unfix(placeholder_array(), :x)))

        @model container_field(p) = p.b ~ Normal()
        @model container_index(p) = p[2] ~ Normal()
        @model nothing_field(p) = p.a ~ Normal()
        @model nothing_index(p) = p[1] ~ Normal()
        for (unused, observed, value) in (
            (container_field, nothing_field, (a=nothing, b=1.0)),
            (container_index, nothing_index, [nothing, 1.0]),
        )
            @test conditioned(unused(value))[@varname(p)] === value
            @test getloglikelihood(VarInfo(Xoshiro(1), unused(value))) ==
                logpdf(Normal(), 1.0)
            @test_throws r"ArgumentError: .*nothing.*decondition" VarInfo(
                Xoshiro(1), observed(value)
            )
        end
        nested = placeholder_scalar(VarNamedTuple(; a=nothing, b=1.0))
        @test conditioned(nested)[@varname(y.a)] === nothing

        @model placeholder_parent(a=nothing) = (
            a ~ to_submodel(decondition(placeholder_scalar())); a
        )
        @model local_placeholder_parent() =
            a ~ to_submodel(decondition(placeholder_scalar()))
        parent = placeholder_parent()
        @test conditioned(parent)[@varname(a)] === nothing
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" VarInfo(
            Xoshiro(1), parent
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" parent(
            Xoshiro(1)
        )
        @test keys(VarInfo(Xoshiro(1), local_placeholder_parent())) == [@varname(a.y)]
        @test local_placeholder_parent()(Xoshiro(1)) == rand(Xoshiro(1), Normal())
        for bind in (condition, fix)
            bound = bind(parent; a=1.0)
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
                Xoshiro(1)
            )
            @test bind(local_placeholder_parent(), @varname(a.y) => 1.0)(Xoshiro(1)) == 1.0
        end

        direct = DynamicPPL.Model{false}(placeholder_array().f, (; x=nothing), (;))
        @test isempty(conditioned(direct))
        @test isempty(DynamicPPL._args_on_lhs(direct))
        @test keys(VarInfo(Xoshiro(1), direct)) == [@varname(x[1]), @varname(x[2])]
    end

    @testset "keyword splat partial bindings are explicit errors" begin
        @model keyword_lhs(; kw...) = (kw[:y] ~ Normal(); return kw[:y])
        for bind in (condition, fix)
            model = keyword_lhs(; y=1.0)
            for original in (model, decondition(model))
                @test_throws r"ArgumentError: .*keyword-splat argument `kw`.*cannot be bound" bind(
                    original, @varname(kw.y) => 2.0
                )
            end
            @test bind(model; kw=(; y=2.0))(Xoshiro(1)) == 2.0
        end
    end

    @testset "keyword splat argument LHS variables reject missing when read" begin
        @model keyword_observation(; kw...) = (kw[:x] ~ Normal(); kw)
        for value in ((; x=1.0, y=missing), (; x=1.0, y=[(a=missing,)]))
            @test keyword_observation(; value...)(Xoshiro(1))[:x] == 1.0
            for bind in (condition, fix), replacement in (value, pairs(value))
                @test bind(keyword_observation(; x=1.0); kw=replacement)(Xoshiro(1))[:x] ==
                    1.0
            end
        end
        model = keyword_observation(; x=missing)
        @test_throws r"ArgumentError: .*`kw\[:x\]`.*decondition" model(Xoshiro(1))
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            bound = bind(keyword_observation(; x=1.0); kw=(; x=missing))
            @test_throws "LHS variable `kw[:x]` contains `missing`; make it latent with `$remove`." bound(
                Xoshiro(1)
            )
        end
    end

    @testset "keyword splat index removal" begin
        @model indexed_keywords(; kwargs...) = (
            kwargs[:x] ~ Normal(); kwargs[:y] ~ Normal(); kwargs
        )
        observed = indexed_keywords(; x=2.0, y=5.0)
        removed = decondition(observed, @varname(kwargs[:x]))
        @test removed(Xoshiro(1))[:x] == rand(Xoshiro(1), Normal())
        @test removed(Xoshiro(1))[:y] == 5.0
        @test observed(Xoshiro(1))[:x] == 2.0
        for value in ((; x=3.0, y=4.0), pairs((; x=3.0, y=4.0)))
            m = fix(indexed_keywords(; x=2.0, y=5.0); kwargs=value)
            if value isa Base.Pairs
                @test_throws r"ArgumentError: .*kwargs.*Base.Pairs.*whole value" unfix(
                    m, @varname(kwargs[:x])
                )
            else
                @test loglikelihood(unfix(m, @varname(kwargs[:x])), (;)) ==
                    logpdf(Normal(), 2.0)
                @test loglikelihood(unfix(m, @varname(kwargs.x)), (;)) ==
                    logpdf(Normal(), 2.0)
            end
            @test loglikelihood(unfix(m, :kwargs), (;)) ==
                logpdf(Normal(), 2.0) + logpdf(Normal(), 5.0)
        end
    end

    @testset "deconditioned arguments retain index bounds" begin
        @model indexed_lhs_argument(x) = (x[1] ~ Normal(); x)
        for bind in (condition, fix), x in ([1.0, 2.0],)
            model = decondition(indexed_lhs_argument(x))
            invalid = DynamicPPL.@vnt begin
                x[3] := 9.0
            end
            for values in ((@varname(x[3]) => 9.0), invalid)
                @test_throws ArgumentError bind(model, values)
            end
            @test bind(model, @varname(x[1]) => 9.0)(Xoshiro(1))[1] == 9.0
        end
    end

    @testset "argument binding templates" begin
        @model template_lhs_variables(y) = (for i in eachindex(y)
            y[i] ~ Normal()
        end;
        y)
        for bind in (condition, fix),
            input in ((@varname(y[2]) => 3.0,), ((@varname(y[2]) => 3.0,),))

            m = @test_logs bind(template_lhs_variables([1.0, 2.0]), input...)
            @test m() == [1.0, 3.0]
        end
    end

    @testset "deconditioned dictionary argument storage" begin
        @testset "keys cannot retain latent storage" begin
            @model function key_storage(x, key)
                μ = x[key][1]
                x[key][1] ~ Normal(μ)
                return x
            end
            @model field_storage(x) = (x.a[1] ~ Normal(); x.a[1])
            for ctor in (Dict, IdDict)
                data = Real[0.0]
                key = Ref(data)
                model = decondition(key_storage(ctor(key => data), key))
                params = (; x=ctor(key => [2.0]))
                for _ in 1:2
                    @test_throws r"ArgumentError:.*argument `x`.*dictionary key" logjoint(
                        model, params
                    )
                    @test data == [0.0]
                end
                model = decondition(field_storage((; a=data, lookup=ctor(data => 1))))
                @test_throws r"ArgumentError:.*argument `x`.*dictionary key" returned(
                    model, (; x=(; a=[2.0]))
                )
                @test data == [0.0]
            end
        end

        @model scalar_dictionary(x) = (x[:a] ~ Normal(); x)
        for T in (Float32, Float64, BigFloat)
            data = Dict(:a => T(1))
            model = decondition(scalar_dictionary(data))
            result = returned(model, (; x=Dict(:a => T(2))))
            @test result == Dict(:a => T(2))
            @test result isa typeof(data)
            @test result !== data
            @test data[:a] == T(1)
        end

        @model function nested_dictionary(x, key)
            x[key][1] ~ Normal()
            return x
        end
        data = pairs((; a=[1.0, 3.0]))
        result = returned(decondition(nested_dictionary(data, :a)), (; x=(; a=[2.0, 3.0])))
        @test result isa typeof(data)
        @test result[:a] == [2.0, 3.0]
        @test data[:a] == [1.0, 3.0]

        @model dictionary_parent(child) = a ~ to_submodel(child)
        # Dictionary keys name storage; identity keys must remain usable in the body.
        for ctor in (Dict, IdDict), key in (Ref(:a), (Ref(:a),))
            data = ctor(key => [1.0, 3.0])
            model = decondition(nested_dictionary(data, key))
            params = (; x=ctor(key => [2.0, 3.0]))
            for (m, p) in ((model, params), (dictionary_parent(model), (; a=params)))
                result = returned(m, p)
                @test result[key] == [2.0, 3.0]
                @test result isa typeof(data)
                @test result !== data
                @test result[key] !== data[key]
                @test data[key] == [1.0, 3.0]
            end
        end
    end

    @testset "partly latent argument storage" begin
        for ctor in ((X, y) -> (; X, y),)
            d = ctor(zeros(1000, 1000), zeros(2))
            m = decondition(selective_copy(d), @varname(d.y))
            result, _ = init!!(
                Xoshiro(1),
                m,
                VarInfo(),
                InitFromParams((; m=0.0, d=(; y=ones(2)))),
                UnlinkAll(),
            )
            m(Xoshiro(1))
            @test (@allocated m(Xoshiro(1))) < sizeof(d.X) ÷ 2
            @test result.X === d.X
            @test result.y == ones(2)
            @test d.y == zeros(2)
            @test result.y !== d.y
        end
    end

    @testset "partial bindings preserve argument element types" begin
        @model elements(y) = (for i in eachindex(y)
            y[i] ~ Normal()
        end;
        y)
        @model typed_elements(y::Vector{Float64}) = (y[1] ~ Normal(); y)
        for bind in (condition, fix), constructor in (elements, typed_elements)
            result = bind(constructor([1.0, 2.0]), @varname(y[1]) => 1)
            @test result() isa Vector{Float64}
            @test result() == [1.0, 2.0]
            @test eltype(DynamicPPL._get_model_binding(result, @varname(y))) === Union{
                DynamicPPL.ModelValue{DynamicPPL.Condition,Float64},
                DynamicPPL.ModelValue{DynamicPPL.ArgumentCondition,Float64},
                DynamicPPL.ModelValue{DynamicPPL.Fix,Float64},
            }
            @test_throws r"ArgumentError: .*represent" bind(
                constructor([1.0, 2.0]), @varname(y[1]) => big(2)^100 + 1
            )()
        end
        @model wrapped_elements() = a ~ to_submodel(elements([1.0, 2.0]))
        @test condition(wrapped_elements(), @varname(a.y[1]) => 1)() isa Vector{Float64}
        for bind in (condition, fix)
            bindings = (@varname(y) => Float32[1, 2], @varname(y[1]) => 3)
            @test bind(elements([1.0, 2.0]), bindings...)() isa Vector{Float32}
            @test bind(elements([1.0, 2.0]), bindings)() isa Vector{Float32}
            inexact = (@varname(y) => [1.0, 2.0], @varname(y[1]) => big(2)^100 + 1)
            @test_throws r"ArgumentError: .*represent" bind(
                elements([1.0, 2.0]), inexact...
            )
            @test_throws r"ArgumentError: .*represent" bind(elements([1.0, 2.0]), inexact)
        end
        for bind in (condition, fix)
            @test_throws InexactError bind(elements([1, 2]), @varname(y[1]) => 1.5)
            @test_throws MethodError bind(elements([1, 2]), @varname(y[1]) => "invalid")
        end
        for T in (Float32, BigFloat)
            @test condition(elements(T[1, 2]), @varname(y[1]) => 3)() isa Vector{T}
        end
    end

    @testset "restore splatted argument-supplied observations" begin
        @model positional(args...) = (args[1] ~ Normal(); args)
        @model keywords(; kwargs...) = (
            kwargs = NamedTuple(kwargs); kwargs.x ~ Normal(); kwargs
        )
        for (m, value, whole, part) in (
            (positional(2.0), (; args=(3.0,)), :args, @varname(args[1])),
            (keywords(; x=2.0), (; kwargs=(; x=3.0)), :kwargs, @varname(kwargs.x)),
        )
            if whole === :args
                @test_throws ArgumentError unfix(fix(m; value...), part)
            end
            for name in (whole === :args ? (whole,) : (whole, part))
                restored = unfix(fix(m; value...), name)
                @test loglikelihood(restored, (;)) ≈ logpdf(Normal(), 2.0)
            end
        end
    end

    @testset "containing removal ranges" begin
        @model range_lhs_variables() = begin
            x = zeros(4)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
        end
        for (bind, remove, query) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))
            m = bind(range_lhs_variables(), @varname(x[1]) => 2.0)
            @test isempty(query(remove(m, @varname(x[1:3]))))
            m = bind(m, @varname(x[4]) => 4.0)
            @test keys(query(remove(m, @varname(x[1:3])))) == [@varname(x[4])]
        end
    end

    @testset "partly unbound LHS variables" begin
        @model mv_argument(x) = x ~ MvNormal(zeros(2), I)
        @model mv_local() = x ~ MvNormal(zeros(2), I)
        @test_throws r"ArgumentError: .*x.*whole" VarInfo(
            decondition(mv_argument([1.0, 2.0]), @varname(x[1]))
        )
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            @test_throws r"ArgumentError: .*x.*whole" VarInfo(
                remove(bind(mv_local(); x=[1.0, 2.0]), @varname(x[1]))
            )
            @test_throws r"ArgumentError: .*x.*whole" VarInfo(
                bind(mv_local(), @varname(x[2]) => 2.0)
            )
        end
    end

    @testset "nonexistent argument fields" begin
        @model nt_fields(p) = (m ~ Normal(); p.a ~ Normal(m); p.b ~ Normal(m))
        for bind in (condition, fix)
            m = nt_fields((; a=1.0, b=2.0))
            @test_throws r"ArgumentError: .*c.*p" bind(m, @varname(p.c) => 0.0)
            @test_throws r"ArgumentError: .*c.*p" bind(
                m, @varname(p.a) => 5.0, @varname(p.c) => 0.0
            )
            @test_throws r"ArgumentError: .*c.*p" bind(
                decondition(m, :p), @varname(p.c) => 0.0
            )
        end
    end

    @testset "fixed coverage errors" begin
        @model uncovered(x) = x[2] ~ Normal()
        @test_throws r"ArgumentError: .*x\[2\].*size and shape" fix(
            uncovered(Any[1.0]); x=Any[2.0]
        )()
        @model field_lhs(p) = p.a ~ Normal()
        @test_throws r"ArgumentError: .*p.a.*size and shape" fix(
            field_lhs((; a=1.0)); p=(; b=2.0)
        )()
        @model scalar_shape(x) = (x = [x]; x ~ MvNormal(zeros(1), 1.0))
        @test_throws r"ArgumentError: .*x.*size and shape" fix(scalar_shape(1.0); x=2.0)()
    end

    @testset "whole fixed bindings cover executed local LHS variables" begin
        @model function local_lhs_variables()
            x = zeros(2)
            for i in 1:2
                x[i] ~ Normal()
            end
            return x
        end
        @model nested_local_lhs_variables() = child ~ to_submodel(local_lhs_variables())
        for model in (
            fix(local_lhs_variables(); x=[2.0]),
            fix(nested_local_lhs_variables(), @varname(child.x) => [2.0]),
        )
            @test_throws r"ArgumentError: .*x\[2\].*coverage.*size and shape" model(
                Xoshiro(1)
            )
        end
        observed = condition(local_lhs_variables(); x=[2.0])
        @test observed(Xoshiro(1)) == [2.0, rand(Xoshiro(1), Normal())]
        @test keys(rand(Xoshiro(1), observed)) == [@varname(x[2])]
    end

    @testset "fixed argument shape" begin
        @model function changed_shape(x, change, index)
            x = change(x)
            x[index] ~ Normal()
            return x
        end
        @model function changed_range(x, change)
            x = change(x)
            x[:] ~ MvNormal(zeros(length(x)), 1.0)
            return x
        end
        for change in (x -> vcat(x, 2.0), x -> x[1:1], x -> reshape(x, 1, 2)),
            constructor in (x -> changed_shape(x, change, 1), x -> changed_range(x, change))

            m = constructor([1.0, 2.0])
            @test_throws r"ArgumentError: .*x.*(size|shape)" fix(m; x=[3.0, 4.0])()
            @test condition(m; x=[3.0, 4.0])() == change([3.0, 4.0])
        end
        @test_throws r"ArgumentError: .*x.*size and shape" fix(
            changed_shape((1.0, 2.0), x -> (x..., 3.0), 1); x=(3.0, 4.0)
        )()
        @model function changed_field(p)
            p = (; a=reshape(p.a, 1, 2))
            p.a[1] ~ Normal()
            return p
        end
        @test_throws r"ArgumentError: .*p.a\[1\].*(size|shape)" fix(
            changed_field((; a=[1.0, 2.0])); p=(; a=[3.0, 4.0])
        )()
    end

    @testset "whole $value arguments reject body replacement" for value in
                                                                  (missing, nothing)
        @model function missing_placeholder(x=missing, ::Type{T}=Float64) where {T}
            if x isa Union{Missing,Nothing}
                x = Vector{T}(undef, 2)
                fill!(x, 7)
            end
            s ~ Normal()
            for i in eachindex(x)
                x[i] ~ Normal(s, 1)
            end
            return x
        end
        model = missing_placeholder(value)
        @test_throws r"ArgumentError: .*`x\[1\]`.*(missing|nothing).*decondition" model(
            Xoshiro(1)
        )
        @test_throws r"ArgumentError: .*`x\[1\]`.*(missing|nothing).*decondition" loglikelihood(
            model, (; s=0.0)
        )
        @test keys(rand(Xoshiro(1), decondition(model))) ==
            [@varname(s), @varname(x[1]), @varname(x[2])]
        @test keys(rand(Xoshiro(1), decondition(model, @varname(x)))) ==
            [@varname(s), @varname(x[1]), @varname(x[2])]
        @test condition(model; x=[1.0, 2.0])(Xoshiro(1)) == [1.0, 2.0]

        @model function missing_keyword(; x)
            x isa Union{Missing,Nothing} && (x = (a=7.0,))
            return x.a ~ Normal()
        end
        @test_throws r"ArgumentError: .*`x.a`.*(missing|nothing).*decondition" missing_keyword(;
            x=value
        )(
            Xoshiro(1)
        )
        @model nested_missing() = child ~ to_submodel(missing_placeholder(value))
        @test_throws r"ArgumentError: .*`child.x\[1\]`.*(missing|nothing).*decondition" nested_missing()(
            Xoshiro(1)
        )
        @model unchanged_missing(x) = x ~ Normal()
        @test_throws "LHS variable `x` contains `$value`; make it latent with `decondition`." unchanged_missing(
            value
        )(
            Xoshiro(1)
        )
        @model unread_missing(x, read) = read ? (x ~ Normal()) : x
        @test unread_missing(value, false)(Xoshiro(1)) === value
    end

    @testset "placeholders are rejected only when an LHS variable reads them" begin
        @model metadata_lhs(p) = (p.a ~ Normal(); p.a)
        @model indexed_observation(y) = begin
            for i in 1:2
                y[i] ~ Normal()
            end
            y
        end
        @model whole_observation(y) = y ~ MvNormal(zeros(2), I)
        @model scalar_observation(y) = y ~ Normal()
        @model product_observation(y) = y ~ product_distribution([Normal(), Normal()])
        for p in (
            (a=1.0, b=missing),
            (a=1.0, b=nothing),
            (a=1.0, b=[(missing,)]),
            MetadataRecord(1.0, missing),
        )
            @test metadata_lhs(p)(Xoshiro(1)) == 1.0
            for bind in (condition, fix)
                @test bind(metadata_lhs((a=1.0, b=2.0)); p)(Xoshiro(1)) == 1.0
            end
        end
        for bind in (
            identity,
            m -> condition(m; y=[1.0, 2.0, missing]),
            m -> fix(m; y=[1.0, 2.0, missing]),
        )
            @test isequal(
                bind(indexed_observation([1.0, 2.0, missing]))(Xoshiro(1)),
                [1.0, 2.0, missing],
            )
        end
        for absent in (missing, nothing),
            (constructor, values, vn) in (
                (metadata_lhs, (; p=(a=absent, b=1.0)), @varname(p.a)),
                (indexed_observation, (; y=[1.0, absent]), @varname(y[2])),
                (whole_observation, (; y=[1.0, absent]), @varname(y)),
                (product_observation, (; y=[1.0, absent]), @varname(y)),
                (scalar_observation, (; y=absent), @varname(y)),
            )

            message = "ArgumentError: LHS variable `$vn` contains `$absent`; make it latent with `decondition`."
            model = constructor(only(values))
            @test_throws message model(Xoshiro(1))
            for bind in (condition, fix)
                bound = bind(model; values...)
                diagnostic = if bind === fix
                    replace(message, "decondition" => "unfix")
                else
                    message
                end
                @test_throws diagnostic bound(Xoshiro(1))
            end
        end
        @test decondition(scalar_observation(missing))(Xoshiro(1)) isa Real
    end

    @testset "named tuple LHS variables require whole bindings" begin
        @model named_lhs() = x ~ product_distribution((a=Normal(), b=Normal()))
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            partial = bind(named_lhs(), @varname(x.a) => 1.0)
            @test_throws r"ArgumentError: .*`x`.*bound as a whole" partial(Xoshiro(1))
            whole = bind(named_lhs(); x=(; a=1.0, b=2.0))
            @test whole(Xoshiro(1)) == (; a=1.0, b=2.0)
            @test bind(whole, @varname(x.a) => 3.0)(Xoshiro(1)) == (; a=3.0, b=2.0)
            @test_throws r"ArgumentError: .*`x`.*bound as a whole" remove(
                whole, @varname(x.b)
            )(
                Xoshiro(1)
            )
        end
    end

    @testset "expanded partial bindings validate their extent" begin
        @model indexed(y) = (y[1] ~ Normal(); y[2] ~ Normal(); y)
        for data in ([1.0, 2.0],), bind in (condition, fix)
            model = bind(indexed(data), @varname(y[1]) => 4.0)
            @test_throws r"ArgumentError: .*`y\[3\]`.*outside.*`y`" bind(
                model, @varname(y[3]) => 9.0
            )
            @test_throws r"ArgumentError: .*outside" bind(
                model, @varname(y[2:3]) => [8.0, 9.0]
            )
            @test bind(model, @varname(y[2]) => 5.0)(Xoshiro(1)) == [4.0, 5.0]
        end
    end

    @testset "ordinary arguments cannot be bound" begin
        @model ordinary(n; scale=1.0) = x ~ Normal(n, scale)
        @model dispatched(x::Real) = x ~ Normal()
        @model dispatched(x::AbstractVector) = y ~ Normal(sum(x))
        for bind in (condition, fix)
            for model in
                (ordinary(1), decondition(ordinary(1)), setthreadsafe(ordinary(1), true))
                for values in ((; n=2), (; n=nothing), (@varname(n) => 2,))
                    @test_throws r"ArgumentError: .*`n`.*left-hand side of `~`.*construct" bind(
                        model, values
                    )
                end
                @test_throws r"ArgumentError: .*`scale`.*left-hand side of `~`" bind(
                    model; scale=2.0
                )
            end
            @test_throws r"ArgumentError: .*`n`.*left-hand side of `~`" bind(
                prefix(ordinary(1), @varname(a)), @varname(a.n) => 2
            )
            @test bind(dispatched(1); x=2)(Xoshiro(1)) == 2
            @test_throws r"ArgumentError: .*`x`.*left-hand side of `~`" bind(
                dispatched([1]); x=[2]
            )
            @test bind(ordinary(1); x=2)(Xoshiro(1)) == 2
            @test_throws ArgumentError bind(ordinary(1); dynamic_lhs=2)
        end
    end

    @testset "removal of valid addresses is idempotent" begin
        @model scalar() = x ~ Normal()
        @model inner_arg(x=1.0) = x ~ Normal()
        @model outer_arg() = a ~ to_submodel(inner_arg())
        for (bind, remove) in ((condition, unfix), (fix, decondition)),
            scope in ((), (DynamicPPL.Recursive(),))

            @test remove(bind(scalar(); x=1.0), scope..., @varname(x))(Xoshiro(1)) == 1.0
            @test keys(rand(Xoshiro(1), remove(scalar(), scope..., @varname(x)))) ==
                [@varname(x)]
            @test_throws r"ArgumentError: .*`unknown`" remove(
                scalar(), scope..., @varname(unknown)
            )
            @test isempty(keys(conditioned(remove(scalar()))))
            @test isempty(keys(fixed(remove(scalar()))))
        end
        @model indexed(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        for data in ([1.0, 2.0],),
            (bind, remove, select) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))

            partial = remove(bind(indexed(data); x=data), @varname(x[2]))
            @test isempty(select(remove(partial, @varname(x[1:2][1]))))
            @test isempty(select(remove(partial, @varname(x[1:2]))))
            @test select(remove(partial, @varname(x[2]))) == select(partial)
        end
        @test decondition(outer_arg(), @varname(a.x))(Xoshiro(1)) == 1.0
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            parent = bind(outer_arg(), @varname(a.x) => 2.0)
            @test remove(parent, @varname(a.x))(Xoshiro(1)) == 1.0
            @test remove(remove(parent, @varname(a.x)), @varname(a.x))(Xoshiro(1)) == 1.0
        end
    end

    @testset "nested slices retain bound siblings" begin
        @model sliced(x) = (x[1:2][1:2][1] ~ Normal(); x[2] ~ Normal(); x)
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            model = remove(bind(sliced([1.0, 2.0]); x=[1.0, 2.0]), @varname(x[2]))
            @test model(Xoshiro(1))[1] == 1.0
            @test loglikelihood(model, (; x=[3.0, 4.0])) ==
                (bind === condition ? logpdf(Normal(), 1.0) : logpdf(Normal(), 2.0))
            @test keys(VarInfo(Xoshiro(1), model)) ==
                (bind === condition ? [@varname(x[2])] : VarName[])
        end
    end

    @testset "accessors return plain partial values" begin
        @model function named_lhs_variables()
            x = (; a=0.0, b=0.0)
            x = (; a=(x.a ~ Normal()), b=(x.b ~ Normal()))
            return x
        end
        @model function tuple_lhs_variables()
            x = (0.0, 0.0)
            x = ((x[1] ~ Normal()), (x[2] ~ Normal()))
            return x
        end
        @model function nested_lhs_variables()
            x = [(; a=0.0, b=0.0)]
            x[1].a ~ Normal()
            x[1].b ~ Normal()
            return x
        end
        @model array_lhs_variables(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        @model inner2() = (x ~ Normal(); y ~ Normal(); (; x, y))
        @model outer() = a ~ to_submodel(inner2())

        for (bind, remove, select, other) in
            ((condition, decondition, conditioned, fix), (fix, unfix, fixed, condition))
            for (name, unbound, whole, removed, retained, template) in (
                (
                    "named tuple",
                    named_lhs_variables(),
                    (; x=(; a=1.0, b=2.0)),
                    @varname(x.a),
                    @varname(x.b),
                    NoTemplate(),
                ),
                (
                    "nested",
                    nested_lhs_variables(),
                    (; x=[(; a=1.0, b=2.0)]),
                    @varname(x[1].a),
                    @varname(x[1].b),
                    [(; a=0.0, b=0.0)],
                ),
                (
                    "array argument",
                    decondition(array_lhs_variables([1.0, 2.0])),
                    (; x=[1.0, 2.0]),
                    @varname(x[1]),
                    @varname(x[2]),
                    zeros(2),
                ),
                (
                    "submodel namespace",
                    outer(),
                    (; a=(; x=1.0, y=2.0)),
                    @varname(a.x),
                    @varname(a.y),
                    NoTemplate(),
                ),
            )
                @testset "$bind $name" begin
                    complete = bind(unbound, whole)
                    root = first(keys(whole)) === :x ? @varname(x) : @varname(a)
                    @test select(complete)[root] === first(whole)
                    expected = DynamicPPL.templated_setindex!!(
                        VarNamedTuple(), 2.0, retained, template
                    )
                    partial = remove(complete, removed)
                    @test select(partial) == select(bind(unbound, expected))
                    @test select(partial) == expected
                    @test select(partial)[retained] == 2.0
                    if template isa NoTemplate
                        @test select(partial)[root] isa VarNamedTuple
                    else
                        @test select(partial).data[first(keys(whole))] isa
                            DynamicPPL.VarNamedTuples.PartialArray
                    end
                    mixed = other(complete, removed => 3.0)
                    @test select(mixed) == (bind === fix ? select(complete) : expected)
                    @test select(remove(mixed, retained)) == (
                        bind === fix ? select(remove(complete, retained)) : VarNamedTuple()
                    )
                    for original in (complete, partial, mixed)
                        rebuilt = fix(
                            condition(unbound, conditioned(original)), fixed(original)
                        )
                        result, vi = init!!(
                            Xoshiro(42), original, VarInfo(), InitFromPrior(), UnlinkAll()
                        )
                        restored, restored_vi = init!!(
                            Xoshiro(42), rebuilt, VarInfo(), InitFromPrior(), UnlinkAll()
                        )
                        @test restored == result
                        @test getlogprior(restored_vi) == getlogprior(vi)
                        @test getloglikelihood(restored_vi) == getloglikelihood(vi)
                    end
                end
            end
        end
    end

    @testset "removal resolves container addresses" begin
        @model function matrix_lhs_variables(x)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        @model field_lhs_variables(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
        @model nested_field_lhs_variables(x) = (x[1].a ~ Normal(); x[1].b ~ Normal(); x)
        for (bind, remove, select) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))
            matrix = bind(matrix_lhs_variables([1.0 3.0; 2.0 4.0]); x=[1.0 3.0; 2.0 4.0])
            for (vn, indices) in (
                (@varname(x[2]), [2]),
                (@varname(x[2, 1]), [2]),
                (@varname(x[2:3]), [2, 3]),
                (@varname(x[begin]), [1]),
                (@varname(x[end]), [4]),
                (@varname(x[:, 1]), [1, 2]),
                (@varname(x[:]), [1, 2, 3, 4]),
                (@varname(x[2:4][2]), [3]),
            )
                changed = remove(matrix, vn)
                latent = rand(Xoshiro(1), changed)
                for i in 1:4
                    @test haskey(latent, @varname(x[i])) ==
                        (bind === condition && i in indices)
                end
                result = changed(Xoshiro(1))
                @test result[setdiff(1:4, indices)] ==
                    [1.0, 2.0, 3.0, 4.0][setdiff(1:4, indices)]
            end
            partial = remove(matrix, @varname(x[2]))
            @test !haskey(select(remove(partial, @varname(x[2:3]))), @varname(x[3]))
            @test_throws ArgumentError remove(matrix, @varname(z))
            @test_throws "Cannot remove `x[8]`" remove(matrix, @varname(x[8]))
            unbound = remove(matrix, @varname(x))
            @test select(remove(unbound, @varname(x[3]))) == select(unbound)
            for (model, first, sibling) in (
                (
                    field_lhs_variables(ComponentVector(; a=1.0, b=2.0)),
                    @varname(x.a),
                    @varname(x.b)
                ),
                (
                    nested_field_lhs_variables([ComponentVector(; a=1.0, b=2.0)]),
                    @varname(x[1].a),
                    @varname(x[1].b)
                ),
            )
                bound = bind(model; x=model.args.x)
                changed = remove(bound, first)
                @test haskey(rand(Xoshiro(1), changed), first) == (bind === condition)
                @test !haskey(select(changed), first)
                @test select(changed)[sibling] == 2.0
                @test select(bound)[sibling] == 2.0
            end
        end
    end

    @testset "unfix restores the last fixed LHS variable's argument-supplied observation" begin
        @model restored_indices(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        for original in ([1.0, 2.0],)
            bound = fix(decondition(restored_indices(original)), @varname(x[1]) => 5.0)
            for names in ((), (@varname(x),), (@varname(x[1]),))
                restored = unfix(bound, names...)
                @test collect(restored(Xoshiro(1))) == rand(Xoshiro(1), Normal(), 2)
            end
        end
        @model restored_fields(x) = (x.a[1] ~ Normal(); x.a[2] ~ Normal(); x)
        restored = unfix(
            fix(decondition(restored_fields((a=[1.0, 2.0],))), @varname(x.a[1]) => 5.0)
        )
        @test restored(Xoshiro(1)).a == rand(Xoshiro(1), Normal(), 2)
    end

    @testset "invalid field and index addresses" begin
        @model function elements(y)
            for i in eachindex(y)
                y[i] ~ Normal()
            end
        end
        for bind in (condition, fix)
            @test_throws "Cannot bind `y[0]`: index is outside the storage at `y`" bind(
                elements(nothing), @varname(y[0]) => 9.0
            )
            @test_throws r"ArgumentError: .*y\[3\]" bind(
                elements([1.0, 2.0]), @varname(y[3]) => 9.0
            )
            @test_throws r"ArgumentError: .*y\[3\]" bind(
                elements((1.0, 2.0)), @varname(y[3]) => 9.0
            )
        end
    end

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
            op(model, (@varname(x) => x,)),
            op(model, @varname(x) => x),
        )
        for transformed in transformed_models
            test_logp_correct(op, transformed, x)
        end
        if op === condition
            test_logp_correct(condition, model | VarNamedTuple(; x), x)
            test_logp_correct(condition, model | (; x), x)
            test_logp_correct(condition, model | (@varname(x) => x,), x)
            test_logp_correct(condition, model | (@varname(x) => x), x)
        end
    end

    @testset "later bindings replace values within their layer" begin
        @model return_x() = x ~ Normal()
        for first_op in (condition, fix), last_op in (condition, fix)
            transformed = last_op(first_op(return_x(); x=1.0); x=2.0)
            isfixed = first_op === fix || last_op === fix
            @test transformed() == (first_op === fix && last_op === condition ? 1.0 : 2.0)
            @test logjoint(transformed, VarNamedTuple()) ==
                (isfixed ? 0.0 : logpdf(Normal(), 2.0))
            @test isempty(conditioned(transformed)) == isfixed
            @test isempty(fixed(transformed)) == !isfixed
        end
    end

    @testset "partial bindings must cover whole LHS variables" begin
        @model whole_vector() = x ~ MvNormal(zeros(2), I)
        @model function ranged_vector()
            x = zeros(3)
            x[1:3] ~ MvNormal(zeros(3), I)
            return x
        end
        @model function disjoint_vector()
            x = zeros(4)
            x[2:4] ~ MvNormal(zeros(3), I)
            return x
        end
        templated = DynamicPPL.@vnt begin
            @template x = zeros(3)
            x[1] := 5.0
        end
        for bind in (condition, fix)
            for model in (whole_vector(), ranged_vector()),
                values in (
                    (@varname(x[1]) => 5.0,),
                    (templated,),
                    (@varname(x[1]) => 5.0, @of(x = of(Array, 3))),
                )

                @test_throws ArgumentError bind(model, values...)(Xoshiro(1))
            end
            @test_throws ArgumentError bind(ranged_vector(), @varname(x[1:2]) => (5.0, 6.0))(
                Xoshiro(1)
            )
            @test bind(disjoint_vector(), @varname(x[1]) => 5.0)(Xoshiro(1)) ==
                disjoint_vector()(Xoshiro(1))
            @test bind(ranged_vector(), @varname(x[1:3]) => ones(3))(Xoshiro(1)) == ones(3)
        end
    end

    @testset "one tilde cannot mix binding roles" begin
        @model joint() = x ~ MvNormal(zeros(2), I)
        for (first_op, last_op) in ((condition, fix), (fix, condition))
            mixed = last_op(first_op(joint(); x=[1.0, 2.0]), @varname(x[1]) => 3.0)
            if first_op === condition
                @test_throws r"ArgumentError: .*condition and fix different parts" mixed(
                    Xoshiro(1)
                )
            else
                @test mixed(Xoshiro(1)) == [1.0, 2.0]
            end
        end
    end

    @testset "unused bindings are ignored" begin
        @model function optional_lhs(active)
            x ~ Normal()
            if active
                y ~ Normal()
            end
            return x
        end
        unused_model = optional_lhs(false)
        for bind in (condition, fix), name in (@varname(y),), data in (1.0, missing)
            bound = bind(unused_model, name => data)
            value, vi = init!!(Xoshiro(1), unused_model, VarInfo(), InitFromPrior())
            bound_value, bound_vi = init!!(Xoshiro(1), bound, VarInfo(), InitFromPrior())
            @test bound_value == value
            @test getlogjoint(bound_vi) == getlogjoint(vi)
        end
    end

    @testset "missing struct fields" begin
        @model field_observation(x) = x.a ~ Normal()
        @model nested_field(child) = inner ~ to_submodel(child)
        for (wrap, message) in (
            (identity, r"ArgumentError: .*`x.a`.*missing.*decondition"),
            (nested_field, r"ArgumentError: .*`inner.x.a`.*missing.*decondition"),
        )
            @test_throws message logjoint(
                wrap(field_observation(MissingRecord(missing))), (;)
            )
            for op in (condition, fix)
                field_model = op(
                    field_observation(MissingRecord(1.0)); x=MissingRecord(missing)
                )
                diagnostic = if op === fix
                    Regex(replace(message.pattern, "decondition" => "unfix"))
                else
                    message
                end
                @test_throws diagnostic logjoint(wrap(field_model), (;))
            end
        end
    end

    @testset "argument storage can be initialized in the model body" begin
        @model function initialize_argument(x)
            fill!(x, [1.0])
            x[1] ~ MvNormal([0.0], [1.0;;])
            return x
        end
        for wrap in (identity, x -> view(x, :))
            data = wrap(Vector{Vector{Float64}}(undef, 1))
            initialized_model = initialize_argument(data)
            @test !isassigned(data, 1)
            @test initialized_model() == [[1.0]]
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
            @test changed()[1] == (first_op === fix && last_op === condition ? 1.0 : 3.0)
            @test changed()[2, 2] == 2.0
            @test original()[1] == 1.0
            replaced = last_op(
                first_op(indexed(), @varname(x[1]) => 3.0); x=[1.0 0.0; 0.0 2.0]
            )
            @test replaced(Xoshiro(1)) == (
                if first_op === fix && last_op === condition
                    [3.0 0.0; 0.0 2.0]
                else
                    [1.0 0.0; 0.0 2.0]
                end
            )
            @test isempty(conditioned(replaced)) == (last_op === fix)
            @test isempty(fixed(replaced)) ==
                (first_op === condition && last_op === condition)
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ==
                (first_op === condition ? logpdf(Normal(), 2.0) : 0.0) + (
                if first_op === condition && last_op === condition
                    logpdf(Normal(), 3.0)
                else
                    0.0
                end
            )
            original = first_op(properties(); x=(; a=1.0, b=2.0))
            changed = last_op(original, @varname(x.a) => 3.0)
            @test changed() ==
                (; a=(first_op === fix && last_op === condition ? 1.0 : 3.0), b=2.0)
            @test original() == (; a=1.0, b=2.0)
        end
    end

    @testset "fixed arguments require coverage of executed LHS variables" begin
        @model grow_scalar(x) = (x = vcat(x, 2.0); x[2] ~ Normal(); return x)
        @model grow_range(x) = (x = vcat(x, 2.0); x[1:2] ~ MvNormal(zeros(2), I); return x)
        for model in (grow_scalar([1.0]), grow_range([1.0]))
            @test_throws r"ArgumentError: .*`x\[(2|1:2)\]`.*size and shape" fix(
                model; x=[3.0]
            )()
            @test condition(model; x=[3.0])() == [3.0, 2.0]
        end
        @test_throws r"ArgumentError: .*`x\[2\]`.*size and shape" fix(
            grow_scalar([1.0]); x=[3.0, 4.0]
        )()
        @test_throws r"Cannot condition and fix different parts" fix(
            grow_range([1.0, 2.0]), @varname(x[1]) => 3.0
        )()
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
                changed_model, VarInfo(), InitFromParams((; x=[7.0, 7.0])), UnlinkAll()
            )
            @test result.before[i] == result.x[i] == 4.0
            @test result.before[3 - i] == 1.0
            @test result.x[3 - i] == 7.0
            @test getloglikelihood(vi) ≈ logpdf(Normal(), 4.0)
            @test data == [0, 0]
        end
        partial_loglik =
            p -> loglikelihood(
                condition(
                    decondition(partial_input(zeros(typeof(p), 2))), @varname(x[1]) => p
                ),
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
            expected = reshape(
                [
                    [T(first_op === fix && third_op === condition ? 10 : 30)],
                    [T(20)],
                    [T(3)],
                    [T(4)],
                ],
                2,
                2,
            )
            @test third() == expected
            expected_parent = reshape([[T(30)], [T(20)], [T(3)], [T(4)]], 2, 2)
            @test third_op(outer_array(second), @varname(a.x[1][1]) => T(30))() ==
                expected_parent
            @test first()[2][1] == T(2)
            @test original() == data
            @test loglikelihood(third, VarNamedTuple()) ≈
                (
                      if first_op === condition && third_op === condition
                          logpdf(Normal(), T(30))
                      else
                          zero(T)
                      end
                  ) +
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
        result, _ = init!!(partial, VarInfo(), InitFromParams((; x=data)), UnlinkAll())
        @test result == reshape([[10.0f0], [20.0f0], [3.0f0], [4.0f0]], 2, 2)

        observations = DynamicPPL.@vnt begin
            @template x = data
            x[1][1] := 10.0f0
        end
        partial = @test_logs condition(decondition(nested_array(data)), observations)
        @test size(conditioned(partial).data.x) == (2, 2)
    end

    @testset "invalid bindings below scalar values" begin
        @model scalar_lhs(a) = a ~ Normal()
        @model scalar_child() = x ~ Normal()
        @model scalar_return(a) = a ~ to_submodel(scalar_child())
        for bind in (condition, fix)
            @test_throws r"ArgumentError: .*`a`.*decondition" bind(
                scalar_lhs(1.0), @varname(a[1]) => 2.0
            )
            @test_throws r"ArgumentError: .*`a`.*decondition" bind(
                scalar_return(0.0), @varname(a.x) => 2.0
            )
        end
    end

    @testset "partial removal expands whole bindings" begin
        @model function partial_observations(x)
            m ~ Normal()
            for i in eachindex(x)
                x[i] ~ Normal(m)
            end
        end
        @model record_observations(t=(; a=1.0, b=2.0)) = (t.a ~ Normal(); t.b ~ Normal(); t)
        for (bind, remove, select) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))
            original = bind(partial_observations([1.0, 2.0]); x=[1.0, 2.0])
            latent = remove(original, @varname(x[2]))
            @test Set(keys(VarInfo(Xoshiro(1), latent))) ==
                Set(bind === condition ? [@varname(m), @varname(x[2])] : [@varname(m)])
            @test loglikelihood(latent, (; m=0.0, x=[1.0, 3.0])) ≈
                (bind === condition ? logpdf(Normal(), 1.0) : logpdf(Normal(), 2.0))
            @test select(original)[@varname(x)] == [1.0, 2.0]
            @test_throws ArgumentError remove(original, @varname(absent))
            record = bind(record_observations(); t=(; a=1.0, b=2.0))
            @test !haskey(select(remove(record, @varname(t.a))), @varname(t.a))
            @test_throws "Cannot remove `t.absent`" remove(record, @varname(t.absent))
            unbound = remove(record, @varname(t))
            @test select(remove(unbound, @varname(t.a))) == select(unbound)
        end
        original = condition(record_observations(); t=(; a=1.0, b=2.0))
        expanded = fix(original, @varname(t.b) => 5.0)
        for model in (original, expanded)
            latent = decondition(model, @varname(t.a))
            @test keys(VarInfo(Xoshiro(1), latent)) == [@varname(t.a)]
        end
        @model nested_removal(x) = (x[1].a[2] ~ Normal(); x)
        original = nested_removal([(; a=[1.0, 2.0])])
        @test keys(VarInfo(Xoshiro(1), decondition(original, @varname(x[1].a[2])))) ==
            [@varname(x[1].a[2])]
    end

    @testset "argument-supplied observations can be replaced and removed" begin
        @model argument_model(x) = x ~ Normal()
        for initial in (1.0f0, 1.0, big"1.0")
            original = argument_model(initial)
            @test original() === initial
            @test isempty(keys(VarInfo(original)))
            @test conditioned(original)[@varname(x)] === initial
            @test loglikelihood(original, VarNamedTuple()) == logpdf(Normal(), initial)
            observed = condition(argument_model(initial); x=2.0)
            @test observed() == 2.0
            @test logjoint(observed, VarNamedTuple()) == logpdf(Normal(), 2.0)
            @test keys(VarInfo(Xoshiro(1), decondition(observed))) == [@varname(x)]
        end

        @model function array_argument(x; config=nothing)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        for data in ([1.0f0, 2.0f0], BigFloat[1, 2], DimArray([1.0, 2.0], X))
            original = array_argument(data)
            @test keys(conditioned(original)) == [@varname(x)]
            latent = decondition(original, @varname(x))
            result, _ = init!!(
                latent, VarInfo(), InitFromParams((; x=[3.0, 4.0])), UnlinkAll()
            )
            @test result == [3.0, 4.0]
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
            @test changed(Xoshiro(1)) === data
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ≈
                (op === condition ? sum(logpdf.(Normal(), data)) : 0.0)
            @test original() == zeros(2)
        end
        loglik =
            x -> loglikelihood(condition(indexed_argument(zeros(2)); x), VarNamedTuple())
        @test ForwardDiff.gradient(loglik, [1.0, 2.0, 3.0]) ≈ [-1.0, -2.0, -3.0]

        @model function read_before_tilde(; x=1.0)
            y ~ Normal(x)
            x ~ Normal()
            return x
        end
        for op in (condition, fix)
            changed = op(read_before_tilde(); x=2.0)
            @test logprior(changed, (; y=2.0)) == logpdf(Normal(2.0), 2.0)
        end
        logp = x -> logprior(condition(read_before_tilde(); x), (; y=2.0))
        @test ForwardDiff.derivative(logp, 1.0) == 1.0
    end

    @testset "property overrides retain replacement containers" begin
        @model fields(x) = (x.a ~ Normal(); return x)
        @model nested_fields(m) = child ~ to_submodel(m)
        for first_op in (condition, fix),
            last_op in (condition, fix),
            (original, replacement) in (
                ((; a=0.0f0), (; a=1.0f0, b=2.0f0)),
                ((; a=big"0", b=big"0"), (; a=big"1", b=big"2")),
            )

            base = first_op(fields(original); x=replacement)
            changed = last_op(base, @varname(x.a) => oftype(replacement.a, 3))
            result = changed()
            @test typeof(result) === typeof(replacement)
            @test result.a == (first_op === fix && last_op === condition ? 1 : 3)
            @test result.b == 2
            @test base().a == replacement.a == 1
            @test loglikelihood(changed, VarNamedTuple()) ≈ (
                first_op === condition && last_op === condition ? logpdf(Normal(), 3) : 0
            )
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
                VarInfo(),
                InitFromParams((; x=(; a=oftype(replacement.a, 7)))),
                UnlinkAll(),
            )
            @test typeof(result) === typeof(replacement)
            @test result.a == (
                if last_op === condition
                    (first_op === fix ? 1 : 7)
                else
                    (first_op === condition ? 1 : original.a)
                end
            )
            @test result.b == 2
        end
        parent = condition(
            nested_fields(fields((; a=0.0))), @varname(child.x) => (; a=1.0, b=2.0)
        )
        parent = fix(parent, @varname(child.x.a) => 3.0)
        result, _ = @inferred evaluate!!(
            parent,
            InitContext(
                InitFromParams(VarNamedTuple(), nothing),
                DynamicPPL.infer_transform_strategy_from_values(VarNamedTuple()),
            ),
            VarInfo(),
        )
        @test result == (; a=3.0, b=2.0)

        @model array_fields(x) = (x[1].a ~ Normal(); return x)
        parent = condition(
            nested_fields(array_fields(Any[(; a=0.0)])),
            @varname(child.x[1]) => (; a=1.0, b=2.0),
        )
        @test fix(parent, @varname(child.x[1].a) => 3.0)() == [(; a=3.0, b=2.0)]
        partial = condition(
            decondition(array_fields(Any[(; a=0.0)])), @varname(x[1]) => (; a=1.0, b=2.0)
        )
        @test fix(partial, @varname(x[1].a) => 3.0)() == [(; a=3.0, b=2.0)]

        replaced = fix(fields(ObservationRecord(1.0, 2.0)); x=(a=3.0, b=4.0))
        @test fix(replaced, @varname(x.a) => 5.0)(Xoshiro(1)) == (a=5.0, b=4.0)
        @test unfix(replaced, @varname(x.a))(Xoshiro(1)) == (a=1.0, b=4.0)

        @model elements(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        replaced_tuple = fix(elements((1.0, 2.0)); x=[3.0, 4.0])
        @test fix(replaced_tuple, @varname(x[1]) => 5.0)(Xoshiro(1)) == [5.0, 4.0]
        for scope in ((), (DynamicPPL.Recursive(),))
            @test unfix(replaced_tuple, scope..., @varname(x[1]))(Xoshiro(1)) == [1.0, 4.0]
            @test unfix(replaced, scope..., @varname(x.a))(Xoshiro(1)) == (a=1.0, b=4.0)
        end
        @test unfix(replaced_tuple, @varname(x))(Xoshiro(1)) == (1.0, 2.0)
        tuple_loglik =
            p -> loglikelihood(
                unfix(fix(elements((p, 2p)); x=[3p, 4p]), @varname(x[1])), (;)
            )
        @test tuple_loglik(2.0) ≈ logpdf(Normal(), 2.0)
        @test ForwardDiff.derivative(tuple_loglik, 2.0) ≈ -2.0

        base = condition(fields(ObservationRecord(0.0, 0.0)); x=ReplacementRecord(1.0, 2.0))
        loglik =
            p -> loglikelihood(
                condition(
                    fields(ReplacementRecord(zero(p), zero(p)));
                    x=ReplacementRecord(p, zero(p)),
                ),
                VarNamedTuple(),
            )
        @test ForwardDiff.derivative(loglik, 3.0) == -3.0
        @test_throws ArgumentError condition(base, @varname(x.a) => 3.0)
        selected = conditioned(condition(base; x=ReplacementRecord(3.0, 2.0)))
        @test conditioned(base)[@varname(x)] isa ReplacementRecord
        @test condition(fields(ObservationRecord(0.0, 0.0)), conditioned(base))() isa
            ReplacementRecord
        @test selected[@varname(x)] isa ReplacementRecord
        @test condition(fields(ObservationRecord(0.0, 0.0)), selected)().a == 3.0
        @test_throws ArgumentError fix(base, @varname(x.a) => 3.0)
        mixed = fix(base; x=ReplacementRecord(3.0, 2.0))
        supplied = merge(conditioned(mixed), fixed(mixed))
        @test supplied[@varname(x)] isa ReplacementRecord
        @test supplied[@varname(x.a)] == 3.0
        @test supplied[@varname(x.b)] == 2.0

        @model namespace_child(x, y) = (x ~ Normal(); y ~ Normal(); return (x, y))
        @model namespace_parent() = child ~ to_submodel(namespace_child(0.0, 0.0))
        parent = condition(namespace_parent(); child=(; x=1.0, y=2.0))
        parent = decondition(fix(parent, @varname(child.x) => 3.0), @varname(child.y))
        @test parent() == (3.0, 0.0)
    end

    @testset "observation role lookup does not copy slices" begin
        @model slices(x) = x[:] ~ Normal()
        function evaluation_bytes(model)
            vi = VarInfo()
            rng = Xoshiro(1)
            _, vi = init!!(rng, model, vi, InitFromPrior(), UnlinkAll())
            return @allocated _, vi = init!!(rng, model, vi, InitFromPrior(), UnlinkAll())
        end
        small, large = slices(zeros(10_000)), slices(zeros(100_000))
        @test evaluation_bytes(large) - evaluation_bytes(small) < 10_000
        @test loglikelihood(large, VarNamedTuple()) ≈ 100_000 * logpdf(Normal(), 0.0)
    end

    @testset "partial bindings retain the rest of a whole argument binding" begin
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
            nested = op(nested_observation(typed_observation([0.0])), @varname(a.x) => T[2])
            @test nested() == (T, T[2])
        end
        @model function ordinary_input(x)
            return y ~ Normal(x)
        end
        @test_throws r"ArgumentError: .*`x`.*left-hand side of `~`" condition(
            ordinary_input(1.0); x=2.0
        )
    end

    @testset "ComponentVector properties and indices share observations" begin
        @model function indexed_fields()
            x = ComponentVector(; a=0.0, b=0.0)
            x[1] ~ Normal()
            x.b ~ Normal()
            return x
        end
        @model joint_draw(n=2) = x ~ MvNormal(zeros(n), I)
        for op in (condition, fix)
            data = ComponentVector(; a=1.0, b=2.0)
            original = op(indexed_fields(); x=data)
            changed = op(original, @varname(x.a) => 3.0)
            changed = op(changed, @varname(x[2]) => 4.0)
            changed = op(changed, @varname(x.b) => 5.0)
            @test changed() == ComponentVector(; a=3.0, b=5.0)
            @test isempty(keys(VarInfo(changed)))
            @test original() == data
            joint = op(op(joint_draw(); x=data), @varname(x.a) => 1.0)
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

            original = op(joint_draw(length(data)); x=data)
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

    @testset "ComponentVector partial addresses retain validation" begin
        @model component_argument(x) = (x.a[1] ~ Normal(); x.b ~ Normal(); x)
        data = ComponentVector(; a=[1.0, 2.0], b=3.0)
        for (bind, remove, listing) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))
            for vn in
                (@varname(x.b[1]), @varname(x.a[1][1]), @varname(x.zzz), @varname(x.a[5]))
                @test_throws ArgumentError bind(component_argument(data), vn => 9.0)
                @test_throws ArgumentError remove(component_argument(data), vn)
                @test_throws ArgumentError remove(
                    bind(component_argument(data); x=data), vn
                )
            end
            for vn in (@varname(x.a[1]), @varname(x.b))
                valid = bind(component_argument(data), vn => 9.0)
                @test listing(valid)[vn] == 9.0
                result = valid(Xoshiro(1))
                @test result isa typeof(data)
                @test result.a[2] == 2.0
                @test vn == @varname(x.b) ? result.b == 9.0 : result.a[1] == 9.0
                removed = remove(valid, vn)
                @test !haskey(listing(removed), vn)
                if bind === condition
                    @test haskey(rand(Xoshiro(1), removed), vn)
                else
                    @test removed(Xoshiro(1)) == data
                end
            end
        end
        for vn in (@varname(x.a[1]), @varname(x.b))
            @test haskey(rand(Xoshiro(1), decondition(component_argument(data), vn)), vn)
            @test unfix(component_argument(data), vn)(Xoshiro(1)) == data
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

    @testset "unfix restores argument defaults" begin
        @model argument_lhs(x) = x ~ Normal()
        @model argument_indices(x) = (
            for i in eachindex(x)
                x[i] ~ Normal()
            end
        )
        m = argument_indices([1.0, 2.0])
        u = unfix(fix(m, @varname(x[1]) => 5.0), @varname(x[1]))
        @test logjoint(u, (;)) ≈ sum(logpdf.(Normal(), [1.0, 2.0]))
        u = unfix(fix(m; x=[5.0, 6.0]), @varname(x[1]))
        @test logjoint(u, (;)) ≈ logpdf(Normal(), 1.0)
        @test fixed(u)[@varname(x[2])] == 6.0
        @test logjoint(unfix(u), (;)) ≈ sum(logpdf.(Normal(), [1.0, 2.0]))
        @model argument_parent(child) = a ~ to_submodel(child)
        @test logjoint(argument_parent(unfix(fix(m; x=[5.0, 6.0]))), (;)) ≈
            sum(logpdf.(Normal(), [1.0, 2.0]))
        p = DynamicPPL.prefix(argument_lhs(1.0), @varname(a))
        @test logjoint(unfix(fix(p, @varname(a.x) => 5.0), @varname(a.x)), (;)) ≈
            logpdf(Normal(), 1.0)
    end

    @testset "partial removal retains replacement shape" begin
        @model function replacement_shape(x)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return x
        end
        for n in (2, 3), k in (1, n)
            m = replacement_shape(zeros(5 - n))
            observed = decondition(condition(m; x=ones(n)), @varname(x[k]))
            expected = ones(n)
            expected[k] = 2.0
            @test returned(observed, (; x=fill(2.0, n))) == expected
            @test logjoint(observed, (; x=fill(2.0, n))) ≈ sum(logpdf.(Normal(), expected))
            @test LogDensityProblems.dimension(LogDensityFunction(observed)) == 1
            restored = unfix(fix(m; x=ones(n)), @varname(x[k]))
            expected[k] = k <= 5 - n ? 0.0 : 2.0
            @test returned(restored, (; x=fill(2.0, n))) == expected
            @test logjoint(restored, (; x=fill(2.0, n))) ≈ logpdf(Normal(), expected[k])
        end
    end

    @testset "unfix restores index-prefixed arguments" begin
        @model prefixed_argument(x=1.0) = x ~ Normal()
        m = DynamicPPL.prefix(
            fix(prefixed_argument(); x=3.0), @varname(p[2]); template=zeros(2)
        )
        restored = unfix(m)
        @test conditioned(restored)[@varname(p[2].x)] == 1.0
        @test returned(restored, (;)) == 1.0
        @test loglikelihood(restored, (;)) ≈ logpdf(Normal(), 1.0)
    end

    @testset "unfix restores nested arguments beside fixed bindings" begin
        @model function nested_argument(x)
            x.a[1] ~ Normal()
            x.a[2] ~ Normal()
            return x
        end
        m = fix(decondition(nested_argument((; a=[1.0, 2.0]))), @varname(x.a) => [5.0, 6.0])
        restored = unfix(m, @varname(x.a[1]))
        @test returned(restored, (; x=(; a=[1.0, 2.0]))) == (; a=[1.0, 6.0])
        @test isempty(conditioned(restored))
        @test fixed(restored)[@varname(x.a[2])] == 6.0
        @test logjoint(restored, (; x=(; a=[1.0, 2.0]))) ≈ logpdf(Normal(), 1.0)
        @model restored_parent(child) = a ~ to_submodel(child)
        @test returned(restored_parent(restored), (; a=(; x=(; a=[1.0, 2.0])))) ==
            (; a=[1.0, 6.0])
        @test logjoint(restored_parent(restored), (; a=(; x=(; a=[1.0, 2.0])))) ≈
            logpdf(Normal(), 1.0)
    end

    @testset "decondition and unfix" begin
        conditioned_model = condition(model; x=1.0, y=2.0)
        @test isempty(keys(VarInfo(conditioned_model)))
        @test keys(VarInfo(decondition(conditioned_model))) == [@varname(x), @varname(y)]
        @test isempty(keys(conditioned(decondition(conditioned_model))))
        @test keys(conditioned(decondition(conditioned_model, :x))) == [@varname(y)]
        @test keys(conditioned(decondition(conditioned_model, @varname(x)))) ==
            [@varname(y)]

        fixed_model = fix(model; x=1.0, y=2.0)
        @test isempty(keys(fixed(unfix(fixed_model))))
        @test keys(fixed(unfix(fixed_model, :x))) == [@varname(y)]
        @test keys(fixed(unfix(fixed_model, @varname(x)))) == [@varname(y)]

        nested_values = VarNamedTuple((@varname(a.x) => 1.0, @varname(b) => 2.0))
        @test_throws ArgumentError condition(model, nested_values)
        @test_throws ArgumentError fix(model, nested_values)

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
        accs = VarInfo(RawValueAccumulator(false))
        retval, accs = init!!(Xoshiro(1), fixed_model, accs, InitFromPrior(), UnlinkAll())
        @test retval == 5.0
        @test get_raw_values(accs)[@varname(m)] isa Real
    end

    @testset "multivariate values" begin
        @model function mvnorm()
            x ~ MvNormal(zeros(3), I)
            return x
        end
        templated = DynamicPPL.@vnt begin
            @template x = zeros(3)
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        untemplated = DynamicPPL.@vnt begin
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        for op in (condition, fix)
            @test op(mvnorm(), templated)() == [1.0, 2.0, 3.0]
            @test op(mvnorm(), @varname(x[:]) => [1.0, 2.0, 3.0], @of(x = of(Array, 3)))() ==
                [1.0, 2.0, 3.0]
            @test_throws ArgumentError op(mvnorm(), untemplated)()
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
        next_values = DynamicPPL.@vnt begin
            @template x = zeros(2, 2)
            x[2, 1] := 2.5
        end
        for op in (condition, fix)
            model = partial_array_model()
            nested = op(partial_array_parent(model), (@varname(child.x[1, 1]) => 1.5))
            model = op(model, (@varname(x[1, 1]) => 1.5))
            for (partial_model, values) in
                ((model, next_values), (nested, VarNamedTuple(; child=next_values)))
                @test returned(op(partial_model, values), VarNamedTuple()) ==
                    [1.5 0.0; 2.5 0.0]
            end
        end
    end
end

@testset "fixed bindings shadow observations" begin
    @model layered(x) = x ~ Normal()
    for names in ((), (:x,), (@varname(x),))
        latent = decondition(layered(1.0), :x)
        @test isempty(conditioned(unfix(fix(latent; x=5.0), names...)))
        observed = condition(layered(1.0); x=2.0)
        @test conditioned(unfix(fix(observed; x=5.0), names...))[@varname(x)] == 2.0
        @test isempty(conditioned(unfix(decondition(fix(observed; x=5.0)), names...)))
    end
    @test fix(condition(fix(layered(1.0); x=5.0); x=2.0); x=4.0)(Xoshiro(1)) == 4.0
    @test unfix(condition(fix(layered(1.0); x=5.0); x=2.0))(Xoshiro(1)) == 2.0
    @model layered_array(x) = (for i in eachindex(x)
        x[i] ~ Normal()
    end;
    x)
    resized = fix(layered_array(zeros(2)); x=ones(3))
    # The observation still owns length two beneath the resized fixed binding.
    @test_throws "outside the storage" condition(resized, @varname(x[3]) => 4.0)
    @test unfix(resized)(Xoshiro(1)) == zeros(2)
    @model layer_parent(child) = a ~ to_submodel(child)
    observed = condition(layered(1.0); x=2.0)
    @test layer_parent(unfix(fix(observed; x=5.0)))(Xoshiro(1)) == 2.0
    prefixed = DynamicPPL.prefix(fix(observed; x=5.0), @varname(a))
    @test unfix(prefixed, @varname(a.x))(Xoshiro(1)) == 2.0
end

@testset "binding input forms are ordered" begin
    @model input_forms(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    @test_throws MethodError input_forms(zeros(2)) | Dict(@varname(x) => [2.0, 3.0])
    @test (input_forms(zeros(2)) | (; x=[2.0, 3.0]))(Xoshiro(1)) == [2.0, 3.0]
    for bind in (condition, fix)
        for invalid in (Dict(@varname(x) => [2.0, 3.0]), 1)
            @test_throws ArgumentError bind(input_forms([0.0, 0.0]), invalid)
        end
        m = bind(
            input_forms([0.0, 0.0]),
            ((; x=[1.0, 2.0]), @varname(x[1]) => 3.0, VarNamedTuple(; x=[4.0, 5.0])),
        )
        @test m(Xoshiro(1)) == [4.0, 5.0]
        @test bind(input_forms([0.0, 0.0]), :x => [2.0, 3.0])(Xoshiro(1)) == [2.0, 3.0]
    end
end

@testset "binding addresses are checked at construction" begin
    @model address_model(x, n=1) = (x[1] ~ Normal(n); false && (branch ~ Normal()); x)
    @model address_child() = y ~ Normal()
    @model address_parent() = a ~ to_submodel(address_child())
    @model address_unprefixed() = a ~ to_submodel(address_child(), false)
    @model address_truncated() = x ~ truncated(Normal(); lower=0)
    @model address_filldist() = x ~ filldist(Normal(), 2)
    @model address_runtime(rhs) = x ~ rhs
    @model address_dynamic_flag(flag) = a ~ to_submodel(address_child(), flag)
    @model address_qualified() = a ~ DynamicPPL.to_submodel(address_child(), false)
    for bind in (condition, fix)
        for (model, value) in (
            (address_truncated(), 1.0),
            (address_filldist(), [1.0, 2.0]),
            (address_runtime(Normal()), 1.0),
        )
            @test_throws r"Cannot bind `typo`.*not an LHS top symbol" bind(model; typo=1.0)
            @test bind(model; x=value)(Xoshiro(1)) == value
        end
        @test_throws r"Cannot bind `y`.*not an LHS top symbol" bind(address_parent(); y=2.0)
        @test_throws r"Cannot bind `y`.*not an LHS top symbol" bind(
            address_dynamic_flag(false); y=2.0
        )
        @test bind(address_dynamic_flag(true), @varname(a.y) => 2.0)(Xoshiro(1)) == 2.0
        @test bind(address_qualified(); y=2.0)(Xoshiro(1)) == 2.0
        @test_throws ArgumentError bind(address_model([0.0]); z=1.0)
        @test_throws ArgumentError bind(
            DynamicPPL.prefix(address_model([0.0]), @varname(p)); z=1.0
        )
        @test_throws ArgumentError bind(address_model([0.0]); n=1.0)
        @test_throws ArgumentError bind(address_model([0.0]), @varname(x[2]) => 1.0)
        @test bind(address_model([0.0]); branch=2.0)(Xoshiro(1)) == [0.0]
        @test bind(address_parent(), @varname(a.y) => 2.0)(Xoshiro(1)) == 2.0
        @test_throws ArgumentError bind(address_parent(), @varname(a.z) => 2.0)(Xoshiro(1))
        @test bind(address_unprefixed(); y=2.0)(Xoshiro(1)) == 2.0
        @test bind(address_unprefixed(); z=2.0)(Xoshiro(1)) ==
            address_unprefixed()(Xoshiro(1))
    end
end

@testset "bindings fit declared and template types" begin
    @model typed_binding(x::Float64) = x ~ Normal()
    @model untyped_binding(x) = x ~ Normal()
    @model typed_field(p) = (p.a ~ Normal(); p.a)
    @model local_binding() = x ~ Normal()
    @model local_type_child() = z ~ Normal()
    @model local_type_parent() = (x ~ Normal(); a ~ to_submodel(local_type_child()))
    @test_throws ArgumentError condition(condition(local_type_parent(); x=1.0); x=2)
    for bind in (condition, fix)
        @test_throws ArgumentError bind(typed_binding(1.0); x=2)
        @test bind(untyped_binding(1.0); x=2)(Xoshiro(1)) === 2
        @test_throws ArgumentError bind(bind(local_binding(); x=1.0); x=2)
        @test_throws ArgumentError bind(typed_field((a=0.0f0,)), @varname(p.a) => 0.1)
        @test bind(typed_field((a=0.0,)), @varname(p.a) => 1)(Xoshiro(1)) === 1.0
    end
end

@testset "runtime bindings preserve AD values" begin
    @model runtime_child(y) = (y[1] ~ Normal(); y[2] ~ Normal())
    @model function runtime_parent(bind, partial)
        m ~ Normal()
        child = runtime_child(fill(zero(m), 2))
        bound = partial ? bind(child, @varname(y[1]) => 2m) : bind(child; y=[2m, zero(m)])
        a ~ to_submodel(bound)
        return m
    end
    for bind in (condition, fix), partial in (false, true)
        ldf = LogDensityFunction(runtime_parent(bind, partial); adtype=AutoForwardDiff())
        _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3])
        @test gradient ≈ [bind === condition ? -1.5 : -0.3]
    end
end

@testset "bindings own shape at their address" begin
    @model shape_nested(x) = (for i in eachindex(x), j in eachindex(x[i])
        x[i][j] ~ Normal()
    end;
    x)
    @model shape_field(p) = (for i in eachindex(p.a)
        p.a[i] ~ Normal()
    end;
    p)
    m = condition(shape_nested([[0.0, 0.0]]), @varname(x[1]) => [1.0, 2.0, 3.0])
    @test returned(decondition(m, @varname(x[1][3])), (; x=[[7.0, 8.0, 9.0]])) ==
        [[1.0, 2.0, 9.0]]
    @test condition(m, @varname(x[1][3]) => 4.0)(Xoshiro(1)) == [[1.0, 2.0, 4.0]]
    @test unfix(fix(m, @varname(x[1]) => zeros(4)), @varname(x[1]))(Xoshiro(1)) ==
        [[1.0, 2.0, 3.0]]
    @test unfix(fix(shape_nested([[0.0, 0.0]]), @varname(x[1]) => ones(3)), @varname(x[1]))(
        Xoshiro(1)
    ) == [[0.0, 0.0]]
    p = condition(shape_field((a=zeros(2),)), @varname(p.a) => ones(3))
    @test returned(decondition(p, @varname(p.a[3])), (; p=(a=fill(2.0, 3),))) ==
        (a=[1.0, 1.0, 2.0],)
    @test unfix(fix(p, @varname(p.a) => zeros(4)), @varname(p.a))(Xoshiro(1)) ==
        (a=ones(3),)
    for bind in (condition, fix)
        @test_throws DimensionMismatch bind(
            shape_nested([SVector(0.0, 0.0)]), @varname(x[1]) => ones(3)
        )
        for vn in (@varname(x[1:2]), @varname(x[:]), @varname(x[[true, true]]))
            @test_throws DimensionMismatch bind(
                shape_nested([[0.0], [0.0]]), vn => [[1.0], [2.0], [3.0]]
            )
        end
    end
end

@testset "partial removal of shapeless observations" begin
    @model allocated_placeholder(x=missing) = (
        (ismissing(x) || x === nothing) && (x = zeros(2)); x[1] ~ Normal(); x
    )
    for value in (missing, nothing)
        @test_throws r"Cannot remove part.*no template.*decondition" decondition(
            allocated_placeholder(value), @varname(x[1])
        )
        @test length(decondition(allocated_placeholder(value), @varname(x))(Xoshiro(1))) ==
            2
    end
end

@testset "keyword splat index binding diagnostic" begin
    @model keyword_index(; x...) = (x[:a] ~ Normal(); x[:a])
    for bind in (condition, fix), wrap in (identity, m -> DynamicPPL.prefix(m, @varname(p)))
        m = wrap(keyword_index(; a=1.0))
        vn = wrap === identity ? @varname(x[:a]) : @varname(p.x[:a])
        @test_throws r"keyword-splat argument `x`.*replace the whole argument" bind(
            m, vn => 2.0
        )
    end
end

@testset "local owners supply indexed binding storage" begin
    @testset "unrelated storage does not increase preparation allocations" begin
        @model function storage_locality(y)
            z = zeros(1)
            z[1] ~ Normal()
            return y[1] ~ Normal()
        end
        function preparation_allocations(model)
            pair = @varname(y[1]) => 0.3
            DynamicPPL._make_condfix_values(model, pair)
            return @allocated DynamicPPL._make_condfix_values(model, pair)
        end
        small = condition(
            storage_locality([0.0]), @of(z = of(Array, 1000)), @varname(z[1]) => 0.0
        )
        large = condition(
            storage_locality([0.0]), @of(z = of(Array, 10000)), @varname(z[1]) => 0.0
        )
        preparation_allocations(small)
        preparation_allocations(large)
        @test preparation_allocations(large) <= preparation_allocations(small) + 1024
    end

    @model function local_storage()
        z = zeros(3)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model function local_offset_storage()
        z = OffsetArray(zeros(3), -1:1)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model function local_nested_storage()
        z = (a=[zeros(2), zeros(3)], b=zeros(2))
        for i in eachindex(z.a), j in eachindex(z.a[i])
            z.a[i][j] ~ Normal()
        end
        for i in eachindex(z.b)
            z.b[i] ~ Normal()
        end
        return z
    end
    @model function local_storage_parent()
        a ~ to_submodel(local_storage())
        return a
    end
    for op in (condition, fix)
        sampled = rand(Xoshiro(1), local_storage())
        observed = conditioned(condition(local_storage(); z=ones(3)))
        for input in (:z => ones(3), (z=ones(3),), sampled, observed)
            original = input === sampled ? sampled[@varname(z)] : ones(3)
            @test op(local_storage(), input, @varname(z[end]) => 2.0)(Xoshiro(1)) ==
                [original[1], original[2], 2.0]
            @test op(local_storage(), input, @varname(z[:]) => fill(2.0, 3))(Xoshiro(1)) ==
                fill(2.0, 3)
        end
        owned = op(local_storage(); z=ones(3))
        @test op(owned, @varname(z[end]) => 2.0)(Xoshiro(1)) == [1.0, 1.0, 2.0]
        @test_throws ArgumentError op(owned, @varname(z[4]) => 2.0)
        @test owned(Xoshiro(1)) == ones(3)
        offset = OffsetArray(ones(3), -1:1)
        result = op(local_offset_storage(), :z => offset, @varname(z[end]) => 2.0)(
            Xoshiro(1)
        )
        @test axes(result) == axes(offset)
        @test result == OffsetArray([1.0, 1.0, 2.0], -1:1)
        nested = (a=[ones(2), ones(3)], b=ones(2))
        result = op(local_nested_storage(), :z => nested, @varname(z.a[end][end]) => 2.0)(
            Xoshiro(1)
        )
        @test result == (a=[ones(2), [1.0, 1.0, 2.0]], b=ones(2))
        @test op(
            local_storage_parent(), @varname(a.z) => ones(3), @varname(a.z[end]) => 2.0
        )(
            Xoshiro(1)
        ) == [1.0, 1.0, 2.0]
        prefixed = DynamicPPL.prefix(local_storage(), @varname(a.b))
        @test op(prefixed, @varname(a.b.z) => ones(3), @varname(a.b.z[end]) => 2.0)(
            Xoshiro(1)
        ) == [1.0, 1.0, 2.0]
    end
end

@testset "binding schemas" begin
    @model function schema_local(n=3, T=Float64)
        z = zeros(T, n)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model schema_arg(z) = z ~ MvNormal(zeros(length(z)), ones(length(z)))
    for op in (condition, fix)
        schema = @of(z = of(Array, 3))
        pair = @varname(z[2]) => 1.0
        for inputs in (
            (schema, pair),
            (pair, schema),
            ((pair,), schema),
            (pair, schema, @varname(z[1]) => 2.0),
        )
            m = op(schema_local(), inputs...)
            @test m(Xoshiro(1))[2] == 1.0
            @test length(op === condition ? conditioned(m) : fixed(m)) >= 1
        end
        @test op(schema_local(), pair, of((z=of(Array, 3),)))(Xoshiro(1))[2] == 1.0
        @test_throws ArgumentError op(schema_local(), schema)
        @test_throws ArgumentError op(schema_local(), pair, schema, schema)
        @test_throws ArgumentError op(schema_local(), pair, @of(w = of(Array, 3)))
        @test_throws ArgumentError op(schema_local(), pair, @of(n = of(Int)))
        @test_throws ArgumentError op(schema_arg(zeros(3)), pair, schema)
        owned = op(schema_local(); z=[2.0, 3.0, 4.0])
        @test op(owned, pair, schema)(Xoshiro(1)) == [2.0, 1.0, 4.0]
        @test_throws ArgumentError op(owned, pair, @of(z = of(Array, 4)))
        @test_throws ArgumentError op(owned, pair, @of(z = of(Array, Float32, 3)))
        @test op(schema_local(), (z=ones(3),), pair, schema)(Xoshiro(1)) == [1.0, 1.0, 1.0]
        @test op(schema_local(), @varname(z[end]) => 2.0, schema)(Xoshiro(1))[3] == 2.0
        @test isempty(conditioned(fix(schema_local(), pair, schema)))
        @test_throws ArgumentError schema_local() | (pair, schema)
        partial = op(schema_local(), pair, schema)
        @test op(partial, (z=ones(3),), schema)(Xoshiro(1)) == ones(3)
        @test op(partial, @varname(z[3]) => 4.0, schema)(Xoshiro(1))[2:3] == [1.0, 4.0]
        prefixed = DynamicPPL.prefix(schema_local(), @varname(a))
        @test op(prefixed, @varname(a.z[2]) => 1.0, schema)(Xoshiro(1))[2] == 1.0
    end
    # A fixed layer must not determine an observation's storage, or vice versa.
    m = fix(condition(schema_local(); z=ones(3)); z=ones(4))
    @test (x -> x(Xoshiro(1)) == [1.0, 2.0, 1.0])(
        unfix(condition(m, @varname(z[2]) => 2.0, @of(z = of(Array, 3))))
    )
    @test_throws ArgumentError condition(m, @varname(z[2]) => 2.0, @of(z = of(Array, 4)))
    @model function schema_parent(op, n)
        m ~ Normal()
        a ~ to_submodel(
            op(
                schema_local(n, typeof(m)),
                @varname(z[2]) => m,
                @of(z = of(Array, typeof(m), n))
            ),
        )
        return a
    end
    for op in (condition, fix)
        @test ForwardDiff.derivative(0.3) do m
            logjoint(schema_parent(op, 3), (m=m, a=(z=[0.0, m, 0.0],)))
        end ≈ (op === condition ? -0.6 : -0.3)
    end
end

@testset "whole binding schemas under prefixes" begin
    @model function whole_schema_local()
        x = zeros(2)
        x[1] ~ Normal()
        x[2] ~ Normal()
        return x
    end
    @model whole_schema_parent(child) = a ~ to_submodel(child)
    plain = whole_schema_local()
    prefixed = prefix(plain, @varname(p))
    nested = prefix(prefixed, @varname(q))
    for op in (condition, fix),
        (model, address) in
        ((plain, @varname(x)), (prefixed, @varname(p.x)), (nested, @varname(q.p.x)))

        schema = @of(x = of(Array, 2))
        bound = op(model, schema, address => [1.0, 2.0])
        @test bound(Xoshiro(1)) == [1.0, 2.0]
        @test whole_schema_parent(bound)(Xoshiro(1)) == [1.0, 2.0]
        @test_throws ArgumentError op(model, schema, address => ones(Float32, 2))
        @test_throws ArgumentError op(model, schema, address => ones(3))
    end
end

@testset "binding schemas resolve dynamic prefix indices" begin
    @model function end_schema_local()
        x = zeros(3)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model end_schema_parent(child) = a ~ to_submodel(child)
    model = prefix(end_schema_local(), @varname(p[end]); template=zeros(2))
    nested = prefix(model, @varname(q[end]); template=zeros(2))
    for bind in (condition, fix),
        (m, dynamic, concrete) in (
            (model, @varname(p[end].x[3]), @varname(p[2].x[3])),
            (model, @varname(p[end].x[end]), @varname(p[2].x[3])),
            (nested, @varname(q[end].p[end].x[3]), @varname(q[2].p[2].x[3])),
        )

        schema = @of(x = of(Array, 3))
        bound = bind(m, schema, dynamic => 9.0)
        @test bound(Xoshiro(1)) == bind(m, schema, concrete => 9.0)(Xoshiro(1))
        @test bound(Xoshiro(1))[3] == 9.0
        @test end_schema_parent(bound)(Xoshiro(1))[3] == 9.0
    end
    for bind in (condition, fix)
        @test_throws ArgumentError bind(
            model, @of(x = of(Array, 3)), @varname(p[1].x[3]) => 9.0
        )
    end
end

@testset "binding schemas beside argument storage" begin
    @model function mixed_schema(y)
        x = zeros(3)
        x[3] ~ Normal()
        y[2] ~ Normal()
        return (x[3], y[2])
    end
    model = mixed_schema(zeros(2))
    for bind in (condition, fix)
        @test bind(
            model, @of(x = of(Array, 3)), @varname(y[end]) => 4.0, @varname(x[3]) => 9.0
        )(
            Xoshiro(1)
        ) == (9.0, 4.0)
        @test bind(
            prefix(model, @varname(p[end]); template=zeros(2)),
            @of(x = of(Array, 3)),
            @varname(p[end].y[end]) => 4.0,
            @varname(p[end].x[3]) => 9.0,
        )(
            Xoshiro(1)
        ) == (9.0, 4.0)
    end
end

@testset "schema exact conversion" begin
    @model function typed_schema()
        z = zeros(3)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    for op in (condition, fix)
        @test_throws InexactError op(
            typed_schema(), @varname(z[2]) => 1.5, @of(z = of(Array, Int, 3))
        )
        @test_throws ArgumentError op(
            typed_schema(), @varname(z[2]) => 0.1, @of(z = of(Array, Float32, 3))
        )
        m = op(typed_schema(), @varname(z[2]) => 1, @of(z = of(Array, 3)))
        @test (op === condition ? conditioned(m) : fixed(m))[@varname(z[2])] === 1.0
        @test_throws ArgumentError op(
            typed_schema(), @varname(z) => ones(Float32, 3), @of(z = of(Array, 3))
        )
        @test_throws ArgumentError op(
            typed_schema(), (z=ones(Float32, 3),), @of(z = of(Array, 3))
        )
        @test_throws ArgumentError op(
            typed_schema(),
            @varname(z[:]) => [0.1, 0.2, 0.3],
            @of(z = of(Array, Float32, 3))
        )
        m32 = op(typed_schema(), @varname(z[2]) => 1.0, @of(z = of(Array, Float32, 3)))
        @test (op === condition ? conditioned(m32) : fixed(m32))[@varname(z[2])] === 1.0f0
        @test_throws ArgumentError op(
            typed_schema(), VarNamedTuple(; z=ones(Float32, 3)), @of(z = of(Array, 3))
        )
    end
end

@testset "schema storage ownership" begin
    @model function schema_fields()
        z = (a=zeros(3), b=zeros(2))
        for i in eachindex(z.a)
            z.a[i] ~ Normal()
        end
        for i in eachindex(z.b)
            z.b[i] ~ Normal()
        end
        return z
    end
    @model function two_locals()
        z = zeros(3)
        w = zeros(2)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        for i in eachindex(w)
            w[i] ~ Normal()
        end
        return z, w
    end
    for op in (condition, fix)
        schema = @of(z = @of(a = of(Array, 3), b = of(Array, 2)))
        owned = op(schema_fields(); z=(a=zeros(2), b=zeros(2)))
        resized = op(owned, @varname(z.a) => ones(3))
        @test op(resized, @varname(z.a[2]) => 2.0, schema)(Xoshiro(1)).a == [1.0, 2.0, 1.0]
        m = op(schema_fields(), @varname(z.a[2]) => 2.0, schema)
        @test op(m, @varname(z.a[3]) => 3.0, schema)(Xoshiro(1)).a[2:3] == [2.0, 3.0]
        @test op(m, @varname(z.b[2]) => 4.0, schema)(Xoshiro(1)).b[2] == 4.0
        @test op(
            two_locals(),
            @of(z = of(Array, 3), w = of(Array, 2)),
            @varname(z[2]) => 2.0,
            @varname(w[2]) => 4.0,
        )(
            Xoshiro(1)
        )[2][2] == 4.0
        @test_throws ArgumentError op(
            two_locals(), @varname(z) => ones(4), @of(z = of(Array, 3))
        )
        @test_throws ArgumentError op(
            two_locals(),
            @varname(z[2]) => 1.0,
            @of(z = of(Array, 3)),
            @varname(w[2]) => 2.0,
            @of(w = of(Array, 2))
        )
        m = op(two_locals(), @varname(z[2]) => 2.0, @of(z = of(Array, 3)))
        @test op(m, @varname(z[3]) => 3.0)(Xoshiro(1))[1][2:3] == [2.0, 3.0]
    end
end

@model function mixed_storage()
    z = zeros(3)
    w = zeros(5)
    for i in eachindex(z)
        z[i] ~ Normal()
    end
    for i in eachindex(w)
        w[i] ~ Normal()
    end
    return z, w
end
@testset "schemas preserve other input storage" begin
    v = DynamicPPL.@vnt begin
        @template w = zeros(5)
        w[2] := 4.0
    end
    for op in (condition, fix)
        m = op(mixed_storage(), @of(z = of(Array, 3)), @varname(z[2]) => 2.0, v)
        values = op === condition ? conditioned(m) : fixed(m)
        @test subset(values, [@varname(w)]) == v
    end
end

@model function keyword_schema()
    z = zeros(3)
    for i in eachindex(z)
        z[i] ~ Normal()
    end
    w ~ Normal()
    return z, w
end
@testset "schemas with keyword binding data" begin
    for op in (condition, fix)
        @test op(keyword_schema(), @of(z = of(Array, 3)), @varname(z[2]) => 1.0; w=2.0)(
            Xoshiro(1)
        )[2] === 2.0
    end
end

@testset "fixed NamedTuple owner fields" begin
    @model function changed_fields(p, change)
        p = change(p)
        p.a ~ Normal()
        return p
    end
    @model function changed_nested_fields(p, change)
        p = (child=change(p.child),)
        p.child.a ~ Normal()
        return p
    end
    @model function changed_keyword_fields(change; kw...)
        kw = pairs(change(values(kw)))
        kw[:a] ~ Normal()
        return kw
    end
    @model fields_parent(child) = child_result ~ to_submodel(child)
    value = (a=3.0, b=4.0)
    reordered = changed_fields(value, p -> (b=p.b, a=p.a))
    @test fix(reordered; p=value)(Xoshiro(1)) == (b=4.0, a=3.0)
    for change in (p -> (a=p.a,), p -> (; p..., c=5.0), p -> (a=p.a, c=p.b))
        @test_throws r"ArgumentError: .*kw.*static size and shape" fix(
            changed_keyword_fields(change; value...); kw=value
        )(
            Xoshiro(1)
        )
        for (model, address) in (
            (changed_fields(value, change), @varname(p)),
            (changed_nested_fields((child=value,), change), @varname(p.child)),
        )
            bound = fix(model, address => value)
            @test_throws r"ArgumentError: .*p.*static size and shape" bound(Xoshiro(1))
            @test_throws r"ArgumentError: .*p.*static size and shape" fields_parent(bound)(
                Xoshiro(1)
            )
            # A partial fix does not own the enclosing NamedTuple's fields.
            leaf = address == @varname(p) ? @varname(p.a) : @varname(p.child.a)
            partial = fix(model, leaf => value.a)
            observed = condition(model, address => value)
            @test partial(Xoshiro(1)) == observed(Xoshiro(1))
            @test observed(Xoshiro(1)) ==
                (address == @varname(p) ? change(value) : (child=change(value),))
        end
    end
end

@testset "partial fixed shape ownership" begin
    @model function partial_shape(x, change)
        x = change(x)
        x[1] ~ Normal()
        return x
    end
    @model shape_parent(child) = a ~ to_submodel(child)
    for change in (x -> x[1:1], x -> vcat(x, 3.0), x -> reshape(x, 1, 2))
        m = fix(partial_shape([1.0, 2.0], change), @varname(x[1]) => 8.0)
        @test m(Xoshiro(1)) == change([8.0, 2.0])
        @test shape_parent(m)(Xoshiro(1)) == change([8.0, 2.0])
    end
    @model function nested_partial_shape(x, change)
        x = change(x)
        x[1][1] ~ Normal()
        return x
    end
    m = fix(
        nested_partial_shape([[1.0, 2.0], [3.0]], x -> x[1:1]), @varname(x[1]) => [8.0, 9.0]
    )
    @test m(Xoshiro(1)) == [[8.0, 9.0]]
    m = fix(
        nested_partial_shape([[1.0, 2.0]], x -> [x[1][1:1]]), @varname(x[1]) => [8.0, 9.0]
    )
    @test_throws r"ArgumentError: .*x\[1\]\[1\].*size and shape" m(Xoshiro(1))
end

@testset "layer overlay preserves whole-binding extents" begin
    @model function child(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model parent(child) = a ~ to_submodel(child)
    for shape in (identity, x -> reshape(x, :, 1)), recursive in (false, true)
        scope = recursive ? (DynamicPPL.Recursive(),) : ()
        base = child(shape(zeros(2)))
        base = recursive ? parent(base) : base
        whole = recursive ? @varname(a.x) : @varname(x)
        first = recursive ? @varname(a.x[1]) : @varname(x[1])
        second = recursive ? @varname(a.x[2]) : @varname(x[2])
        m = condition(base, whole => shape([1.0, 2.0, 3.0]))
        m = fix(m, first => 9.0, second => 9.0)
        @test m(StableRNG(1)) == shape([9.0, 9.0, 3.0])
        @test logjoint(m, (;)) ≈ logpdf(Normal(), 3.0)
        for (observed, fixed) in
            (([1.0, 2.0, 3.0], [4.0, 5.0]), ([1.0, 2.0], [4.0, 5.0, 6.0]))
            for order in (false, true)
                m = if order
                    condition(fix(base, whole => shape(fixed)), whole => shape(observed))
                else
                    fix(condition(base, whole => shape(observed)), whole => shape(fixed))
                end
                m = unfix(m, scope..., second)
                expected = copy(fixed)
                expected[2] = observed[2]
                @test m(StableRNG(1)) == shape(expected)
                @test logjoint(m, (;)) ≈ logpdf(Normal(), observed[2])
                released = unfix(m, scope..., whole)
                @test released(StableRNG(1)) == shape(observed)
                @test logjoint(released, (;)) ≈ sum(logpdf.(Normal(), observed))
            end
        end
    end
    @model function shared(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        a ~ to_submodel(child([0.0, 0.0]), false)
        return (x, a)
    end
    # A whole fixed owner also supplies the unprefixed child's shape.
    m = fix(shared([0.0, 0.0]); x=[4.0, 5.0])
    m = condition(m, @varname(x) => [1.0, 2.0, 3.0])
    m = fix(m, @varname(x[1]) => 9.0)
    @test m(StableRNG(1)) == ([9.0, 5.0], [9.0, 5.0])
    m = fix(m, @varname(x[2]) => 8.0)
    @test m(StableRNG(1)) == ([9.0, 8.0], [9.0, 8.0])
end

@testset "incomplete local owner conversion" begin
    @model function localz()
        z = zeros(3)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model local_owner_parent(child) = a ~ to_submodel(child)
    for (bind, remove, select) in
        ((condition, decondition, conditioned), (fix, unfix, fixed))
        m = remove(bind(localz(); z=ones(Float32, 3)), @varname(z[2]))
        @test_throws ArgumentError bind(m, @varname(z[2]) => 0.1)
        changed = bind(m, @varname(z[2]) => 0.5)
        @test select(changed)[@varname(z[2])] === 0.5f0
        @test local_owner_parent(changed)(Xoshiro(1)) == [1.0, 0.5, 1.0]
        m = bind(localz(), @of(z = of(Array, Int, 3)), @varname(z[1]) => 1)
        @test_throws InexactError bind(m, @varname(z[2]) => 1.5)
        @test select(bind(m, @varname(z[2]) => 2.0))[@varname(z[2])] === 2
        # Reference-valued storage retains its type without reading the removed entry.
        m = remove(bind(localz(); z=ones(BigFloat, 3)), @varname(z[2]))
        changed = bind(m, @varname(z[2]) => 2)
        @test select(changed)[@varname(z[2])] isa BigFloat
        @test changed(Xoshiro(1)) == [1.0, 2.0, 1.0]
    end
end

@testset "first partial fix owns its layer" begin
    @model function fixed_layer_array(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model fixed_layer_parent(child) = a ~ to_submodel(child)
    observed = condition(fixed_layer_array(zeros(3)); x=ones(1))
    m = fix(observed, @varname(x[3]) => 9.0)
    @test fixed(m)[@varname(x[3])] == 9.0
    @test keys(conditioned(m)) == [@varname(x[1])]
    @test conditioned(m)[@varname(x[1])] == 1.0
    @test returned(m, (; x=[5.0, 6.0, 7.0])) == [1.0, 6.0, 9.0]
    @test returned(fixed_layer_parent(m), (; a=(x=[5.0, 6.0, 7.0],))) == [1.0, 6.0, 9.0]
    @test fix(m, @varname(x[2]) => 8.0)(Xoshiro(1)) == [1.0, 8.0, 9.0]
    @test unfix(m)(Xoshiro(1)) == [1.0]
    @test unfix(m, @varname(x[3]))(Xoshiro(1)) == [1.0]
    @test_throws ArgumentError fix(observed, @varname(x[4]) => 9.0)
    observed = condition(fixed_layer_array(zeros(3, 1)); x=ones(1, 3))
    m = fix(observed, @varname(x[3, 1]) => 9.0)
    expected = fill(2.0, 3, 3)
    expected[1, :] .= 1.0
    expected[3, 1] = 9.0
    @test returned(m, (; x=fill(2.0, 3, 3))) == expected
    @test unfix(m)(Xoshiro(1)) == ones(1, 3)
    @model owned_shrink(x) = (x = x[1:1]; x[1] ~ Normal(); x)
    m = fix(fix(owned_shrink([1.0, 2.0]); x=[3.0, 4.0]), @varname(x[1]) => 8.0)
    @test_throws "static size and shape" m(Xoshiro(1))
    @test_throws "static size and shape" fixed_layer_parent(m)(Xoshiro(1))
    @model fixed_layer_fields(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
    m = fix(condition(fixed_layer_fields((a=0.0, b=0.0)); x=(a=1.0,)), @varname(x.b) => 9.0)
    @test m(Xoshiro(1)) == (a=1.0, b=9.0)
    @test fixed_layer_parent(m)(Xoshiro(1)) == (a=1.0, b=9.0)
end

@testset "binding edits use their layer's owner" begin
    @model layer_array(x) = (for i in eachindex(x)
        x[i] ~ Normal()
    end;
    x)
    @model function layer_flexible(x)
        if x isa NamedTuple
            x.a ~ Normal()
        else
            x[1] ~ Normal()
            x[2] ~ Normal()
        end
        return x
    end
    @model layer_local() = (z = zeros(3);
    for i in eachindex(z)
        z[i] ~ Normal()
    end;
    z)
    @model layer_record() = (z = (a=1.0, b=2.0); z.a ~ Normal(); z.b ~ Normal(); z)
    @model layer_parent(child) = a ~ to_submodel(child)

    m = fix(condition(layer_array(zeros(2)); x=ones(3)); x=zeros(2))
    for edit in
        (m -> condition(m, @varname(x[3]) => 7.0), m -> m | (@varname(x[3]) => 7.0,))
        changed = edit(m)
        @test changed(Xoshiro(1)) == zeros(2)
        @test unfix(changed)(Xoshiro(1)) == [1.0, 1.0, 7.0]
        @test layer_parent(unfix(changed))(Xoshiro(1)) == [1.0, 1.0, 7.0]
    end
    m = fix(condition(layer_array(zeros(2)); x=ones(2)); x=zeros(3))
    @test_throws ArgumentError condition(m, @varname(x[3]) => 7.0)
    m = fix(condition(layer_array(zeros(2)); x=Float32[1, 2]); x=zeros(2))
    @test_throws ArgumentError condition(m, @varname(x[1]) => 0.1)
    @test unfix(condition(m, @varname(x[1]) => 0.5))(Xoshiro(1)) == Float32[0.5, 2]
    @test eltype(unfix(condition(m, @varname(x[1]) => 0.5))(Xoshiro(1))) === Float32

    m = condition(fix(layer_array(zeros(2)); x=zeros(3)); x=ones(2))
    @test fix(m, @varname(x[3]) => 7.0)(Xoshiro(1)) == [0.0, 0.0, 7.0]
    @test unfix(m)(Xoshiro(1)) == ones(2)
    m = fix(condition(layer_local(); z=ones(Float32, 3)); z=zeros(3))
    @test_throws "template type" condition(m; z=ones(3))
    @test_throws "template type" fix(m; z=ones(Float32, 3))
    @test unfix(condition(m; z=fill(2.0f0, 3)))(Xoshiro(1)) == fill(2.0f0, 3)

    m = decondition(fix(layer_flexible([1.0, 2.0]); x=(a=1.0,)), @varname(x[1]))
    @test m(Xoshiro(1)) == (a=1.0,)
    @test returned(unfix(m), (; x=[8.0, 9.0])) == [8.0, 2.0]
    @test_throws "Cannot remove `x[1]`: integer indexing into a NamedTuple" unfix(
        fix(layer_flexible([1.0, 2.0]); x=(a=1.0,)), @varname(x[1])
    )

    for (bind, remove) in ((condition, decondition), (fix, unfix))
        whole = bind(layer_local(); z=ones(3))
        for partial in (
            remove(whole, @varname(z[1])),
            bind(whole, @varname(z[1]) => 2.0),
            bind(layer_local(), @of(z = of(Array, 3)), @varname(z[1]) => 2.0),
        )
            @test_throws "template type" bind(partial; z=ones(Float32, 3))
            @test bind(partial; z=fill(3.0, 3))(Xoshiro(1)) == fill(3.0, 3)
        end
        record = remove(bind(layer_record(); z=(a=1.0, b=2.0)), @varname(z.a))
        @test_throws "template type" bind(record; z=(a=1.0f0, b=2.0f0))
        @test bind(record; z=(a=3.0, b=4.0))(Xoshiro(1)) == (a=3.0, b=4.0)
    end
end

@testset "binding and removal address validation agree" begin
    @model removal_addresses(x, t, p) = (
        x[1] ~ Normal(); x[3] ~ Normal(); t.present ~ Normal(); p.b[1] ~ Normal()
    )
    @model removal_child(c) = a ~ to_submodel(c)
    @model removal_unprefixed(c) = a ~ to_submodel(c, false)
    @model local_fields() = (t = (present=1.0,); t.present ~ Normal())
    @model local_array() = (x = zeros(3); x[1] ~ Normal(); x)
    @model indexed_child(c) = p[1] ~ to_submodel(c)
    @model removal_matrix(x) = (x[1, 1] ~ Normal(); x[2, 2] ~ Normal())
    @model removal_tuple(p) = (p[1] ~ Normal(); p[2] ~ Normal())
    m = removal_addresses(zeros(3), (present=1.0,), (b=[1.0, 2.0],))
    addresses = (
        (@varname(x[3]), true),
        (@varname(x[8]), false),
        (@varname(t.present), true),
        (@varname(t.absent), false),
        (@varname(t.present[1]), false),
        (@varname(x.a), false),
        (@varname(t[1]), false),
        (@varname(p[1]), false),
        (@varname(p.b[2]), true),
        (@varname(p.b[3]), false),
        (@varname(absent), false),
    )
    for (bind, remove, select) in
        ((condition, decondition, conditioned), (fix, unfix, fixed)),
        scope in ((), (DynamicPPL.Recursive(),))

        unbound = remove(m)
        @test_throws "Cannot bind parts below `t.present` with value of type Float64. For other bindings, use `decondition(model, @varname(t.present))` first" bind(
            m, @varname(t.present[1]) => 2.0
        )
        @test_throws "Cannot remove `x.a`: cannot partially edit array owner `x` at `x.a` through container type Vector{Float64}; remove the whole value `x` instead." remove(
            m, scope..., @varname(x.a)
        )
        @test_throws "Cannot remove `t[1]`: integer indexing into a NamedTuple is unsupported; use `t.present` instead." remove(
            m, scope..., @varname(t[1])
        )
        for (vn, valid) in addresses
            if valid
                @test haskey(select(bind(unbound, vn => 2.0)), vn)
                removed = remove(unbound, scope..., vn)
                @test select(removed) == select(unbound)
                @test select(remove(removed, scope..., vn)) == select(unbound)
            else
                @test_throws ArgumentError bind(unbound, vn => 2.0)
                @test_throws "Cannot remove `$vn`" remove(unbound, scope..., vn)
                @test_throws "Cannot remove `$vn`" remove(m, scope..., vn)
            end
        end
        # Dynamic addresses use argument storage even when the edited layer is empty.
        for (original, dynamic, concrete) in (
            (m, @varname(x[end]), @varname(x[3])),
            (m, @varname(p.b[end]), @varname(p.b[2])),
            (removal_matrix(zeros(2, 2)), @varname(x[end, end]), @varname(x[2, 2])),
        )
            empty_layer = remove(original)
            @test select(bind(empty_layer, dynamic => 2.0)) ==
                select(bind(empty_layer, concrete => 2.0))
            for model in (original, empty_layer, bind(empty_layer, concrete => 2.0))
                removed = remove(model, scope..., dynamic)
                @test select(removed) == select(remove(model, scope..., concrete))
                @test select(remove(removed, scope..., dynamic)) == select(removed)
            end
        end
        # The edited layer's current owner supplies bounds, including below a field.
        owned = bind(m; x=ones(5), p=(b=[1.0, 2.0, 3.0],))
        for vn in (@varname(x[5]), @varname(p.b[3]))
            @test bind(owned, vn => 2.0) isa Model
            @test !haskey(select(remove(owned, scope..., vn)), vn)
        end
        for vn in (@varname(x[6]), @varname(p.b[4]))
            @test_throws ArgumentError bind(owned, vn => 2.0)
            @test_throws "Cannot remove `$vn`" remove(owned, scope..., vn)
        end
        prefixed_owner = prefix(owned, @varname(q[2]))
        @test !haskey(
            select(remove(prefixed_owner, scope..., @varname(q[2].x))), @varname(q[2].x)
        )
        @test_throws "Cannot remove" remove(prefixed_owner, scope..., @varname(q[2].x[6]))
        prefixed = prefix(unbound, @varname(q[2]))
        @test_throws ArgumentError(
            "Cannot bind `q[1].x`: it is outside this model's prefix `q[2]`."
        ) bind(prefixed, @varname(q[1].x) => ones(3))
        @test select(remove(prefixed, scope..., @varname(q[2].x[3]))) == select(prefixed)
        @test_throws "Cannot remove" remove(prefixed, scope..., @varname(q[1].x))
        @test_throws "Cannot remove" remove(prefixed, scope..., @varname(x))
        @test_throws "Cannot remove" remove(prefixed, scope..., @varname(q[2].x[8]))
        # Child storage is unknown here, and literal unprefixed children relax top symbols.
        for (parent, vn) in
            ((removal_child(m), @varname(a.typo)), (removal_unprefixed(m), @varname(typo)))
            @test haskey(select(bind(parent, vn => 2.0)), vn)
            @test isempty(select(remove(parent, scope..., vn)))
        end
        local_prefixed = prefix(bind(local_array(); x=ones(3)), @varname(q[2]))
        @test_throws ArgumentError bind(local_prefixed, @varname(q[2].x[8]) => 2.0)
        @test_throws "Cannot remove" remove(local_prefixed, scope..., @varname(q[2].x[8]))
        indexed = bind(indexed_child(local_array()), @varname(p[1].x) => ones(3))
        @test_throws ArgumentError bind(indexed, @varname(p[1].x[8]) => 2.0)
        @test_throws "Cannot remove" remove(indexed, scope..., @varname(p[1].x[8]))
        @test !haskey(
            select(remove(indexed, scope..., @varname(p[1].x[3]))), @varname(p[1].x[3])
        )
        local_bound = bind(local_fields(); t=(present=1.0,))
        @test_throws ArgumentError bind(local_bound, @varname(t.absent) => 2.0)
        @test_throws "Cannot remove `t.absent`" remove(
            local_bound, scope..., @varname(t.absent)
        )
        namespace = bind(removal_child(m); a=(x=ones(3),))
        @test bind(namespace, @varname(a.typo) => 2.0) isa Model
        @test select(remove(namespace, scope..., @varname(a.typo))) == select(namespace)
    end
    # An empty fixed layer has no owner; overlaying it must still fit the stored observation.
    for (original, whole, valid, invalid) in (
        (local_array(), @varname(x), @varname(x[3]), @varname(x[8])),
        (removal_child(m), @varname(a.x), @varname(a.x[3]), @varname(a.x[8])),
    )
        observed = condition(original, whole => ones(3))
        for bind in (condition, fix)
            @test_throws ArgumentError(
                "Cannot bind `$invalid`: index is outside the storage at `$(DynamicPPL.getsym(invalid))`",
            ) bind(observed, invalid => 1.0)
        end
        bound = fix(observed, valid => 2.0)
        @test fixed(bound)[valid] == 2.0
        @test conditioned(unfix(bound, valid)) == conditioned(observed)
        for scope in ((), (DynamicPPL.Recursive(),))
            @test_throws "index is outside the storage" decondition(
                observed, scope..., invalid
            )
            @test isempty(fixed(unfix(observed, scope..., invalid)))
        end
    end
end

struct PlaceholderState{T}
    value::T
end
struct PlaceholderStateNormal <: Distribution{Univariate,Continuous} end
function Distributions.logpdf(::PlaceholderStateNormal, p::PlaceholderState)
    return logpdf(Normal(), p.value)
end
Distributions.loglikelihood(d::PlaceholderStateNormal, p::PlaceholderState) = logpdf(d, p)

@testset "placeholders in whole struct LHS values" begin
    @model state_lhs(p) = p ~ PlaceholderStateNormal()
    @model state_field(p) = (p.a ~ Normal(); p.a)
    @model state_parent(child) = a ~ to_submodel(child)
    for value in (nothing, missing)
        original = state_lhs(PlaceholderState(value))
        for m in (
            original,
            condition(original; p=PlaceholderState(value)),
            fix(original; p=PlaceholderState(value)),
        )
            @test_throws "LHS variable `p` contains `$value`" logjoint(m, (;))
            @test_throws "LHS variable `a.p` contains `$value`" logjoint(
                state_parent(m), (;)
            )
        end
        # A field not read by a tilde may still contain a placeholder.
        for bind in (condition, fix)
            p = ObservationRecord(1.0, PlaceholderState(value))
            @test bind(state_field(p); p=p)(Xoshiro(1)) == 1.0
        end
    end
    for T in (Float32, Float64, BigFloat)
        value = PlaceholderState(T(1))
        @test logjoint(state_lhs(value), (;)) ≈ logpdf(Normal(), T(1))
        @test fix(state_lhs(value); p=value)(Xoshiro(1)) === value
    end
    @test ForwardDiff.derivative(x -> logjoint(state_lhs(PlaceholderState(x)), (;)), 1.0) ≈
        -1.0
end

@testset "aliases in latent argument storage" begin
    @model function aliased_argument(x)
        x.a[1] ~ Normal()
        0.0 ~ Normal(x.b[1], 1)
        return (x.a === x.b, x.a[1], x.b[1])
    end
    v = [0.0]
    m = decondition(aliased_argument((a=v, b=v)))
    p = (x=(a=[2.0],),)
    @test returned(m, p) == (true, 2.0, 2.0)
    @test loglikelihood(m, p) ≈ logpdf(Normal(2.0, 1), 0.0)
    @test v == [0.0]
    copied = @inferred DynamicPPL._copy_model_argument((a=v, b=v))
    @test copied.a === copied.b
    @test copied.a !== v
    cycle = Any[nothing]
    cycle[1] = cycle
    copied_cycle = DynamicPPL._copy_model_argument(cycle)
    @test copied_cycle[1] === copied_cycle
    @test copied_cycle !== cycle
end

mutable struct InnerConstructorState
    x::Float64
    InnerConstructorState() = new(0.0)
end
@testset "latent structs with inner constructors" begin
    @model sample_inner_state(s) = (s.x ~ Normal(); s.x)
    s = InnerConstructorState()
    @test returned(decondition(sample_inner_state(s)), (s=(x=2.0,),)) == 2.0
    @test s.x == 0.0
end

@testset "partial edits admit named array storage families" begin
    @model function array_storage(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model storage_parent(child) = a ~ to_submodel(child)
    @model nested_storage(x) = (x.a[1] ~ Normal(); x.a[2] ~ Normal(); x)
    @model indexed_storage(x) = (x[1][1] ~ Normal(); x[1][2] ~ Normal(); x)
    components = ComponentVector(; a=1.0, b=2.0)
    supported = (
        [1.0, 2.0], OffsetArray([1.0, 2.0], 0:1), components, DimArray([1.0, 2.0], X)
    )
    rejected = (
        view([1.0, 2.0], :),
        reshape(view([1.0, 2.0], :), 2, 1),
        Transpose([1.0, 2.0]),
        Adjoint([1.0, 2.0]),
        SVector(1.0, 2.0),
        1.0:2.0,
        MVector(1.0, 2.0),
        SizedArray{Tuple{2}}([1.0, 2.0]),
        BitArray([true, false]),
        OffsetArray(view([1.0, 2.0], :), 0:1),
        ComponentVector(view([1.0, 2.0], :), getaxes(components)),
        DimArray(view([1.0, 2.0], :), X),
    )
    for data in rejected,
        (bind, remove, listing) in
        ((condition, decondition, conditioned), (fix, unfix, fixed))
        # Each rejected family retains whole bindings and whole removal.
        whole = bind(array_storage(data); x=data)
        @test listing(whole)[@varname(x)] === data
        @test !haskey(listing(remove(whole, @varname(x))), @varname(x))
        i = lastindex(data)
        vn = @varname(x[i])
        message = r"x.*container type.*whole.*collect"
        @test_throws message bind(array_storage(data), vn => 1)
        @test_throws message remove(whole, vn)
        removal_message = r"ArgumentError: Cannot remove.*x.*container type.*whole.*collect"
        @test_throws removal_message remove(array_storage(data), vn)
        @test_throws removal_message remove(array_storage(data), DynamicPPL.Recursive(), vn)
        @test_throws removal_message remove(
            storage_parent(array_storage(data)), DynamicPPL.Recursive(), @varname(a.x[i])
        )(
            Xoshiro(1)
        )
        @test_throws removal_message remove(
            storage_parent(whole), DynamicPPL.Recursive(), @varname(a.x[i])
        )(
            Xoshiro(1)
        )
        @test_throws message bind(
            storage_parent(array_storage(data)), @varname(a.x[i]) => 1
        )(
            Xoshiro(1)
        )
        @test_throws message bind(nested_storage((a=data,)), @varname(x.a[i]) => 1)
        @test_throws message remove(
            bind(nested_storage((a=data,)); x=(a=data,)), @varname(x.a[i])
        )
        @test_throws message bind(indexed_storage([data]), @varname(x[1][i]) => 1)
        @test_throws message decondition(indexed_storage([data]), @varname(x[1][i]))
        # Untouched leaves need no reconstruction, even when they are unsupported arrays.
        @test listing(bind(nested_storage((a=data,)), @varname(x.a) => data))[@varname(
            x.a
        )] === data
    end
    for good in supported,
        (bind, remove, listing) in
        ((condition, decondition, conditioned), (fix, unfix, fixed))

        j = lastindex(good)
        @test unfix(array_storage(good), @varname(x[j]))(Xoshiro(1)) == good
        @test unfix(array_storage(good), DynamicPPL.Recursive(), @varname(x[j]))(
            Xoshiro(1)
        ) == good
        @test unfix(
            storage_parent(array_storage(good)), DynamicPPL.Recursive(), @varname(a.x[j])
        )(
            Xoshiro(1)
        ) == good
        valid = bind(bind(array_storage(good); x=good), @varname(x[j]) => 3.0)
        @test listing(valid)[@varname(x[j])] == 3.0
        @test axes(listing(valid)[@varname(x)]) == axes(good)
        expected = copy(good)
        expected[j] = 3.0
        @test valid(Xoshiro(1)) == expected
        @test valid(Xoshiro(1)) isa typeof(good)
        @test storage_parent(valid)(Xoshiro(1)) == expected
        removed = remove(storage_parent(valid), DynamicPPL.Recursive(), @varname(a.x[j]))
        @test returned(removed, Dict(@varname(a.x[j]) => 4.0))[j] ==
            (bind === condition ? 4.0 : good[j])
        @test !haskey(listing(remove(valid, @varname(x[j]))), @varname(x[j]))
        @test !haskey(
            conditioned(decondition(array_storage(good), @varname(x[j]))), @varname(x[j])
        )
        @test listing(bind(nested_storage((a=good,)), @varname(x.a[j]) => 3.0))[@varname(
            x.a[j]
        )] == 3.0
        @test listing(bind(indexed_storage([good]), @varname(x[1][j]) => 3.0))[@varname(
            x[1][j]
        )] == 3.0
    end
end

@testset "untouched unassigned argument entries" begin
    @model assigned_argument(x) = (x[1] ~ MvNormal(zeros(1), ones(1)); x)
    x = Vector{Vector{Float64}}(undef, 2)
    x[1] = [1.0]
    for bind in (condition, fix)
        result = bind(assigned_argument(x), @varname(x[1]) => [2.0])(Xoshiro(1))
        @test result[1] == [2.0]
        @test !isassigned(result, 2)
    end
    result = returned(decondition(assigned_argument(x), @varname(x[1])), (x=[[3.0]],))
    @test result[1] == [3.0]
    @test !isassigned(result, 2)
    @test x[1] == [1.0]
    @test !isassigned(x, 2)
end

mutable struct CyclicObservation
    value::Bool
    next::Any
end
struct CyclicObservationDistribution <: DiscreteUnivariateDistribution end
function Distributions.logpdf(::CyclicObservationDistribution, x::CyclicObservation)
    return logpdf(Bernoulli(0.5), x.value)
end
function Distributions.loglikelihood(d::CyclicObservationDistribution, x::CyclicObservation)
    return logpdf(d, x)
end
@testset "cyclic structured observations" begin
    @model cyclic_observation(x) = x ~ CyclicObservationDistribution()
    x = CyclicObservation(true, nothing)
    x.next = x
    @test logjoint(cyclic_observation(x), (;)) ≈ log(0.5)
    for placeholder in (nothing, missing)
        x.next = (x, placeholder)
        @test_throws ArgumentError logjoint(cyclic_observation(x), (;))
    end
end
@testset "partial latent aliases and views" begin
    @model function copy_aliases(x)
        x.a[1] ~ Normal()
        x.c ~ Normal()
        0.0 ~ Normal(x.b[1])
        return (x.a === x.b, x.a[1], x.b[1])
    end
    v = Real[0.0]
    source = copy_aliases((a=v, b=v, c=0.0))
    models = (
        decondition(source, @varname(x.a), @varname(x.b)),
        condition(decondition(source), @varname(x.c) => 0.0),
    )
    for model in models
        @test returned(model, (x=(a=[2.0],),)) == (true, 2.0, 2.0)
        @test loglikelihood(model, (x=(a=[2.0],),)) ≈
            logpdf(Normal(), 0.0) + logpdf(Normal(2.0), 0.0)
        @test v == [0.0]
    end
    @model function copy_views(x)
        x.a[2] ~ Normal()
        0.0 ~ Normal(x.b[1])
        return (x.a[2], x.b[1])
    end
    data = Real[0, 0, 0]
    model = decondition(copy_views((a=view(data, 1:2), b=view(data, 2:3))))
    @test returned(model, (x=(a=[0.0, 2.0],),)) == (2.0, 2.0)
    @test loglikelihood(model, (x=(a=[0.0, 2.0],),)) ≈ logpdf(Normal(2.0), 0.0)
    @test data == [0, 0, 0]
end
@testset "new binding extents preserve surviving latent storage" begin
    @model function stale_shape(x)
        before = x[1]
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return before
    end
    @model parent(child) = a ~ to_submodel(child)
    for T in (Float32, Float64, BigFloat), original in (T[5], T[5, 6, 7])
        model = decondition(condition(stale_shape(original); x=T[1, 2, 3]), @varname(x[1]))
        @test returned(model, (x=T[9, 0, 0],)) == T(5)
        nested = decondition(
            condition(parent(stale_shape(original)), @varname(a.x) => T[1, 2, 3]),
            @varname(a.x[1]),
        )
        @test returned(nested, (a=(x=T[9, 0, 0],),)) == T(5)
    end
    derivative = ForwardDiff.derivative(5.0) do x
        model = decondition(condition(stale_shape([x]); x=[1.0, 2.0, 3.0]), @varname(x[1]))
        returned(model, (x=[9.0, 0.0, 0.0],))
    end
    @test derivative == 1.0
end
@testset "child bindings promote mixed payload types" begin
    @model function promote_child(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model promote_parent(child) = a ~ to_submodel(child)
    model = decondition(
        condition(
            promote_parent(promote_child(zeros(3))), @varname(a.x) => Float32[1, 2, 3]
        ),
        @varname(a.x[1]),
    )
    @test returned(model, (a=(x=[9.0, 0.0, 0.0],),)) isa Vector{Float64}
    @model function promote_body(n)
        m ~ Normal()
        a ~ to_submodel(promote_child(fill(m, n)))
        return a
    end
    body = decondition(
        condition(promote_body(3), @varname(a.x) => [1.0, 2.0, 3.0]), @varname(a.x[1])
    )
    ForwardDiff.derivative(0.5) do m
        value = returned(body, (; m))
        @test eltype(value) <: ForwardDiff.Dual
        sum(value)
    end
end

@testset "bound view parents are not latent storage" begin
    @model function latent_view(x)
        x.a[1] ~ Normal()
        0.0 ~ Normal(x.b[1])
        return (x.a[1], x.b[1])
    end
    data = Real[0.0]
    for value in ((a=view(data, :), b=data), (a=data, b=view(data, :)))
        model = decondition(latent_view(value), @varname(x.a))
        @test returned(model, (x=(a=[2.0],),)) == (2.0, 0.0)
        @test data == [0.0]
    end
    @model function partial_latent_view(x)
        x.a[1] ~ Normal()
        x.a[2] ~ Normal()
        return (x.a[2], x.b[2])
    end
    data = Real[0.0, 0.0]
    removed = decondition(partial_latent_view((a=view(data, :), b=data)), @varname(x.a))
    @test_throws r"x.a\[1\].*SubArray.*whole" condition(removed, @varname(x.a[1]) => 1.0)
    model = condition(
        decondition(partial_latent_view((a=data, b=data)), @varname(x.a)),
        @varname(x.a[1]) => 1.0,
    )
    @test returned(model, (x=(a=[0.0, 2.0],),)) == (2.0, 0.0)
    @test data == [0.0, 0.0]
end

mutable struct PartialInnerState
    x::Float64
    y::Float64
    PartialInnerState() = new(0.0, 1.0)
end
struct ImmutableInnerState
    x::Vector{Float64}
    ImmutableInnerState() = new([0.0])
end
struct AbstractFieldInnerState
    x::Float64
    y::Any
    AbstractFieldInnerState(x::Float64, ::Symbol) = new(x, 0.0)
end
struct CustomConstructorState
    x::Float64
    y::Any
    CustomConstructorState(x::Float64, y::Float64, ::Nothing) = new(x, y)
end
function DynamicPPL.ConstructionBase.constructorof(::Type{CustomConstructorState})
    return (x::Float64, y::Float64) -> CustomConstructorState(x, y, nothing)
end
struct CustomSetterState
    x::Float64
    CustomSetterState(x::Float64, ::Nothing) = new(x)
end
function DynamicPPL.ConstructionBase.setproperties(
    ::CustomSetterState, patch::NamedTuple{(:x,)}
)
    return CustomSetterState(patch.x, nothing)
end
mutable struct UndefinedInnerState
    x::Float64
    unused::Vector{Float64}
    UndefinedInnerState() = new(1.0)
end
mutable struct ConstBindingState
    const offset::Float64
    x::Float64
    ConstBindingState() = new(0.0, 0.0)
end
mutable struct ConstNestedBindingState
    const x::Vector{Float64}
end
@testset "nested bindings check enclosing field replacement" begin
    @model nested_field_binding(s) = (s.x[1] ~ Normal(); s)
    for source in (ConstNestedBindingState([0.0]), ImmutableInnerState()),
        bind in (condition, fix)

        @test_throws r"ArgumentError: .*whole value" bind(
            nested_field_binding(source), @varname(s.x[1]) => 2.0
        )
        whole = bind(nested_field_binding(source); s=source)
        @test (bind === condition ? conditioned : fixed)(whole)[@varname(s)] === source
        @test source.x == [0.0]
    end
end

@testset "const fields require a whole replacement" begin
    @model const_binding_state(s) = (s.x ~ Normal(s.offset); s)
    for bind in (condition, fix)
        @test_throws r"ArgumentError: .*ConstBindingState" bind(
            const_binding_state(ConstBindingState()), @varname(s.offset) => 0.0
        )
        @test bind(const_binding_state(ConstBindingState()); s=ConstBindingState())(
            Xoshiro(1)
        ).x == 0.0
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

@testset "bindings in shared submodel namespaces" begin
    @model child(y=2.0) = y ~ Normal()
    @model parent() = a ~ to_submodel(child())
    @model function shared(y)
        y ~ Normal()
        a ~ to_submodel(child(), false)
        return (y, a)
    end
    for bind in (condition, fix)
        @test bind(parent(), @varname(a.y) => 3.0)(Xoshiro(1)) == 3.0
        @test_throws ArgumentError bind(child(); unknown=3.0)
        @test bind(child(); y=3.0)(Xoshiro(1)) == 3.0
        @test_throws ArgumentError bind(child(), DynamicPPL.Recursive(); y=3.0)
        @test bind(shared(1.0); y=3.0)(Xoshiro(1)) == (3.0, 3.0)
        @test bind(prefix(child(), @varname(a)), @varname(a.y) => 3.0)() == 3.0
    end
    @test fix(condition(shared(1.0); y=4.0); y=3.0)() == (3.0, 3.0)
    @test (parent() | (@varname(a.y) => 3.0))(Xoshiro(1)) == 3.0
    @test conditioned(parent()) == VarNamedTuple()
    @test_throws MethodError conditioned(parent(), DynamicPPL.Recursive())
    @test fixed(parent()) == VarNamedTuple()
    @test_throws MethodError fixed(parent(), DynamicPPL.Recursive())
end

@testset "own LHS bindings beside submodel namespaces" begin
    @model binding_leaf(x=2.0) = x ~ Normal()
    @model function binding_branch(a, observed)
        if observed
            a ~ Normal()
        else
            a ~ to_submodel(binding_leaf())
        end
        return a
    end
    @model function binding_siblings(a)
        a.child ~ to_submodel(binding_leaf())
        a.obs ~ Normal()
        return a
    end
    @model function binding_indices(a)
        a[1] ~ to_submodel(binding_leaf())
        a[2] ~ Normal()
        return a
    end
    @model function binding_dynamic_index(a, index)
        a[index] ~ to_submodel(binding_leaf())
        a[2] ~ Normal()
        return a
    end
    @model function local_siblings()
        a = (obs=0.0, child=0.0)
        a.obs ~ Normal()
        a.child ~ to_submodel(binding_leaf())
        return a
    end
    @model binding_parent(m) = b ~ to_submodel(m)
    for bind in (condition, fix)
        bound = bind(binding_dynamic_index(zeros(2), 1), @varname(a[2]) => 3.0)
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
            Xoshiro(1)
        )
        bound = bind(binding_dynamic_index(zeros(2), 1), @varname(a[1]) => 3.0)
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
            Xoshiro(1)
        )
        @test bind(local_siblings(), @varname(a.obs) => 3.0)(Xoshiro(1)) ==
            (obs=3.0, child=2.0)
        @test bind(binding_branch(1.0, true); a=3.0)(Xoshiro(1)) == 3.0
        return_branch = bind(binding_branch(1.0, false); a=3.0)
        @test_throws ArgumentError return_branch(Xoshiro(1))
        for (m, address, child_address) in (
            (binding_siblings((child=0.0, obs=1.0)), @varname(a.obs), @varname(a.child)),
            (binding_indices(zeros(2)), @varname(a[2]), @varname(a[1])),
        )
            bound = bind(m, address => 3.0)
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
                Xoshiro(1)
            )
            bound = bind(m, child_address => 3.0)
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
                Xoshiro(1)
            )
            pm = prefix(m, @varname(p))
            bound = bind(pm, AbstractPPL.prefix(address, @varname(p)) => 3.0)
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" bound(
                Xoshiro(1)
            )
        end
    end
end

@testset "recursive unfix releases fixed shape ownership" begin
    @model function grows(x)
        x = vcat(x, 0.0)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    old = fix(grows(zeros(2)); x=ones(2))
    for remove in
        (m -> unfix(m, @varname(x)), m -> unfix(m, DynamicPPL.Recursive(), @varname(x)))
        model = fix(remove(old), @varname(x[1]) => 2.0)
        @test model(Xoshiro(1))[1:2] == [2.0, 0.0]
    end
end

@testset "NamedTuple fields use property addresses" begin
    @model fields(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
    @model nested_fields(x) = (x.a[1].b ~ Normal(); x)
    @model indexed(x, i) = (x[i] ~ Normal(); x)
    @model local_indexed(i) = (x = (a=1.0, b=2.0); x[i] ~ Normal(); x)
    @model parent(c) = child ~ to_submodel(c)
    @model local_fields() = (x = (a=0.0, b=0.0); x.a ~ Normal(); x.b ~ Normal(); x)
    data = (a=1.0, b=2.0)
    for (index, message, removal) in (
        (
            1,
            "Integer indexing into a NamedTuple at `x[1]` is unsupported; use `x.a` instead.",
            "Cannot remove `x[1]`: integer indexing into a NamedTuple is unsupported; use `x.a` instead.",
        ),
        (
            :a,
            "Symbol indexing into a NamedTuple at `x[:a]` is unsupported; use `x.a` instead.",
            "Cannot remove `x[:a]`: Symbol indexing into a NamedTuple is unsupported; use `x.a` instead.",
        ),
    )
        vn = @varname(x[index])
        @test_throws ArgumentError(message) indexed(data, index)(Xoshiro(1))
        @test_throws ArgumentError(message) local_indexed(index)(Xoshiro(1))
        @test fields(data)(Xoshiro(1)) == data
        # Ordinary indexing in the body is unaffected.
        @model body_index(x, i) = (x.a ~ Normal(); x[i])
        @test body_index(data, index)(Xoshiro(1)) == 1.0
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            for model in (fields(data), bind(local_fields(); x=data))
                @test_throws ArgumentError(message) bind(model, vn => 3.0)
                @test bind(model, @varname(x.a) => 3.0)(Xoshiro(1)) == (a=3.0, b=2.0)
                for scope in ((), (DynamicPPL.Recursive(),))
                    @test_throws ArgumentError(removal) remove(model, scope..., vn)
                    @test remove(model, scope..., @varname(x.a))(Xoshiro(1)).b == 2.0
                end
            end
            child_vn = AbstractPPL.append_optic(
                @varname(child), AbstractPPL.varname_to_optic(vn)
            )
            @test_throws ArgumentError bind(parent(fields(data)), child_vn => 3.0)(
                Xoshiro(1)
            )
            @test bind(parent(fields(data)), @varname(child.x.a) => 3.0)(Xoshiro(1)) ==
                (a=3.0, b=2.0)
            @test_throws ArgumentError remove(
                parent(bind(fields(data); x=data)), DynamicPPL.Recursive(), child_vn
            )(
                Xoshiro(1)
            )
            @test remove(
                parent(bind(fields(data); x=data)),
                DynamicPPL.Recursive(),
                @varname(child.x.a)
            )(
                Xoshiro(1)
            ).b == 2.0
        end
    end
    for (vn, message, removal) in (
        (
            @varname(x[1, end]),
            "Indexing into a NamedTuple at `x[1, DynamicIndex(end)]` is unsupported; use a field name instead.",
            "Cannot remove `x[1, DynamicIndex(end)]`: indexing into a NamedTuple is unsupported; use a field name instead.",
        ),
        (
            @varname(x[1:2]),
            "Indexing into a NamedTuple at `x[1:2]` is unsupported; use a field name instead.",
            "Cannot remove `x[1:2]`: indexing into a NamedTuple is unsupported; use a field name instead.",
        ),
        (
            @varname(x[:]),
            "Indexing into a NamedTuple at `x[:]` is unsupported; use a field name instead.",
            "Cannot remove `x[:]`: indexing into a NamedTuple is unsupported; use a field name instead.",
        ),
        (
            @varname(x[[1]]),
            "Indexing into a NamedTuple at `x[[1]]` is unsupported; use a field name instead.",
            "Cannot remove `x[[1]]`: indexing into a NamedTuple is unsupported; use a field name instead.",
        ),
        (
            @varname(x[end]),
            "Integer indexing into a NamedTuple at `x[DynamicIndex(end)]` is unsupported; use `x.b` instead.",
            "Cannot remove `x[DynamicIndex(end)]`: integer indexing into a NamedTuple is unsupported; use `x.b` instead.",
        ),
    )
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            for model in (fields(data), bind(local_fields(); x=data))
                @test_throws ArgumentError(message) bind(model, vn => 3.0)
                for scope in ((), (DynamicPPL.Recursive(),))
                    @test_throws ArgumentError(removal) remove(model, scope..., vn)
                end
            end
            @test_throws ArgumentError(message) bind(local_fields(), vn => 3.0)(Xoshiro(1))
        end
    end
    for (vn, message) in (
            (
                @varname(x[1]),
                "Integer indexing into a NamedTuple at `x[1]` is unsupported; use `x.a` instead.",
            ),
            (
                @varname(x[:a]),
                "Symbol indexing into a NamedTuple at `x[:a]` is unsupported; use `x.a` instead.",
            ),
        ),
        bind in (condition, fix)

        @test_throws ArgumentError(message) bind(local_fields(), vn => 3.0)(Xoshiro(1))
    end
    # Parent bindings cannot inspect the child's local storage until evaluation.
    for bind in (condition, fix)
        @test_throws ArgumentError(
            "Integer indexing into a NamedTuple at `child.x[1]` is unsupported; use `child.x.a` instead.",
        ) bind(parent(local_fields()), @varname(child.x[1]) => 3.0)(Xoshiro(1))
        @test_throws ArgumentError(
            "Integer indexing into a NamedTuple at `x[1]` is unsupported; use `x.a` instead.",
        ) bind(local_fields(), VarNamedTuple((@varname(x[1]) => 3.0,)))
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        model = bind(nested_fields((a=[(b=1.0,)],)); x=(a=[(b=1.0,)],))
        for vn in (@varname(x[:a][1].b), @varname(x.a[end][:b]), @varname(x.a[1][1]))
            @test_throws ArgumentError bind(model, vn => 2.0)
            @test bind(model, @varname(x.a[1].b) => 2.0)(Xoshiro(1)) == (a=[(b=2.0,)],)
            @test_throws ArgumentError remove(model, vn)
            @test remove(model, @varname(x.a[1].b))(Xoshiro(1)).a[1].b isa Real
        end
    end
end

@testset "partial edit errors name the enclosing owner" begin
    @model nested(x) = (x.t[1] ~ Normal(); x)
    @model local_nested() = (x = (t=(1.0, 2.0),); x.t[1] ~ Normal(); x)
    @model parent(c) = child ~ to_submodel(c)
    data = (t=(1.0, 2.0),)
    for (bind, remove) in ((condition, decondition), (fix, unfix)),
        model in (nested(data), bind(local_nested(); x=data))

        @test_throws ArgumentError(
            "Cannot partially bind tuple owner `x.t` at `x.t[1]` through container type Tuple{Float64, Float64}; bind or decondition the whole value `x.t` instead.",
        ) bind(model, @varname(x.t[1]) => 3.0)
        @test bind(model, @varname(x.t) => (3.0, 4.0))(Xoshiro(1)).t[1] == 3.0
        bound = bind(model; x=data)
        for scope in ((), (DynamicPPL.Recursive(),))
            @test_throws ArgumentError(
                "Cannot remove `x.t[1]`: cannot partially edit tuple owner `x.t` at `x.t[1]` through container type Tuple{Float64, Float64}; remove the whole value `x.t` instead.",
            ) remove(bound, scope..., @varname(x.t[1]))
            @test isempty(
                (bind === condition ? conditioned : fixed)(
                    remove(bound, scope..., @varname(x.t))
                ),
            )
        end
        stored = bind(parent(nested(data)), @varname(child.x) => data)
        @test_throws ArgumentError(
            "Cannot partially bind tuple owner `child.x.t` at `child.x.t[1]` through container type Tuple{Float64, Float64}; bind or decondition the whole value `child.x.t` instead.",
        ) bind(stored, @varname(child.x.t[1]) => 3.0)
        @test bind(stored, @varname(child.x.t) => (3.0, 4.0))(Xoshiro(1)) == (t=(3.0, 4.0),)
    end
end

@testset "replacement arrays overlay supported observations" begin
    @model elements(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    @model parent(c) = child ~ to_submodel(c)
    @model nested(x) = (x.t[1] ~ Normal(); x)
    @model prefixed_child() = a ~ to_submodel(nested((t=(1.0, 2.0),)))
    for original in ((1.0, 2.0), [1.0, 2.0]), replacement in ([3.0, 4.0], [3.0, 4.0, 5.0])
        expected = copy(replacement)
        expected[1] = 1.0
        for model in (
            unfix(fix(elements(original); x=replacement), @varname(x[1])),
            unfix(
                fix(parent(elements(original)), @varname(child.x) => replacement),
                @varname(child.x[1])
            ),
            unfix(
                parent(fix(elements(original); x=replacement)),
                DynamicPPL.Recursive(),
                @varname(child.x[1])
            ),
        )
            @test model(Xoshiro(1)) == expected
            @test loglikelihood(model, (;)) == logpdf(Normal(), 1.0)
        end
    end
    for (original, message) in (
        (
            Dict(1 => 1.0, 2 => 2.0),
            "Cannot partially bind dictionary owner `x` at `x[2]` through container type Dict{Int64, Float64}; bind or decondition the whole value `x` instead. Or use a NamedTuple/array argument.",
        ),
        (
            (a=1.0, b=2.0),
            "Integer indexing into a NamedTuple at `x[2]` is unsupported; use `x.b` instead.",
        ),
        (
            ObservationRecord(1.0, 2.0),
            "Cannot partially bind struct owner `x` at `x[2]` through container type Main.DynamicPPLConditionFixTests.ObservationRecord{Float64, Float64}; bind or decondition the whole value `x` instead.",
        ),
    )
        model = fix(elements(original); x=[3.0, 4.0])
        @test model(Xoshiro(1)) == [3.0, 4.0]
        @test_throws ArgumentError(message) unfix(model, @varname(x[1]))(Xoshiro(1))
    end
    @test_throws ArgumentError(
        "Cannot partially bind dictionary owner `x` at `x[1]` through container type Dict{Int64, Float64}; bind or decondition the whole value `x` instead. Or use a NamedTuple/array argument.",
    ) fix(elements(Dict(1 => 1.0, 2 => 2.0)), @varname(x[1]) => 3.0)
    for bind in (condition, fix)
        @test_throws ArgumentError(
            "Cannot partially bind tuple owner `a.x.t` at `a.x.t[1]` through container type Tuple{Float64, Float64}; bind or decondition the whole value `a.x.t` instead.",
        ) bind(prefixed_child(), @varname(a.x.t[1]) => 3.0)(Xoshiro(1))
    end
end

# --- Binding contract: independent oracle, generator, and implementation adapter ---

# Independent oracle based on the binding rules in docs/src/conditionfix.md,
# refined through differential checks. It uses no DynamicPPL APIs.
struct BCCase
    kind::Symbol                 # scalar, array, tuple, named, multivariate
    argument::Bool
    depth::Int
    childfixed::Bool
    runtime::Bool
end
struct BCOp
    verb::Symbol
    recursive::Bool              # decondition/unfix only
    path::Union{Nothing,Tuple}
    value::Any
end
bc_contains(p, q) = length(p) <= length(q) && q[1:length(p)] == p
bc_related(p, q) = bc_contains(p, q) || bc_contains(q, p)
bc_rootname(c) = c.kind in (:named, :namedwhole, :nested_named) ? :t : :x
bc_namespace(c) =
    if c.depth == 0
        ()
    elseif c.depth == 1
        (:a,)
    else
        (:a, :b)
    end
bc_basepath(c) = (bc_namespace(c)..., bc_rootname(c))
function bc_basevalue(c)
    return if c.kind == :nested_named
        (a=[2.0, 3.0], b=4.0)
    elseif c.kind == :nested_tuple
        ([2.0, 3.0], 4.0)
    elseif c.kind == :scalar
        2.0
    elseif c.kind in (:named, :namedwhole)
        (a=2.0, b=3.0)
    elseif c.kind in (:tuple, :tuplewhole)
        (2.0, 3.0)
    else
        [2.0, 3.0]
    end
end
function bc_leaves(p, x)
    x isa NamedTuple && return reduce(
        vcat, [bc_leaves((p..., k), v) for (k, v) in pairs(x)]; init=Pair{Tuple,Any}[]
    )
    (x isa Tuple || x isa AbstractArray) && return reduce(
        vcat,
        [bc_leaves((p..., i), v) for (i, v) in enumerate(x)];
        init=Pair{Tuple,Any}[],
    )
    return Pair{Tuple,Any}[p => x]
end
function bc_at(value, path)
    for k in path
        value = k isa Symbol ? getproperty(value, k) : value[k]
    end
    return value
end
function bc_replace_at(value, path, replacement)
    isempty(path) && return replacement
    k = first(path)
    child = bc_replace_at(bc_at(value, (k,)), Base.tail(path), replacement)
    value isa NamedTuple && return merge(value, NamedTuple{(k,)}((child,)))
    value isa Tuple && return ntuple(i -> i == k ? child : value[i], length(value))
    result = copy(value)
    result[k] = child
    return result
end
function bc_shape(c, owners)
    result = bc_basevalue(c)
    bp = bc_basepath(c)
    for p in sort!(collect(keys(owners)); by=length)
        bc_contains(bp, p) &&
            (result = bc_replace_at(result, p[(length(bp) + 1):end], owners[p]))
    end
    return result
end
mutable struct BCLayer
    obs::Dict{Tuple,Any}
    fixed::Dict{Tuple,Any}
    noobs::Set{Tuple}
    nofixed::Set{Tuple}
    owners::Dict{Tuple,Any}
    fixowners::Dict{Tuple,Any}
end
BCLayer() = BCLayer(Dict(), Dict(), Set(), Set(), Dict(), Dict())
mutable struct BCReference
    case::BCCase
    layers::Vector{BCLayer}
    prefixed::Bool
end
function bc_reference(c)
    ls = [BCLayer() for _ in 0:(c.depth)]
    p = bc_basepath(c)
    v = c.runtime ? :twice_parent : bc_basevalue(c)
    if c.argument
        merge!(ls[end].obs, Dict(bc_leaves(p, v)))
        ls[end].owners[p] = bc_basevalue(c)
    end
    if c.childfixed
        v = if c.kind == :scalar
            4.0
        elseif c.kind in (:named, :namedwhole)
            (a=4.0, b=5.0)
        elseif c.kind in (:tuple, :tuplewhole)
            (4.0, 5.0)
        else
            [4.0, 5.0]
        end
        merge!(ls[end].fixed, Dict(bc_leaves(p, v)))
        ls[end].fixowners[p] = v
    end
    return BCReference(c, ls, false)
end
function bc_universe(c)
    p = bc_basepath(c)
    ps = if c.kind in (:nested_named, :nested_tuple)
        first.(bc_leaves(p, bc_basevalue(c)))
    elseif c.kind == :scalar
        [p]
    elseif c.kind in (:named, :namedwhole)
        [(p..., :a), (p..., :b)]
    else
        [(p..., i) for i in 1:4]
    end
    ps = Tuple[ps...]
    c.depth > 0 && push!(ps, (:m,))
    c.depth > 1 && push!(ps, (:a, :q))
    return ps
end
function bc_lookup(r, p)
    blockobs = false
    blockfix = false
    for l in r.layers
        !blockfix && haskey(l.fixed, p) && return (:fixed, l.fixed[p])
        !blockobs && haskey(l.obs, p) && return (:observed, l.obs[p])
        blockobs |= p in l.noobs
        blockfix |= p in l.nofixed
    end
    return (:latent, 0.25)
end
function bc_apply_reference!(r, op)
    op.verb == :prefix && (r.prefixed = true; return :ok)
    p = op.path
    if p !== nothing && r.prefixed
        (isempty(p) || first(p) != :p) && return :call
        p = Base.tail(p)
    end
    c, l = r.case, first(r.layers)
    adding = op.verb in (:condition, :fix)
    fixed = op.verb in (:fix, :unfix)
    d = fixed ? l.fixed : l.obs
    masks = fixed ? l.nofixed : l.noobs
    owners = fixed ? l.fixowners : l.owners
    p === nothing && adding && return :ok
    addresses = bc_universe(c)
    for layer in r.layers
        append!(addresses, keys(layer.obs), keys(layer.fixed), layer.noobs, layer.nofixed)
        for owners in (layer.owners, layer.fixowners), (address, value) in owners
            append!(addresses, first.(bc_leaves(address, value)))
        end
    end
    unique!(addresses)
    targets = p === nothing ? addresses : filter(q -> bc_related(p, q), addresses)
    # Bindings and named removals accept the same addresses.
    if p !== nothing
        isempty(targets) && return c.depth == 0 ? :call : :evaluation
        bp = bc_basepath(c)
        for layer in r.layers
            layer === l || (adding || op.recursive) || continue
            candidates = fixed ? merge(layer.owners, layer.fixowners) : layer.owners
            for (owner, value) in candidates
                bc_contains(owner, p) && length(p) > length(owner) || continue
                !fixed &&
                    layer !== l &&
                    any(q -> bc_contains(q, p), keys(l.fixowners)) &&
                    continue
                for key in p[(length(owner) + 1):end]
                    value isa Tuple && return layer === l ? :call : :evaluation
                    if value isa NamedTuple
                        key isa Symbol && haskey(value, key) || break
                    elseif value isa AbstractArray
                        key isa Int && checkbounds(Bool, value, key) || break
                    else
                        break
                    end
                    value = bc_at(value, (key,))
                end
            end
        end
        if c.argument &&
            bc_contains(bp, p) &&
            length(p) > length(bp) &&
            bc_basevalue(c) isa Tuple &&
            !(c.depth > 0 && !fixed && any(q -> bc_contains(q, p), keys(l.fixowners)))
            return c.depth == 0 ? :call : :evaluation
        end
        # Supported partial paths must fit the edited layer's latest shape owner.
        if bc_contains(bp, p) && length(p) > length(bp)
            template = bc_shape(c, owners)
            template = bc_at(template, p[(length(bp) + 1):(end - 1)])
            k = last(p)
            if template isa NamedTuple
                k isa Symbol && haskey(template, k) || return :call
            elseif template isa Tuple || template isa AbstractArray
                k isa Int && 1 <= k <= length(template) || return :call
            else
                return :call
            end
        end
    end
    if adding
        for q in collect(keys(d))
            bc_contains(p, q) && delete!(d, q)
        end
        merge!(d, Dict(bc_leaves(p, op.value)))
        # Any binding in this layer replaces the removal at its address.
        for q in targets
            delete!(masks, q)
        end
        for q in collect(keys(owners))
            bc_contains(p, q) && delete!(owners, q)
        end
        owners[p] = op.value
    else
        for q in targets
            delete!(d, q)
            op.recursive && push!(masks, q)
        end
        for q in collect(keys(owners))
            (p === nothing || bc_contains(p, q)) && delete!(owners, q)
        end
    end
    return :ok
end
function bc_expected(r)
    c = r.case
    bp = bc_basepath(c)
    # Argument storage follows the highest surviving shape owner; locals keep body storage.
    template = bc_basevalue(c)
    if c.argument
        for l in reverse(r.layers)
            for owners in (l.owners, l.fixowners)
                for p in sort!(collect(keys(owners)); by=length)
                    bc_contains(bp, p) && (
                        template = bc_replace_at(
                            template, p[(length(bp) + 1):end], owners[p]
                        )
                    )
                end
            end
        end
    end
    ps = Tuple[first.(bc_leaves(bp, template))...]
    c.depth > 0 && pushfirst!(ps, (:m,))
    c.depth > 1 && insert!(ps, 2, (:a, :q))
    out = Dict{Tuple,Tuple{Symbol,Any}}()
    for p in ps
        role, value = bc_lookup(r, p)
        value === :twice_parent && (value = 2 * out[(:m,)][2])
        out[p] = (role, value)
    end
    if c.kind in (:multivariate, :tuplewhole, :namedwhole)
        ks = c.kind == :namedwhole ? (:a, :b) : (1, 2)
        entries = [out[(bp..., i)] for i in ks]
        all(e -> e[1] == entries[1][1], entries) || return (:evaluation, out)
        # Current implementation limitation: growable indexed storage cannot
        # establish a whole multivariate binding, even when all elements are bound.
        if c.kind == :multivariate && !c.argument && entries[1][1] != :latent
            any(l -> haskey(l.owners, bp) || haskey(l.fixowners, bp), r.layers) ||
                return (:evaluation, out)
        end
        for i in ks
            delete!(out, (bp..., i))
        end
        out[bp] = (entries[1][1], bc_structured(c, last.(entries)))
    end
    if r.prefixed
        out = Dict((:p, k...) => v for (k, v) in out)
    end
    return (:ok, out)
end
function bc_predict(c, ops)
    r = bc_reference(c)
    for (i, op) in enumerate(ops)
        stage = bc_apply_reference!(r, op)
        stage == :call && return (stage, i, nothing)
        stage == :evaluation && return (:evaluation, length(ops), nothing)
    end
    stage, out = bc_expected(r)
    return (stage, length(ops), out)
end
function bc_structured(c, xs)
    return if c.kind == :namedwhole
        (a=xs[1], b=xs[2])
    elseif c.kind == :tuplewhole
        Tuple(xs)
    else
        xs
    end
end

# Implementation adapter: models return all reached values, including fixed ones.
@model function bc_scalar_arg(x)
    x ~ Normal()
    return (; x)
end
@model function bc_scalar_local()
    x ~ Normal()
    return (; x)
end
@model function bc_array_arg(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return (; x)
end
@model function bc_array_local()
    x = zeros(2)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return (; x)
end
@model function bc_tuple_arg(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return (; x)
end
@model function bc_tuple_local()
    x = (0.0, 0.0)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return (; x)
end
@model function bc_named_arg(t)
    t.a ~ Normal()
    t.b ~ Normal()
    return (; t)
end
@model function bc_named_local()
    t = (a=0.0, b=0.0)
    t.a ~ Normal()
    t.b ~ Normal()
    return (; t)
end
@model function bc_nested_named_arg(t)
    for i in eachindex(t.a)
        t.a[i] ~ Normal()
    end
    t.b ~ Normal()
    return (; t)
end
@model function bc_nested_tuple_arg(x)
    for i in eachindex(x[1])
        x[1][i] ~ Normal()
    end
    x[2] ~ Normal()
    return (; x)
end
@model function bc_mv_arg(x)
    x ~ MvNormal(zeros(2), 1.0)
    return (; x)
end
@model function bc_mv_local()
    x ~ MvNormal(zeros(2), 1.0)
    return (; x)
end
# A tuple-valued product of two standard normals, used only with BCQuarter initialisation.
struct BCTupleNormal <: Distributions.ContinuousMultivariateDistribution end
Base.length(::BCTupleNormal) = 2
Distributions.logpdf(::BCTupleNormal, x::Tuple) = sum(logpdf.(Normal(), x))
Distributions.loglikelihood(d::BCTupleNormal, x::Tuple) = logpdf(d, x)
@model function bc_tuple_whole(x)
    x ~ BCTupleNormal()
    return (; x)
end
@model function bc_named_whole(t)
    t ~ product_distribution((a=Normal(), b=Normal()))
    return (; t)
end
@model function bc_middle(child)
    q ~ Normal()
    b ~ to_submodel(child)
    return (; q, b)
end
@model function bc_outer(child)
    m ~ Normal()
    a ~ to_submodel(child)
    return (; m, a)
end
@model function bc_runtime_outer()
    m ~ Normal()
    a ~ to_submodel(condition(bc_scalar_arg(2.0); x=2m))
    return (; m, a)
end
function bc_model(c)
    c.runtime && return bc_runtime_outer()
    constructors = Dict(
        :tuplewhole => (bc_tuple_whole, bc_tuple_whole),
        :namedwhole => (bc_named_whole, bc_named_whole),
        :nested_named => (bc_nested_named_arg, bc_nested_named_arg),
        :nested_tuple => (bc_nested_tuple_arg, bc_nested_tuple_arg),
        :scalar => (bc_scalar_arg, bc_scalar_local),
        :array => (bc_array_arg, bc_array_local),
        :tuple => (bc_tuple_arg, bc_tuple_local),
        :named => (bc_named_arg, bc_named_local),
        :multivariate => (bc_mv_arg, bc_mv_local),
    )
    f, g = constructors[c.kind]
    m = c.argument ? f(bc_basevalue(c)) : g()
    if c.childfixed
        v = if c.kind == :scalar
            4.0
        elseif c.kind in (:named, :namedwhole)
            (a=4.0, b=5.0)
        elseif c.kind in (:tuple, :tuplewhole)
            (4.0, 5.0)
        else
            [4.0, 5.0]
        end
        m = fix(m, (bc_rootname(c) == :x ? @varname(x) : @varname(t)) => v)
    end
    c.depth == 2 && (m = bc_middle(m))
    c.depth > 0 && (m = bc_outer(m))
    return m
end
# Addresses are built once via @varname; the reference uses only tuples.
const BC_ADDRESSES = let d = Dict{Tuple,Any}()
    for (p, v) in [
        ((:x,), @varname(x)),
        ((:t,), @varname(t)),
        ((:m,), @varname(m)),
        ((:a, :q), @varname(a.q)),
        ((:t, :a), @varname(t.a)),
        ((:t, :b), @varname(t.b)),
        ((:t, :absent), @varname(t.absent)),
        ((:a, :x), @varname(a.x)),
        ((:a, :t), @varname(a.t)),
        ((:a, :t, :a), @varname(a.t.a)),
        ((:a, :t, :b), @varname(a.t.b)),
        ((:a, :b, :x), @varname(a.b.x)),
        ((:a, :b, :t), @varname(a.b.t)),
        ((:a, :b, :t, :a), @varname(a.b.t.a)),
        ((:a, :b, :t, :b), @varname(a.b.t.b)),
    ]
        d[p] = v
    end
    for i in 1:4
        d[(:t, :a, i)] = @varname(t.a[i])
        d[(:x, 1, i)] = @varname(x[1][i])
        d[(:x, i)] = @varname(x[i])
        d[(:t, i)] = @varname(t[i])
        d[(:a, :x, i)] = @varname(a.x[i])
        d[(:a, :b, :x, i)] = @varname(a.b.x[i])
    end
    for (p, v) in collect(d)
        if first(p) in (:x, :t)
            d[(:a, p...)] = AbstractPPL.prefix(v, @varname(a))
            d[(:a, :b, p...)] = AbstractPPL.prefix(v, @varname(a.b))
        end
    end
    for (p, v) in collect(d)
        d[(:p, p...)] = AbstractPPL.prefix(v, @varname(p))
    end
    d
end
function bc_apply_actual(m, op)
    @nospecialize m
    op.verb == :prefix && return prefix(m, @varname(p))
    f = getfield(DynamicPPL, op.verb)
    args =
        op.verb in (:decondition, :unfix) && op.recursive ? (DynamicPPL.Recursive(),) : ()
    op.path === nothing && return f(m, args...)
    v = BC_ADDRESSES[op.path]
    return if op.verb in (:condition, :fix)
        f(m, args..., v => deepcopy(op.value))
    else
        f(m, args..., v)
    end
end
struct BCTrace <: DynamicPPL.AbstractAccumulator
    events::Vector{Tuple{String,Symbol,Any}}
end
BCTrace() = BCTrace(Tuple{String,Symbol,Any}[])
DynamicPPL.accumulator_name(::BCTrace) = :BCTrace
DynamicPPL.reset(::BCTrace) = BCTrace()
Base.copy(t::BCTrace) = BCTrace(deepcopy(t.events))
function DynamicPPL.accumulate_assume!!(t::BCTrace, val, tval, jac, vn, dist, template)
    push!(t.events, (string(vn), :latent, deepcopy(val)))
    return t
end
function DynamicPPL.accumulate_observe!!(t::BCTrace, dist, val, vn, template)
    push!(t.events, (string(vn), :observed, deepcopy(val)))
    return t
end
struct BCQuarter <: DynamicPPL.AbstractInitStrategy end
function DynamicPPL.init(rng, vn, dist::Distribution, ::BCQuarter)
    v = if dist isa BCTupleNormal
        (0.25, 0.25)
    elseif dist isa Distributions.ProductNamedTupleDistribution
        (a=0.25, b=0.25)
    elseif dist isa MultivariateDistribution
        fill(0.25, length(dist))
    else
        0.25
    end
    return DynamicPPL.TransformedValue(v, DynamicPPL.NoTransform())
end
function bc_actual(c, ops, m=bc_model(c))
    for (i, op) in enumerate(ops)
        try
            m = bc_apply_actual(m, op)
        catch e
            return (:call, i, e)
        end
    end
    try
        vi = VarInfo(BCTrace(), LogPriorAccumulator(), LogLikelihoodAccumulator())
        value, vi = init!!(StableRNG(17), m, vi, BCQuarter(), UnlinkAll())
        trace = DynamicPPL.getacc(vi, Val(:BCTrace))
        vals = Dict(bc_leaves((), value))
        if c.kind in (:multivariate, :tuplewhole, :namedwhole)
            bp = bc_basepath(c)
            ks = c.kind == :namedwhole ? (:a, :b) : (1, 2)
            vals[bp] = bc_structured(c, [pop!(vals, (bp..., i)) for i in ks])
        end
        any(o -> o.verb == :prefix, ops) &&
            (vals = Dict((:p, p...) => v for (p, v) in vals))
        roles = Dict(vn => role for (vn, role, _) in trace.events)
        result = Dict(
            p => (get(roles, string(BC_ADDRESSES[p]), :fixed), v) for (p, v) in vals
        )
        # Returned observed/latent values must agree with what the evaluator scored.
        scored = Dict(vn => v for (vn, _, v) in trace.events)
        for (p, (_, value)) in result
            vn = string(BC_ADDRESSES[p])
            haskey(scored, vn) &&
                scored[vn] != value &&
                error("BCTrace/return mismatch at $vn")
        end
        return (:ok, length(ops), (result, getlogprior(vi), getloglikelihood(vi)))
    catch e
        return (:evaluation, length(ops), e)
    end
end
function bc_agrees(pred, act)
    pred[1:2] == act[1:2] || return false
    pred[1] != :ok && return act[3] isa ArgumentError || act[3] isa BoundsError
    exp, (got, lp, ll) = pred[3], act[3]
    exp == got || return false
    function bc_density(role)
        return sum(
            (
                sum(logpdf(Normal(), x) for (_, x) in bc_leaves((), v)) for
                (r, v) in values(exp) if r == role
            );
            init=0.0,
        )
    end
    return isapprox(lp, bc_density(:latent); atol=1e-12) &&
           isapprox(ll, bc_density(:observed); atol=1e-12)
end
function bc_sequence(rng, c, n; invalid_removals=false)
    ops = BCOp[]
    prefixed = false
    for _ in 1:n
        if !prefixed && rand(rng) < 0.07
            push!(ops, BCOp(:prefix, false, nothing, nothing))
            prefixed = true
            continue
        end
        verb = rand(rng, (:condition, :fix, :decondition, :unfix))
        bp = bc_basepath(c)
        paths = if c.kind == :scalar
            [bp]
        elseif c.kind in (:named, :namedwhole)
            [bp, (bp..., :a), (bp..., :b)]
        else
            [bp, (bp..., 1), (bp..., 2)]
        end
        paths = Tuple[paths...]
        c.depth > 0 && push!(paths, (:m,))
        c.depth > 1 && push!(paths, (:a, :q))
        p = rand(rng) < 0.15 ? nothing : rand(rng, paths)
        v = Float64(rand(rng, -3:3))
        if p == bp && c.kind != :scalar
            v = if c.kind in (:named, :namedwhole)
                (a=v, b=v + 1)
            elseif c.kind in (:tuple, :tuplewhole)
                (v, v + 1)
            else
                [v, v + 1]
            end
        end
        if invalid_removals &&
            verb in (:decondition, :unfix) &&
            c.kind != :scalar &&
            rand(rng) < 0.25
            p = (bp..., c.kind in (:named, :namedwhole) ? :absent : 4)
        end
        p !== nothing && prefixed && (p = (:p, p...))
        recursive = rand(rng, Bool) && verb in (:decondition, :unfix)
        push!(ops, BCOp(verb, recursive, p, v))
    end
    return ops
end
function bc_corpus()
    rng = StableRNG(1501)
    cs = vcat(
        [
            BCCase(k, a, 0, false, false) for
            k in (:scalar, :array, :tuple, :named, :multivariate) for a in (false, true)
        ],
        [
            BCCase(:namedwhole, true, 0, false, false),
            BCCase(:tuplewhole, true, 0, false, false),
            BCCase(:scalar, true, 1, true, false),
            BCCase(:array, true, 2, false, false),
            BCCase(:scalar, true, 1, false, true),
        ],
    )
    # Keep longer histories and local no-name removals; directed cases cover short edits.
    # Generate skipped histories too, so trimming preserves the StableRNG stream.
    return [
        (c, ops) for c in cs for i in 1:6 for
        ops in (bc_sequence(rng, c, i),) if i > 2 || any(
            op -> op.verb in (:decondition, :unfix) && !op.recursive && op.path === nothing,
            ops,
        )
    ]
end

function bc_invalid_removal_corpus()
    # A separate stream leaves every existing random history unchanged.
    rng = StableRNG(1502)
    return [
        (c, bc_sequence(rng, c, n; invalid_removals=true)) for
        kind in (:array, :tuple, :named, :multivariate, :tuplewhole, :namedwhole) for
        c in (BCCase(kind, true, 0, false, false),) for n in 1:6
    ]
end

function bc_shape_corpus()
    samples = Tuple{BCCase,Vector{BCOp}}[]
    for kind in (:array, :tuple, :nested_named, :nested_tuple)
        c = BCCase(kind, true, 0, false, false)
        p = if kind == :nested_named
            (:t, :a)
        elseif kind == :nested_tuple
            (:x, 1)
        else
            (:x,)
        end
        whole = kind == :tuple ? (4.0, 5.0, 6.0) : [4.0, 5.0, 6.0]
        for bind in (:condition, :fix)
            remove = bind == :condition ? :decondition : :unfix
            ops = [BCOp(bind, false, p, whole), BCOp(remove, false, (p..., 3), nothing)]
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(bind, false, (p..., 3), 7.0))
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(remove, false, p, nothing))
            push!(samples, (c, copy(ops)))
        end
    end
    # Cross-layer owners of different extents, including a complete partial mask.
    for kind in (:array, :tuple, :nested_named, :nested_tuple), depth in (0, 1, 2)
        c = BCCase(kind, true, depth, false, false)
        p = (bc_namespace(c)..., (
            if kind == :nested_named
                (:t, :a)
            elseif kind == :nested_tuple
                (:x, 1)
            else
                (:x,)
            end
        )...)
        shape(x) = kind == :tuple ? Tuple(x) : x
        for bind in (:condition, :fix)
            other = bind == :condition ? :fix : :condition
            ops = [
                BCOp(bind, false, p, shape([4.0, 5.0, 6.0])),
                BCOp(other, false, (p..., 1), 7.0),
                BCOp(other, false, (p..., 2), 8.0),
            ]
            push!(samples, (c, ops))
        end
        for (nobs, nfix) in ((3, 2), (2, 3)), reverse_order in (false, true)
            ops = [
                BCOp(:condition, false, p, shape(collect(1.0:nobs))),
                BCOp(:fix, false, p, shape(collect(4.0:(3 + nfix)))),
            ]
            reverse_order && reverse!(ops)
            push!(ops, BCOp(:unfix, false, (p..., 2), nothing))
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(:unfix, false, p, nothing))
            push!(samples, (c, copy(ops)))
        end
    end
    return samples
end

function bc_contract_corpus()
    samples = Tuple{BCCase,Vector{BCOp}}[]
    for fixed in (false, true), recursive in (false, true)
        c = BCCase(:scalar, true, 1, fixed, false)
        for (bind, remove) in ((:condition, :decondition), (:fix, :unfix))
            ops = [
                BCOp(bind, false, (:a, :x), 3.0), BCOp(remove, recursive, (:a, :x), nothing)
            ]
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(remove, recursive, (:a, :x), nothing))
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(bind, false, (:a, :x), 7.0))
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(remove, false, (:a, :x), nothing))
            push!(samples, (c, copy(ops)))
        end
    end
    # Recursive removals must reach bindings two submodels below the caller.
    for fixed in (false, true), remove in (:decondition, :unfix)
        c = BCCase(:scalar, true, 2, fixed, false)
        for p in ((:a, :b, :x), nothing)
            push!(samples, (c, [BCOp(remove, true, p, nothing)]))
        end
    end
    # Clearing the local table must preserve earlier recursive removal markers.
    for fixed in (false, true), remove in (:decondition, :unfix)
        c = BCCase(:scalar, true, 1, fixed, false)
        ops = [BCOp(remove, true, (:a, :x), nothing), BCOp(remove, false, nothing, nothing)]
        push!(samples, (c, ops))
    end
    for verb in (:condition, :fix)
        for argument in (false, true)
            c = BCCase(:multivariate, argument, 0, false, false)
            ops = [BCOp(verb, false, (:x, 1), 2.0), BCOp(verb, false, (:x, 2), 3.0)]
            push!(samples, (c, ops))
            push!(samples, (c, [BCOp(verb, false, (:x,), [2.0, 3.0])]))
        end
    end
    for verb in (:condition, :fix, :decondition, :unfix)
        push!(
            samples,
            (
                BCCase(:named, true, 0, false, false),
                [BCOp(verb, false, (:t, :absent), 1.0)],
            ),
        )
        push!(
            samples,
            (BCCase(:named, true, 0, false, false), [BCOp(verb, false, (:t, 1), 1.0)]),
        )
        push!(
            samples,
            (BCCase(:array, true, 0, false, false), [BCOp(verb, false, (:x, 4), 1.0)]),
        )
    end
    return samples
end

@model function bc_shared_namespace(x, child)
    x ~ Normal()
    a ~ to_submodel(child, false)
    return (; x, a)
end

@model function bc_named_branch(t, observed)
    if observed
        t.a ~ Normal()
        t.b ~ Normal()
    else
        t ~ to_submodel(bc_scalar_arg(2.0))
    end
    return (; t)
end

@testset "binding contract: randomized differential test" begin
    samples = vcat(
        bc_corpus(), bc_shape_corpus(), bc_contract_corpus(), bc_invalid_removal_corpus()
    )
    # Local indexed bindings intentionally exercise the documented growable-array fallback.
    with_logger(NullLogger()) do
        for (c, ops) in samples
            predicted, observed = bc_predict(c, ops), bc_actual(c, ops)
            @test bc_agrees(predicted, observed)
        end
    end
    # Recursive scope belongs only to removals; bindings already accept child addresses.
    for bind in (condition, fix)
        m = bc_outer(bc_scalar_arg(2.0))
        @test bind(m, @varname(a.x) => 3.0)(StableRNG(1)).a.x == 3.0
        @test_throws ArgumentError bind(m, DynamicPPL.Recursive(), @varname(a.x) => 3.0)
    end
    # Partial edits are valid until an argument-rooted submodel tilde is reached.
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        observed = bind(bc_named_branch((a=1.0, b=2.0), true), @varname(t.a) => 3.0)
        @test observed(StableRNG(1)).t.a == 3.0
        @test remove(observed, @varname(t.a))(StableRNG(1)).t.a isa Real
        submodel = bind(bc_named_branch((a=1.0, b=2.0), false), @varname(t.a) => 3.0)
        @test_throws r"ArgumentError: Submodel tilde .*model argument `t`.*local LHS" submodel(
            StableRNG(1)
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `t`.*local LHS" remove(
            submodel, @varname(t.a)
        )(
            StableRNG(1)
        )
    end
    let
        c = BCCase(:scalar, false, 1, false, false)
        for verb in (:decondition, :unfix)
            ops = [BCOp(verb, true, (:m,), nothing)]
            observed = bc_actual(c, ops)
            @test bc_agrees(bc_predict(c, ops), observed)
        end
    end
    let
        for (child, expected_child) in
            ((bc_scalar_arg(3.0), (:observed, 3.0)), (bc_scalar_local(), (:latent, 0.25)))
            m = bc_shared_namespace(2.0, child)
            value, vi = init!!(
                StableRNG(1), m, VarInfo(BCTrace()), BCQuarter(), UnlinkAll()
            )
            events = DynamicPPL.getacc(vi, Val(:BCTrace)).events
            @test events == [("x", :observed, 2.0), ("x", expected_child...)]
            @test value.x == 2.0
            @test value.a.x == expected_child[2]
        end
        # Any binding replaces the marker, including in a shared namespace.
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            child = bind(bc_scalar_arg(3.0); x=4.0)
            m = remove(bc_shared_namespace(2.0, child), DynamicPPL.Recursive())
            m = bind(m; x=7.0)
            value = m(StableRNG(1))
            @test value.x == value.a.x == 7.0
            # Removing only the new binding uncovers the child's binding again.
            @test remove(m, @varname(x))(StableRNG(1)).a.x == 4.0
        end
        # A removal on the child cannot erase a binding held by its parent.
        child = decondition(bc_scalar_arg(2.0), DynamicPPL.Recursive(), @varname(x))
        m = condition(bc_outer(child), @varname(a.x) => 3.0)
        value, vi = init!!(StableRNG(1), m, VarInfo(BCTrace()), BCQuarter(), UnlinkAll())
        @test value.a.x == 3.0
        @test last(DynamicPPL.getacc(vi, Val(:BCTrace)).events) == ("a.x", :observed, 3.0)
    end
end

# --- End binding contract ---

@model closed_leaf(x=3.0) = x ~ Normal()
@model function closed_branch(a, run)
    if run
        a[1] ~ Normal()
    else
        a ~ to_submodel(closed_leaf())
    end
    return a
end
@model function closed_sibling(a)
    a.obs ~ Normal()
    a.child ~ to_submodel(closed_leaf())
    return a
end
@testset "argument submodel tildes after partial edits" begin
    @model function local_sibling()
        a = (obs=0.0, child=0.0)
        a.obs ~ Normal()
        a.child ~ to_submodel(closed_leaf())
        return a
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        observed = bind(closed_branch([1.0, 2.0], true), @varname(a[1]) => 4.0)
        @test observed(StableRNG(1)) == [4.0, 2.0]
        @test remove(observed, @varname(a[1]))(StableRNG(1))[2] == 2.0
        submodel = bind(closed_branch([1.0, 2.0], false), @varname(a[1]) => 4.0)
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" submodel(
            StableRNG(1)
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" remove(
            submodel, @varname(a[1])
        )(
            StableRNG(1)
        )
    end
    sibling = decondition(closed_sibling((obs=1.0, child=0.0)), @varname(a.obs))
    @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" sibling(
        StableRNG(1)
    )
    @test condition(local_sibling(), @varname(a.obs) => 2.0)(StableRNG(1)) ==
        (obs=2.0, child=3.0)
end

@testset "tuple and struct owners require whole edits" begin
    @model element(x) = (x[1] ~ Normal(); x)
    @model field(x) = (x.a ~ Normal(); x)
    @model array_tuple(x) = (x[1][1] ~ Normal(); x)
    @model named_tuple(x) = (x.a[1] ~ Normal(); x)
    @model tuple_array(x) = (x[1][1] ~ Normal(); x)
    @model array_struct(x) = (x[1].a ~ Normal(); x)
    @model struct_array(x) = (x.a[1] ~ Normal(); x)
    @model named_struct(x) = (x.a.a ~ Normal(); x)
    @model parent(child) = a ~ to_submodel(child)
    cases = (
        (element, (1.0, 2.0), @varname(x[1]), @varname(x), (3.0, 2.0)),
        (
            field,
            ObservationRecord(1.0, 2.0),
            @varname(x.a),
            @varname(x),
            ObservationRecord(3.0, 2.0),
        ),
        (array_tuple, [(1.0, 2.0)], @varname(x[1][1]), @varname(x[1]), (3.0, 2.0)),
        (named_tuple, (a=(1.0, 2.0),), @varname(x.a[1]), @varname(x.a), (3.0, 2.0)),
        (tuple_array, ([1.0, 2.0],), @varname(x[1][1]), @varname(x), ([3.0, 2.0],)),
        (
            array_struct,
            [ObservationRecord(1.0, 2.0)],
            @varname(x[1].a),
            @varname(x[1]),
            ObservationRecord(3.0, 2.0),
        ),
        (
            struct_array,
            ObservationRecord([1.0, 2.0], 0.0),
            @varname(x.a[1]),
            @varname(x),
            ObservationRecord([3.0, 2.0], 0.0),
        ),
        (
            named_struct,
            (a=ObservationRecord(1.0, 2.0),),
            @varname(x.a.a),
            @varname(x.a),
            ObservationRecord(3.0, 2.0),
        ),
    )
    for (make, data, address, owner, replacement) in cases
        model = make(data)
        @test loglikelihood(model, (;)) ≈ logpdf(Normal(), 1.0)
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            for base in (model, bind(model; x=data), decondition(model))
                @test_throws r"ArgumentError: .*x.*(Tuple|Record).*whole value" bind(
                    base, address => 9.0
                )
                whole = bind(base, owner => replacement)
                @test loglikelihood(whole, (;)) ≈
                    (bind === condition ? logpdf(Normal(), 3.0) : 0.0)
            end
            bound = bind(model; x=data)
            for scope in ((), (DynamicPPL.Recursive(),))
                @test_throws r"ArgumentError: .*x.*(Tuple|Record).*whole value" remove(
                    bound, scope..., address
                )
                @test isempty(
                    (bind === condition ? conditioned : fixed)(
                        remove(bound, scope..., @varname(x))
                    ),
                )
            end
            child_address = AbstractPPL.append_optic(
                @varname(a), AbstractPPL.varname_to_optic(address)
            )
            @test_throws ArgumentError bind(parent(model), child_address => 9.0)(Xoshiro(1))
            @test loglikelihood(bind(parent(model), @varname(a.x) => data), (;)) ≈
                (bind === condition ? logpdf(Normal(), 1.0) : 0.0)
        end
    end
    @model pairs_argument(x) = (x[:a] ~ Normal(); x)
    data = pairs((a=1.0, b=2.0))
    model = pairs_argument(data)
    @test loglikelihood(model, (;)) ≈ logpdf(Normal(), 1.0)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        @test_throws r"ArgumentError: .*Base.Pairs.*whole value" bind(
            model, @varname(x[:a]) => 3.0
        )
        whole = bind(model; x=data)
        @test (bind === condition ? conditioned : fixed)(whole)[@varname(x)] === data
        @test_throws r"ArgumentError: .*Base.Pairs.*whole value" remove(
            whole, @varname(x[:a])
        )
        @test isempty(
            (bind === condition ? conditioned : fixed)(remove(whole, @varname(x)))
        )
    end
    @model local_fields() = (
        x = ObservationRecord(0.0, 0.0); x.a ~ Normal(); x.b ~ Normal(); x
    )
    @model local_tuple() = (x = (0.0, 0.0); x = ((x[1] ~ Normal()), (x[2] ~ Normal())); x)
    @model local_array() = (x = zeros(2); x[1] ~ Normal(); x[2] ~ Normal(); x)
    for (model, producer) in
        ((local_fields(), local_fields()), (local_tuple(), local_array()))
        values = rand(Xoshiro(42), producer)
        for bind in (condition, fix)
            @test isempty(keys(VarInfo(Xoshiro(42), bind(model, values))))
            for (address, value) in pairs(values)
                @test (bind === condition ? conditioned : fixed)(
                    bind(model, address => value)
                )[address] == value
            end
        end
    end
    @model struct_observation(s) = (s.x ~ Normal(); s)
    for source in (
        PartialInnerState(),
        AbstractFieldInnerState(0.0, :init),
        CustomConstructorState(0.0, 1.0, nothing),
        CustomSetterState(0.0, nothing),
        UndefinedInnerState(),
    )
        model = struct_observation(source)
        for bind in (condition, fix)
            @test_throws r"ArgumentError: .*whole value" bind(model, @varname(s.x) => 2.0)
            whole = bind(model; s=source)
            @test (bind === condition ? conditioned : fixed)(whole)[@varname(s)] === source
        end
    end
    @model immutable_inner(s) = (original = s; s = 0.0; s ~ Normal(); original)
    source = ImmutableInnerState()
    result = returned(decondition(immutable_inner(source)), (s=2.0,))
    @test result.x == [0.0]
    @test result.x !== source.x
end

end
