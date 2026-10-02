module DynamicPPLConditionFixTests

using AbstractPPL: of, @of
using Dates: now
using ADTypes: AutoForwardDiff
using ComponentArrays: ComponentVector
using Distributions
using DimensionalData: DimArray, X
using DynamicPPL
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using LogDensityProblems: LogDensityProblems
using OffsetArrays: OffsetArray
using Test
using Random: Xoshiro
using StaticArrays: SVector

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

@testset "condition and fix" begin
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
                @test_throws message bind(model, templated)
                @test_throws message bind(model, of((;)), @varname(x[1]) => 2.0)
                @test_throws message model | (@varname(x[1]) => 2.0)
                @test_throws message model | templated
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
        for (model, value, address, message) in (
            (
                integer_lhs,
                (a=1.0, b=2.0),
                @varname(x[1]),
                "ArgumentError: Integer indexing into a NamedTuple at `x[1]` is unsupported; use `x.a` instead.",
            ),
            (
                nested_integer_lhs,
                (p=(a=1.0, b=2.0),),
                @varname(x.p[1]),
                "ArgumentError: Integer indexing into a NamedTuple at `x.p[1]` is unsupported; use `x.p.a` instead.",
            ),
            (
                array_integer_lhs,
                [(a=1.0, b=2.0)],
                @varname(x[1][1]),
                "ArgumentError: Integer indexing into a NamedTuple at `x[1][1]` is unsupported; use `x[1].a` instead.",
            ),
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
                    expected = if remove === unfix && origin !== fix
                        "no fixed binding is stored"
                    elseif remove === decondition && origin === fix
                        "supplies no template"
                    else
                        message
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
                expected = if remove === unfix && bind === condition
                    "no fixed binding is stored"
                else
                    message
                end
                @test_throws expected remove(m, @varname(x[1]))
            end
        end
        for value in ((1.0, 2.0), [1.0, 2.0]),
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
                    (
                        placeholder_array(),
                        placeholder_array(nothing),
                        placeholder_keyword_array(),
                        placeholder_keyword_array(; x=nothing),
                    ),
                    @varname(x),
                    [@varname(x[1]), @varname(x[2])],
                    [1.0, 2.0],
                ),
                (
                    (
                        placeholder_scalar(),
                        placeholder_scalar(nothing),
                        placeholder_keyword_scalar(),
                        placeholder_keyword_scalar(; y=nothing),
                    ),
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
        @test_throws "Cannot remove `x`: no fixed binding is stored at this address." unfix(
            placeholder_array(), :x
        )

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
        parent = placeholder_parent()
        @test conditioned(parent)[@varname(a)] === nothing
        @test keys(VarInfo(Xoshiro(1), parent)) == [@varname(a.y)]
        @test parent(Xoshiro(1)) == rand(Xoshiro(1), Normal())
        for bind in (condition, fix)
            @test_throws ArgumentError bind(parent; a=1.0)(Xoshiro(1))
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
        for value in ((; x=3.0, y=4.0), pairs((; x=3.0, y=4.0)))
            m = fix(indexed_keywords(; x=2.0, y=5.0); kwargs=value)
            @test loglikelihood(unfix(m, @varname(kwargs[:x])), (;)) ==
                logpdf(Normal(), 2.0)
            @test loglikelihood(unfix(m, :kwargs), (;)) ==
                logpdf(Normal(), 2.0) + logpdf(Normal(), 5.0)
        end
    end

    @testset "deconditioned arguments retain index bounds" begin
        @model indexed_lhs_argument(x) = (x[1] ~ Normal(); x)
        for bind in (condition, fix), x in ([1.0, 2.0], (1.0, 2.0))
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
        for ctor in (Dict, IdDict)
            key = Ref(:a)
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
        for ctor in ((X, y) -> (; X, y), LatentRecord, MutableLatentRecord)
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
            for name in (whole, part)
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
        for change in (
                x -> vcat(x, 2.0),
                x -> x[1:1],
                x -> reshape(x, 1, 2),
                x -> resize!(copy(x), 1),
            ),
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

    @testset "missing is rejected only when an LHS variable reads it" begin
        @model metadata_lhs(p) = (p.a ~ Normal(); p.a)
        @model indexed_observation(y) = begin
            for i in 1:2
                y[i] ~ Normal()
            end
            y
        end
        @model whole_observation(y) = y ~ MvNormal(zeros(2), I)
        @model scalar_observation(y) = y ~ Normal()
        for p in ((a=1.0, b=missing), (a=1.0, b=[(missing,)]), MetadataRecord(1.0, missing))
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
        for (constructor, values, message) in (
            (
                metadata_lhs,
                (; p=(a=missing, b=1.0)),
                r"ArgumentError: .*`p.a`.*missing.*latent.*decondition",
            ),
            (
                indexed_observation,
                (; y=[1.0, missing, 3.0]),
                r"ArgumentError: .*`y\[2\]`.*missing.*latent.*decondition",
            ),
            (
                indexed_observation,
                (; y=[1.0, missing]),
                r"ArgumentError: .*`y\[2\]`.*missing.*latent.*decondition",
            ),
            (
                whole_observation,
                (; y=[1.0, missing]),
                r"ArgumentError: .*`y`.*missing.*latent.*decondition",
            ),
            (
                scalar_observation,
                (; y=missing),
                r"ArgumentError: .*`y`.*missing.*latent.*decondition",
            ),
        )
            model = constructor(only(values))
            @test_throws message model(Xoshiro(1))
            for bind in (condition, fix)
                bound = bind(model; values...)
                diagnostic = if bind === fix
                    Regex(replace(message.pattern, "decondition" => "unfix"))
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
        for data in ([1.0, 2.0], (1.0, 2.0)), bind in (condition, fix)
            model = bind(indexed(data), @varname(y[1]) => 4.0)
            @test_throws r"ArgumentError: .*`y\[3\]`.*outside.*`y`" bind(
                model, @varname(y[3]) => 9.0
            )
            @test_throws r"ArgumentError: .*outside" bind(
                model, @varname(y[2:3]) => [8.0, 9.0]
            )
            @test bind(model, @varname(y[2]) => 5.0)(Xoshiro(1)) ==
                (data isa Tuple ? (4.0, 5.0) : [4.0, 5.0])
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

    @testset "removal requires a stored binding of the requested role" begin
        @model scalar() = x ~ Normal()
        @model inner_arg(x=1.0) = x ~ Normal()
        @model outer_arg() = a ~ to_submodel(inner_arg())
        for (bind, remove, other_role) in
            ((condition, unfix, "conditioned"), (fix, decondition, "fixed"))
            @test_throws r"ArgumentError: .*`x`.*no .* binding" remove(
                bind(scalar(); x=1.0), :x
            )
            @test_throws r"ArgumentError: .*`unknown`" remove(scalar(), :unknown)
            @test isempty(keys(conditioned(remove(scalar()))))
            @test isempty(keys(fixed(remove(scalar()))))
        end
        @model indexed(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
        for data in ([1.0, 2.0], (1.0, 2.0)),
            (bind, remove, select) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))

            partial = remove(bind(indexed(data); x=data), @varname(x[2]))
            @test isempty(select(remove(partial, @varname(x[1:2][1]))))
            @test isempty(select(remove(partial, @varname(x[1:2]))))
            @test_throws r"ArgumentError: .*`x\[2\]`" remove(partial, @varname(x[2]))
        end
        @test_throws r"ArgumentError: .*`a.x`.*[Dd]econdition.*child.*to_submodel" decondition(
            outer_arg(), @varname(a.x)
        )
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            parent = bind(outer_arg(), @varname(a.x) => 2.0)
            @test remove(parent, @varname(a.x))(Xoshiro(1)) == 1.0
            @test_throws r"ArgumentError: .*`a.x`" remove(
                remove(parent, @varname(a.x)), @varname(a.x)
            )
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
                    "tuple",
                    tuple_lhs_variables(),
                    (; x=(1.0, 2.0)),
                    @varname(x[1]),
                    @varname(x[2]),
                    zeros(2),
                ),
                (
                    "tuple trailing removal",
                    tuple_lhs_variables(),
                    (; x=(2.0, 1.0)),
                    @varname(x[2]),
                    @varname(x[1]),
                    zeros(1),
                ),
                (
                    "tuple argument",
                    decondition(array_lhs_variables((1.0, 2.0))),
                    (; x=(1.0, 2.0)),
                    @varname(x[1]),
                    @varname(x[2]),
                    zeros(2),
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
                        @test getlogjoint(restored_vi) == getlogjoint(vi)
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
            @test_throws ArgumentError remove(matrix, @varname(x[8]))
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
        for original in ([1.0, 2.0], (1.0, 2.0))
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
            op(model, (@varname(x) => x,)),
        )
        for transformed in transformed_models
            test_logp_correct(op, transformed, x)
        end
        if op === condition
            test_logp_correct(condition, model | VarNamedTuple(; x), x)
            test_logp_correct(condition, model | (; x), x)
            test_logp_correct(condition, model | (@varname(x) => x,), x)
            test_logp_correct(condition, model | (@varname(x) => x), x)
            test_logp_correct(condition, model | (@varname(x) => x,), x)
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

    @testset "partial bindings of static-array arguments" begin
        @model function static_argument(x)
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end
        @model static_parent(child) = a ~ to_submodel(child)
        data = SVector(1.0, 2.0)
        original = static_argument(data)
        for bind in (condition, fix)
            changed = bind(original, @varname(x[1]) => 3.0)
            @test changed(Xoshiro(1)) == SVector(3.0, 2.0)
            @test static_parent(changed)(Xoshiro(1)) == SVector(3.0, 2.0)
            @test original(Xoshiro(1)) === data
            removed = decondition(changed, @varname(x[2]))
            @test !haskey(conditioned(removed), @varname(x[2]))
            @test bind(changed, @varname(x[2]) => 4.0f0)(Xoshiro(1)) == SVector(3.0, 4.0)
            @test loglikelihood(changed, VarNamedTuple()) ≈
                logpdf(Normal(), 2.0) + (bind === condition ? logpdf(Normal(), 3.0) : 0.0)
        end
        latent = decondition(original, @varname(x[1]))
        @test !haskey(conditioned(latent), @varname(x[1]))
        @test conditioned(latent)[@varname(x[2])] == 2.0
        @test Set(keys(rand(Xoshiro(1), latent))) == Set([@varname(x[1])])
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
        for bind in (condition, fix),
            (model, vn) in
            ((scalar_lhs(1.0), @varname(a[1])), (scalar_return(0.0), @varname(a.x)))

            @test_throws r"ArgumentError: .*`a`.*decondition" bind(model, vn => 2.0)
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
            @test_throws ArgumentError remove(record, @varname(t.absent))
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

    @testset "partial property overrides preserve struct fields" begin
        @model function record_lhs_variables(x)
            x.a ~ Normal()
            x.b[1] ~ Normal()
            x.b[2] ~ Normal()
            return x
        end
        data = ObservationRecord(1.0, [2.0, 3.0])
        for first_op in (condition, fix), last_op in (condition, fix)
            original = first_op(record_lhs_variables(data); x=data)
            changed = last_op(original, @varname(x.a) => 4.0, @varname(x.b[1]) => 5.0)
            result = changed()
            @test result isa ObservationRecord
            @test result.a == (first_op === fix && last_op === condition ? 1.0 : 4.0)
            @test result.b ==
                (first_op === fix && last_op === condition ? [2.0, 3.0] : [5.0, 3.0])
            @test isempty(keys(VarInfo(changed)))
            @test logjoint(changed, VarNamedTuple()) ≈
                (first_op === condition ? logpdf(Normal(), 3.0) : 0.0) + (
                if first_op === condition && last_op === condition
                    sum(logpdf.(Normal(), [4.0, 5.0]))
                else
                    0.0
                end
            )
            @test original().a == data.a == 1.0
            @test original().b == data.b == [2.0, 3.0]
        end
        @test_throws ArgumentError condition(
            record_lhs_variables(data), @varname(x.unknown) => 1.0
        )
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

        base = condition(fields(ObservationRecord(0.0, 0.0)); x=ReplacementRecord(1.0, 2.0))
        loglik =
            p -> loglikelihood(
                condition(fields(ReplacementRecord(zero(p), zero(p))), @varname(x.a) => p),
                VarNamedTuple(),
            )
        @test ForwardDiff.derivative(loglik, 3.0) == -3.0
        selected = conditioned(condition(base, @varname(x.a) => 3.0))
        @test conditioned(base)[@varname(x)] isa ReplacementRecord
        @test condition(fields(ObservationRecord(0.0, 0.0)), conditioned(base))() isa
            ReplacementRecord
        @test selected[@varname(x)] isa ReplacementRecord
        @test condition(fields(ObservationRecord(0.0, 0.0)), selected)().a == 3.0
        mixed = fix(base, @varname(x.a) => 3.0)
        supplied = merge(conditioned(mixed), fixed(mixed))
        @test supplied[@varname(x)] isa VarNamedTuple
        @test supplied[@varname(x.a)] == 3.0
        @test supplied[@varname(x.b)] == 2.0

        @model namespace_child(x, y) = (x ~ Normal(); y ~ Normal(); return (x, y))
        @model namespace_parent() = child ~ to_submodel(namespace_child(0.0, 0.0))
        parent = condition(namespace_parent(); child=(; x=1.0, y=2.0))
        parent = decondition(fix(parent, @varname(child.x) => 3.0), @varname(child.y))
        @test parent() == (3.0, 0.0)
    end

    @testset "indexed tuple bindings preserve tuples and roles" begin
        @model tuple_lhs_variables(x) = (x[1] ~ Normal(); x[2] ~ Normal(); return x)
        @model nested_tuple(m) = child ~ to_submodel(m)
        for T in (Float32, BigFloat),
            first_op in (condition, fix),
            last_op in (condition, fix)

            original = tuple_lhs_variables((T(1), T(2)))
            first = first_op(original, @varname(x[1]) => T(3))
            changed = last_op(first, @varname(x[2]) => T(4))
            @test changed() == (T(3), T(4))
            @test Tuple(merge(conditioned(changed), fixed(changed))[@varname(x)]) ==
                (T(3), T(4))
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
            @test isempty(conditioned(conditioned_model))
            @test conditioned(unfix(conditioned_model))[vn] == 3.0
            @test merge(conditioned(fixed_model), fixed(fixed_model))[@varname(x)] ==
                expected
            rebuilt = fix(
                condition(decondition(original), conditioned(fixed_model)),
                fixed(fixed_model),
            )
            @test rebuilt(Xoshiro(42)) == expected
            nested = nested_tuple(fixed_model)
            @test decondition(nested)() == expected
        end
        loglik =
            p -> loglikelihood(
                condition(
                    tuple_lhs_variables((zero(p), oftype(p, 2))), @varname(x[1]) => p
                ),
                VarNamedTuple(),
            )
        @test ForwardDiff.derivative(loglik, 3.0) == -3.0
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
        m = argument_lhs(1.0)
        for original in (m, condition(m; x=2.0), decondition(m, :x))
            u = unfix(fix(original; x=5.0), :x)
            @test conditioned(u) == conditioned(original)
            @test rand(Xoshiro(1), u) == rand(Xoshiro(1), original)
        end
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

    @testset "unfix restores slice-prefixed arguments" begin
        @model slice_argument(x=1.0) = x ~ Normal()
        m = DynamicPPL.prefix(
            fix(slice_argument(); x=3.0), @varname(p[1:2]); template=zeros(2)
        )
        restored = unfix(m)
        @test conditioned(restored)[@varname(p[1:2].x)] == 1.0
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

@testset "nothing is rejected only where a tilde reads it" begin
    @model absence_field(p) = p.a ~ Normal()
    @model absence_array(x) = x ~ product_distribution([Normal(), Normal()])
    for absent in (missing, nothing)
        message = "LHS variable `p.a` contains `$absent`; make it latent with `decondition`."
        @test_throws message absence_field((a=absent, b=1.0))(Xoshiro(1))
        @test absence_field((a=1.0, b=absent))(Xoshiro(1)) == 1.0
        @test_throws "LHS variable `x` contains `$absent`" absence_array([1.0, absent])(
            Xoshiro(1)
        )
        @test_throws "make it latent with `unfix`" fix(
            absence_field((a=1.0, b=2.0)); p=(a=absent, b=2.0)
        )(
            Xoshiro(1)
        )
    end
end

@testset "binding input forms are ordered" begin
    @model input_forms(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    for bind in (condition, fix)
        for invalid in (
            Dict(@varname(x) => [2.0, 3.0]),
            Dict(:x => [2.0, 3.0]),
            pairs((; x=[2.0, 3.0])),
            [@varname(x) => [2.0, 3.0]],
            1,
        )
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
    @model address_callable(Normal) = a ~ Normal()
    @test condition(address_callable(() -> to_submodel(address_child(), false)); y=2.0)(
        Xoshiro(1)
    ) == 2.0
    for bind in (condition, fix)
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

@testset "binding schemas under slice namespaces" begin
    @model function slice_schema_local()
        z = zeros(3)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model function slice_schema_parent(child)
        b ~ to_submodel(child)
        return b
    end
    schema = @of(z = of(Array, 3))
    for op in (condition, fix)
        for (prefix, address) in (
            (@varname(a[:]), @varname(a[:].z[2])),
            (@varname(a[1:2]), @varname(a[1:2].z[2])),
            (@varname(a[:]), @varname(a[1:2].z[2])),
            (@varname(a[2:2]), @varname(a[2:2].z[2])),
        )
            model = DynamicPPL.prefix(slice_schema_local(), prefix; template=zeros(2))
            bound = op(model, address => 1.0, schema)
            @test bound(Xoshiro(1)) == op(model, address => 1.0)(Xoshiro(1))
            @test slice_schema_parent(bound)(Xoshiro(1)) == bound(Xoshiro(1))
        end
        model = DynamicPPL.prefix(slice_schema_local(), @varname(a[:]); template=zeros(2))
        @test op(model, @varname(a[:].z[end]) => 1.0, schema)(Xoshiro(1))[3] == 1.0
        @test op(model, @varname(a[:].z[:]) => ones(3), schema)(Xoshiro(1)) == ones(3)
        @test_throws ArgumentError op(model, @varname(a[:].z[4]) => 1.0, schema)
        @test_throws ArgumentError op(
            model, @varname(a[:].z[2]) => 0.1, @of(z = of(Array, Float32, 3))
        )
        nested = DynamicPPL.prefix(model, @varname(b[1:2]); template=zeros(2))
        @test op(nested, @varname(b[1:2].a[:].z[2]) => 1.0, schema)(Xoshiro(1))[2] == 1.0
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
    @test_throws "Integer indexing into a NamedTuple" unfix(
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

@testset "removal through whole bindings with nested tuples" begin
    @model nested_tuple_fields(p) = (p.b[1] ~ Normal(); p.b[2] ~ Normal(); p)
    @model tuple_removal_parent(child) = a ~ to_submodel(child)
    data = (b=(1.0, 2.0),)
    m = nested_tuple_fields(data)
    latent = Dict(@varname(p.b[1]) => 8.0)
    @test returned(decondition(m, @varname(p.b[1])), latent) == (b=(8.0, 2.0),)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        whole = bind(decondition(m); p=data)
        partial = remove(whole, @varname(p.b[1]))
        @test returned(partial, latent) == (b=(8.0, 2.0),)
        @test returned(tuple_removal_parent(partial), Dict(@varname(a.p.b[1]) => 8.0)) ==
            (b=(8.0, 2.0),)
        @test_throws r"no .* binding is stored" remove(whole, @varname(p.b[3]))
    end
    @test unfix(fix(m; p=(b=(3.0, 4.0),)), @varname(p.b[1]))(Xoshiro(1)) == (b=(1.0, 4.0),)

    @model deeper_tuple(p) = (p[1].b[1][1] ~ Normal(); p)
    deep = deeper_tuple(((b=((1.0, 2.0),),),))
    @test returned(
        decondition(deep, @varname(p[1].b[1][1])), Dict(@varname(p[1].b[1][1]) => 8.0)
    ) == ((b=((8.0, 2.0),),),)
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

@testset "partial bindings preserve array types" begin
    @model typed_static_argument(x::SVector{2,Float64}) = (
        x[1] ~ Normal(); x[2] ~ Normal(); x
    )
    @model typed_view_argument(x::SubArray) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    for (model, x) in (
        (typed_static_argument, SVector(1.0, 2.0)),
        (typed_view_argument, view([1.0, 2.0], :)),
    )
        for bind in (condition, fix)
            result = bind(model(x), @varname(x[1]) => 3.0)(Xoshiro(1))
            @test result isa typeof(x)
            @test result == [3.0, 2.0]
            @test x == [1.0, 2.0]
        end
    end
    @model replacement_array(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    for owner in (Float32[1, 2], SVector(1.0f0, 2.0f0), view(Float32[1, 2], :))
        for bind in (condition, fix)
            changed = bind(replacement_array(view([1.0, 2.0], :)); x=owner)
            changed = bind(changed, @varname(x[1]) => 3)
            changed = bind(changed, @varname(x[2]) => 4)
            result = changed(Xoshiro(1))
            @test result isa typeof(owner)
            @test result == [3, 4]
        end
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
@info "Completed $(@__FILE__) in $(now() - __now__)."

end
