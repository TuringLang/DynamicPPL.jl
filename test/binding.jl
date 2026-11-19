module BindingStorageTests

using DynamicPPL, Distributions, Test, LinearAlgebra
using AbstractPPL: AbstractPPL, @of, of
using Accessors: @set
using ComponentArrays: ComponentVector
using DimensionalData: DimArray, X
using OffsetArrays: OffsetArray
using Random: Xoshiro

@model function element_storage(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
@model field_storage(x) = (x.a[1] ~ Normal(); x)
@model function read_argument(x)
    if false
        x ~ Normal()
    end
    return x
end

struct Record{T}
    a::T
end
mutable struct MutableRecord{T}
    a::T
end
struct CustomProperties
    a::Float64
end
Base.propertynames(::CustomProperties) = (:b,)
function Base.getproperty(x::CustomProperties, n::Symbol)
    return n === :b ? getfield(x, :a) : getfield(x, n)
end
struct KeywordOnly
    a::Float64
    KeywordOnly(; a) = new(a)
end
function DynamicPPL._element_optic(
    ::Union{Record,MutableRecord,CustomProperties,KeywordOnly}
)
    return AbstractPPL.Property
end

@testset "opted-in field storage" begin
    for T in (Record, MutableRecord)
        data = T([1.0, 2.0])
        model = condition(decondition(field_storage(data)), @varname(x.a[1]) => 3.0)
        result = model(Xoshiro(1))
        @test result isa T && result.a == [3.0, 2.0]
        @test decondition(model, @varname(x.a[1]))(Xoshiro(1)).a[1] ==
            rand(Xoshiro(1), Normal())
        @test data.a == [1.0, 2.0]
    end
    @model whole_record(r) = r ~ product_distribution((a=Normal(),))
    for T in (Record, MutableRecord)
        latent = decondition(whole_record(T(1.0)))
        whole = fix(latent; r=T(2.0))(Xoshiro(1))
        parts = fix(latent, @varname(r.a) => 2.0)(Xoshiro(1))
        @test typeof(parts) === typeof(whole) && parts.a == whole.a
    end
    @test_throws ErrorException decondition(read_argument(CustomProperties(1.0)))
    @test_throws MethodError decondition(read_argument(KeywordOnly(; a=1.0)))
end

struct FieldArray{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::FieldArray) = size(x.data)
Base.getindex(x::FieldArray, i::Int) = x.data[i]
DynamicPPL._element_optic(::FieldArray) = AbstractPPL.Property
@model field_array(x) = (x.data[1] ~ Normal(); x)

@testset "arrays addressed by fields" begin
    data = FieldArray([1.0])
    result = decondition(field_array(data), @varname(x.data[1]))(Xoshiro(1))
    @test result isa FieldArray && result.data == [rand(Xoshiro(1), Normal())]
    @test data.data == [1.0]
    @test_throws ArgumentError decondition(field_array(data), @varname(x[1]))
end

struct ForeignNumber <: Number
    value::Float64
end
mutable struct FieldStorage
    a::Vector{Float64}
end
@model inserted_dictionary(x, d) = (x[1] = d; x[1][:a] ~ Normal(); x)

@testset "structural leaves" begin
    data = Any[ForeignNumber(1.0), :tag, Float64]
    result = decondition(read_argument(data))(Xoshiro(1))
    @test result !== data && all(result .=== data)
    @test decondition(read_argument(Float64))(Xoshiro(1)) === Float64
    record = FieldStorage([1.0])
    @test_throws "unsupported binding storage" decondition(field_storage(record))(
        Xoshiro(1)
    )
    @test record.a == [1.0]
    d = Dict(:a => 1.0)
    @test_throws "unsupported binding storage" decondition(inserted_dictionary(Any[0.0], d))(
        Xoshiro(1)
    )
    @test d[:a] == 1.0
end

@testset "partial wrapper argument bindings are rejected" begin
    for data in (OffsetArray([1.0, 2.0], 0:1), DimArray([1.0, 2.0], X))
        model = element_storage(data)
        address = @varname(x[firstindex(data)])
        @test_throws ArgumentError condition(model, address => 3.0)
        @test_throws ArgumentError decondition(model, address)
    end
end

@testset "ComponentArray index bindings and copies" begin
    data = ComponentVector(; a=1.0, b=2.0)
    latent = decondition(element_storage(data))
    @test condition(latent, @varname(x[1]) => 3.0)(Xoshiro(1))[1] == 3.0
    @test_throws ArgumentError condition(latent, @varname(x.a) => 3.0)
    result = latent(Xoshiro(1))
    result[1] = 100.0
    @test data == ComponentVector(; a=1.0, b=2.0)
end

@testset "complete record bindings" begin
    @model record_model() = r ~ product_distribution((a=Normal(), b=Normal()))
    @model record_storage(r) = r ~ product_distribution((a=Normal(), b=Normal()))
    template = @of(r = @of(a = of(Float64), b = of(Float64)))
    for op in (condition, fix)
        whole = op(record_model(); r=(a=1.0, b=2.0))
        unshaped = op(record_model(), @varname(r.a) => 1.0, @varname(r.b) => 2.0)
        @test_throws r"whole array for array elements.*flat records also accept an @of template" unshaped(
            Xoshiro(1)
        )
        partial = op(record_model(), @varname(r.a) => 1.0, template)
        @test_throws "cannot mix bound and latent fields" partial(Xoshiro(1))
        parts = op(partial, @varname(r.b) => 2.0, template)
        @test parts(Xoshiro(1)) == whole(Xoshiro(1))
        @test logjoint(parts, (;)) == logjoint(whole, (;))
        reversed = op(
            op(record_model(), @varname(r.b) => 2.0, template),
            @varname(r.a) => 1.0,
            template,
        )
        @test reversed(Xoshiro(1)) === whole(Xoshiro(1))
        for shaped in (whole, decondition(record_storage((a=0.0, b=0.0))))
            bound = op(shaped, @varname(r.b) => 2.0, @varname(r.a) => 1.0)
            @test bound(Xoshiro(1)) === whole(Xoshiro(1))
        end
    end
    mixed = condition(
        fix(record_model(), @varname(r.a) => 1.0, template), @varname(r.b) => 2.0, template
    )
    @test_throws "cannot mix binding roles" mixed(Xoshiro(1))
end

struct RecordPair{T}
    a::T
    b::T
end
DynamicPPL._element_optic(::RecordPair) = AbstractPPL.Property
struct RecordArrayDistribution <: ContinuousMultivariateDistribution end
Base.length(::RecordArrayDistribution) = 2
function Distributions.logpdf(::RecordArrayDistribution, x::Union{AbstractVector,Tuple})
    return sum(
        r -> logpdf(Normal(), r.a) + (r isa RecordPair ? 2 : 1) * logpdf(Normal(), r.b), x
    )
end
function Distributions.loglikelihood(
    d::RecordArrayDistribution, x::Union{AbstractVector,Tuple}
)
    return logpdf(d, x)
end

@testset "whole-array record bindings" begin
    @model array_argument(r) = (r ~ RecordArrayDistribution(); r)
    @model function array_local(data)
        r = data
        r ~ RecordArrayDistribution()
        return r
    end
    fields = (
        @varname(r[1].b) => 2.0,
        @varname(r[1].a) => 1.0,
        @varname(r[2].a) => 3.0,
        @varname(r[2].b) => 4.0,
    )
    for op in (condition, fix)
        for data in (
            [(a=1.0, b=2.0), (a=3.0, b=4.0)],
            [RecordPair(1.0, 2.0), RecordPair(3.0, 4.0)],
            ((a=1.0, b=2.0), (a=3.0, b=4.0)),
            (RecordPair(1.0, 2.0), RecordPair(3.0, 4.0)),
            Any[RecordPair(1.0, 2.0), RecordPair(3.0, 4.0)],
            Union{RecordPair{Float64},NamedTuple{(:a, :b),Tuple{Float64,Float64}}}[
                RecordPair(1.0, 2.0), (a=3.0, b=4.0)
            ],
        )
            latent = decondition(array_argument(data))
            changed = @set data[2].b = 9.0
            for (whole, parts) in (
                (op(latent; r=data), op(latent, fields...)),
                (
                    op(array_local(data); r=changed),
                    op(op(array_local(data); r=data), @varname(r[2].b) => 9.0),
                ),
            )
                result, expected = parts(Xoshiro(1)), whole(Xoshiro(1))
                @test isequal(result, expected)
                @test typeof(result) === typeof(expected)
                @test logjoint(parts, (;)) == logjoint(whole, (;))
            end
        end
    end
    latent = decondition(array_argument(((a=1.0, b=2.0), (a=3.0, b=4.0))), @varname(r[1].a))
    @test_throws "record fields need stored shape" latent(Xoshiro(1))
end

@testset "indexed tuple binding return type" begin
    @model function tuple_tilde()
        x = (0.0, 0.0)
        x ~ MvNormal(zeros(2), I)
        return x
    end
    for op in (condition, fix)
        partial = op(tuple_tilde(), @varname(x[1]) => 1.0, @of(x = of(Array, 2)))
        @test_throws r"a tilde cannot mix bound and latent positions$" partial(Xoshiro(1))
        model = op(op(tuple_tilde(); x=(1.0, 2.0)), @varname(x[1]) => 3.0)
        @test model(Xoshiro(1)) === (3.0, 2.0)
        # The template gives only the length; the tuple comes from the model.
        full = op(
            tuple_tilde(),
            @varname(x[1]) => 1.0,
            @varname(x[2]) => 2.0,
            @of(x = of(Array, 2))
        )
        @test full(Xoshiro(1)) === (1.0, 2.0)
    end
end

@testset "mixed tuple bindings" begin
    @model function mixed_argument(x)
        x[1] ~ Normal()
        x[2] ~ Poisson()
        return x
    end
    model = decondition(mixed_argument((0.5, 3)))
    @test condition(model, @varname(x[1]) => 1.5)(Xoshiro(1)) isa Tuple{Float64,Int}
    @test fix(model, @varname(x[2]) => 4)(Xoshiro(1))[2] === 4
    @test_throws "storage element type Int64 is inexact" fix(model, @varname(x[2]) => 4.5)
    partial = decondition(mixed_argument((0.5, 3)), @varname(x[1]))
    listed = condition(model, conditioned(partial))
    @test isequal(listed(Xoshiro(1)), partial(Xoshiro(1)))
    @test logjoint(listed, (; x=(0.2, 3))) == logjoint(partial, (; x=(0.2, 3)))
    @test fix(partial, @varname(x[1]) => 2.0)(Xoshiro(1)) === (2.0, 3)
    fixed_only = fix(model, @varname(x[2]) => 4)
    @test isempty(conditioned(fixed_only))
    @test decondition(fixed_only)(Xoshiro(1))[2] === 4
    observed_only = condition(model, @varname(x[1]) => 1.5)
    @test isempty(fixed(observed_only))
    @test unfix(observed_only)(Xoshiro(1))[1] === 1.5

    @model tuple_records(x) = (x[1].a ~ Normal(); x[2].a ~ Normal(); x)
    records = condition(
        decondition(tuple_records(((a=0.0,), (a=0.0,)))), @varname(x[2].a) => 2.0
    )
    @test records(Xoshiro(1))[2] === (a=2.0,)
    @test keys(rand(Xoshiro(1), records)) == [@varname(x[1].a)]

    @model tuple_of_vector(x) = (x[1][1] ~ Normal(); x[1][2] ~ Normal(); x)
    partial_slot = decondition(tuple_of_vector(([1.0, 2.0],)), @varname(x[1][1]))
    @test keys(rand(Xoshiro(1), partial_slot)) == [@varname(x[1][1])]
    @model tuple_of_record() = (x = ((a=0.0,), 0.0); x[1].a ~ Normal(); x)
    bound = decondition(
        condition(tuple_of_record(); x=([1.0, 2.0], 3.0)), @varname(x[1][2])
    )
    @test_throws "address kind disagrees" bound()

    @model vector_of_tuples(x) = (x[1][1] ~ Normal(); x)
    m = decondition(vector_of_tuples([(0.5, 3)]))
    @test_throws "storage type Tuple{Float64, Int64} cannot hold [1.0, 2.0]" condition(
        m, @varname(x[1]) => [1.0, 2.0]
    )(
        Xoshiro(1)
    )

    @model inner() = (y ~ Normal(); y)
    @model function tuple_submodel()
        x = (0.0, 0.0)
        x[1] ~ to_submodel(inner())
        return x
    end
    draw = rand(Xoshiro(1), tuple_submodel())
    @test draw.data.x isa DynamicPPL.VarNamedTuples.PartialTuple
    @test keys(draw) == [@varname(x[1].y)]
end

@testset "tuple slices" begin
    model = decondition(element_storage((0.0, 0.0, 0.0)))
    for (op, value) in ((condition, (1.0, 2.0)), (fix, [1.0, 2.0]))
        @test op(model, @varname(x[1:2]) => value)(Xoshiro(1))[1:2] === (1.0, 2.0)
    end
    @test fix(model, @varname(x[:]) => [1.0, 2.0, 3.0])(Xoshiro(1)) === (1.0, 2.0, 3.0)
    @test_throws "selection requires a tuple or array value" condition(
        model, @varname(x[1:2]) => 1.0
    )
    @test_throws "selection and value lengths must match" condition(
        model, @varname(x[1:2]) => (1.0,)
    )
    @model tuple_slice(x) = (x[1:2] ~ MvNormal(zeros(2), I); x)
    @test tuple_slice((1.0, 2.0))(Xoshiro(1)) === (1.0, 2.0)
    for m in (
        decondition(tuple_slice((1.0, 2.0))),
        decondition(tuple_slice((1.0, 2.0)), @varname(x[2])),
    )
        @test_throws "Cannot index a tuple with `x[1:2]`" m(Xoshiro(1))
    end
end

@testset "record argument provenance" begin
    @model record_argument(r) = r ~ product_distribution((a=Normal(), b=Normal()))
    @model wrap_record(c) = child ~ to_submodel(c)
    data = (a=1.0, b=2.0)
    original = record_argument(data)
    edited = condition(decondition(original, @varname(r.a)), @varname(r.a) => 1.0)
    @test edited(Xoshiro(1)) == original(Xoshiro(1))
    @test logjoint(edited, (;)) == logjoint(original, (;))
    for child in (original, edited)
        @test_throws "submodels cannot observe through arguments" wrap_record(child)(
            Xoshiro(1)
        )
    end
    explicit = condition(decondition(edited, @varname(r.b)), @varname(r.b) => 2.0)
    for child in
        (explicit, condition(original; r=data), condition(decondition(original); r=data))
        @test wrap_record(child)(Xoshiro(1)) == data
    end
    ancestor = condition(
        decondition(wrap_record(original), @varname(child.r.a)), @varname(child.r.a) => 1.0
    )
    @test_throws "submodels cannot observe through arguments" ancestor(Xoshiro(1))
    @model nested_record(r) =
        r ~ product_distribution((
            a=product_distribution((left=Normal(), right=Normal())), b=Normal()
        ))
    nested_data = (a=(left=1.0, right=2.0), b=3.0)
    nested = nested_record(nested_data)
    edited_nested = condition(
        decondition(nested, @varname(r.a.left)), @varname(r.a.left) => 1.0
    )
    @test edited_nested(Xoshiro(1)) == nested_data
    @test logjoint(edited_nested, (;)) == logjoint(nested, (;))
    @test_throws "submodels cannot observe through arguments" wrap_record(edited_nested)(
        Xoshiro(1)
    )
end

@testset "half-missing vector argument allocations" begin
    @model function vector_observations(y)
        for i in eachindex(y)
            y[i] ~ MvNormal(zeros(2), I)
        end
    end
    data = Union{Missing,Vector{Float64}}[zeros(2) for _ in 1:1000]
    data[1:2:end] .= missing
    model = vector_observations(data)
    vi = VarInfo(Xoshiro(1), model)
    vi = DynamicPPL.evaluate!!(model, vi)[2]
    vi = DynamicPPL.evaluate!!(model, vi)[2]
    @test (@allocated DynamicPPL.evaluate!!(model, vi)) < 2_000_000
end

end
