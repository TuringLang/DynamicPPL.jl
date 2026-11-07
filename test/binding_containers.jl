module BindingContainerTests

using DynamicPPL, Distributions, Test
using AbstractPPL: AbstractPPL, @of, of
using ForwardDiff: ForwardDiff
using DimensionalData: DimArray, X
using OffsetArrays: OffsetArray
using SparseArrays: sparse
using Random: Xoshiro

const BINDING_ROLES = ((condition, decondition, conditioned), (fix, unfix, fixed))

mutable struct FieldStorage
    a::Vector{Float64}
end
struct CustomVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::CustomVector) = size(x.data)
Base.getindex(x::CustomVector, i::Int) = x.data[i]
Base.setindex!(x::CustomVector, v, i::Int) = (x.data[i] = v; x)

@model function element_storage(x, entered)
    entered[] = true
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
@model field_storage(x, entered) = (entered[] = true; x.a[1] ~ Normal(); x)
@model dictionary_storage(x, entered) = (entered[] = true; x[:a] ~ Normal(); x)
@model covariate_storage(x) = (y ~ Normal(); x)

@testset "partial wrapper bindings are rejected" begin
    for data in (OffsetArray([1.0, 2.0], 0:1), DimArray([1.0, 2.0], X))
        for (bind, remove, listing) in BINDING_ROLES
            entered = Ref(false)
            original = element_storage(data, entered)
            source = bind === condition ? original : element_storage(nothing, entered)
            whole = bind(source; x=data)
            @test listing(whole)[@varname(x)] === data
            @test whole(Xoshiro(1)) == data
            entered[] = false
            address = @varname(x[firstindex(data)])
            @test_throws ArgumentError bind(source, address => 3.0)
            @test !entered[]
            @test_throws ArgumentError remove(whole, address)
            @test !entered[]
        end
    end
end

@testset "latent containers fail when unbound" begin
    for data in (
        Dict(:a => 1.0),
        view([1.0, 2.0], :),
        sparse([1.0, 2.0]),
        CustomVector([1.0, 2.0]),
        OffsetArray([1.0, 2.0], 0:1),
        DimArray([1.0, 2.0], X),
    )
        @testset "$(typeof(data))" begin
            entered = Ref(false)
            constructor = if data isa Dict
                dictionary_storage
            else
                element_storage
            end
            @test_throws ArgumentError decondition(constructor(data, entered))
            @test !entered[]
            @test covariate_storage(data)(Xoshiro(1)) === data
        end
    end
end
@testset "unsupported array storage remedy" begin
    for data in (view(zeros(2), :), OffsetArray(zeros(2), 0:1), 1.0:2.0),
        (bind, remove, _) in BINDING_ROLES

        entered = Ref(false)
        model = if bind === condition
            element_storage(data, entered)
        else
            fix(element_storage(nothing, entered); x=data)
        end
        err = try
            if bind === condition
                remove(model)
            else
                remove(model, @varname(x[firstindex(data)]))
            end
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(
            "use collect or pass a plain Array before constructing the model",
            sprint(showerror, err),
        )
        plain = decondition(element_storage(collect(data), entered))
        @test length(plain(Xoshiro(1))) == 2
    end
end
@testset "latent tuples rebuild at the write" begin
    entered = Ref(false)
    data = (1.0, 2.0)
    model = decondition(element_storage(data, entered))
    @test model(Xoshiro(1)) == Tuple(rand(Xoshiro(1), Normal(), 2))
    @test entered[]
    @test covariate_storage(data)(Xoshiro(1)) === data
end
struct ForeignNumber <: Number
    value::Float64
end
@model function read_argument(x)
    if false
        x ~ Normal()
    end
    return x
end
@model replace_argument(x) = (x ~ Normal(); x)
@model function inserted_dictionary(x, d, write)
    x[1] = d
    if write
        x[1][:a] ~ Normal()
    end
    return x[1]
end
@testset "structural leaf protocol" begin
    @testset "foreign leaves" begin
        data = Any[ForeignNumber(1.0), :tag, Float64]
        result = decondition(read_argument(data))(Xoshiro(1))
        @test result !== data
        @test all(result .=== data)
        result[1] = 3.0
        @test data[1] isa ForeignNumber
    end
    @testset "unclaimed storage has no recursive shape" begin
        for (a, b) in (
            (FieldStorage([1.0]), FieldStorage([1.0, 2.0])),
            (view([1.0], :), view([1.0, 2.0], :)),
        )
            @test DynamicPPL._same_shape(Any[a], Any[b])
        end
        @test DynamicPPL._shape_signature(view(Any[[1.0]], :)) === nothing
    end
    @test !DynamicPPL._same_shape(Any[([1.0],)], Any[([1.0, 2.0],)])
    @testset "latent Type values" begin
        @test decondition(read_argument(Float64))(Xoshiro(1)) === Float64
    end
    @testset "whole leaf replacement" for data in (FieldStorage([1.0]), (1.0, 2.0))
        @test decondition(replace_argument(data))(Xoshiro(1)) == rand(Xoshiro(1), Normal())
    end
    @testset "argument writes reject shared containers" begin
        data = FieldStorage([1.0])
        entered = Ref(false)
        model = decondition(field_storage(data, entered))
        @test_throws ArgumentError(
            "`x.a[1]`: unsupported binding storage $(typeof(data)); use supported Array/NamedTuple storage and matching index/field addresses",
        ) model(Xoshiro(1))
        @test entered[]
        @test data.a == [1.0]
        @test covariate_storage(data)(Xoshiro(1)) === data
    end
    @testset "inserted dictionary writes" begin
        d = Dict(:a => 1.0)
        model = decondition(inserted_dictionary(Any[0.0], d, true))
        @test_throws ArgumentError(
            "`x[1][:a]`: unsupported binding storage $(typeof(d)); use supported Array/NamedTuple storage and matching index/field addresses",
        ) model(Xoshiro(1))
        @test d[:a] == 1.0
    end
    @testset "shared reads remain unchanged" begin
        @test read_argument(Float64)(Xoshiro(1)) === Float64
        d = Dict(:a => 1.0)
        @test decondition(inserted_dictionary(Any[0.0], d, false))(Xoshiro(1)) === d
    end
end

struct FieldArray{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::FieldArray) = size(x.data)
Base.getindex(x::FieldArray, i::Int) = x.data[i]
DynamicPPL._binding_optic(::FieldArray) = AbstractPPL.Property

@model field_array(x) = (x.data[1] ~ Normal(); x)

@testset "one binding optic for an array addressed by fields" begin
    data = FieldArray([1.0])
    @test DynamicPPL._binding_optic(data) === AbstractPPL.Property
    @test DynamicPPL._split_bound_value(
        DynamicPPL.BoundValue{DynamicPPL.ConditionRole}(data), @varname(x)
    ) isa VarNamedTuple
    @test DynamicPPL._shape_signature(data) ==
        DynamicPPL._shape_signature((; data=data.data))
    for model in (
        decondition(field_array(data)), decondition(field_array(data), @varname(x.data[1]))
    )
        result = model(Xoshiro(1))
        @test result isa FieldArray
        @test result.data == [rand(Xoshiro(1), Normal())]
        @test result.data !== data.data
        @test data.data == [1.0]
    end
    @test_throws ArgumentError decondition(field_array(data), @varname(x[1]))
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
function DynamicPPL._binding_optic(
    ::Union{Record,MutableRecord,CustomProperties,KeywordOnly}
)
    return DynamicPPL.AbstractPPL.Property
end
@model field_storage(x) = (x.a[1] ~ Normal(); x)
@testset "opted-in field storage" begin
    @testset "unchanged reconstruction: $T" for T in (Record, MutableRecord)
        data = T([1.0, 2.0])
        for (bind, remove, listing) in BINDING_ROLES
            model = bind(decondition(field_storage(data)), @varname(x.a[1]) => 3.0)
            @test listing(model)[@varname(x.a[1])] == 3.0
            result = model(Xoshiro(1))
            @test result isa T
            @test result.a == [3.0, 2.0]
            @test result.a !== data.a
            @test result !== data
            result = remove(model, @varname(x.a[1]))(Xoshiro(1))
            @test result.a[1] == rand(Xoshiro(1), Normal())
            @test data.a == [1.0, 2.0]
        end
    end
    @testset "invalid reconstruction fails when made latent" begin
        for (data, error) in
            ((CustomProperties(1.0), ErrorException), (KeywordOnly(; a=1.0), MethodError))
            @test_throws error decondition(read_argument(data))
        end
    end
    @testset "whole bindings remain unchanged" begin
        for data in (CustomProperties(1.0), KeywordOnly(; a=1.0))
            @test read_argument(data)(Xoshiro(1)) === data
            @test fix(read_argument(nothing); x=data)(Xoshiro(1)) === data
        end
    end
    @testset "observed sibling reconstruction remains unnecessary" begin
        for data in (CustomProperties(1.0), KeywordOnly(; a=1.0))
            original = (a=[1.0], observed=data)
            model = decondition(field_storage(original), @varname(x.a[1]))
            result = model(Xoshiro(1))
            @test result.observed === data
            @test result.a[1] == rand(Xoshiro(1), Normal())
            @test original.a == [1.0]
        end
    end
end
@model tuple_argument(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
@model named_tuple_argument(x) = (x.a[1] ~ Normal(); x.a[2] ~ Normal(); x)
@model tuple_arrays_argument(x) = (x[1][1] ~ Normal(); x[2][1] ~ Normal(); x)
@testset "tuple argument bindings" begin
    @test DynamicPPL._binding_optic(()) === AbstractPPL.Index
    @test DynamicPPL.VarNamedTuples._check_tuple_storage(()) === nothing
    @test_throws BoundsError DynamicPPL._setindex!!((), 1, 1)
    @test decondition(element_storage((), Ref(false)))(Xoshiro(1)) === ()
    for (data, constructor, names) in (
        ((1.0f0, 2.0f0), tuple_argument, (@varname(x[1]), @varname(x[2]))),
        ((big"1.0", big"2.0"), tuple_argument, (@varname(x[1]), @varname(x[2]))),
        (
            (ForwardDiff.Dual(1.0, 1.0), ForwardDiff.Dual(2.0, 0.0)),
            tuple_argument,
            (@varname(x[1]), @varname(x[2])),
        ),
        ((a=(1.0, 2.0),), named_tuple_argument, (@varname(x.a[1]), @varname(x.a[2]))),
        (([1.0], [2.0]), tuple_arrays_argument, (@varname(x[1][1]), @varname(x[2][1]))),
    )
        @testset "$(typeof(data))" begin
            snapshot = deepcopy(data)
            latent = decondition(constructor(data))
            params = VarNamedTuple(; x=data)
            result, vi = init!!(
                Xoshiro(1),
                latent,
                VarInfo(DynamicPPL.RawValueAccumulator(false)),
                InitFromParams(params, nothing),
            )
            @test result == data
            @test typeof(result) === typeof(data)
            repeated, _ = init!!(
                Xoshiro(1), latent, VarInfo(), InitFromParams(get_raw_values(vi), nothing)
            )
            @test repeated == data
            @test typeof(repeated) === typeof(data)
            for (bind, remove, listing) in BINDING_ROLES
                whole = bind(
                    latent, names[1] => params[names[1]], names[2] => params[names[2]]
                )
                @test whole(Xoshiro(1)) == data
                @test typeof(whole(Xoshiro(1))) === typeof(data)
                released = remove(whole, names[1])
                @test !haskey(listing(released), names[1])
                @test listing(released)[names[2]] == params[names[2]]
            end
            if data isa Tuple{<:AbstractArray,<:AbstractArray}
                @test result[1] !== data[1] && result[2] !== data[2]
                result[1][1] = 3.0
            end
            @test data == snapshot
        end
    end
    latent = decondition(tuple_argument((1.0f0, 2.0f0)))
    for selector in (@varname(x[1:2]), @varname(x[:]))
        for values in ((3, 4), [3, 4])
            whole = condition(latent, selector => values)
            @test whole(Xoshiro(1)) === (3.0f0, 4.0f0)
            @test decondition(whole, selector)(Xoshiro(1)) == latent(Xoshiro(1))
        end
        @test_throws ArgumentError condition(latent, selector => (3,))
    end
    @test_throws ArgumentError condition(latent, @varname(x[1]) => 0.1)
    @test_throws ArgumentError condition(latent, @varname(x.a) => 1.0)
    @test_throws ArgumentError condition(latent, @varname(x[3]) => 1.0)
    restriction = ArgumentError(
        "same-type tuples only; give the elements one type or use an array"
    )
    @test_throws restriction decondition(tuple_argument((1, 2.0)))
    @test_throws restriction condition(tuple_argument((1, 2.0)), @varname(x[1]) => 3)
    result, _ = @inferred init!!(
        Xoshiro(1),
        decondition(tuple_argument((1.0, 2.0))),
        VarInfo(),
        InitFromParams((; x=(3.0, 4.0)), nothing),
    )
    @test result === (3.0, 4.0)
    tuple = DynamicPPL._setindex!!((0, 0), 1, 1)
    array = DynamicPPL._setindex!!([0, 0], 1, 1)
    tuple = DynamicPPL._setindex!!(tuple, 2.5, 2)
    array = DynamicPPL._setindex!!(array, 2.5, 2)
    @test tuple === Tuple(array) === (1.0, 2.5)
end
@model mixed_whole(x) = (x ~ product_distribution([Normal(), Normal()]); x)
@model mixed_positions() = (x = (0.0, 0.0); x[1] ~ Normal(); x[2] ~ Normal(); x)
@testset "mixed tuple binding reads" begin
    mixed = (1.0f0, big"2.0")
    restriction = ArgumentError(
        "same-type tuples only; give the elements one type or use an array"
    )
    @test_throws restriction condition(mixed_positions(); x=mixed)(Xoshiro(1))
    for bind in (fix, condition)
        @test bind(mixed_whole(missing); x=mixed)(Xoshiro(1)) === mixed
    end
end
end

module StaticBindingTests
using DynamicPPL, Distributions, StaticArrays, Test
using Random: Xoshiro
using ForwardDiff: ForwardDiff
@model static_covariate(x) = (y ~ Normal(); x)
@model static_argument(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
@testset "static argument bindings" begin
    for data in (SVector(1.0f0, 2.0f0), SVector(big"1.0", big"2.0"), MVector(1.0f0, 2.0f0))
        @testset "$(typeof(data))" begin
            snapshot = copy(data)
            @test static_covariate(data)(Xoshiro(1)) === data
            latent = decondition(static_argument(data))
            result, _ = @inferred init!!(
                Xoshiro(1), latent, VarInfo(), InitFromParams((; x=data), nothing)
            )
            @test result == data
            @test typeof(result) === typeof(data)
            for (bind, remove) in ((condition, decondition), (fix, unfix))
                partial = bind(latent, @varname(x[1]) => 3)
                @test partial(Xoshiro(1))[1] == 3
                whole = bind(partial, @varname(x[2]) => 4)
                @test whole(Xoshiro(1)) == typeof(data)(3, 4)
                @test typeof(whole(Xoshiro(1))) === typeof(data)
                @test remove(whole, @varname(x[1]))(Xoshiro(1))[2] == 4
                @test remove(whole, @varname(x))(Xoshiro(1)) == latent(Xoshiro(1))
            end
            for selector in (@varname(x[1:2]), @varname(x[:]))
                bound = condition(latent, selector => [3, 4])
                @test bound(Xoshiro(1)) == typeof(data)(3, 4)
                @test decondition(bound, selector)(Xoshiro(1)) == latent(Xoshiro(1))
                @test_throws ArgumentError condition(latent, selector => [3])
            end
            if data isa MVector
                @test axes(result) == axes(data)
                @test result !== data
                result[1] = 10
            end
            @test data == snapshot
        end
    end
    for data in (SVector(1.0, 2.0), MVector(1.0, 2.0))
        dual = ForwardDiff.Dual(3.0, 1.0)
        result = DynamicPPL._setindex!!(data, dual, 1)
        @test eltype(result) === typeof(dual)
        @test result[1] === dual
        @test result[2] === ForwardDiff.Dual(2.0, 0.0)
        @test data == [1.0, 2.0]
    end
    @test_throws ArgumentError decondition(static_argument(MVector(big"1.0", big"2.0")))
end
end
