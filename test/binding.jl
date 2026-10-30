module ResolvedBindingTests
using DynamicPPL, AbstractPPL, Test
using DynamicPPL:
    Condition, Fix, ResolvedBindings, _bind_resolved, _select_model_values, _unbind_resolved
using OffsetArrays, ComponentArrays, DimensionalData, ForwardDiff
using DynamicPPL: _model_role_at, _model_argument_binding, ModelValue, NoModelBinding
using DynamicPPL.VarNamedTuples: PartialArray
struct ShapeRecord{T}
    value::T
end

@testset "scalar array index dimensions preserve Julia bounds" begin
    array = zeros(2, 3)
    for indices in ((1,), (2, 3), (CartesianIndex(2, 3),))
        @test DynamicPPL.VarNamedTuples._ndims(array, indices...) == 0
    end
    @test_throws BoundsError DynamicPPL.VarNamedTuples._ndims(array, 7)
    @test_throws BoundsError DynamicPPL.VarNamedTuples._ndims(array, 3, 1)
    @test_throws ArgumentError DynamicPPL.VarNamedTuples._ndims(array, true)
end

@testset "whole binding storage identity" begin
    for role in (Condition, Fix)
        value = ones(8)
        table = _bind_resolved(ResolvedBindings(), role; x=value)
        @test table.storage.data.x === value
        removed = _unbind_resolved(table, role, @varname(x[1:4]))
        removed = _unbind_resolved(removed, role, @varname(x[5:8]))
        @test _select_model_values(role, removed.values) == VarNamedTuple()
        @test _select_model_values(role, table.values)[@varname(x)] === value
    end
end

@testset "batched indexed storage copies" begin
    for role in (Condition, Fix)
        x = ones(100_000)
        table = _bind_resolved(ResolvedBindings(), role; x)
        edit() = _bind_resolved(table, role, @varname(x[1]) => 2.0, @varname(x[2]) => 3.0)
        edit()
        @test (@allocated edit()) <= 4sizeof(x)
        @test edit().storage.data.x[1:3] == [2.0, 3.0, 1.0]
        @test x == ones(length(x))

        shared = [1.0, 2.0]
        table = _bind_resolved(
            ResolvedBindings(; roots=(@varname(x),)), role; x=(a=shared, b=shared)
        )
        edited = _bind_resolved(
            table, role, @varname(x.a[1]) => 3.0, @varname(x.b[2]) => 4.0
        )
        @test edited.storage.data.x.a == [3.0, 2.0]
        @test edited.storage.data.x.b == [1.0, 4.0]
        @test shared == [1.0, 2.0]
    end
end

@testset "resolved role table" begin
    @testset "resolved bindings" begin
        empty = ResolvedBindings()
        for role in (Condition, Fix)
            table = _bind_resolved(empty, role, (x=1.0,))
            @test _select_model_values(role, table.values)[@varname(x)] == 1.0
            @test isempty(empty.values)
            @test_throws ArgumentError _bind_resolved(empty, role, @varname(x[1]) => 2.0)
            table = _bind_resolved(
                empty, role, @varname(x[1]) => 2.0, @of(x = of(Array, 2))
            )
            @test _select_model_values(role, table.values)[@varname(x[1])] == 2.0
            @test size(table.storage.data.x) == (2,)
            @test_throws ArgumentError _bind_resolved(
                table, role === Fix ? Condition : Fix, @varname(x[1]) => 3.0
            )
            table = _bind_resolved(table, role, @varname(x[2]) => 3.0)
            @test _select_model_values(role, table.values)[@varname(x)] == [2.0, 3.0]
            removed = _unbind_resolved(table, role, @varname(x[1]))
            @test _select_model_values(role, removed.values)[@varname(x[2])] == 3.0
            @test !haskey(_select_model_values(role, removed.values), @varname(x[1]))
            @test size(removed.storage.data.x) == (2,)
            @test _select_model_values(role, table.values)[@varname(x)] == [2.0, 3.0]
            cleared = _unbind_resolved(removed, role, @varname(x))
            @test isempty(cleared.values)
            @test size(cleared.storage.data.x) == (2,)
            @test_throws ArgumentError _bind_resolved(
                empty,
                role,
                (@varname(x[1]) => 1.0, @varname(x[1, 1]) => 2.0),
                @of(x = of(Array, 2, 2))
            )
            @test_throws ArgumentError _bind_resolved(empty, role, @varname(p[1].x) => 1.0)
            table = _bind_resolved(empty, role, @varname(p.x) => (a=1.0,))
            @test_throws ArgumentError _bind_resolved(
                table, role, @varname(p.x) => (b=2.0,)
            )
        end
    end

    @testset "binding input adapters and ownership" begin
        for R in (Condition, Fix)
            inputs = ((x=1.0,), :x => 1.0, @varname(x) => 1.0, ((:x => 1.0,),))
            for input in inputs
                t = _bind_resolved(ResolvedBindings(), R, input)
                @test _select_model_values(R, t.values)[@varname(x)] === 1.0
            end
            @test _select_model_values(
                R, _bind_resolved(ResolvedBindings(), R; x=2).values
            )[@varname(x)] === 2
            @test_throws ArgumentError _bind_resolved(ResolvedBindings(), R, Dict(:x => 1))
            for value in (missing, nothing, [missing], (a=nothing,), Any[1, (b=missing,)])
                @test_throws ArgumentError _bind_resolved(ResolvedBindings(), R; x=value)
            end
            t = _bind_resolved(ResolvedBindings(), R; x=1.0)
            @test _select_model_values(R, _bind_resolved(t, R; x=2).values)[@varname(x)] ===
                2
            @test _select_model_values(R, t.values)[@varname(x)] === 1.0
            other = R === Fix ? Condition : Fix
            @test_throws ArgumentError _bind_resolved(t, other; x=3)
            @test isempty(_select_model_values(R, _unbind_resolved(t, R).values))
            @test _select_model_values(R, _unbind_resolved(t, other).values)[@varname(
                x
            )] === 1.0
        end
        @test_throws ArgumentError ResolvedBindings(
            VarNamedTuple((x=1,)),
            VarNamedTuple(),
            VarNamedTuple(),
            VarNamedTuple(),
            VarNamedTuple(),
            VarName[],
        )
    end

    @testset "whole and field edits require root structure" begin
        for R in (Condition, Fix)
            empty = ResolvedBindings()
            whole = _bind_resolved(empty, R, @varname(a.b) => (x=1.0,))
            @test_throws ArgumentError _bind_resolved(whole, R, @varname(a.b.y) => 2.0)
            @test_throws ArgumentError _bind_resolved(whole, R, @varname(a.b.x) => 2.0)
            @test_throws ArgumentError _bind_resolved(whole, R, @varname(a.b) => (y=2.0,))
            @test_throws ArgumentError _unbind_resolved(whole, R, @varname(a.b.x))
            same = _bind_resolved(whole, R, @varname(a.b) => (x=2.0,))
            @test _select_model_values(R, same.values)[@varname(a.b)] == (x=2.0,)
            field = _bind_resolved(empty, R, @varname(a.b.y) => 2.0)
            @test_throws ArgumentError _bind_resolved(field, R, @varname(a.b) => (x=1.0,))
            shaped = _bind_resolved(
                empty, R, @varname(a.b) => (x=1.0,), @of(a = @of(b = @of(x = Float64)))
            )
            @test_throws ArgumentError _bind_resolved(shaped, R, @varname(a.b.x) => 2.0)
            known = ResolvedBindings(; arguments=VarNamedTuple((q=(x=0.0, y=0.0),)))
            known = _bind_resolved(known, R; q=(x=1.0, y=2.0))
            replaced = _bind_resolved(known, R, @varname(q.x) => 3.0)
            @test _select_model_values(R, replaced.values)[@varname(q)] == (x=3.0, y=2.0)
            @test_throws ArgumentError _bind_resolved(known, R, @varname(q.z) => 3.0)
            removed = _unbind_resolved(known, R, @varname(q.x))
            @test !haskey(_select_model_values(R, removed.values), @varname(q.x))
            @test _select_model_values(R, removed.values)[@varname(q.y)] == 2.0
            rebound = _bind_resolved(removed, R, @varname(q.x) => 3.0)
            @test _select_model_values(R, rebound.values)[@varname(q)] === (x=3.0, y=2.0)
        end
    end

    @testset "binding storage lifetimes" begin
        for R in (Condition, Fix)
            other = R === Fix ? Condition : Fix
            empty = ResolvedBindings()
            t = _bind_resolved(empty, R; x=ones(2))
            cleared = _unbind_resolved(t, R)
            @test _select_model_values(
                R, _bind_resolved(cleared, R, @varname(x[end]) => 3.0).values
            )[@varname(x[2])] == 3.0
            @test_throws ArgumentError _bind_resolved(cleared, R, @varname(x[3]) => 3.0)
            @test _select_model_values(R, _bind_resolved(cleared, R; x=ones(3)).values)[@varname(
                x
            )] == ones(3)
            @test_throws ArgumentError _bind_resolved(t, other; x=ones(3))
            templated = _bind_resolved(
                empty, R, @varname(x) => ones(2), @of(x = of(Array, 2))
            )
            for cleared in
                (_unbind_resolved(templated, R), _unbind_resolved(templated, R, :x))
                @test_throws ArgumentError _bind_resolved(cleared, R; x=ones(3))
                @test_throws ArgumentError _bind_resolved(cleared, other; x=ones(3))
                updated = _bind_resolved(
                    cleared, R, @varname(x) => ones(3), @of(x = of(Array, 3))
                )
                @test _select_model_values(R, updated.values)[@varname(x)] == ones(3)
            end
            nested = _bind_resolved(empty, R; x=[[1.0, 2.0, 3.0]])
            nested = _unbind_resolved(nested, R, @varname(x[1]))
            @test_throws ArgumentError _bind_resolved(nested, R, @varname(x[1]) => ones(4))
            @test _select_model_values(
                R, _bind_resolved(nested, R, @varname(x[1]) => ones(3)).values
            )[@varname(x[1])] == ones(3)
            source = Vector{Vector{Float64}}(undef, 1)
            nested = _bind_resolved(empty, R; x=source)
            nested = _bind_resolved(nested, other, @varname(x[1]) => ones(2))
            nested = _unbind_resolved(_unbind_resolved(nested, R), other)
            @test_throws ArgumentError _bind_resolved(nested, R, @varname(x[1]) => ones(3))
            @test _select_model_values(
                R, _bind_resolved(nested, R, @varname(x[1][end]) => 4.0).values
            )[@varname(x[1][2])] == 4.0
            @test !isassigned(source, 1)
            p = _bind_resolved(
                empty, R, @varname(p[1].z) => ones(2), @of(p = of(Array, 2, 2))
            )
            p = _unbind_resolved(p, R)
            p = _bind_resolved(p, R, @varname(p[end].z) => 4.0)
            @test _select_model_values(R, p.values)[@varname(p[2, 2].z)] == 4.0
            @test_throws ArgumentError _bind_resolved(
                p, R, @varname(p[1].z) => ones(2), @of(p = of(Array, 3, 2))
            )
            p = _bind_resolved(
                _unbind_resolved(p, R), R, @varname(p[2].z) => 1.0, @of(p = of(Array, 3, 2))
            )
            @test _select_model_values(
                R, _bind_resolved(p, R, @varname(p[1].z[end]) => 4.0).values
            )[@varname(p[1].z[2])] == 4.0
        end
    end

    @testset "canonical selections and atomic failure" begin
        for R in (Condition, Fix)
            empty = ResolvedBindings(; arguments=VarNamedTuple((x=zeros(2, 2),)))
            for addresses in (
                (@varname(x[1]), @varname(x[1, 1])),
                (@varname(x[1:2]), @varname(x[2])),
                (@varname(x[:][1]), @varname(x[1, 1])),
                (@varname(x[CartesianIndex(1, 1)]), @varname(x[1])),
            )
                a, b = addresses
                value = a == @varname(x[1:2]) ? ones(2) : 1.0
                @test_throws ArgumentError _bind_resolved(empty, R, a => value, b => 2.0)
                @test isempty(empty.values)
                @test empty.arguments[@varname(x)] == zeros(2, 2)
            end
            @test_throws ArgumentError _bind_resolved(
                empty, R, @varname(x[[1, 1]]) => ones(2)
            )
            @test_throws ArgumentError _bind_resolved(
                empty, R, @varname(x[falses(3)]) => Float64[]
            )
            @test_throws ArgumentError _bind_resolved(
                empty, R, @varname(x[:, 1]) => ones(3)
            )
            t = _bind_resolved(empty, R, @varname(x[:][2]) => 3.0)
            @test _select_model_values(R, t.values)[@varname(x[2, 1])] == 3.0
            t = _bind_resolved(t, R, @varname(x[:, 2]) => [4.0, 5.0])
            @test _select_model_values(R, t.values)[@varname(x[2, 2])] == 5.0
            removed = _unbind_resolved(t, R, @varname(x[:][2:3]))
            @test !haskey(_select_model_values(R, removed.values), @varname(x[2, 1]))
            @test !haskey(_select_model_values(R, removed.values), @varname(x[1, 2]))
            @test _select_model_values(R, removed.values)[@varname(x[2, 2])] == 5.0
            @test_throws ArgumentError _unbind_resolved(t, R, @varname(x[5]))
            @test _select_model_values(R, t.values)[@varname(x[2, 1])] == 3.0
        end
    end

    @testset "partial container and numeric contract" begin
        for R in (Condition, Fix)
            for source in
                (zeros(Float32, 2), zeros(BigFloat, 2), zeros(2), reshape([0.0], ()))
                table = ResolvedBindings(; arguments=VarNamedTuple((x=source,)))
                t = _bind_resolved(table, R, @varname(x[1]) => 1)
                @test _select_model_values(R, t.values)[@varname(x[1])] ==
                    one(eltype(source))
                @test _select_model_values(R, t.values)[@varname(x[1])] isa eltype(source)
                @test source == zero(source)
            end
            table = ResolvedBindings(; arguments=VarNamedTuple((p=(a=0.0f0,),)))
            @test_throws ArgumentError _bind_resolved(table, R, @varname(p.a) => 0.1)
            @test _select_model_values(
                R, _bind_resolved(table, R, @varname(p.a) => 1).values
            )[@varname(p.a)] === 1.0f0
            for source in (1.0:3.0, view(zeros(3), :), (1.0, 2.0), Dict(1 => 0.0))
                table = _bind_resolved(ResolvedBindings(), R; x=source)
                @test_throws ArgumentError _bind_resolved(table, R, @varname(x[1]) => 2.0)
            end
            for source in (
                OffsetArray(zeros(3), -1:1),
                ComponentArray(; a=0.0, b=[0.0, 0.0]),
                DimArray(zeros(3), (X,)),
            )
                table = ResolvedBindings(; arguments=VarNamedTuple((x=source,)))
                index = firstindex(source)
                t = _bind_resolved(table, R, @varname(x[index]) => 2.0)
                @test _select_model_values(R, t.values)[@varname(x[index])] == 2.0
                @test axes(t.storage.data.x) == axes(source)
            end
            source = ComponentArray(; a=0.0, b=[0.0, 0.0])
            table = ResolvedBindings(; arguments=VarNamedTuple((x=source,)))
            t = _bind_resolved(table, R, @varname(x.b[2]) => 2.0)
            @test _select_model_values(R, t.values)[@varname(x.b[2])] == 2.0
            @test_throws ArgumentError _bind_resolved(
                table, R, @varname(x.b) => ones(2), @varname(x[2]) => 3.0
            )
        end
    end
    @testset "supplied root structure and sparse batches" begin
        for R in (Condition, Fix)
            layout = VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=0.0, y=0.0)),)),))
            t = _unbind_resolved(_bind_resolved(ResolvedBindings(), R, layout), R)
            whole = _bind_resolved(t, R, @varname(a.b) => (x=1.0,))
            whole = _bind_resolved(whole, R, @varname(a.b.y) => 2.0)
            @test _select_model_values(R, whole.values)[@varname(a.b.x)] == 1.0
            @test _select_model_values(R, whole.values)[@varname(a.b.y)] == 2.0
            field = _bind_resolved(t, R, @varname(a.b.y) => 2.0)
            field = _bind_resolved(field, R, @varname(a.b) => (x=1.0,))
            @test _select_model_values(R, field.values)[@varname(a.b.x)] == 1.0
            @test _select_model_values(R, field.values)[@varname(a.b.y)] == 2.0
            for wrap in (identity, VarNamedTuple)
                t = _bind_resolved(
                    ResolvedBindings(),
                    R,
                    wrap((x=PartialArray([1.0, 2.0, 3.0], [true, false, true]),)),
                )
                listed = _select_model_values(R, t.values)
                @test listed[@varname(x[1])] == 1.0
                @test !haskey(listed, @varname(x[2]))
                @test listed[@varname(x[3])] == 3.0
                @test size(t.storage.data.x) == (3,)
            end
            allocations = map((100, 1000)) do n
                input = VarNamedTuple((x=PartialArray(ones(n), trues(n)),))
                table = _bind_resolved(ResolvedBindings(), R, input)
                _bind_resolved(table, R, input)
                @allocated _bind_resolved(table, R, input)
            end
            @test allocations[2] <= 15 * allocations[1]
        end
    end

    @testset "empty selections retain concrete extent" begin
        for R in (Condition, Fix)
            t = _bind_resolved(
                ResolvedBindings(),
                R,
                @varname(x[falses(3)]) => Float64[],
                @of(x = of(Array, 3)),
            )
            @test isempty(_select_model_values(R, t.values))
            @test size(t.storage.data.x) == (3,)
            @test _select_model_values(
                R, _bind_resolved(t, R, @varname(x[end]) => 1.0).values
            )[@varname(x[3])] == 1.0
            @test_throws ArgumentError _bind_resolved(
                ResolvedBindings(),
                R,
                @varname(x[falses(3)]) => Float64[],
                @of(x = of(Array, 2)),
            )
            for wrap in (identity, VarNamedTuple)
                t = _bind_resolved(
                    ResolvedBindings(), R, wrap((x=PartialArray(zeros(3), falses(3)),))
                )
                @test isempty(_select_model_values(R, t.values))
                @test _select_model_values(
                    R, _bind_resolved(t, R, @varname(x[end]) => 1.0).values
                )[@varname(x[3])] == 1.0
            end
        end
    end

    @testset "two-position role and removal oracle" begin
        roles = (nothing, Condition, Fix)
        for first_role in roles, second_role in roles, R in (Condition, Fix), target in 1:2
            t = ResolvedBindings(; arguments=VarNamedTuple((x=zeros(2),)))
            assigned = (first_role, second_role)
            for (i, role) in enumerate(assigned)
                role === nothing && continue
                t = _bind_resolved(t, role, @varname(x[i]) => float(i))
            end
            if assigned[target] !== nothing && assigned[target] !== R
                @test_throws ArgumentError _bind_resolved(t, R, @varname(x[target]) => 4.0)
            else
                updated = _bind_resolved(t, R, @varname(x[target]) => 4.0)
                @test _select_model_values(R, updated.values)[@varname(x[target])] == 4.0
            end
            optic = AbstractPPL.varname_to_optic(@varname(x))
            if first_role === second_role
                role = _model_role_at(t.values, optic, @varname(x))
                @test role === nothing ? first_role === nothing : role isa first_role
            else
                @test_throws ArgumentError _model_role_at(t.values, optic, @varname(x))
            end
            cleared = _unbind_resolved(t, R, @varname(x[target]))
            for i in 1:2, role in (Condition, Fix)
                expected = assigned[i] === role && !(i == target && R === role)
                listed = _select_model_values(role, cleared.values)
                @test haskey(listed, @varname(x[i])) == expected
                expected && @test listed[@varname(x[i])] == i
            end
            @test _select_model_values(Condition, t.values) ==
                _select_model_values(Condition, _unbind_resolved(t, Fix, :absent).values)
        end
        t = _bind_resolved(ResolvedBindings(), Condition; (x=1.0,)...)
        @test (@inferred _select_model_values(Condition, t.values))[@varname(x)] == 1.0
    end
    @testset "field roots survive role removal" begin
        t = _bind_resolved(ResolvedBindings(), Condition, @varname(q.a) => 1.0)
        @test_throws ArgumentError _bind_resolved(
            _unbind_resolved(t, Condition), Condition; q=(a=2.0,)
        )
        cleared = _unbind_resolved(t, Condition, :q)
        @test _select_model_values(
            Condition, _bind_resolved(cleared, Condition; q=(a=2.0,)).values
        )[@varname(q)] == (a=2.0,)
    end
    @testset "slice assignment follows Julia dimensions" begin
        for R in (Condition, Fix)
            source = zeros(2, 3, 2)
            for (indices, value) in (
                ((:, 1:1, 1), [1.0, 2.0]),
                ((:, 1:2, :), reshape(collect(1.0:8.0), 2, 4)),
                (([1, 2], 2, :), reshape(collect(1.0:4.0), 2, 2)),
            )
                expected = copy(source)
                expected[indices...] = value
                t = _bind_resolved(ResolvedBindings(), R; x=source)
                t = _bind_resolved(t, R, @varname(x[indices...]) => value)
                @test _select_model_values(R, t.values)[@varname(x)] == expected
                @test source == zeros(2, 3, 2)
            end
            unsupported = ComponentArray(
                view(zeros(3), :), (ComponentArrays.Axis(; a=1, b=2:3),)
            )
            t = ResolvedBindings(; arguments=VarNamedTuple((x=unsupported,)))
            @test_throws ArgumentError _bind_resolved(t, R, @varname(x.b[1]) => 1.0)
        end
    end
    @testset "positional template is last" begin
        for R in (Condition, Fix)
            empty = ResolvedBindings()
            template = @of(x = of(Array, 2))
            value = @varname(x[1]) => 1.0
            for inputs in (
                (template, value),
                (value, template, (:z => 2.0,)),
                ((template,), value),
                (template, template),
            )
                @test_throws r"positional binding template must be the last argument" _bind_resolved(
                    empty, R, inputs...
                )
                @test isempty(empty.values)
            end
            @test _select_model_values(
                R, _bind_resolved(empty, R, value, template).values
            )[@varname(x[1])] == 1.0
            @test _select_model_values(
                R, _bind_resolved(empty, R, template; x=ones(2)).values
            )[@varname(x)] == ones(2)
        end
    end

    @testset "whole storage compares recursive shape" begin
        for R in (Condition, Fix)
            t = _bind_resolved(ResolvedBindings(), R; x=(1.0, 2.0))
            @test _select_model_values(R, _bind_resolved(t, R; x=[3.0, 4.0]).values)[@varname(
                x
            )] == [3.0, 4.0]
            t = _bind_resolved(ResolvedBindings(), R; x=ShapeRecord(zeros(2)))
            @test_throws ArgumentError _bind_resolved(t, R; x=ShapeRecord(zeros(3)))
            @test _select_model_values(
                R, _bind_resolved(t, R; x=ShapeRecord(ones(2))).values
            )[@varname(x)].value == ones(2)
            cycle = Any[]
            push!(cycle, cycle)
            t = _bind_resolved(ResolvedBindings(), R; x=cycle)
            copied = deepcopy(cycle)
            t = _bind_resolved(t, R; x=copied)
            @test _select_model_values(R, t.values)[@varname(x)] === copied
            @test t.storage.data.x[1] === t.storage.data.x
            @test t.storage.data.x === copied
        end
    end
end
end

module ArgumentBindingTests
using DynamicPPL, AbstractPPL, Test, StaticArrays, ForwardDiff, ReverseDiff
using DynamicPPL: Condition, Fix, ResolvedBindings, _bind_resolved, _copy_model_argument
using Distributions
@model function argument_preparation(x; data=nothing, y=nothing, z=nothing)
    return (; x, data, y, z)
    return x ~ Normal()
end
function prepared_arguments(table, args)
    model = argument_preparation(args.x; Base.tail(args)...)
    model = DynamicPPL._finish_binding(model, table, nothing)
    return NamedTuple{keys(args)}(model())
end

struct BufferNumber <: Number
    data::Vector{Float64}
end
mutable struct CopyProbe
    value::Vector{Float64}
end
Base.deepcopy_internal(value::CopyProbe, memo::IdDict) = CopyProbe([only(value.value) + 1])
@testset "isolated argument values" begin
    x = [1.0, 2.0]
    args = (x=x,)
    table = ResolvedBindings(; arguments=VarNamedTuple(args))
    prepared = prepared_arguments(table, args)
    @test prepared == args
    @test prepared.x !== x
    prepared.x[1] = 9
    @test x == [1, 2]
    @test prepared_arguments(table, args).x == x
    bound = _bind_resolved(table, Fix, @varname(x[1]) => 3.0)
    prepared = prepared_arguments(bound, args)
    @test prepared.x == [3, 2]
    @test prepared.x !== x
    replacement = [4.0, 5.0]
    bound = _bind_resolved(table, Fix, (x=replacement,))
    @test prepared_arguments(bound, args).x === replacement
    @test prepared_arguments(table, (x=x, z=[8.0])).z == [8.0]
    @test_throws ArgumentError prepared_arguments(table, (x=x, z=x))
    @test prepared_arguments(bound, (x=x, z=x)).z === x
    @test_throws ArgumentError _copy_model_argument(SVector(1.0, 2.0), @varname(x))
    @test_throws ArgumentError _copy_model_argument(BufferNumber([1.0]), @varname(x))
    @test_throws ArgumentError _copy_model_argument(Dict([1.0] => 2.0), @varname(x))
    @test _copy_model_argument(Dict(:x => [1.0]), @varname(x)) == Dict(:x => [1.0])
    @test _copy_model_argument(CopyProbe([1.0]), @varname(x)).value == [2.0]
    cyclic = Any[]
    push!(cyclic, cyclic)
    copied = _copy_model_argument(cyclic, @varname(x))
    @test copied !== cyclic
    @test copied[1] === copied
    b = big"1.5"
    @test _copy_model_argument(b, @varname(x)) === b
end

@testset "argument copy AD" begin
    for ad in (ForwardDiff.gradient, ReverseDiff.gradient)
        original = [2.0, 3.0]
        gradient = ad(original) do x
            copied = _copy_model_argument((x=x,), @varname(x))
            if ad === ForwardDiff.gradient
                @test copied.x !== x
            end
            scalar = x[1]
            @test _copy_model_argument((scalar=scalar,), @varname(x)).scalar === scalar
            if ad === ReverseDiff.gradient
                @test_throws ArgumentError _copy_model_argument(
                    (tracked=x, raw=ReverseDiff.value(x)), @varname(x)
                )
            end
            sum(abs2, copied.x)
        end
        @test gradient == [4.0, 6.0]
        @test original == [2.0, 3.0]
    end
end

using SparseArrays, OffsetArrays, ComponentArrays, DimensionalData
using DynamicPPL: _model_argument_value, ModelValue, ArgumentCondition
using DynamicPPL.VarNamedTuples: PartialArray

mutable struct CyclicStorage
    value::Vector{Float64}
    next::Any
end
struct WrappedStorage{T}
    value::T
end
mutable struct MutableScalar <: Number
    value::Float64
end
mutable struct InnerState
    value::Vector{Float64}
    InnerState() = new([0.0])
end

@testset "native copying and supported latent storage" begin
    for source in (
        Float32[1, 2],
        BigFloat[1, 2],
        [1.0, 2.0],
        reshape([1.0, 2.0], 1, 2),
        OffsetArray([1.0, 2.0], 0:1),
        ComponentArray(; a=1.0, b=[2.0]),
        DimArray([1.0, 2.0], (X,)),
        sparse([1.0, 2.0]),
        sparse(reshape([1.0, 2.0], 1, 2)),
        view([1.0, 2.0], :),
    )
        copied = _copy_model_argument(source, @varname(x))
        @test copied == source
        @test copied !== source
        @test typeof(copied) === typeof(source)
        index = firstindex(copied)
        original = source[index]
        copied[index] = 9
        @test source[index] == original
    end
    for source in (
        SVector(1.0, 2.0),
        1.0:2.0,
        view(SVector(1.0, 2.0), :),
        (a=SVector(1.0, 2.0),),
        [SVector(1.0, 2.0)],
    )
        @test_throws ArgumentError _copy_model_argument(source, @varname(x))
    end
    scalar = MutableScalar(2.0)
    @test _copy_model_argument((scalar=scalar,), @varname(x)).scalar === scalar
    for scalar in (big"1.25", big"12", Complex(big"1.25", big"2.25"))
        @test _copy_model_argument(scalar, @varname(x)) === scalar
    end
    @test_throws ArgumentError _copy_model_argument((a=BufferNumber([1.0]),), @varname(x))
    state = InnerState()
    copied = _copy_model_argument(state, @varname(x))
    copied.value[1] = 3
    @test state.value == [0.0]
    data = Vector{Vector{Float64}}(undef, 2)
    data[2] = [1.0]
    copied = _copy_model_argument(data, @varname(x))
    @test !isassigned(copied, 1)
    @test copied[2] !== data[2]
    @test copied[2] == data[2]
end

@testset "dictionary key policy" begin
    for key in (:a, "a", 1, (1, 2), 1.0)
        source = Dict(key => [1.0])
        copied = _copy_model_argument(source, @varname(x))
        @test copied == source
        @test copied[key] !== source[key]
    end
    for key in ([1.0], Ref(1), big"12", SubString("abc", 1, 2))
        @test_throws ArgumentError _copy_model_argument(Dict(key => [1.0]), @varname(x))
    end
    covariate = Dict([1.0] => [2.0])
    args = (x=[0.0], data=covariate)
    table = ResolvedBindings(; arguments=VarNamedTuple((x=args.x,)))
    @test prepared_arguments(table, args).data === covariate
end

@testset "latent aliases and independent branches" begin
    v = [1.0, 2.0]
    for other in (v, reshape(v, 1, 2), view(v, :))
        args = (x=v, y=other)
        table = ResolvedBindings(; arguments=VarNamedTuple((x=v,)))
        @test_throws ArgumentError prepared_arguments(table, args)
    end
    GC.@preserve v begin
        alias = unsafe_wrap(Array, pointer(v), length(v))
        table = ResolvedBindings(; arguments=VarNamedTuple((x=v,)))
        @test_throws ArgumentError prepared_arguments(table, (x=v, y=alias))
    end
    table = ResolvedBindings(; arguments=VarNamedTuple((x=(a=v, b=v),)))
    @test_throws ArgumentError prepared_arguments(table, (x=(a=v, b=v),))
    for R in (Condition, Fix)
        independent = _bind_resolved(table, R, @varname(x.b) => copy(v))
        prepared = prepared_arguments(independent, (x=(a=v, b=v),))
        @test prepared.x.a !== v
        @test prepared.x.a == v
        @test prepared.x.b !== v
        replacement = (a=SVector(1.0, 2.0), b=v)
        bound = _bind_resolved(
            ResolvedBindings(; arguments=VarNamedTuple((x=replacement,))),
            R,
            @varname(x.a) => replacement.a,
        )
        prepared = prepared_arguments(bound, (x=replacement,))
        @test prepared.x.a === replacement.a
        @test prepared.x.b !== v
    end
    copied = @inferred _copy_model_argument((a=v, b=v))
    @test copied.a === copied.b
    @test copied.a !== v
    cycle = CyclicStorage(v, nothing)
    cycle.next = cycle
    copied = _copy_model_argument(cycle)
    @test copied.next === copied
    @test copied !== cycle
    table = ResolvedBindings(; arguments=VarNamedTuple((x=cycle,)))
    @test_throws ArgumentError prepared_arguments(table, (x=cycle,))
end

@testset "partial preparation and repeated isolation" begin
    for R in (Condition, Fix),
        T in (Float32, Float64, BigFloat, ForwardDiff.Dual{Nothing,Float64,1})

        source = (x=(a=T[1, 2], b=T[3, 4]),)
        table = ResolvedBindings(; arguments=VarNamedTuple(source))
        table = _bind_resolved(table, R, @varname(x.a[1]) => T(5))
        first = @inferred prepared_arguments(table, source)
        @test first.x.a == T[5, 2]
        @test first.x.b == source.x.b
        first.x.a[2] = 9
        first.x.b[1] = 9
        second = prepared_arguments(table, source)
        @test second.x.a == T[5, 2]
        @test second.x.b == T[3, 4]
        @test source.x.a == T[1, 2]
        @test source.x.b == T[3, 4]
    end
    args = (x=1.0, y=2)
    table = ResolvedBindings(; arguments=VarNamedTuple((x=args.x,)))
    @test (@inferred prepared_arguments(table, args)) === args
    table = _bind_resolved(table, Fix; x=3.0)
    @test (@inferred prepared_arguments(table, args)) === (x=3.0, y=2)
end

@testset "nested tracer copying and overlays" begin
    for gradient in (ForwardDiff.gradient, ReverseDiff.gradient), T in (Float32, Float64)
        x = T[2, 3]
        objective = function (z)
            for wrap in (identity, x -> (a=x,), Ref, x -> Dict(:a => x), WrappedStorage)
                wrapped = wrap(z)
                copied = _copy_model_argument(wrapped, @varname(x))
                value = if wrap === identity
                    copied
                elseif copied isa NamedTuple
                    copied.a
                elseif copied isa Ref
                    copied[]
                elseif copied isa Dict
                    copied[:a]
                else
                    copied.value
                end
                if gradient === ForwardDiff.gradient
                    @test value !== z
                end
                @test value == z
            end
            values = [ModelValue{Condition}(3z[1]), ModelValue{Condition}(z[2])]
            bindings = PartialArray(values, [true, false])
            copied = _model_argument_value(bindings, z, @varname(x))
            @test copied !== z
            return sum(abs2, copied)
        end
        @test gradient(objective, x) ≈ T[36, 6]
        @test x == T[2, 3]
    end
end

@testset "retained covariates and empty storage aliases" begin
    v = [1.0]
    table = ResolvedBindings(; arguments=VarNamedTuple((x=v,)))
    for ctor in (Dict, IdDict)
        @test_throws r"shared by `x` and `data`" prepared_arguments(
            table, (x=v, data=ctor(v => 1))
        )
        @test prepared_arguments(table, (x=v, data=ctor(copy(v) => 1))).x == v
    end
    empty = Float64[]
    for other in (Float64[], copy(empty))
        args = (x=(a=empty, b=other),)
        @test prepared_arguments(ResolvedBindings(; arguments=VarNamedTuple(args)), args) ==
            args
    end
    args = (x=(a=empty, b=empty),)
    @test_throws ArgumentError prepared_arguments(
        ResolvedBindings(; arguments=VarNamedTuple(args)), args
    )
    args = (x=(a=v, b=v),)
    table = ResolvedBindings(; arguments=VarNamedTuple(args))
    observed = DynamicPPL.setindex!!(
        VarNamedTuple(), ModelValue{ArgumentCondition}(v), @varname(x.b)
    )
    table = ResolvedBindings(
        observed,
        table.storage,
        table.observed_templates,
        table.fixed_templates,
        table.arguments,
        table.roots,
    )
    @test_throws ArgumentError prepared_arguments(table, args)
end

@testset "surviving argument values after removal" begin
    for T in (Float32, Float64, BigFloat), R in (Condition, Fix)
        args = (x=T[5, 6, 7],)
        table = _bind_resolved(
            ResolvedBindings(; arguments=VarNamedTuple(args)), R; x=T[1, 2, 3]
        )
        table = DynamicPPL._unbind_resolved(table, R, @varname(x[1]))
        @test prepared_arguments(table, args).x == T[5, 2, 3]
        @test args.x == T[5, 6, 7]
    end
end

@testset "tracked view copies and retained buffers" begin
    for wrap in (x -> view(x, :), x -> view(reshape(x, 1, 2), 1, :))
        objective = x -> sum(abs2, _copy_model_argument((a=wrap(x),), @varname(x)).a)
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(objective, [2.0, 3.0]))
        for x in ([2.0, 3.0], [-0.4, 0.2], [2.0, 3.0])
            result = zeros(2)
            ReverseDiff.gradient!(result, tape, x)
            @test result ≈ 2x
            @test ReverseDiff.gradient(objective, x) ≈ ForwardDiff.gradient(objective, x)
        end
    end
    ReverseDiff.gradient([2.0, 3.0]) do x
        for buffer in (ReverseDiff.value(x), ReverseDiff.deriv(x)),
            wrap in (identity, x -> view(x, :), x -> reshape(x, 1, 2))

            @test_throws r"retained storage.*latent path" _copy_model_argument(
                (tracked=x, raw=wrap(buffer)), @varname(x)
            )
            copied = _copy_model_argument((tracked=x, raw=copy(buffer)), @varname(x))
            @test copied.raw !== buffer
        end
        scalar = x[1]
        copied = _copy_model_argument([scalar, scalar], @varname(x))
        @test copied[1] === scalar
        @test copied[2] === scalar
        sum(x)
    end
end

@testset "partly latent argument storage allocations" begin
    source = (x=(X=zeros(1000, 1000), y=zeros(2)),)
    table = ResolvedBindings(; arguments=VarNamedTuple(source))
    table = _bind_resolved(table, Condition, @varname(x.X) => source.x.X)
    model = DynamicPPL._finish_binding(argument_preparation(source.x), table, nothing)
    model()
    @test (@allocated model()) < sizeof(source.x.X) ÷ 2
    result = model()
    @test result.x.X === source.x.X
    @test result.x.y !== source.x.y
    result.x.y .= 1
    @test source.x.y == zeros(2)
end

end

module BindingTests

using Test
using DynamicPPL, Distributions, Random, LinearAlgebra
using StableRNGs: StableRNG
using AbstractPPL: @of, of

# Set DYNAMICPPL_EXHAUSTIVE_TESTS=true to run the complete binding occupancy enumeration.
const EXHAUSTIVE = get(ENV, "DYNAMICPPL_EXHAUSTIVE_TESTS", "false") == "true"

@model function oracle_argument(x, whole, entered)
    entered[] = true
    if whole
        x ~ MvNormal(zeros(2), I)
    else
        x[1] ~ Normal()
        x[2] ~ Normal()
    end
    return x
end
@model function oracle_local(whole, entered)
    x = zeros(2)
    entered[] = true
    if whole
        x ~ MvNormal(zeros(2), I)
    else
        x[1] ~ Normal()
        x[2] ~ Normal()
    end
    return x
end
@model oracle_parent(c) = a ~ to_submodel(c)
struct OracleInit <: DynamicPPL.AbstractInitStrategy end
function DynamicPPL.init(rng, vn, d::Distribution, ::OracleInit)
    value = d isa MultivariateDistribution ? fill(0.25, length(d)) : 0.25
    return DynamicPPL.TransformedValue(value, DynamicPPL.NoTransform())
end
const ROOTS = (@varname(x), @varname(a.x), @varname(a.a.x))
const PARTS = (
    (@varname(x[1]), @varname(x[2])),
    (@varname(a.x[1]), @varname(a.x[2])),
    (@varname(a.a.x[1]), @varname(a.a.x[2])),
)
selected(mask, i) = !iszero(mask & (1 << (i - 1)))
const SCHEMAS = (
    @of(x = of(Array, 2)),
    @of(a = @of(x = of(Array, 2))),
    @of(a = @of(a = @of(x = of(Array, 2)))),
)
function bindmask(m, depth, mask, role, value)
    inputs = Tuple(PARTS[depth][i] => value + i for i in 1:2 if selected(mask, i))
    isempty(inputs) && return m
    schema = depth == 1 ? () : (SCHEMAS[depth],)
    return (role == 1 ? condition : fix)(m, inputs..., schema...)
end
function removemask(m, depth, mask, role, recursive)
    names = Tuple(PARTS[depth][i] for i in 1:2 if selected(mask, i))
    isempty(names) && return m
    f = role == 1 ? decondition : unfix
    return f(m, names...)
end
function base_model(storage, whole, initial, entered)
    m = if storage == :argument
        oracle_argument([1.0, 2.0], whole, entered)
    elseif storage == :template
        used = condition(oracle_local(whole, entered), @of(x = of(Array, 2)); x=zeros(2))
        decondition(used, @varname(x))
    else
        condition(oracle_local(whole, entered); x=zeros(2))
    end
    for i in 1:2
        if storage != :argument || initial[i] != 1
            m = decondition(m, PARTS[1][i])
        end
    end
    return m
end
function predict(q, initial, firstop, secondop, removal, whole; removal_scope=q)
    roles = zeros(Int, q, 2)
    values = zeros(q, 2)
    for i in 1:2
        initial[i] == 0 && continue
        owner, role = (initial[i] + 1) ÷ 2, 1 + (initial[i] - 1) % 2
        roles[owner, i] = role
        values[owner, i] = Float64(i)
    end
    mask, role, recursive = removal
    for scope in 1:q
        if scope == removal_scope
            for owner in 1:scope, i in 1:2
                selected(mask, i) && roles[owner, i] == role && (roles[owner, i] = 0)
            end
        end
        for (owner, subset, newrole, value) in (firstop, secondop)
            owner == scope || continue
            for i in 1:2
                selected(subset, i) || continue
                old = roles[scope, i]
                old != 0 && old != newrole && return (:call, string(PARTS[scope][i]))
                roles[scope, i] = newrole
                values[scope, i] = value + i
            end
        end
    end
    resultroles = zeros(Int, 2)
    resultvalues = fill(0.25, 2)
    for i in 1:2
        owners = [s for s in 1:q if roles[s, i] != 0]
        length(owners) > 1 && return (:entry, string(ROOTS[q]))
        if !isempty(owners)
            owner = only(owners)
            resultroles[i] = roles[owner, i]
            resultvalues[i] = values[owner, i]
        end
    end
    whole && resultroles[1] != resultroles[2] && return (:tilde, string(ROOTS[q]))
    density(x) = -(log(2pi) + x * x) / 2
    prior = sum((density(resultvalues[i]) for i in 1:2 if resultroles[i] == 0); init=0.0)
    likelihood = sum(
        (density(resultvalues[i]) for i in 1:2 if resultroles[i] == 1); init=0.0
    )
    listing = Tuple((roles[q, i], roles[q, i] == 0 ? 0.0 : values[q, i]) for i in 1:2)
    latent = if whole
        (all(iszero, resultroles) ? [string(ROOTS[q])] : String[])
    else
        [string(PARTS[q][i]) for i in 1:2 if resultroles[i] == 0]
    end
    return (:ok, (resultvalues, prior, likelihood, prior + likelihood, listing, latent))
end
oracle_inputs(x) = deepcopy(x)
oracle_inputs(::Base.RefValue{Bool}) = nothing
oracle_inputs(x::NamedTuple) = map(oracle_inputs, x)
function oracle_inputs(m::Model)
    return (oracle_inputs(m.args), deepcopy(conditioned(m)), deepcopy(fixed(m)))
end
function actual(storage, whole, q, initial, firstop, secondop, removal; removal_scope=q)
    entered = Ref(false)
    m = base_model(storage, whole, initial, entered)
    q > 1 && (m = condition(m, conditioned(m)))
    snapshots = []
    snapshot(m) = (deepcopy(conditioned(m)), deepcopy(fixed(m)), oracle_inputs(m.args))
    intact() = all(isequal(snapshot(model), before) for (model, before) in snapshots)
    push!(snapshots, (m, snapshot(m)))
    try
        for scope in 1:q
            scope > 1 && (m = oracle_parent(m))
            for i in 1:2
                initial[i] == 0 && continue
                owner, role = (initial[i] + 1) ÷ 2, 1 + (initial[i] - 1) % 2
                storage == :argument && owner == 1 && role == 1 && continue
                if owner == scope
                    push!(snapshots, (m, snapshot(m)))
                    m = bindmask(m, scope, 1 << (i - 1), role, 0.0)
                end
            end
            if scope == removal_scope
                push!(snapshots, (m, snapshot(m)))
                m = removemask(m, scope, removal...)
            end
            for (owner, subset, role, value) in (firstop, secondop)
                owner == scope || continue
                push!(snapshots, (m, snapshot(m)))
                m = bindmask(m, scope, subset, role, value)
            end
        end
    catch err
        err isa InterruptException && rethrow()
        intact() || return (:mutated, nothing)
        return (:call, (nameof(typeof(err)), sprint(showerror, err)))
    end
    push!(snapshots, (m, snapshot(m)))
    try
        before = snapshot(m)
        listing = Tuple(
            map(1:2) do i
                vn = PARTS[q][i]
                o, f = haskey(before[1], vn), haskey(before[2], vn)
                o && f && return (-1, 0.0)
                return if o
                    (1, before[1][vn])
                elseif f
                    (2, before[2][vn])
                else
                    (0, 0.0)
                end
            end,
        )
        value, vi = init!!(
            Xoshiro(1),
            m,
            VarInfo(
                RawValueAccumulator(false),
                LogPriorAccumulator(),
                LogLikelihoodAccumulator(),
            ),
            OracleInit(),
            UnlinkAll(),
        )
        intact() || return (:mutated, nothing)
        latent = sort!(string.(collect(keys(get_raw_values(vi)))))
        return (
            :ok,
            (
                value,
                getlogprior(vi),
                getloglikelihood(vi),
                getlogjoint(vi),
                listing,
                latent,
            ),
        )
    catch err
        err isa InterruptException && rethrow()
        intact() || return (:mutated, nothing)
        return (entered[] ? :tilde : :entry, (nameof(typeof(err)), sprint(showerror, err)))
    end
end
function agrees(expected, observed)
    expected[1] == observed[1] || return false
    if expected[1] != :ok
        return observed[2][1] === :ArgumentError && occursin(expected[2], observed[2][2])
    end
    e, o = expected[2], observed[2]
    return e[1] == o[1] &&
           all(isapprox(e[i], o[i]; atol=1e-12) for i in 2:4) &&
           e[5] == o[5] &&
           sort(e[6]) == o[6]
end
function run_oracle()
    failures, cases = 0, 0
    rng = StableRNG(1001)
    categories = Dict{Tuple{Symbol,Symbol},Int}()
    for storage in (:argument, :template, :held), whole in (false, true), q in 1:3
        startfail = failures
        for initial in Iterators.product(0:(2q), 0:(2q))
            for owner1 in 1:q, owner2 in 1:q, subset1 in 0:3, subset2 in 0:3
                for role1 in 1:2,
                    role2 in 1:2,
                    removal in 0:3,
                    removerole in 1:2,
                    recursive in (false, true),
                    offsets in ((2.0, 2.0), (2.0, 4.0), (4.0, 2.0))

                    mandatory =
                        initial == (0, 0) &&
                        owner1 == 1 &&
                        owner2 == q &&
                        subset1 == 3 &&
                        subset2 == 3 &&
                        role1 == role2 &&
                        removal == 3 &&
                        removerole == role1
                    !EXHAUSTIVE && !mandatory && rand(rng) > 0.00001 && continue
                    op1, op2 = (owner1, subset1, role1, offsets[1]),
                    (owner2, subset2, role2, offsets[2])
                    rem = (removal, removerole, recursive)
                    expected = predict(q, initial, op1, op2, rem, whole)
                    observed = actual(storage, whole, q, initial, op1, op2, rem)
                    cases += 1
                    valid = agrees(expected, observed)
                    if !valid
                        failures += 1
                        key = (expected[1], observed[1])
                        categories[key] = get(categories, key, 0) + 1
                        if failures <= 15
                            println(
                                "MISMATCH ",
                                (storage, whole, q, initial, op1, op2, rem),
                                " expected=",
                                expected,
                                " observed=",
                                observed,
                            )
                        end
                    end
                end
            end
        end
        println(
            "GROUP ",
            (storage, whole, q),
            " cases=",
            cases,
            " failures=",
            failures - startfail,
        )
        flush(stdout)
    end
    println(
        "TOTAL cases=",
        cases,
        " failures=",
        failures,
        " categories=",
        sort!(collect(categories); by=first),
    )
    @test failures == 0
end
function run_scope_and_batch_oracle()
    cases = failures = 0
    rng = StableRNG(1002)
    for storage in (:argument, :template, :held),
        whole in (false, true),
        q in 1:3,
        initial in Iterators.product(0:(2q), 0:(2q)),
        remover in 1:q,
        mask in 0:3,
        role in 1:2,
        recursive in (false, true),
        owner in 1:q,
        insertrole in 1:2

        mandatory =
            initial == (0, 0) &&
            remover == 1 &&
            owner == q &&
            mask == 3 &&
            role == insertrole &&
            recursive
        !EXHAUSTIVE && !mandatory && rand(rng) > 0.00001 && continue
        op = (owner, 3, insertrole, 2.0)
        noop = (owner, 0, insertrole, 2.0)
        removal = (mask, role, recursive)
        expected = predict(q, initial, op, noop, removal, whole; removal_scope=remover)
        observed = actual(
            storage, whole, q, initial, op, noop, removal; removal_scope=remover
        )
        cases += 1
        if !agrees(expected, observed)
            failures += 1
            failures <= 15 && println(
                "SCOPE MISMATCH ",
                (storage, whole, q, initial, remover, removal, op),
                " expected=",
                expected,
                " observed=",
                observed,
            )
        end
    end
    for storage in (:argument, :template, :held),
        whole in (false, true),
        q in 1:3,
        role in 1:2,
        position in 1:2,
        second in (3.0, 4.0)

        m = base_model(storage, whole, (0, 0), Ref(false))
        for scope in 2:q
            m = oracle_parent(m)
        end
        before = oracle_inputs(m)
        vn = PARTS[q][position]
        result = try
            (role == 1 ? condition : fix)(m, vn => 3.0, vn => second, SCHEMAS[q])
            false
        catch err
            err isa InterruptException && rethrow()
            err isa ArgumentError && occursin(string(vn), sprint(showerror, err))
        end
        cases += 1
        if !result || !isequal(before, oracle_inputs(m))
            failures += 1
            failures <= 15 &&
                println("BATCH MISMATCH ", (storage, whole, q, role, position, second))
        end
    end
    println("SCOPE_AND_BATCH cases=", cases, " failures=", failures)
    @test failures == 0
end
@testset "binding occupancy across model scopes" begin
    elapsed = @elapsed begin
        run_oracle()
        run_scope_and_batch_oracle()
    end
    println("Binding oracle seconds=", elapsed)
end

end
