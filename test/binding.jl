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
