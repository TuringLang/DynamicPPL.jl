module DynamicPPLBindingTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using DynamicPPL: @vnt
using AbstractPPL: @of, of
using Distributions
using LinearAlgebra: I
using AbstractPPL: AbstractPPL
using StableRNGs: StableRNG
using Random: Xoshiro
using StaticArrays: SVector
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector, Axis
using Test
using Unitful: @u_str

@testset "condition and fix" begin
    @testset "external facing condition & fix API" begin
        @model function demo_cond_fix()
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = demo_cond_fix()

        # Assume that we've either conditioned or fixed `x` inside `op_model`.
        function test_logp_correct(
            op::Union{typeof(condition),typeof(fix)}, op_model::Model, x
        )
            y = 1.0
            eval_values = VarNamedTuple(; y=y)
            @test (op === condition ? conditioned : fixed)(op_model) == VarNamedTuple(; x=x)
            if op === condition
                # x should contribute to the log-likelihood, but `y` should still be
                # in the log-prior.
                @test logprior(op_model, eval_values) == logpdf(Normal(x), y)
                @test loglikelihood(op_model, eval_values) == logpdf(Normal(), x)
            else
                # `x` should not contribute at all
                @test logprior(op_model, eval_values) == logpdf(Normal(x), y)
                @test iszero(loglikelihood(op_model, eval_values))
            end
        end

        x = 0.5
        @testset "$op" for op in (condition, fix)
            @testset "Keyword arguments" begin
                test_logp_correct(op, op(model; x=x), x)
            end
            @testset "$label as positional argument" for (label, input) in (
                ("VarNamedTuple", VarNamedTuple(; x=x)),
                ("NamedTuple", (; x=x)),
                ("Pair", @varname(x) => x),
            )
                test_logp_correct(op, op(model, input), x)
                if op === condition
                    test_logp_correct(condition, model | input, x)
                end
            end
            @testset "Dict as positional argument" begin
                @test_throws ArgumentError op(model, Dict(@varname(x) => x))
                if op === condition
                    @test_throws ArgumentError model | Dict(@varname(x) => x)
                end
            end
        end
    end

    @testset "Can fix immutable data safely" begin
        # extra regression test for
        # https://github.com/TuringLang/DynamicPPL.jl/issues/1176#issuecomment-3648871018
        @model function ntfix()
            m ~ Normal()
            data = (; x=undef)
            data.x ~ Normal(m, 1.0)
            return data.x
        end
        fixm = DynamicPPL.fix(ntfix(), (; data=(; x=5.0)))
        accs = VarInfo(RawValueAccumulator(false))
        retval, accs = DynamicPPL.init!!(fixm, accs, InitFromPrior(), UnlinkAll())
        # The fixed data should overwrite the NamedTuple that came before it
        @test retval == 5.0
        # `m` should still be sampled
        @test get_raw_values(accs)[@varname(m)] isa Real
    end

    @testset "can condition/fix on each individual part of a multivariate" begin
        @model function mvnorm()
            x ~ MvNormal(zeros(3), I)
            return x
        end
        @testset "with templating" begin
            vnt = @vnt begin
                @template x = zeros(3)
                x[1] := 1.0
                x[2] := 2.0
                x[3] := 3.0
            end
            for op in (condition, fix)
                op_model = op(mvnorm(), vnt)
                @test op_model() == [1.0, 2.0, 3.0]
            end
        end
        @testset "without templating" begin
            vnt = @vnt begin
                x[1] := 1.0
                x[2] := 2.0
                x[3] := 3.0
            end
            for op in (condition, fix)
                @test_throws ArgumentError op(mvnorm(), vnt)
            end
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
            nested = op(
                partial_array_parent(model),
                @varname(child.x[1, 1]) => 1.5,
                @of(child = @of(x = of(Array, 2, 2)))
            )
            model = op(model, @varname(x[1, 1]) => 1.5, @of(x = of(Array, 2, 2)))
            for (partial_model, values) in
                ((model, next_values), (nested, VarNamedTuple(; child=next_values)))
                @test returned(op(partial_model, values), VarNamedTuple()) ==
                    [1.5 0.0; 2.5 0.0]
            end
        end
    end
end

@model function binding_array(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
@model function binding_local()
    x = zeros(2)
    x[1] ~ Normal()
    x[2] ~ Normal()
    return x
end
@model binding_scalar(x=missing) = (x ~ Normal(); x)
@model binding_mv(x=missing) = (x ~ MvNormal(zeros(2), I); x)
@model binding_fields(x) = (x.a[1] ~ Normal(); x.a[2] ~ Normal(); x.b ~ Normal(); x)
const BINDING_ROLES = ((condition, decondition, conditioned), (fix, unfix, fixed))

function binding_error(f, address, rule)
    err = try
        f()
    catch e
        e
    end
    @test err isa ArgumentError
    if err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(address, message)
        @test occursin(rule, message)
    end
end

@testset "binding inputs and ownership" begin
    for (bind, remove, listing) in BINDING_ROLES
        m = binding_scalar()
        expected = bind(m; x=0.5)
        for input in (:x => 0.5, ((x=0.5,),), (:x => 0.5,))
            bound = bind(m, input)
            @test listing(bound) == listing(expected)
            @test logjoint(bound, (;)) == logjoint(expected, (;))
        end
        @test listing(bind(expected; x=1.0))[@varname(x)] == 1.0
        @test listing(expected)[@varname(x)] == 0.5
        @test isempty(listing(remove(expected)))
        @test_throws ArgumentError bind(m, :x => 1.0, @varname(x) => 2.0)
        @test_throws ArgumentError bind(m, Dict(:x => 1.0))
        @test_throws ArgumentError bind(m, TransformedValue([0.5], Unlink()))
        @test_throws ArgumentError bind(m, VarInfo())
    end
    for input in (:x => 0.5, (@varname(x) => 0.5,))
        @test conditioned(binding_scalar() | input) == VarNamedTuple(; x=0.5)
    end
    original = binding_scalar(1.0)
    @test conditioned(original)[@varname(x)] == 1.0
    @test isempty(
        conditioned(contextualize(binding_scalar(), condition(original; x=2.0).context))
    )
    for api in (logprior, loglikelihood, logjoint, returned)
        @test_throws ArgumentError api(binding_scalar(), Dict(@varname(x) => 1.0))
    end
    vi = VarInfo(
        Xoshiro(1), binding_scalar(), InitFromParams(Dict(@varname(x) => 1.0), nothing)
    )
    @test getlogjoint(vi) == logpdf(Normal(), 1.0)
    direct = Model{false}(original.f, original.args, original.defaults; args_on_lhs=(:x,))
    @test direct(Xoshiro(1)) == 1.0
    @test conditioned(direct) == conditioned(original)
    @test_throws MethodError Model{false,()}(original.f, original.args, original.defaults)
end

@testset "binding storage and density roles" begin
    for (bind, remove, listing) in BINDING_ROLES
        entered = Ref(false)
        @model function entered_array(entered)
            entered[] = true
            x = zeros(2)
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end
        m = entered_array(entered)
        schema = @of(x = of(Array, 2))
        binding_error(() -> bind(m, @varname(x[1]) => 0.5), r"\bx\b", "storage")
        @test !entered[]
        for args in
            ((schema, @varname(x[1]) => 0.5), (@varname(x[1]) => 0.5, schema, schema))
            @test_throws ArgumentError bind(m, args...)
            @test !entered[]
        end
        stored = @vnt begin
            @template x = zeros(2)
            x[1] := 0.5
        end
        sampled = rand(Xoshiro(1), binding_local())
        block = subset(sampled, [@varname(x[1])])
        for bound in (
            bind(decondition(binding_array(zeros(2))), @varname(x[1]) => 0.5),
            remove(bind(binding_local(); x=[0.5, 1.0]), @varname(x[2])),
            bind(m, stored),
            bind(m, @varname(x[1]) => 0.5, schema),
            bind(
                binding_array(Union{Missing,Float64}[missing, missing]),
                @varname(x[1]) => 0.5,
                schema,
            ),
        )
            @test listing(bound)[@varname(x[1])] == 0.5
            @test !haskey(listing(bound), @varname(x[2]))
            @test returned(bound, (; x=[0.5, 0.25])) == [0.5, 0.25]
            @test logprior(bound, (; x=[0.5, 0.25])) ≈ logpdf(Normal(), 0.25)
            @test loglikelihood(bound, (; x=[0.5, 0.25])) ≈
                (bind === condition ? logpdf(Normal(), 0.5) : 0.0)
        end
        @test bind(binding_local(), block)(Xoshiro(2))[1] == block[@varname(x[1])]
        @test bind(binding_local(), @varname(x[1]) => 0.5, of(block))(Xoshiro(2))[1] == 0.5
        @test isempty(listing(bind(binding_local(), schema)))
        @test_throws ArgumentError bind(binding_mv(), @varname(x[1]) => 0.5, schema)
        onepart = bind(decondition(binding_mv(zeros(2))), @varname(x[1]) => 0.5, schema)
        @test_throws ArgumentError onepart(Xoshiro(1))
        @test bind(onepart, @varname(x[2]) => 1.0)(Xoshiro(1)) == [0.5, 1.0]
        other = bind === condition ? fix : condition
        @test_throws ArgumentError other(onepart, @varname(x[1]) => 1.0)
        @test_throws ArgumentError other(onepart, @varname(x[2]) => 1.0)(Xoshiro(1))
        @test other(remove(onepart); x=[1.0, 2.0])(Xoshiro(1)) == [1.0, 2.0]
    end
end

@testset "binding selectors" begin
    for (bind, remove, listing) in BINDING_ROLES
        m = decondition(binding_array(zeros(2, 2)))
        for (address, value, indices) in (
            (@varname(x[1]), 0.5, [1]),
            (@varname(x[CartesianIndex(2, 1)]), 0.5, [2]),
            (@varname(x[begin:end]), fill(0.5, 4), 1:4),
            (@varname(x[:, 2]), fill(0.5, 2), [3, 4]),
            (@varname(x[[true, false, true, false]]), fill(0.5, 2), [1, 3]),
            (@varname(x[1:3][2]), 0.5, [2]),
        )
            bound = bind(m, address => value)
            result = bound(Xoshiro(1))
            @test all(result[i] == 0.5 for i in indices)
            @test all(listing(bound)[@varname(x[i])] == 0.5 for i in indices)
            @test isempty(listing(remove(bound, address)))
        end
        for address in (
            @varname(x[5]), @varname(x[0]), @varname(x[1:2][3]), @varname(x[[true, false]])
        )
            @test_throws ArgumentError bind(m, address => 0.5)
            @test_throws ArgumentError remove(m, address)
        end
        released = remove(bind(binding_local(); x=ones(3)))
        @test_throws ArgumentError remove(released, @varname(x[1:2][3]))
        @test_throws ArgumentError bind(m, @varname(x[:, 1]) => ones(3))
        @test_throws ArgumentError bind(m, @varname(x[2]) => 0.5, @varname(x[2, 1]) => 0.5)
        @test_throws ArgumentError bind(m, @varname(x[[1, 1]]) => [0.5, 0.5])
        bound = bind(m, @varname(x[1]) => 0.5)
        for inds in (Int[], 2:1, falses(4))
            @test listing(bind(bound, @varname(x[inds]) => Float64[])) == listing(bound)
            @test listing(remove(bound, @varname(x[inds]))) == listing(bound)
            @test_throws ArgumentError bind(bound, @varname(x[inds][1]) => 1.0)
        end
        fields = decondition(binding_fields((a=zeros(2), b=0.0)))
        @test bind(fields, @varname(x.a[end]) => 0.5)(Xoshiro(1)).a[2] == 0.5
        @test_throws ArgumentError bind(fields, @varname(x[1]) => [1.0, 2.0])
        @test_throws ArgumentError bind(fields, @varname(x[:a]) => [1.0, 2.0])
        @model zero_array() = (x = fill(0.0); x[] ~ Normal(); x)
        for address in (@varname(x[]), @varname(x[CartesianIndex()]))
            b = bind(zero_array(), address => 0.5, @of(x = of(fill(0.0))))
            @test b(Xoshiro(1)) == fill(0.5)
            @test remove(b, address)(Xoshiro(1)) == fill(rand(Xoshiro(1), Normal()))
        end
    end
end

@model function interface_storage(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end

@testset "container interface" begin
    @testset "ComponentArray index bindings" begin
        data = ComponentVector(; a=1.0, b=2.0)
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            source = decondition(interface_storage(data))
            partial = bind(source, @varname(x[1]) => 3.0)
            result = partial(Xoshiro(1))
            @test typeof(result) === typeof(data)
            @test axes(result) == axes(data)
            @test ComponentArrays.getaxes(result) == ComponentArrays.getaxes(data)
            @test result[1] == 3.0
            whole = bind(partial, @varname(x[2]) => 4.0)
            @test whole(Xoshiro(1)) == ComponentVector(; a=3.0, b=4.0)
            @test typeof(whole(Xoshiro(1))) === typeof(data)
            released = remove(whole, @varname(x[1]))
            @test released(Xoshiro(1))[2] == 4.0
            @test typeof(released(Xoshiro(1))) === typeof(data)
            @test remove(released, @varname(x[2]))(Xoshiro(1)) == source(Xoshiro(1))
            @test data == ComponentVector(; a=1.0, b=2.0)
            @test_throws ArgumentError bind(source, @varname(x.a) => 3.0)
            @test_throws ArgumentError remove(whole, @varname(x.a))
        end
    end
end

@testset "ComponentArray buffer ownership" begin
    data = ComponentArray(reshape([1.0, 2.0], 1, 2), Axis(; a=1:1), Axis(; b=1:2))
    result = DynamicPPL._fill_latent_storage(nothing, data, @varname(x), data)
    result[1] = 3.0
    @test data[1] == 1.0
end

@testset "ComponentArray latent copies" begin
    data = ComponentVector(; a=1.0, b=2.0)
    original = copy(data)
    result = decondition(interface_storage(data))(Xoshiro(1))
    @test typeof(result) === typeof(data)
    @test axes(result) == axes(data)
    @test result !== data
    result[1] = 100.0
    @test data == original
end

@testset "Unitful latent copies" begin
    data = [1.0, 2.0]u"m"
    model = decondition(interface_storage(data))
    result = DynamicPPL.prepare_model_argument(model, @varname(x), data)
    @test result == data
    @test typeof(result) === typeof(data)
    @test axes(result) == axes(data)
    @test !Base.mightalias(result, data)
    result[1] = 3.0u"m"
    @test data == [1.0, 2.0]u"m"
end

@testset "Unitful exact partial conversion" begin
    for (bind, _, listing) in BINDING_ROLES,
        (T, inexact) in ((Float32, 0.1u"m"), (Int, 0.5u"m"))

        data = T[1, 2]u"m"
        model = decondition(interface_storage(data))
        bound = bind(model, @varname(x[1]) => 3.0u"m")
        @test listing(bound)[@varname(x[1])] === T(3)u"m"
        result = DynamicPPL.prepare_model_argument(bound, @varname(x), data)
        @test result == T[3, 2]u"m"
        @test eltype(result) === eltype(data)
        @test_throws ArgumentError bind(model, @varname(x[1]) => inexact)
    end
end

@model interface_nested(x) = (x[1][1] ~ Normal(); x)
@testset "latent container validation" begin
    @testset "validate retained latent storage" begin
        data = Any[1.0, 2.0, 3.0]
        bound = condition(decondition(interface_storage(data)); x=[5.0, 6.0])
        data[3] = SVector(3.0, 4.0)
        @test decondition(bound, @varname(x[1]))(Xoshiro(1)) ==
            [rand(Xoshiro(1), Normal()), 6.0]
    end
    @testset "unsupported latent storage at both boundaries" begin
        data = Any[[1.0, 2.0]]
        latent = decondition(interface_nested(data))
        bound = condition(latent; x=[[5.0, 6.0]])
        data[1] = view([1.0, 2.0], :)
        message = "`x[1]`: unsupported binding storage $(typeof(data[1])); use collect or pass a plain Array before constructing the model (axes or metadata may be lost)"
        @test_throws ArgumentError(message) latent(Xoshiro(1))
        @test decondition(bound, @varname(x[1][1]))(Xoshiro(1)) ==
            [[rand(Xoshiro(1), Normal()), 6.0]]
        for bind in (condition, fix)
            @test_throws ArgumentError(message) bind(latent)
            @test_throws ArgumentError(message) bind(latent, @of(x = of(Array, Float64, 1)))
        end
        @test_throws ArgumentError(message) latent | NamedTuple()
        @test_throws ArgumentError decondition(interface_storage(view([1.0, 2.0], :)))
    end
end

@testset "binding placeholders" begin
    for (bind, _, _) in BINDING_ROLES,
        payload in (
            missing,
            nothing,
            [missing],
            (a=[nothing],),
            [[missing]],
            TransformedValue((0.5, missing), NoTransform()),
            TransformedValue([nothing], Unlink()),
        )

        @test_throws ArgumentError bind(binding_scalar(), :x => payload)
    end
    data = Union{Missing,Float64}[missing, 1.0]
    m = binding_array(data)
    @test conditioned(m)[@varname(x[2])] == 1.0
    @test Set(keys(rand(Xoshiro(1), m))) == Set([@varname(x[1])])
    @test ismissing(data[1])
    for placeholder in (missing, nothing)
        @test isempty(conditioned(binding_scalar(placeholder)))
        @test rand(Xoshiro(1), binding_scalar(placeholder))[@varname(x)] isa Real
    end
    @model mutated_observation(x) = (x[1] = missing; x[1] ~ Normal())
    @test_throws ArgumentError mutated_observation(Any[1.0])(Xoshiro(1))
    for (bind, _, _) in BINDING_ROLES
        payload = Any[1.0, 2.0]
        b = bind(binding_local(); x=payload)
        payload[1] = missing
        @test_throws ArgumentError b(Xoshiro(1))
    end
end

@testset "inexact binding diagnostics" begin
    for (bind, _, _) in BINDING_ROLES
        for data in ([1, 2], (1, 2), Float32[1, 2])
            value = eltype(data) === Float32 ? 0.1 : 1.5
            binding_error(
                "`x[1]`",
                "conversion of $value into storage element type $(eltype(data)) is inexact",
            ) do
                bind(decondition(binding_array(data)), @varname(x[1]) => value)
            end
        end
        for (storage, value) in ((1, 1.5), (1.0f0, 0.1))
            binding_error(
                "`x.b`",
                "conversion of $value into storage element type $(typeof(storage)) is inexact",
            ) do
                bind(
                    decondition(binding_fields((a=zeros(2), b=storage))),
                    @varname(x.b) => value,
                )
            end
        end
    end
end

@testset "local shape diagnostics" begin
    @model local_fields() = (x = (a=zeros(2),); x.a[1] ~ Normal(); x)
    for (bind, _, _) in BINDING_ROLES
        binding_error("`x`:", "bound axes (Base.OneTo(3),), local axes (Base.OneTo(2),)") do
            logjoint(bind(binding_local(); x=ones(3)), (;))
        end
        binding_error(
            "`x.a`:", "bound axes (Base.OneTo(3),), local axes (Base.OneTo(2),)"
        ) do
            logjoint(bind(local_fields(), @varname(x.a) => ones(3)), (;))
        end
    end
end

@testset "exact conversion and storage lifetime" begin
    @model nested_arrays(x) = (x[1][1] ~ Normal(); x)
    for (bind, remove, listing) in BINDING_ROLES
        nested = bind(nested_arrays(nothing); x=[SVector(0.0, 0.0)])
        @test_throws DimensionMismatch bind(nested, @varname(x[1]) => ones(3))
        for T in (Float32, BigFloat)
            m = bind(binding_local(), @varname(x[1]) => T(0.5), @of(x = of(Array, T, 2)))
            @test listing(m)[@varname(x[1])] isa T
            @test listing(m)[@varname(x[1])] == T(0.5)
            @test m(Xoshiro(1))[1] == T(0.5)
        end
        for (T, value) in ((Float32, 0.1), (Int, 0.5))
            @test_throws ArgumentError bind(
                binding_local(), @varname(x[1]) => value, @of(x = of(Array, T, 2))
            )
        end
        bound = bind(binding_array(nothing); x=Float32[0.5, 1.0])
        @test_throws ArgumentError remove(bound, @varname(x[1]))
        partial = remove(
            bind(decondition(binding_array(zeros(3))); x=Float32[0.5, 1.0]), @varname(x[1])
        )
        @test_throws ArgumentError bind(partial, @varname(x[1]) => 0.1)
        @test_throws ArgumentError bind(partial; x=ones(3))
        allparts = remove(partial, @varname(x[2]))
        @test isempty(listing(allparts))
        @test length(allparts(Xoshiro(1))) == 2
        @test_throws ArgumentError bind(allparts, @varname(x[1]) => 0.1)
        cleared = remove(bound, @varname(x))
        @test_throws ArgumentError bind(cleared, @varname(x[1]) => 0.5)
        @test bind(cleared; x=[0.1, 0.2, 0.3])(Xoshiro(1)) == [0.1, 0.2, 0.3]
        observed = binding_array([1.0, 2.0])
        replaced = condition(decondition(observed); x=[3.0, 4.0, 5.0])
        @test length(decondition(replaced)(Xoshiro(1))) == 2
        @test isempty(conditioned(decondition(replaced)))
        listed = listing(partial)
        replay = bind(binding_local(), listed)
        @test replay(Xoshiro(1)) == partial(Xoshiro(1))
        @test listing(replay) == listed
    end
    observed = decondition(condition(binding_local(); x=ones(Float32, 2)), @varname(x[2]))
    split = fix(observed, @varname(x[2]) => 0.25)
    @test conditioned(split)[@varname(x[1])] isa Float32
    @test fixed(split)[@varname(x[2])] isa Float64
    @test_throws ArgumentError condition(split, @varname(x[1]) => 0.1)
    @test fixed(fix(split, @varname(x[2]) => 0.1))[@varname(x[2])] == 0.1
end

@model function prepared_scalar(x)
    before = x
    x = x + 1
    x ~ Normal()
    return before, x
end
@model typed_positional(x::Float64) = (x ~ Normal(); x)
@model typed_keyword(; x::Float64=0.0) = (x ~ Normal(); x)
@model typed_splat(x::Float64, ys::Float64...) = (x ~ Normal(); return (x, ys))
@testset "argument preparation and copying" begin
    observed = condition(prepared_scalar(0.0); x=2.0)
    held = fix(decondition(prepared_scalar(0.0)); x=2.0)
    @test observed(Xoshiro(1)) == (2.0, 3.0)
    @test loglikelihood(observed, (;)) == logpdf(Normal(), 3.0)
    @test held(Xoshiro(1)) == (2.0, 2.0)
    @test logjoint(held, (;)) == 0.0
    for model in (typed_positional(0.0), typed_keyword(), typed_splat(0.0, 1.0))
        for (bind, _, _) in BINDING_ROLES
            binding_error(() -> bind(decondition(model); x=1), "x", "type")
            @test first(bind(decondition(model); x=1.0)(Xoshiro(1))) == 1.0
        end
    end
    @model typed_rest(xs::Float64...) = (xs[1] ~ Normal(); xs)
    @test_throws ArgumentError condition(typed_rest(1.0); xs=(1,))
    for storage in ([[1.0, 2.0]], (a=[1.0, 2.0], b=3.0))
        @model nested_array(x) = (x[1][1] ~ Normal(); x[1][2] ~ Normal(); x)
        model = storage isa NamedTuple ? binding_fields(storage) : nested_array(storage)
        original = deepcopy(storage)
        latent = decondition(model)
        first_run = deepcopy(latent(Xoshiro(1)))
        @test latent(Xoshiro(1)) == first_run
        @test storage == original
        result = latent(Xoshiro(2))
        result isa NamedTuple ? (result.a[1] = 100.0) : (result[1][1] = 100.0)
        @test storage == original
    end
    @model shared(x, y) = (x[1] ~ Normal(); y)
    shared_data = [1.0]
    @test_throws ArgumentError decondition(shared(shared_data, shared_data))(Xoshiro(1))
    @model shared_fields(x) = (x.a[1] ~ Normal(); x.b)
    @test_throws ArgumentError decondition(shared_fields((a=shared_data, b=shared_data)))(
        Xoshiro(1)
    )
    @model covariate(x) = (y ~ Normal(); x)
    @test covariate(shared_data)(Xoshiro(1)) === shared_data
    model = condition(binding_scalar(); x=1.0)
    _, vi = @inferred init!!(Xoshiro(1), model, VarInfo(), InitFromPrior(), UnlinkAll())
    @test getloglikelihood(vi) == logpdf(Normal(), 1.0)
end

@testset "partial bindings preserve placeholder argument branches" begin
    @model function comp(x::Union{Missing,AbstractMatrix{<:Integer}})
        generator = ismissing(x)
        if generator
            x = zeros(Int, 2, 2)
            x[1, :] ~ product_distribution([Poisson(3.0), Poisson(3.0)])
            x[2, :] .= 10 .- x[1, :]
        else
            @addlogprob! -Float64(sum(x))
        end
        return (; generator, x)
    end
    m = comp(missing)
    theta = rand(Xoshiro(1), m)
    for bind in (fix, condition)
        @test_throws ArgumentError bind(m, theta, @of(x = of(Array, Int, 2, 2)))
    end
    x = m(Xoshiro(1)).x
    @test fix(m; x)(Xoshiro(2)) == (; generator=false, x)
end

@testset "placeholder argument binding routes" begin
    @model fields(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
    @model nested(x) = (x.a.b ~ Normal(); x.a.c ~ Normal(); x)
    stored = @vnt begin
        @template x = zeros(2)
        x[1] := 1.0
    end
    for placeholder in (missing, nothing), (bind, remove, _) in BINDING_ROLES
        m = binding_array(placeholder)
        for template in ((), (@of(x = of(Array, 2)),), (of(stored),))
            @test_throws ArgumentError bind(m, @varname(x[1]) => 1.0, template...)
        end
        @test_throws ArgumentError bind(m, stored)
        @test_throws ArgumentError m | stored
        @test bind(m, @varname(x[:]) => [1.0, 2.0], @of(x = of(Array, 2)))(Xoshiro(1)) ==
            [1.0, 2.0]
        schema = of((x=(a=0.0, b=0.0),))
        @test_throws ArgumentError bind(fields(placeholder), @varname(x.a) => 1.0, schema)
        @test bind(fields(placeholder), @varname(x.a) => 1.0, @varname(x.b) => 2.0, schema)(
            Xoshiro(1)
        ) == (a=1.0, b=2.0)
        for (model, vn) in (
            (nested((a=placeholder,)), @varname(x.a.b)),
            (decondition(nested((a=placeholder,))), @varname(x.a.b)),
            (binding_fields((a=placeholder, b=0.0)), @varname(x.a[1])),
            (decondition(binding_fields((a=placeholder, b=0.0))), @varname(x.a[1])),
        )
            binding_error(() -> bind(model, vn => 1.0), "x.a", "bind the whole value")
        end
        held = bind(decondition(nested((a=placeholder,))); x=(a=(b=1.0, c=2.0),))
        @test_throws ArgumentError remove(held, @varname(x.a.b))
    end
end

@testset "removing all placeholder positions" begin
    @model function two(x)
        if x isa Union{Missing,Nothing}
            x = zeros(2)
        end
        x[1] ~ Normal()
        x[2] ~ Normal()
        return x
    end
    for placeholder in (missing, nothing), (bind, remove, _) in BINDING_ROLES
        m = bind(two(placeholder); x=[1.0, 2.0])
        for names in ((@varname(x[1]), @varname(x[2])), (@varname(x[:]),))
            @test remove(m, names...)(Xoshiro(1)) == remove(m, @varname(x))(Xoshiro(1))
        end
    end
    for (bind, remove, _) in BINDING_ROLES
        m = decondition(two(zeros(2)))
        @test bind(remove(m, @varname(x[1])), @varname(x[1]) => 1.0)(Xoshiro(1))[1] == 1.0
    end
end

@testset "absent nested argument storage" begin
    @model nested(x) = (x[1][1] ~ Normal(); x)
    for (bind, _, _) in BINDING_ROLES
        binding_error(
            () -> bind(nested([[1.0], [2.0]]), @varname(x[3][1]) => 0.5),
            "x",
            "index is outside retained storage",
        )
        for placeholder in (missing, nothing)
            binding_error(
                () -> bind(nested([placeholder]), @varname(x[1][1]) => 0.5),
                "x",
                "bind the whole value",
            )
        end
    end
end

@testset "binding identity" begin
    for (bind, remove, listing) in BINDING_ROLES
        for (a, b) in (
            (bind(binding_scalar(); x=NaN), bind(binding_scalar(); x=NaN)),
            (bind(binding_local(); x=ones(2)), bind(binding_local(); x=ones(2))),
            (
                bind(
                    binding_local(),
                    @varname(x[1]) => 1.0,
                    @varname(x[2]) => 2.0,
                    @of(x = of(Array, 2))
                ),
                bind(
                    binding_local(),
                    @varname(x[2]) => 2.0,
                    @varname(x[1]) => 1.0,
                    @of(x = of(Array, 2))
                ),
            ),
        )
            @test a == b
            @test isequal(a, b)
            @test hash(a) == hash(b)
            @test b in Set([a])
        end
        @test bind(binding_scalar(); x=0.0) != bind(binding_scalar(); x=-0.0)
        a = remove(bind(binding_array(nothing); x=ones(2)), @varname(x[1]), @varname(x[2]))
        b = remove(
            bind(binding_array(nothing); x=ones(3)),
            @varname(x[1]),
            @varname(x[2]),
            @varname(x[3])
        )
        @test isempty(listing(a)) && isempty(listing(b))
        @test a != b
        a = bind(binding_local(), @varname(x[1]) => 0.5, @of(x = of(Array, Float32, 2)))
        b = bind(binding_local(), @varname(x[1]) => 0.5, @of(x = of(Array, Float64, 2)))
        @test a != b
    end
    @test condition(binding_scalar(); x=1.0) != fix(binding_scalar(); x=1.0)
    @test decondition(binding_scalar()) != unfix(binding_scalar())
    @test binding_array(ones(2)) == binding_array(ones(2))
end

@model prepared_fields(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
@testset "partial prepared arguments infer" begin
    for T in (Float32, BigFloat)
        data = (a=T(1), b=T(2))
        model = fix(decondition(prepared_fields(data)); x=(b=T(4),))
        params = VarNamedTuple((@varname(x.a) => T(3),))
        value, vi = @inferred init!!(
            Xoshiro(1), model, VarInfo(), InitFromParams(params, nothing)
        )
        @test (value.a, value.b) == (T(3), T(4))
        @test value.a isa T && value.b isa T
        @test getlogprior(vi) ≈ logpdf(Normal(), T(3))
        @test getloglikelihood(vi) == 0.0
        @test data == (a=T(1), b=T(2))
    end
end

struct BCCase
    kind::Symbol
    argument::Bool
    depth::Int
    childfixed::Bool
    runtime::Bool
end
struct BCOp
    verb::Symbol
    skip_short::Bool
    path::Union{Nothing,Tuple}
    value::Any
end
BCOp(verb, path, value=nothing; skip=false) = BCOp(verb, skip, path, value)
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
    else
        bc_structured(c, [2.0, 3.0])
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
mutable struct BCLayer
    obs::Dict{Tuple,Any}
    fixed::Dict{Tuple,Any}
    owners::Dict{Tuple,Any}
    fixowners::Dict{Tuple,Any}
    shapes::Dict{Tuple,Any}
    removals::Vector{Tuple{Bool,Union{Nothing,Tuple}}}
end
BCLayer() = BCLayer(Dict(), Dict(), Dict(), Dict(), Dict(), [])
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
        ls[end].shapes[p] = bc_basevalue(c)
    end
    if c.childfixed
        empty!(ls[end].obs)
        empty!(ls[end].owners)
        v = bc_structured(c, [4.0, 5.0])
        merge!(ls[end].fixed, Dict(bc_leaves(p, v)))
        ls[end].fixowners[p] = v
        ls[end].shapes[p] = v
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
    found = (:latent, 0.25)
    for l in r.layers
        for (values, role) in ((l.obs, :observed), (l.fixed, :fixed))
            if haskey(values, p)
                found[1] == :latent || return (:collision, nothing)
                found = (role, values[p])
            end
        end
    end
    return found
end
function bc_same_shape(a, b)
    if a isa NamedTuple || b isa NamedTuple
        return a isa NamedTuple &&
               b isa NamedTuple &&
               keys(a) == keys(b) &&
               all(bc_same_shape(x, y) for (x, y) in zip(a, b))
    elseif a isa Union{AbstractArray,Tuple} || b isa Union{AbstractArray,Tuple}
        return a isa Union{AbstractArray,Tuple} &&
               b isa Union{AbstractArray,Tuple} &&
               axes(a) == axes(b) &&
               all(bc_same_shape(x, y) for (x, y) in zip(a, b))
    end
    return true
end
function bc_bind_shape!(l, c, p, value)
    for (root, template) in l.shapes
        if bc_contains(root, p)
            previous = bc_at(template, p[(length(root) + 1):end])
            if !bc_same_shape(previous, value)
                occupied =
                    any(q -> bc_contains(root, q), keys(l.owners)) ||
                    any(q -> bc_contains(root, q), keys(l.fixowners))
                (p != root || occupied) && return :call
                l.shapes[root] = value
            end
            return :ok
        elseif bc_contains(p, root)
            return :call
        end
    end
    any(k -> k isa Int, p) && return :call
    any(k -> k isa Int, p) || (l.shapes[p] = value)
    return :ok
end
function bc_path_valid(shapes, p)
    for (root, value) in shapes
        bc_contains(root, p) || continue
        for key in p[(length(root) + 1):end]
            value isa Tuple && !allequal(typeof.(value)) && return false
            if value isa NamedTuple
                key isa Symbol && haskey(value, key) || return false
            elseif value isa Union{AbstractArray,Tuple}
                key isa Int && key in eachindex(value) || return false
            else
                return false
            end
            value = bc_at(value, (key,))
        end
    end
    return true
end
function bc_remove!(l, fixed, p, c, argument)
    values = fixed ? l.fixed : l.obs
    owners = fixed ? l.fixowners : l.owners
    for q in collect(keys(values))
        (p === nothing || bc_contains(p, q)) && delete!(values, q)
    end
    for q in collect(keys(owners))
        (p === nothing || bc_contains(p, q)) && delete!(owners, q)
    end
    bp = bc_basepath(c)
    if argument &&
        (p === nothing || bc_contains(p, bp)) &&
        !any(q -> bc_related(bp, q), keys(l.owners)) &&
        !any(q -> bc_related(bp, q), keys(l.fixowners))
        empty!(l.shapes)
        l.shapes[bp] = bc_basevalue(c)
    end
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
    owners = fixed ? l.fixowners : l.owners
    p === nothing && adding && return :ok
    if adding &&
        c.depth > 0 &&
        p !== nothing &&
        bc_contains(bc_basepath(c), p) &&
        any(k -> k isa Int, p)
        if c.kind in (:tuple, :tuplewhole, :nested_tuple)
            any(q -> bc_contains(q, p), keys(l.shapes)) || return :call
        end
        root = bc_basepath(c)
        any(q -> bc_contains(q, p), keys(l.shapes)) || (l.shapes[root] = bc_basevalue(c))
    end
    if p !== nothing
        addresses = bc_universe(c)
        append!(addresses, keys(l.obs), keys(l.fixed))
        c.depth == 0 && !any(q -> bc_related(p, q), addresses) && return :call
        bc_path_valid(l.shapes, p) || return :call
    end
    if adding
        opposite = fixed ? l.obs : l.fixed
        any(haskey(opposite, q) for (q, _) in bc_leaves(p, op.value)) && return :call
        bc_bind_shape!(l, c, p, op.value) == :call && return :call
        for q in collect(keys(d))
            bc_contains(p, q) && delete!(d, q)
        end
        merge!(d, Dict(bc_leaves(p, op.value)))
        for q in collect(keys(owners))
            bc_contains(p, q) && delete!(owners, q)
        end
        owners[p] = op.value
    else
        bc_remove!(l, fixed, p, c, c.argument && c.depth == 0)
        if p === nothing
            filter!(marker -> first(marker) != fixed, l.removals)
            push!(l.removals, (fixed, p))
        elseif !any(
            marker ->
                first(marker) == fixed &&
                    (last(marker) === nothing || bc_contains(last(marker), p)),
            l.removals,
        )
            push!(l.removals, (fixed, p))
        end
    end
    return :ok
end
function bc_project(r)
    r = deepcopy(r)
    c, bp = r.case, bc_basepath(r.case)
    for (i, parent) in enumerate(r.layers), j in (i + 1):length(r.layers)
        child = r.layers[j]
        for (fixed, p) in parent.removals
            p === nothing || bc_path_valid(child.shapes, p) || return nothing
            bc_remove!(child, fixed, p, c, c.argument && j == length(r.layers))
        end
    end
    for (i, parent) in enumerate(r.layers), j in (i + 1):length(r.layers)
        child = r.layers[j]
        for owners in (parent.owners, parent.fixowners), (p, value) in owners
            bc_path_valid(child.shapes, p) || return nothing
            for (root, template) in child.shapes
                bc_contains(root, p) || continue
                held =
                    any(q -> bc_related(root, q), keys(child.owners)) ||
                    any(q -> bc_related(root, q), keys(child.fixowners))
                (held || p != root) &&
                    !bc_same_shape(bc_at(template, p[(length(root) + 1):end]), value) &&
                    return nothing
            end
        end
    end
    allpaths = unique(
        vcat(
            [collect(keys(l.obs)) for l in r.layers]...,
            [collect(keys(l.fixed)) for l in r.layers]...,
        ),
    )
    any(p -> first(bc_lookup(r, p)) == :collision, allpaths) && return nothing
    if !c.argument && c.depth > 0
        for l in r.layers, owners in (l.owners, l.fixowners), p in keys(owners)
            bc_contains(bp, p) && any(k -> k isa Int, p[(length(bp) + 1):end]) || continue
            any(layer -> any(root -> bc_contains(root, p), keys(layer.shapes)), r.layers) ||
                return nothing
        end
    end
    return r
end
function bc_expected(original)
    r = bc_project(original)
    r === nothing && return (:evaluation, nothing)
    c = r.case
    bp = bc_basepath(c)
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
    if !c.argument
        for layer in r.layers, (root, value) in layer.owners
            root == bp && !bc_same_shape(template, value) && return (:evaluation, nothing)
        end
    end
    ps = Tuple[first.(bc_leaves(bp, template))...]
    c.depth > 0 && pushfirst!(ps, (:m,))
    c.depth > 1 && insert!(ps, 2, (:a, :q))
    out = Dict{Tuple,Tuple{Symbol,Any}}()
    for p in ps
        role, value = bc_lookup(r, p)
        role == :collision && return (:evaluation, nothing)
        value === :twice_parent && (value = 2 * out[(:m,)][2])
        out[p] = (role, value)
    end
    if c.kind in (:multivariate, :tuplewhole, :namedwhole)
        ks = c.kind == :namedwhole ? (:a, :b) : (1, 2)
        entries = [out[(bp..., i)] for i in ks]
        all(e -> e[1] == entries[1][1], entries) || return (:evaluation, out)
        if c.kind == :multivariate && !c.argument && entries[1][1] != :latent
            any(l -> haskey(l.shapes, bp), r.layers) || return (:evaluation, out)
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
    return if c.kind == :scalar
        xs[1]
    elseif c.kind in (:named, :namedwhole)
        (a=xs[1], b=xs[2])
    elseif c.kind in (:tuple, :tuplewhole)
        Tuple(xs)
    else
        xs
    end
end

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
        :tuple => (bc_array_arg, bc_tuple_local),
        :named => (bc_named_arg, bc_named_local),
        :multivariate => (bc_mv_arg, bc_mv_local),
    )
    f, g = constructors[c.kind]
    m = c.argument ? f(bc_basevalue(c)) : g()
    if c.argument && c.depth > 0
        root = bc_rootname(c) == :x ? @varname(x) : @varname(t)
        m = condition(m, root => bc_basevalue(c))
    end
    if c.childfixed
        v = bc_structured(c, [4.0, 5.0])
        root = bc_rootname(c) == :x ? @varname(x) : @varname(t)
        m = fix(decondition(m, root), root => v)
    end
    c.depth == 2 && (m = bc_middle(m))
    c.depth > 0 && (m = bc_outer(m))
    return m
end
function bc_varname(p)
    return foldl(Base.tail(p); init=VarName{first(p)}()) do vn, k
        AbstractPPL.append_optic(
            vn, k isa Symbol ? AbstractPPL.Property{k}() : AbstractPPL.Index((k,), (;))
        )
    end
end
function bc_apply_actual(m, op, c, storage)
    @nospecialize m
    op.verb == :prefix && return prefix(m, @varname(p))
    f = getfield(DynamicPPL, op.verb)
    op.path === nothing && return f(m)
    v = bc_varname(op.path)
    return if op.verb in (:condition, :fix)
        root = bc_basepath(c)
        first(op.path) == :p && (root = (:p, root...))
        if c.depth > 0 && op.path == root
            storage[] = deepcopy(op.value)
            input = if op.value isa NamedTuple
                VarNamedTuple((v => deepcopy(op.value),))
            else
                v => deepcopy(op.value)
            end
            f(m, input)
        elseif c.depth > 0 &&
            bc_contains(root, op.path) &&
            any(k -> k isa Int, op.path) &&
            !(c.kind in (:tuple, :tuplewhole, :nested_tuple))
            schema = of(VarNamedTuple((bc_varname(root) => deepcopy(storage[]),)))
            f(m, v => deepcopy(op.value), schema)
        else
            f(m, v => deepcopy(op.value))
        end
    else
        f(m, v)
    end
end
function bc_varinfo()
    return VarInfo(
        RawValueAccumulator(false), LogPriorAccumulator(), LogLikelihoodAccumulator()
    )
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
    storage = Ref{Any}(bc_basevalue(c))
    snapshots = []
    for (i, op) in enumerate(ops)
        try
            push!(snapshots, (m, deepcopy(conditioned(m)), deepcopy(fixed(m))))
            m = bc_apply_actual(m, op, c, storage)
        catch e
            e isa InterruptException && rethrow()
            return (:call, i, e)
        end
    end
    try
        vi = bc_varinfo()
        value, vi = init!!(StableRNG(17), m, vi, BCQuarter(), UnlinkAll())
        vals = Dict(bc_leaves((), value))
        if c.kind in (:multivariate, :tuplewhole, :namedwhole)
            bp = bc_basepath(c)
            ks = c.kind == :namedwhole ? (:a, :b) : (1, 2)
            vals[bp] = bc_structured(c, [pop!(vals, (bp..., i)) for i in ks])
        end
        any(o -> o.verb == :prefix, ops) &&
            (vals = Dict((:p, p...) => v for (p, v) in vals))
        for (old, observed, held) in snapshots
            @test isequal(conditioned(old), observed)
            @test isequal(fixed(old), held)
        end
        latent = Set(keys(get_raw_values(vi)))
        return (:ok, length(ops), (vals, latent, getlogprior(vi), getloglikelihood(vi)))
    catch e
        e isa InterruptException && rethrow()
        return (:evaluation, length(ops), e)
    end
end
function bc_agrees(pred, act)
    pred[1:2] == act[1:2] || return false
    pred[1] != :ok && return act[3] isa ArgumentError || act[3] isa BoundsError
    exp, (got, latent, lp, ll) = pred[3], act[3]
    Dict(p => v for (p, (_, v)) in exp) == got || return false
    Set(bc_varname(p) for (p, (r, _)) in exp if r == :latent) == latent || return false
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
            push!(ops, BCOp(:prefix, nothing))
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
            v = bc_structured(c, [v, v + 1])
        end
        if invalid_removals &&
            verb in (:decondition, :unfix) &&
            c.kind != :scalar &&
            rand(rng) < 0.25
            p = (bp..., c.kind in (:named, :namedwhole) ? :absent : 4)
        end
        p !== nothing && prefixed && (p = (:p, p...))
        skip_short = rand(rng, Bool) && verb in (:decondition, :unfix)
        push!(ops, BCOp(verb, skip_short, p, v))
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
    return [
        (c, ops) for c in cs for i in (1:6..., 16, 32) for
        ops in (bc_sequence(rng, c, i),) if i > 2 || any(
            op ->
                op.verb in (:decondition, :unfix) && !op.skip_short && op.path === nothing,
            ops,
        )
    ]
end

function bc_invalid_removal_corpus()
    rng = StableRNG(1502)
    return [
        (c, bc_sequence(rng, c, n; invalid_removals=true)) for
        kind in (:array, :tuple, :named, :multivariate, :tuplewhole, :namedwhole) for
        c in (BCCase(kind, true, 0, false, false),) for n in 1:6
    ]
end

function bc_shape_path(kind)
    kind == :nested_named && return (:t, :a)
    return kind == :nested_tuple ? (:x, 1) : (:x,)
end
function bc_shape_corpus()
    samples = Tuple{BCCase,Vector{BCOp}}[]
    for kind in (:array, :tuple, :nested_named, :nested_tuple)
        c = BCCase(kind, true, 0, false, false)
        p = bc_shape_path(kind)
        whole = kind == :tuple ? (4.0, 5.0, 6.0) : [4.0, 5.0, 6.0]
        for bind in (:condition, :fix)
            remove = bind == :condition ? :decondition : :unfix
            ops = [
                BCOp(bind, p, whole),
                BCOp(remove, (p..., 3)),
                BCOp(bind, (p..., 3), 7.0),
                BCOp(remove, p),
            ]
            append!(samples, [(c, ops[1:n]) for n in 2:4])
        end
    end
    for kind in (:array, :tuple, :nested_named, :nested_tuple), depth in (0, 1, 2)
        c = BCCase(kind, true, depth, false, false)
        p = (bc_namespace(c)..., bc_shape_path(kind)...)
        shape(x) = kind == :tuple ? Tuple(x) : x
        for bind in (:condition, :fix)
            other = bind == :condition ? :fix : :condition
            ops = [
                BCOp(bind, p, shape([4.0, 5.0, 6.0])),
                BCOp(other, (p..., 1), 7.0),
                BCOp(other, (p..., 2), 8.0),
            ]
            push!(samples, (c, ops))
        end
        for (nobs, nfix) in ((3, 2), (2, 3)), reverse_order in (false, true)
            ops = [
                BCOp(:condition, p, shape(collect(1.0:nobs))),
                BCOp(:fix, p, shape(collect(4.0:(3 + nfix)))),
            ]
            reverse_order && reverse!(ops)
            push!(ops, BCOp(:unfix, (p..., 2)))
            push!(samples, (c, copy(ops)))
            push!(ops, BCOp(:unfix, p))
            push!(samples, (c, copy(ops)))
        end
    end
    return samples
end

function bc_contract_corpus()
    samples = Tuple{BCCase,Vector{BCOp}}[]
    for fixed in (false, true)
        c = BCCase(:scalar, true, 1, fixed, false)
        for (bind, remove) in ((:condition, :decondition), (:fix, :unfix))
            ops = [
                BCOp(bind, (:a, :x), 3.0),
                BCOp(remove, (:a, :x)),
                BCOp(remove, (:a, :x)),
                BCOp(bind, (:a, :x), 7.0),
                BCOp(remove, (:a, :x)),
            ]
            append!(samples, [(c, ops[1:n]) for n in 2:5])
        end
    end
    for fixed in (false, true), remove in (:decondition, :unfix)
        c = BCCase(:scalar, true, 2, fixed, false)
        for p in ((:a, :b, :x), nothing)
            push!(samples, (c, [BCOp(remove, p; skip=true)]))
        end
    end
    for fixed in (false, true), remove in (:decondition, :unfix)
        c = BCCase(:scalar, true, 1, fixed, false)
        ops = [BCOp(remove, (:a, :x); skip=true), BCOp(remove, nothing)]
        push!(samples, (c, ops))
    end
    for verb in (:condition, :fix)
        for argument in (false, true)
            c = BCCase(:multivariate, argument, 0, false, false)
            ops = [BCOp(verb, (:x, 1), 2.0), BCOp(verb, (:x, 2), 3.0)]
            push!(samples, (c, ops))
            push!(samples, (c, [BCOp(verb, (:x,), [2.0, 3.0])]))
        end
    end
    for verb in (:condition, :fix, :decondition, :unfix),
        (kind, p) in ((:named, (:t, :absent)), (:named, (:t, 1)), (:array, (:x, 4)))

        push!(samples, (BCCase(kind, true, 0, false, false), [BCOp(verb, p, 1.0)]))
    end
    ops = [
        BCOp(:decondition, (:a, :x); skip=true),
        BCOp(:condition, (:a, :x), [4.0, 5.0, 6.0]),
        BCOp(:decondition, (:a, :x, 3); skip=true),
    ]
    push!(samples, (BCCase(:array, true, 1, false, false), ops))
    ops = [BCOp(:unfix, (:a, :x); skip=true), BCOp(:condition, (:a, :x), ones(3))]
    push!(samples, (BCCase(:array, false, 1, true, false), ops))
    for (bind, remove) in ((:condition, :decondition), (:fix, :unfix))
        ops = [BCOp(bind, (:a, :t, :absent), 1.0), BCOp(remove, (:a, :t, :absent))]
        push!(samples, (BCCase(:named, true, 1, false, false), ops))
    end
    return samples
end

@testset "random binding histories" begin
    for (c, ops) in vcat(
        bc_corpus(), bc_shape_corpus(), bc_contract_corpus(), bc_invalid_removal_corpus()
    )
        @testset "$c, $ops" begin
            @test bc_agrees(bc_predict(c, ops), bc_actual(c, ops))
        end
    end
end

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
function removemask(m, depth, mask, role)
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
        condition(decondition(used, @varname(x)), @of(x = of(Array, 2)))
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
    mask, role = removal
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
snapshot(m) = (deepcopy(conditioned(m)), deepcopy(fixed(m)))
function actual(storage, whole, q, initial, firstop, secondop, removal; removal_scope=q)
    entered = Ref(false)
    m = base_model(storage, whole, initial, entered)
    q > 1 && (m = condition(m, conditioned(m)))
    snapshots = []
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
        listing = map(PARTS[q]) do vn
            o, f = haskey(before[1], vn), haskey(before[2], vn)
            o && f && return (-1, 0.0)
            o && return (1, before[1][vn])
            f && return (2, before[2][vn])
            return (0, 0.0)
        end
        value, vi = init!!(Xoshiro(1), m, bc_varinfo(), BCQuarter(), UnlinkAll())
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
        return observed[2][1] === :ArgumentError
    end
    e, o = expected[2], observed[2]
    return e[1] == o[1] &&
           all(isapprox(e[i], o[i]; atol=1e-12) for i in 2:4) &&
           e[5] == o[5] &&
           sort(e[6]) == o[6]
end
@testset "binding occupancy across model scopes" begin
    rng = StableRNG(1001)
    for storage in (:argument, :template, :held), whole in (false, true), q in 1:3
        cases = []
        for role in 1:2, offset in (2.0, 4.0), remover in 1:q
            push!(
                cases, ((0, 0), (1, 3, role, offset), (q, 3, role, 2.0), (3, role), remover)
            )
        end
        for _ in 1:300
            op() = (rand(rng, 1:q), rand(rng, 0:3), rand(rng, 1:2), rand(rng, (2.0, 4.0)))
            push!(
                cases,
                (
                    Tuple(rand(rng, 0:(2q), 2)),
                    op(),
                    op(),
                    (rand(rng, 0:3), rand(rng, 1:2)),
                    rand(rng, 1:q),
                ),
            )
        end
        for (initial, firstop, secondop, removal, remover) in cases
            @testset "$storage, $whole, $q, $initial, $firstop, $secondop, $removal, $remover" begin
                expected = predict(
                    q, initial, firstop, secondop, removal, whole; removal_scope=remover
                )
                observed = actual(
                    storage,
                    whole,
                    q,
                    initial,
                    firstop,
                    secondop,
                    removal;
                    removal_scope=remover,
                )
                @test agrees(expected, observed)
            end
        end
        for role in 1:2, position in 1:2, second in (3.0, 4.0)
            m = base_model(storage, whole, (0, 0), Ref(false))
            for _ in 2:q
                m = oracle_parent(m)
            end
            before = snapshot(m)
            vn = PARTS[q][position]
            @test_throws ArgumentError (role == 1 ? condition : fix)(
                m, vn => 3.0, vn => second, SCHEMAS[q]
            )
            @test isequal(snapshot(m), before)
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

@testset "whole removal releases typed templates" begin
    @model typed_storage(y) = (y .~ Normal(); y)
    for (bind, remove, listing) in BINDING_ROLES
        m = bind(
            typed_storage(missing),
            @varname(y[:]) => Float32[0.5, 0, 0],
            @of(y = of(Array, Float32, 3))
        )
        @test_throws ArgumentError remove(m, @varname(y[1]))
        cleared = unfix(decondition(m, @varname(y)), @varname(y))
        @test isempty(conditioned(cleared)) && isempty(fixed(cleared))
        for y in ([0.1, 0.2, 0.3], [0.1, 0.2])
            rebound = bind(cleared; y)
            @test listing(rebound)[@varname(y)] === y
            @test rebound(Xoshiro(1)) == y
        end
    end
end

@testset "resupplying released template storage" begin
    for (bind, remove, listing) in BINDING_ROLES
        schema = @of(x = of(Array, Float32, 2))
        used = bind(binding_local(), @varname(x[1]) => 0.5, schema)
        restored = bind(remove(used, @varname(x)), schema)
        @test isempty(listing(restored))
        @test listing(bind(restored, @varname(x[1]) => 0.5))[@varname(x[1])] === 0.5f0
        @test_throws ArgumentError bind(restored, @varname(x[1]) => 0.1)
    end
end

end # module
