module DynamicPPLBindingTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using AbstractPPL: @of, of
using Distributions
using LogDensityProblems
using LinearAlgebra: I
using AbstractPPL: AbstractPPL
using Random: Xoshiro
using StaticArrays: SVector
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

    @testset "decondition transposed argument storage" begin
        @model function transposed(x)
            x[1, 1] ~ Normal()
            return x
        end
        data = transpose(reshape([1.0, 2.0], 2, 1))
        result = decondition(transposed(data))(Xoshiro(1))
        @test result == [rand(Xoshiro(1), Normal()) 2.0]
        @test result isa Matrix
        @test data == [1.0 2.0]
        for unsupported in (
            transpose([1.0, 2.0]),
            transpose([[1.0, 2.0], [3.0, 4.0]]),
            transpose(reshape([[1.0, 2.0], [3.0, 4.0]], 2, 1)),
        )
            @test_throws "unsupported binding storage" decondition(transposed(unsupported))
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
            bind(binding_array([missing, missing]), @varname(x[1]) => 0.5),
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
            (@varname(x[1:1, 1:2]), fill(0.5, 2), [1, 3]),
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
    for (bind, _, _) in BINDING_ROLES, data in ([nothing, nothing], (missing, missing))
        binding_error("`x[1]`", "storage element type $(eltype(data)) cannot hold 5.0") do
            bind(binding_array(data), @varname(x[1]) => 5.0)
        end
    end
    binding_error("`x[1:2]`", "is inexact") do
        storage = Union{Missing,Int}[missing, missing]
        condition(binding_array(storage), @varname(x[1:2]) => [5.5, 6.0])
    end
end

@testset "scalar NamedTuple bindings outside the fast path" begin
    @model splat_scalar(x=missing; kw...) = (x ~ Normal(); x)
    @model splat_lhs(; kw...) = (kw ~ product_distribution((y=Normal(),)); kw)
    @model splat_unprefixed() = a ~ to_submodel(splat_lhs(), false)
    @model splat_prefixed() = a ~ to_submodel(splat_lhs())
    @model splat_retained() = a ~ to_submodel(decondition(splat_lhs(; y=0.0)))
    for (bind, _, _) in BINDING_ROLES
        @test bind(splat_scalar(), (x=1.0,))(Xoshiro(1)) == 1.0
        binding_error("`x`", "address is outside model prefix") do
            bind(prefix(binding_scalar(), :p), (x=1.0,))
        end
        binding_error("`x`", "inputs overlap in this call") do
            bind(binding_scalar(), (x=1.0,); x=2.0)
        end
        # Keyword splats accept only NamedTuples, including through a child.
        for (m, vn, value) in (
            (splat_lhs(), @varname(kw), Dict(:y => 2.0)),
            (splat_unprefixed(), @varname(kw), 1.0),
            (splat_prefixed(), @varname(a.kw), 1.0),
        )
            binding_error("`$vn`", "binding requires declared argument type NamedTuple") do
                logjoint(bind(m, vn => value), (;))
            end
        end
        # A field binding covering every retained field is still partial.
        binding_error("`a.kw`", "splat arguments require whole bindings") do
            logjoint(bind(splat_retained(), @varname(a.kw.y) => 2.0), (;))
        end
    end
    # A keyword splat is observed as a NamedTuple; an empty one is latent.
    @model splat_pair(x=missing; kw...) = (x ~ Normal(); kw ~ Normal(); kw)
    @test logjoint(splat_lhs(; y=2.0), (;)) ≈ logpdf(Normal(), 2.0)
    @test logjoint(condition(splat_lhs(; y=0.0); kw=(; y=2.0)), (;)) ≈ logpdf(Normal(), 2.0)
    @test logjoint(condition(splat_lhs(); kw=(; y=2.0)), (;)) ≈ logpdf(Normal(), 2.0)
    @test logjoint(fix(splat_lhs(); kw=(; y=2.0)), (;)) == 0
    @test haskey(rand(Xoshiro(1), splat_lhs()), @varname(kw))
    @test splat_lhs(; y=2.0)() === (y=2.0,)
    @test splat_lhs()(Xoshiro(1)) isa @NamedTuple{y::Float64}
    @test haskey(rand(Xoshiro(1), fix(splat_pair(); x=1.0)), @varname(kw))
    # A positional splat is an ordinary tuple, so its elements bind like a tuple's.
    @model function splat_elements(xs...)
        for i in eachindex(xs)
            xs[i] ~ Normal()
        end
        z ~ Normal()
        return xs
    end
    m = splat_elements(0.0, 0.0)
    expected = logpdf(Normal(), 1.0) + 2 * logpdf(Normal(), 0.0)
    @test logjoint(condition(m, @varname(xs[1]) => 1.0), (z=0.0,)) ≈ expected
    removed = decondition(m, @varname(xs[1]))
    @test logjoint(condition(removed, @varname(xs[1]) => 1.0), (z=0.0,)) ≈ expected
    @test haskey(rand(Xoshiro(1), condition(removed; z=0.0)), @varname(xs[1]))
end

@testset "local shape diagnostics" begin
    @model local_fields() = (x = (a=zeros(2),); x.a[1] ~ Normal(); x)
    for (bind, _, _) in BINDING_ROLES
        binding_error(
            "`x[1]`:", "bound axes (Base.OneTo(3),), local axes (Base.OneTo(2),)"
        ) do
            logjoint(bind(binding_local(); x=ones(3)), (;))
        end
        binding_error(
            "`x.a[1]`:", "bound axes (Base.OneTo(3),), local axes (Base.OneTo(2),)"
        ) do
            logjoint(bind(local_fields(), @varname(x.a) => ones(3)), (;))
        end
        binding_error("`x[1]`:", "bound a tuple, local Vector{Float64}") do
            logjoint(bind(binding_local(); x=(1.0, 2.0)), (;))
        end
        partial_tuple = @vnt begin
            @template x = (1.0, 2.0)
            x[1] := 5.0
        end
        binding_error("`x[1]`:", "bound a tuple, local Vector{Float64}") do
            logjoint(bind(binding_local(), partial_tuple), (;))
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
    @model function copied_nested(y)
        y[1][1] += 1
        for i in eachindex(y)
            y[i] ~ MvNormal(zeros(2), I)
        end
        return y[1][1]
    end
    data = Union{Missing,Vector{Float64}}[[1.0, 2.0], missing]
    original = deepcopy(data)
    model = copied_nested(data)
    vi = VarInfo(Xoshiro(1), model)
    for _ in 1:2
        value, vi = init!!(Xoshiro(1), model, vi, InitFromPrior(), UnlinkAll())
        @test value == 2.0
        @test isequal(data, original)
        value, vi = DynamicPPL.evaluate!!(model, vi)
        @test value == 2.0
        @test isequal(data, original)
    end
    ldf = LogDensityFunction(model)
    params = zeros(LogDensityProblems.dimension(ldf))
    @test LogDensityProblems.logdensity(ldf, params) ==
        LogDensityProblems.logdensity(ldf, params)
    @test isequal(data, original)
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
    model = fix(decondition(binding_array((1.0, 2.0))), @varname(x[2]) => 0.7)
    @test returned(model, VarNamedTuple((@varname(x[1]) => 0.3,))) === (0.3, 0.7)
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
    data = fill(missing, 3)
    model = condition(
        fix(binding_array(data), @varname(x[1]) => 0.5), @varname(x[2]) => 1.5
    )
    value, _ = @inferred init!!(Xoshiro(1), model, VarInfo(), InitFromPrior(), UnlinkAll())
    @test value[1:2] == [0.5, 1.5]
    exact = fix(binding_array(data), @varname(x[1]) => 2^53 + 1, @varname(x[2]) => 0.5)
    @test exact(Xoshiro(1))[1] === 2^53 + 1
    @model local_fields() = (x = (a=1.0,); x.b ~ Normal(); x)
    @test_throws "no field" local_fields()(Xoshiro(1))
    # Thread-safe evaluation adds the field too, then rejects the replaced container.
    model = fix(decondition(prepared_fields((a=1.0, b=2.0))); x=(b=4.0,))
    @test_throws "cannot replace the container" setthreadsafe(model, true)(Xoshiro(1))
end

@testset "multivariate tildes over missing argument entries" begin
    @model function mv_rows(y)
        for i in axes(y, 1)
            y[i, :] ~ MvNormal(zeros(2), I)
        end
        return y
    end
    rule = "a multivariate tilde cannot impute `missing` argument entries"
    for x in ([1.0, missing], [missing, missing], (1.0, missing), (missing, missing))
        binding_error(() -> binding_mv(x)(Xoshiro(1)), "`x`", rule)
        binding_error(() -> decondition(binding_mv(x))(Xoshiro(1)), "`x`", rule)
    end
    binding_error(() -> mv_rows([1.0 2.0; missing missing])(Xoshiro(1)), "`y[2, :]`", rule)
    # Missing entries written by the body are not imputed either.
    @model mutated_mv(x) = (x[2] = missing; x ~ MvNormal(zeros(2), I); x)
    err = @test_throws ArgumentError mutated_mv(Any[1.0, 2.0])(Xoshiro(1))
    @test occursin("write one tilde per imputed entry", sprint(showerror, err.value))
    @test !occursin("decondition", sprint(showerror, err.value))
    @test binding_mv((1.0, 2.0))(Xoshiro(1)) === (1.0, 2.0)
end

@testset "repeated named removals keep the model type" begin
    for (bind, remove, _) in BINDING_ROLES
        once = remove(bind(binding_scalar(); x=1.0), :x)
        @test typeof(remove(once, :x)) === typeof(once)
        @test typeof(remove(bind(once; x=1.0), :x)) === typeof(once)
        @test typeof(remove(bind(binding_scalar(); x=1.0), :x, @varname(x))) ===
            typeof(once)
    end
end

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

@testset "binding regressions" begin
    @testset "selection iteration order" begin
        @model function selection_argument(x)
            for i in CartesianIndices(x)
                x[i] ~ Normal()
            end
            return x
        end
        for (shape, vn, indices, values) in (
                ((3,), @varname(x[1:2:3]), (1:2:3,), [11.0, 12.0]),
                (
                    (3, 3),
                    @varname(x[1:2:3, 3:-2:1]),
                    (1:2:3, 3:-2:1),
                    [11.0 13.0; 12.0 14.0],
                ),
                ((2, 2), @varname(x[1:1, 1:2]), (1:1, 1:2), [11.0, 12.0]),
                ((2, 2), @varname(x[1:2, 1:1]), (1:2, 1:1), [11.0, 12.0]),
                ((2, 2), @varname(x[:, 2]), (:, 2), [11.0, 12.0]),
                ((2, 2), @varname(x[CartesianIndex(2, 1)]), (CartesianIndex(2, 1),), 11.0),
            ),
            bind in (condition, fix)

            @testset "$bind $vn" begin
                result = bind(decondition(selection_argument(zeros(shape))), vn => values)(
                    Xoshiro(1)
                )
                @test vec([result[indices...]...]) == vec([values...])
            end
        end
    end
    @testset "initially empty retained roots" begin
        @model empty_root() = (x = zeros(2); x[1] ~ Normal(); x[2] ~ Normal(); x)
        values = DynamicPPL.templated_setindex!!(
            DynamicPPL.VarNamedTuple(), 1.0, @varname(x[1]), zeros(2)
        )
        model = condition(empty_root(), DynamicPPL.empty!!(values))
        for remove in (decondition, unfix)
            @test_throws "retained storage" condition(
                remove(model, @varname(x[1])), @of(x = of(Array, 3))
            )
        end
    end
    @testset "submodel argument validation" begin
        @model validation_parent(child) = (a ~ to_submodel(child); a)
        @model validation_leaf(x, z) = (x[1][1] ~ Normal(); (x, z))
        x, z = [zeros(1)], ones(1)
        child = decondition(validation_leaf(x, z))
        x[1] = z
        @test_throws "aliases" validation_parent(child)(Xoshiro(1))
        x = [zeros(1)]
        original = decondition(validation_leaf(x, ones(1)))
        child = DynamicPPL.Model{false}(
            original.f,
            (; x, z=x[1]),
            original.defaults,
            original.context,
            original.values;
            args_on_lhs=DynamicPPL._model_metadata(original),
        )
        @test_throws "aliases" validation_parent(child)(Xoshiro(1))
        @model validation_view(x, z) = (x[1] ~ Normal(); (x, z))
        @model validation_kwview(x; z) = (x[1] ~ Normal(); (x, z))
        for make_child in (
            (x, z) -> validation_view(x, z),
            (x, z) -> validation_leaf([x], z),
            (x, z) -> validation_kwview(x; z=z),
        )
            x = zeros(1)
            @test_throws ArgumentError("`z`: latent storage aliases another argument path") validation_parent(
                decondition(make_child(x, view(x, :)))
            )(
                Xoshiro(1)
            )
            child = decondition(make_child(x, view(ones(1), :)))
            @test validation_parent(child)(Xoshiro(1))[2] == ones(1)
        end
        @model validation_scalar(y) = (y ~ Normal(); y)
        original = fix(validation_scalar(missing); y=3.0)
        child = DynamicPPL.Model{false}(
            original.f,
            original.args,
            original.defaults,
            original.context,
            original.values;
            args_on_lhs=(),
        )
        @test_throws "covariates" validation_parent(child)(Xoshiro(1))
    end
    @testset "linear alias validation" begin
        @model alias_covariates(X, x) = (x[1] ~ Normal(X[1][1]); x)
        M = zeros(2, 1000)
        for X in ([M[:, i] for i in axes(M, 2)], [view(M, :, i) for i in axes(M, 2)])
            m = alias_covariates(X, fill(missing, 1000))
            condition(m, @varname(x[1]) => 0.0)
            @test (@allocated condition(m, @varname(x[1]) => 0.0)) < 2_000_000
        end
    end
    @testset "field creation requires a fixed ancestor" begin
        @model fields_argument(x) = (x.a ~ Normal(); x.b ~ Normal(); x)
        @model nested_fields_argument(x) = (x.c.a ~ Normal(); x.c.b ~ Normal(); x)
        @model field_parent(child) = (a ~ to_submodel(child); a)
        for model in (
            decondition(fields_argument((b=1.0,))),
            decondition(nested_fields_argument((c=(b=1.0,),))),
        )
            @test_throws "no field" model(Xoshiro(1))
            @test_throws "no field" field_parent(model)(Xoshiro(1))
        end
        @test_throws "no field" fix(
            decondition(fields_argument((b=1.0,))), @varname(x.b) => 4.0
        )(
            Xoshiro(1)
        )
        model = decondition(nested_fields_argument((c=(a=1.0, b=2.0),)))
        result = fix(model; x=(c=(b=4.0,),))(Xoshiro(1))
        @test hasproperty(result.c, :a) && result.c.b == 4.0
    end
    @testset "retained nested child storage" begin
        @model storage_child(y) = (y[1] ~ Normal(); y[2] ~ Normal(); y)
        @model storage_parent(child) = (a ~ to_submodel(child); a)
        @model storage_body(y) = (a ~ to_submodel(storage_child(y)); a)
        @model function storage_indexed(y, child)
            q = Vector{Any}(undef, 1)
            q[1] ~ to_submodel(child)
            y[1] ~ Normal()
            y[2] ~ Normal()
            return y
        end
        for (model, vn) in (
                (
                    storage_parent(prefix(storage_child(fill(missing, 2)), @varname(p))),
                    @varname(a.p.y[2])
                ),
                (
                    storage_parent(storage_parent(storage_child(fill(missing, 2)))),
                    @varname(a.a.y[2])
                ),
                (
                    storage_parent(
                        storage_indexed(fill(missing, 2), storage_child(fill(missing, 2)))
                    ),
                    @varname(a.y[2])
                ),
            ),
            bind in (condition, fix)

            @test bind(model, vn => 5.0)(Xoshiro(1))[2] == 5.0
        end
        @model function storage_mixed(y)
            z = zeros(2)
            for i in eachindex(z)
                z[i] ~ Normal()
                y[i] ~ Normal()
            end
            return (; y, z)
        end
        child = condition(
            storage_parent(storage_mixed(fill(missing, 2))), @of(a = @of(z = of(Array, 2)))
        )
        result = fix(
            storage_parent(child), @varname(a.a.z[2]) => 5.0, @varname(a.a.y[2]) => 6.0
        )(
            Xoshiro(1)
        )
        @test result.z[2] == 5.0 && result.y[2] == 6.0
        @test_throws "template" fix(storage_body(fill(missing, 2)), @varname(a.y[2]) => 5.0)
        @test fix(storage_parent(storage_child(zeros(2))), @varname(a.y) => [5.0, 6.0])(
            Xoshiro(1)
        ) == [5.0, 6.0]
        @test fix(storage_body(zeros(2)), @varname(a.y) => [5.0, 6.0])(Xoshiro(1)) ==
            [5.0, 6.0]
        @test_throws "submodels cannot observe through arguments" fix(
            storage_body(zeros(2)), @varname(a.y[2]) => 5.0, @of(a = @of(y = of(Array, 2)))
        )(
            Xoshiro(1)
        )
        model = storage_parent(storage_child(zeros(2)))
        observed = condition(model, @varname(a.y) => [5.0, 6.0])
        @test observed(Xoshiro(1)) == [5.0, 6.0]
        @test loglikelihood(observed, (;)) == sum(logpdf.(Normal(), [5.0, 6.0]))
        latent = decondition(model, @varname(a.y))
        @test latent(Xoshiro(1)) ==
            storage_parent(decondition(storage_child(zeros(2))))(Xoshiro(1))
        @test loglikelihood(latent, (; a=(; y=[5.0, 6.0]))) == 0.0
    end
    @testset "whole removal after field removal releases shape" begin
        @model resize_argument(t) = (t.a .~ Normal(); t.b ~ Normal(); t)
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            model = decondition(resize_argument((a=zeros(2), b=0.0)))
            model = bind(model, @varname(t.a) => [1.0, 2.0])
            model = remove(remove(model, @varname(t.a)), @varname(t))
            value = (a=[1.0, 2.0, 3.0], b=4.0)
            @test bind(model; t=value)(Xoshiro(1)) == value
        end
    end
end

@testset "binding shape and storage edge cases" begin
    @model function short_local()
        x = (0.0, 0.0, 0.0)
        x[3] ~ Normal()
        return x
    end
    @model nested_local() = (x = (zeros(3), 0.0); x[1][3] ~ Normal(); x)
    @model function tuple_of_vector()
        x = (zeros(2), 0.0)
        x[1][1] ~ Normal()
        x[1][2] ~ Normal()
        x[2] ~ Normal()
        return x
    end
    @model fields_local() = (x = (a=0.0, b=0.0); x.a ~ Normal(); x.b ~ Normal(); x)
    @model function mixed_fields()
        x = (a=zeros(2), b=0.0)
        x.a[1] ~ Normal()
        x.b ~ Normal()
        return x
    end
    for (bind, remove, listing) in BINDING_ROLES
        # A whole value shorter than the local storage must not leave `x[3]` latent.
        binding_error(
            "`x[3]`:", "bound axes (Base.OneTo(2),), local axes (Base.OneTo(3),)"
        ) do
            bind(short_local(); x=(1.0, 2.0))(Xoshiro(1))
        end
        partial = remove(bind(nested_local(); x=([1.0, 2.0], 5.0)), @varname(x[1][1]))
        binding_error(
            "`x[1][3]`:", "bound axes (Base.OneTo(2),), local axes (Base.OneTo(3),)"
        ) do
            partial(Xoshiro(1))
        end
        binding_error("`x.a`:", "address kind disagrees with storage") do
            remove(bind(fields_local(); x=(1.0, 2.0)), @varname(x[1]))(Xoshiro(1))
        end
        model = bind(mixed_fields(), @varname(x.b) => 1.0)
        @test listing(remove(model, @varname(x.a[1]))) == listing(model)
        whole = bind(tuple_of_vector(); x=([1.0, 2.0], 3.0))
        for vn in (@varname(x[1]), @varname(x[1][1]))
            replayed = bind(tuple_of_vector(), listing(remove(whole, vn)))
            @test bind(replayed, @varname(x[1]) => [7.0, 8.0])(Xoshiro(1)) ==
                ([7.0, 8.0], 3.0)
        end
        @test bind(bind(binding_local(); x=[1.0, 2.0]); x=3.0:4.0)(Xoshiro(1)) == [3.0, 4.0]
    end
    @test condition(binding_array([1.0, 2.0]); x=3.0:4.0)(Xoshiro(1)) == 3.0:4.0

    @model scalar_site() = (x ~ Normal(); x)
    @model function indexed_sites()
        z = zeros(2)
        for i in 1:2
            z[i] ~ to_submodel(scalar_site())
        end
        return z
    end
    observed = condition(indexed_sites(); z=[(x=1.0,), (x=2.0,)])
    @test condition(observed, @varname(z[1]) => (x=7.0,))(Xoshiro(1)) == [7.0, 2.0]
    @model function site_beside_field()
        z = (a=zeros(1), b=0.0)
        z.a[1] ~ to_submodel(scalar_site())
        z.b ~ Normal()
        return z
    end
    for (bind, _, _) in BINDING_ROLES
        @test bind(site_beside_field(), @varname(z.b) => 1.0)(Xoshiro(1)).b == 1.0
    end

    @model function vectors_local()
        x = [zeros(2), zeros(1)]
        x[1][1] ~ Normal()
        x[2][1] ~ Normal()
        return x
    end
    @model vectors_parent() = (a ~ to_submodel(vectors_local()); a)
    whole = condition(vectors_parent(), @varname(a.x) => [[1.0, 2.0], [3.0]])
    binding_error("`a.x[1][1]`:", "indexed binding lacks storage") do
        fix(decondition(whole, @varname(a.x[1])), @varname(a.x[1][1]) => 9.0)
    end

    @model record_child(x) = (x.b[1] ~ Normal(); x.b[2] ~ Normal(); x)
    @model record_parent(c) = (a ~ to_submodel(c); a)
    model = record_parent(decondition(record_child((b=[1.0, 2.0],))))
    partial = condition(model, @varname(a.x.b[2]) => 5.0)
    @test condition(model, conditioned(partial))(Xoshiro(1)).b[2] == 5.0
end

end # module
