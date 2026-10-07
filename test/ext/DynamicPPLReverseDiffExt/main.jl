using AbstractPPL: of, @of
using ADTypes: AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface
using DynamicPPL
using DynamicPPL.TestUtils: ALL_MODELS
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult
using Distributions: MvNormal, Normal, logpdf
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro
using ReverseDiff: ReverseDiff
using Test: @test, @testset, @test_throws

ADTYPES = (
    ("ReverseDiff", AutoReverseDiff(; compile=false)),
    ("ReverseDiffCompiled", AutoReverseDiff(; compile=true)),
)

@testset "$ad_key" for (ad_key, ad_type) in ADTYPES
    @testset "$(model.f)" for model in ALL_MODELS
        @test run_ad(model, ad_type) isa Any
    end
end

@testset "ReverseDiff compiled prep reduces repeated-call allocations" begin
    @model f() = x ~ Normal()
    ldf_compiled = LogDensityFunction(
        f(), getlogjoint_internal, LinkAll(); adtype=AutoReverseDiff(; compile=true)
    )
    ldf_uncompiled = LogDensityFunction(
        f(), getlogjoint_internal, LinkAll(); adtype=AutoReverseDiff(; compile=false)
    )
    params = rand(ldf_compiled)

    LogDensityProblems.logdensity_and_gradient(ldf_compiled, params)
    LogDensityProblems.logdensity_and_gradient(ldf_uncompiled, params)

    function repeated_call_allocs(ldf, params)
        GC.gc()
        before = Base.gc_num()
        for _ in 1:100
            LogDensityProblems.logdensity_and_gradient(ldf, params)
        end
        after = Base.gc_num()
        return Base.GC_Diff(after, before).allocd
    end

    allocs_compiled = repeated_call_allocs(ldf_compiled, params)
    allocs_uncompiled = repeated_call_allocs(ldf_uncompiled, params)

    @test allocs_compiled < allocs_uncompiled
end

struct ArgumentRecord{A,B}
    a::A
    b::B
end
mutable struct MutableArgumentRecord{A,B}
    a::A
    b::B
end

@testset "deconditioned argument gradients" begin
    @model function child(y, read)
        μ = read(y)
        x ~ Normal(μ, 1)
        y = [missing]
        return y[1] ~ Normal()
    end
    @model function record_child(y)
        μ = y.a[1]
        x ~ Normal(μ, 1)
        y = (a=[missing], b=y.b)
        return y.a[1] ~ Normal()
    end
    @model function parent(make_child, vn)
        z ~ Normal()
        return a ~ to_submodel(decondition(make_child(z), vn))
    end

    cases = (
        (z -> child([z], first), @varname(y)),
        (z -> child(Real[z], first), @varname(y)),
        (z -> child(Any[z], first), @varname(y)),
        (z -> child(Dict(:a => z), y -> y[:a]), @varname(y)),
        (z -> child(Dict(:a => [z]), y -> y[:a][1]), @varname(y)),
        (z -> child([[z]], y -> y[1][1]), @varname(y)),
        (z -> child(([z],), y -> y[1][1]), @varname(y)),
        (z -> child((a=[z],), y -> y.a[1]), @varname(y)),
        (z -> child([z, zero(z)], first), @varname(y[1])),
        (z -> child([[z], [zero(z)]], y -> y[1][1]), @varname(y[1])),
        (z -> record_child((a=[z], b=[zero(z)])), @varname(y.a)),
        (z -> child(ArgumentRecord([z], [zero(z)]), y -> y.a[1]), @varname(y)),
        (z -> child(ArgumentRecord([z], Float64), y -> y.a[1]), @varname(y)),
        (z -> child(MutableArgumentRecord([z], [zero(z)]), y -> y.a[1]), @varname(y)),
    )
    for value in
        (([0.4], [0.0]), ArgumentRecord([0.4], [0.0]), MutableArgumentRecord([0.4], [0.0]))
        model = value isa Tuple ? child(value, y -> y[1][1]) : record_child(value)
        address = value isa Tuple ? @varname(y[1]) : @varname(y.a)
        @test_throws ArgumentError decondition(model, address)
        @test isempty(conditioned(decondition(model, @varname(y))))
    end
    params = [0.4, 0.7, 0.9]
    for (make_child, vn) in cases, threadsafe in (false, true)
        model = setthreadsafe(parent(make_child, vn), threadsafe)
        _, vi = DynamicPPL.init!!(
            Xoshiro(123456),
            model,
            VarInfo(VectorValueAccumulator()),
            InitFromPrior(),
            UnlinkAll(),
        )
        ldf = LogDensityFunction(
            model, getlogjoint, vi; adtype=AutoReverseDiff(; compile=false)
        )
        value, gradient = LogDensityProblems.logdensity_and_gradient(ldf, params)
        f = x -> LogDensityProblems.logdensity(ldf, x)
        h = 1e-5
        numerical = map(eachindex(params)) do i
            delta = [j == i ? h : 0.0 for j in eachindex(params)]
            (f(params + delta) - f(params - delta)) / (2h)
        end
        @test value ≈ sum(logpdf.(Normal(), [0.4, 0.3, 0.9]))
        @test gradient ≈ [-0.1, -0.3, -0.9]
        @test gradient ≈ ForwardDiff.gradient(f, params)
        @test gradient ≈ numerical
    end
end

@testset "runtime bindings preserve AD values" begin
    @model runtime_child(y) = (y[1] ~ Normal(); y[2] ~ Normal())
    @model function runtime_parent(bind, partial, make_array)
        m ~ Normal()
        child = runtime_child(make_array(m))
        bound = partial ? bind(child, @varname(y[1]) => 2m) : bind(child; y=[2m, zero(m)])
        a ~ to_submodel(bound)
        return m
    end
    for bind in (condition, fix), partial in (false, true)
        ldf = LogDensityFunction(
            runtime_parent(bind, partial, m -> fill(zero(m), 2)); adtype=AutoReverseDiff()
        )
        if partial
            @test_throws r"TrackedArray.*whole" LogDensityProblems.logdensity_and_gradient(
                ldf, [0.3]
            )
        else
            _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3])
            @test gradient ≈ [bind === condition ? -1.5 : -0.3]
        end
        ldf = LogDensityFunction(
            runtime_parent(bind, partial, m -> [zero(m), zero(m)]); adtype=AutoReverseDiff()
        )
        _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3])
        @test gradient ≈ [bind === condition ? -1.5 : -0.3]
    end
end

@testset "runtime bindings retain writable latent argument entries" begin
    @model function child(y)
        y = y .+ 1
        y[1] ~ Normal()
        y[2] ~ Normal()
        return y
    end
    @model function parent(bind, make_array)
        m ~ Normal()
        return a ~ to_submodel(bind(decondition(child(make_array(m))), @varname(y[1]) => m))
    end
    for bind in (condition, fix),
        make_array in (m -> [zero(m), zero(m)], m -> reshape([zero(m), zero(m)], 2, 1))

        model = parent(bind, make_array)
        _, vi = DynamicPPL.init!!(
            Xoshiro(123456),
            model,
            VarInfo(VectorValueAccumulator()),
            InitFromPrior(),
            UnlinkAll(),
        )
        ldf = LogDensityFunction(model, getlogjoint, vi; adtype=AutoReverseDiff())
        params = [2.0, 6.0]
        value, gradient = LogDensityProblems.logdensity_and_gradient(ldf, params)
        expected = logpdf(Normal(), 2.0) + logpdf(Normal(), 6.0)
        bind === condition && (expected += logpdf(Normal(), 3.0))
        @test value ≈ expected
        @test gradient ≈ [bind === condition ? -5.0 : -2.0, -6.0]
        @test gradient ≈
            ForwardDiff.gradient(x -> LogDensityProblems.logdensity(ldf, x), params)
    end
end

@testset "runtime schemas preserve tracked values" begin
    @model function schema_child(T, n)
        z = zeros(T, n)
        for i in eachindex(z)
            z[i] ~ Normal()
        end
        return z
    end
    @model function schema_parent(bind, n)
        m ~ Normal()
        return a ~ to_submodel(
            bind(
                schema_child(typeof(m), n),
                @varname(z[2]) => m,
                @of(z = of(Array, typeof(m), n))
            ),
        )
    end
    for bind in (condition, fix)
        gradient = ReverseDiff.gradient([0.3]) do x
            m = only(x)
            logjoint(schema_parent(bind, 3), (m=m, a=(z=[zero(m), m, zero(m)],)))
        end
        @test gradient ≈ [bind === condition ? -0.6 : -0.3]
    end
end

mutable struct ConstantCopyState{T}
    const offset::T
    x::Real
    ConstantCopyState(offset::T) where {T} = new{T}(offset, zero(offset))
end
@testset "copy preserves AD identity" begin
    adtype = AutoReverseDiff(; compile=false)
    @model function alias_copy(x)
        x.a[1] ~ Normal()
        x.c ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    data = Real[0.0]
    alias_model = condition(
        decondition(alias_copy((a=data, b=data, c=0.0))), @varname(x.c) => 0.0
    )
    ldf = LogDensityFunction(alias_model; adtype)
    for x in ([2.0], [-0.4])
        density, gradient = LogDensityProblems.logdensity_and_gradient(ldf, x)
        @test density ≈ 3logpdf(Normal(), 0.0) - x[1]^2
        @test gradient ≈ -2x
    end

    @model function nested_copy_child(x)
        y ~ Normal(x.b[1])
        x.a[1] ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    @model function nested_copy_parent()
        m ~ MvNormal(zeros(1), ones(1))
        return a ~ to_submodel(decondition(nested_copy_child((a=m, b=m))))
    end
    ldf = LogDensityFunction(nested_copy_parent(); adtype)
    _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3, 0.7, 0.9])
    @test gradient ≈ [0.1, -0.4, -1.8]

    @model function const_copy(s)
        s.x ~ Normal()
        return 0.0 ~ Normal(s.x + s.offset)
    end
    @model function runtime_const_copy()
        m ~ Normal()
        return a ~ to_submodel(decondition(const_copy(ConstantCopyState(m))))
    end
    ldf = LogDensityFunction(runtime_const_copy(); adtype)
    for x in ([0.3, 0.7], [-0.4, 0.2])
        _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, x)
        @test gradient ≈ [-2x[1] - x[2], -x[1] - 2x[2]]
    end
end

@testset "copied tracked views retain parent buffers" begin
    @model function view_child(p)
        y ~ Normal(p.a[1])
        return p.b ~ Normal()
    end
    @model function view_parent(make_view)
        m ~ MvNormal(zeros(1), ones(1))
        return a ~ to_submodel(decondition(view_child((a=make_view(m), b=0.0))))
    end
    for make_view in (m -> view(m, 1:1), m -> view(reshape(m, 1, 1), :, 1)),
        (_, adtype) in ADTYPES

        ldf = LogDensityFunction(view_parent(make_view); adtype)
        for x in ([0.3, 0.7, 0.9], [-0.4, 0.2, -0.3], [0.3, 0.7, 0.9])
            density, gradient = LogDensityProblems.logdensity_and_gradient(ldf, x)
            @test density ≈ sum(logpdf.(Normal(), [x[1], x[2] - x[1], x[3]]))
            @test gradient ≈ [x[2] - 2x[1], x[1] - x[2], -x[3]]
            @test gradient ≈
                ForwardDiff.gradient(z -> LogDensityProblems.logdensity(ldf, z), x)
        end
    end
end

@model function wrapped_child(y, μ)
    for i in eachindex(y)
        y[i] ~ Normal(μ)
    end
    return sum(y)
end
@model function wrapped_parent(nfixed, recursive)
    μ ~ Normal()
    m = condition(wrapped_child([zero(μ), zero(μ)], μ); y=[1 + μ, 1 + 2μ, 1 + 3μ])
    if recursive
        m = wrapped_namespace(m)
        for i in 1:nfixed
            m = fix(m, (@varname(a.y[i])) => (i + 1) * μ)
        end
    else
        for i in 1:nfixed
            m = fix(m, (@varname(y[i])) => (i + 1) * μ)
        end
    end
    s ~ to_submodel(m)
    return 0.7 ~ Normal(s)
end
@model wrapped_namespace(m) = a ~ to_submodel(m)
@testset "runtime wrapped storage keeps observations" begin
    for nfixed in (1, 2), recursive in (false, true), compile in (false, true)
        model = wrapped_parent(nfixed, recursive)
        ldf = LogDensityFunction(model; adtype=AutoReverseDiff(; compile))
        function oracle(μ)
            return logpdf(Normal(), μ) +
                   sum(logpdf(Normal(μ), 1 + i * μ) for i in (nfixed + 1):3) +
                   logpdf(
                       Normal(sum((i <= nfixed ? (i + 1) * μ : 1 + i * μ) for i in 1:3)),
                       0.7,
                   )
        end
        for μ in (0.3, 0.4)
            val, grad = LogDensityProblems.logdensity_and_gradient(ldf, [μ])
            @test val ≈ oracle(μ)
            @test only(grad) ≈ (oracle(μ + 1e-5) - oracle(μ - 1e-5)) / 2e-5
        end
    end
end

@testset "tracked arrays require whole binding operations" begin
    @model tracked_child(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    ReverseDiff.gradient([0.3, 0.4]) do x
        for (bind, remove, listing) in
            ((condition, decondition, conditioned), (fix, unfix, fixed))
            whole = bind(tracked_child(x); x=x)
            @test listing(whole)[@varname(x)] === x
            @test !haskey(listing(remove(whole, @varname(x))), @varname(x))
            @test_throws r"x\[1\].*TrackedArray.*whole" bind(
                tracked_child(x), @varname(x[1]) => x[1]
            )
            @test_throws r"x\[1\].*TrackedArray.*whole" remove(whole, @varname(x[1]))
            @test_throws r"x\[1\].*TrackedArray.*whole" decondition(
                tracked_child(x), @varname(x[1])
            )
            scalars = map(identity, x)
            valid = bind(tracked_child(scalars), @varname(x[1]) => x[1])
            @test listing(valid)[@varname(x[1])] == x[1]
            @test !haskey(listing(remove(valid, @varname(x[1]))), @varname(x[1]))
        end
        sum(x)
    end

    @model function view_argument(x)
        μ = x[1]
        x[1] ~ Normal()
        return 0.0 ~ Normal(μ + x[1])
    end
    density =
        x -> logjoint(decondition(view_argument(view(x, :)), @varname(x)), (; x=[0.7]))
    for x in ([0.3], [-0.4])
        @test density(x) ≈ logpdf(Normal(), 0.7) + logpdf(Normal(), x[1] + 0.7)
        @test ReverseDiff.gradient(density, x) ≈ -x .- 0.7
        @test ReverseDiff.gradient(density, x) ≈ ForwardDiff.gradient(density, x)
        h = 1e-5
        @test only(ReverseDiff.gradient(density, x)) ≈
            (density(x .+ h) - density(x .- h)) / (2h)
    end
end

@model function replaced_argument(x=missing)
    x === missing && (x = zeros(2))
    x[1] ~ Normal()
    return x[2] ~ Normal(x[1])
end
@model function replaced_keyword(; x=missing)
    x === missing && (x = zeros(2))
    x[1] ~ Normal()
    return x[2] ~ Normal(x[1])
end
@model function partly_missing(x)
    x[1] ~ Normal()
    x[2] ~ Normal(x[1])
    x[3] ~ Normal(x[2])
    return x
end
@model function partly_missing_parent(make_data)
    m ~ Normal()
    a ~ to_submodel(partly_missing(make_data(m)))
    return 0.5 ~ Normal(a[3])
end
@testset "placeholder argument gradients" begin
    # The placeholder pattern depends only on the arguments, so compiled tapes are valid.
    a, b, c = 0.3, -0.4, 0.8
    chain = (logpdf(Normal(), a) + logpdf(Normal(a), b), [b - 2a, a - b])
    # The parent observes `a.x[2] = d` with `dd = ∂d/∂m`; parameters are `m`, `a.x[1]`, `a.x[3]`.
    function parent_result(d, dd)
        value = sum(logpdf.(Normal(), [a, b])) + logpdf(Normal(b), d) + logpdf(Normal(d), c)
        value += logpdf(Normal(c), 0.5)
        return value, [-a + dd * (b - d + c - d), d - 2b, d - 2c + 0.5]
    end
    data = Union{Missing,Float64}[missing, 1.5, missing]
    cases = (
        (replaced_argument(), [a, b], chain...),
        (replaced_keyword(), [a, b], chain...),
        (
            partly_missing(data),
            [a, b],
            logpdf(Normal(), a) + logpdf(Normal(a), 1.5) + logpdf(Normal(1.5), b),
            [1.5 - 2a, 1.5 - b],
        ),
        (partly_missing_parent(_ -> data), [a, b, c], parent_result(1.5, 0)...),
        (
            partly_missing_parent(m -> Union{Missing,typeof(m)}[missing, 2m, missing]),
            [a, b, c],
            parent_result(2a, 2)...,
        ),
    )
    for (model, params, value, gradient) in cases,
        adtype in (AutoForwardDiff(), last.(ADTYPES)...)

        test = WithExpectedResult(value, gradient)
        @test run_ad(model, adtype; params, test, verbose=false) isa Any
    end
end

@testset "zero-dimensional LHS gradients" begin
    @model scalar_array() = (x = fill(0.0); x[] ~ Normal(); x)
    @model argument_array(x) = x[] ~ Normal()
    for model in (scalar_array(), decondition(argument_array(fill(0.0)), @varname(x[])))
        @test run_ad(
            model,
            AutoReverseDiff();
            params=[0.5],
            test=WithExpectedResult(logpdf(Normal(), 0.5), [-0.5]),
        ) isa Any
    end
end
