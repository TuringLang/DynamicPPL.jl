using AbstractPPL: of, @of
using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DynamicPPL
using DynamicPPL.TestUtils: ALL_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using Distributions: Normal, logpdf
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro
using ReverseDiff: ReverseDiff
using Test: @test, @testset

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
        (z -> child([[z]], y -> y[1][1]), @varname(y)),
        (z -> child(([z],), y -> y[1][1]), @varname(y)),
        (z -> child((a=[z],), y -> y.a[1]), @varname(y)),
        (z -> child([z, zero(z)], first), @varname(y[1])),
        (z -> child([[z], [zero(z)]], y -> y[1][1]), @varname(y[1])),
        (z -> child(([z], [zero(z)]), y -> y[1][1]), @varname(y[1])),
        (z -> record_child((a=[z], b=[zero(z)])), @varname(y.a)),
        (z -> child(ArgumentRecord([z], [zero(z)]), y -> y.a[1]), @varname(y)),
        (z -> child(ArgumentRecord([z], Float64), y -> y.a[1]), @varname(y)),
        (z -> child(MutableArgumentRecord([z], [zero(z)]), y -> y.a[1]), @varname(y)),
        (z -> record_child(ArgumentRecord([z], [zero(z)])), @varname(y.a)),
        (z -> record_child(MutableArgumentRecord([z], [zero(z)])), @varname(y.a)),
    )
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
    @model function runtime_parent(bind, partial)
        m ~ Normal()
        child = runtime_child(fill(zero(m), 2))
        bound = partial ? bind(child, @varname(y[1]) => 2m) : bind(child; y=[2m, zero(m)])
        a ~ to_submodel(bound)
        return m
    end
    for bind in (condition, fix), partial in (false, true)
        ldf = LogDensityFunction(runtime_parent(bind, partial); adtype=AutoReverseDiff())
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
        make_array in (m -> fill(zero(m), 2), m -> fill(zero(m), 2, 1))

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
