using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DynamicPPL
using DynamicPPL.TestUtils: ALL_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using Distributions: MvNormal, Normal, logpdf
using ForwardDiff: ForwardDiff  # run_ad uses FD for correctness test
using LogDensityProblems: LogDensityProblems
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

@testset "broadcasted tracked arguments remain writable" begin
    @model function broadcast_argument(y)
        y = y .+ 1
        return y[1] ~ Normal()
    end
    f = x -> logjoint(decondition(broadcast_argument(x)), (; y=x))
    @test ReverseDiff.gradient(f, [1.0]) ≈ [-1.0]
end

@testset "repeated tracked numeric leaves" begin
    @model numeric_child(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
    for partial in (false, true)
        function density(θ)
            scalar = θ[1]
            model = numeric_child([scalar, scalar])
            model = partial ? decondition(model, @varname(x[1])) : decondition(model)
            return logjoint(model, (x=[0.7, θ[1]],))
        end
        @test density([0.3]) ≈ logpdf(Normal(), 0.7) + logpdf(Normal(), 0.3)
        @test ReverseDiff.gradient(density, [0.3]) ≈ [-0.3]
    end
end

@testset "copied tracked views retain parent buffers" begin
    @model function view_child(p)
        y ~ Normal(p.a[1])
        return p.b ~ Normal()
    end
    @model function view_parent(make_view)
        m ~ MvNormal(zeros(1), ones(1, 1))
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

@testset "copied tracked arrays keep a shared tape" begin
    @model function child(x)
        y ~ Normal(x.b[1])
        x.a[1] ~ Normal()
        return 0.0 ~ Normal(x.b[1])
    end
    @model function parent()
        m ~ MvNormal(zeros(1), ones(1, 1))
        return a ~ to_submodel(decondition(child((a=m, b=m .+ 0))))
    end
    ldf = LogDensityFunction(parent(); adtype=AutoReverseDiff())
    _, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3, 0.7, 0.9])
    @test gradient ≈ [-0.2, -0.4, -0.9]
end

@testset "retained buffers reached by latent arguments" begin
    @model function retained_buffer(x)
        x.b[1][1] ~ Normal()
        return x
    end
    z = ReverseDiff.track([0.3])
    b = Any[[0.0]]
    model = decondition(retained_buffer((a=z, b=b)))
    b[1] = ReverseDiff.value(z)
    @test_throws r"argument `x`.*retained storage.*latent path" returned(
        model, (x=(b=[[0.7]],),)
    )
    @test ReverseDiff.value(z) == [0.3]
end

mutable struct ConstTrackedOwner
    const a::Any
end
@testset "unsupported tracked argument owner" begin
    @model unsupported(x) = x.a[1] ~ Normal()
    f = z -> logjoint(decondition(unsupported(ConstTrackedOwner(z))), (x=(a=[0.7],),))
    @test_throws ArgumentError(
        "Cannot replace const property `a` of argument type ConstTrackedOwner; bind the whole value instead.",
    ) ReverseDiff.gradient(f, [0.3])
end
