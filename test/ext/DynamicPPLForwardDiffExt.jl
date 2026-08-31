module DynamicPPLForwardDiffExtTests

using DynamicPPL
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using Distributions: MvNormal, Normal
using LinearAlgebra: I
using Test: @test, @test_logs, @testset

mutable struct ReferenceValue{T}
    value::T
end
function Base.convert(::Type{ReferenceValue{T}}, value::T) where {T}
    return ReferenceValue(value)
end

@testset "ForwardDiff tweak_adtype" begin
    MODEL_SIZE = 10
    @model f() = x ~ MvNormal(zeros(MODEL_SIZE), I)
    model = f()
    x = randn(MODEL_SIZE)

    @testset "Chunk size setting" for chunksize in (nothing, 0)
        base_adtype = AutoForwardDiff(; chunksize=chunksize)
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype isa AutoForwardDiff{MODEL_SIZE}
    end

    @testset "Tag setting" begin
        base_adtype = AutoForwardDiff()
        new_adtype = DynamicPPL.tweak_adtype(base_adtype, model, x)
        @test new_adtype.tag isa ForwardDiff.Tag{DynamicPPL.DynamicPPLTag}
    end
end

@testset "input dependency warning" begin
    @model function derived_observation(y, sigma)
        m ~ Normal(0, sigma)
        v = exp(y) + sigma
        return v ~ Normal(m, sigma)
    end
    @test_logs (:warn, r"v.*derived from a model input.*classified as latent") check_model(
        derived_observation(1.0, 1.0)
    )

    @model function ordinary_latent(y)
        x ~ Normal(y)
        return x
    end
    @test_logs check_model(ordinary_latent(1.0))

    @model function preallocated_latent()
        x = zeros(2)
        return x ~ MvNormal(zeros(2), I)
    end
    @test_logs check_model(preallocated_latent())

    @model function derived_index(y)
        x = exp.([y])
        return x[1] ~ Normal()
    end
    @test_logs (:warn, r"Variable x\[1\].*derived from a model input") check_model(
        derived_index(1.0)
    )

    @model function derived_property(y)
        state = (; x=exp(y))
        return state.x ~ Normal()
    end
    @test_logs (:warn, r"Variable state\.x.*derived from a model input") check_model(
        derived_property(1.0)
    )

    @model function derived_dotted(y)
        x = exp.([y, y])
        return x .~ Normal()
    end
    @test_logs (:warn, r"Variable x\[1\].*derived from a model input") (
        :warn, r"Variable x\[2\].*derived from a model input"
    ) check_model(derived_dotted(1.0))

    @model function unassigned_index(::Type{F}=Float64) where {F<:AbstractFloat}
        x = Vector{ReferenceValue{F}}(undef, 1)
        return x[1] ~ Normal()
    end
    @test_logs check_model(unassigned_index())

    @model function derived_inner(y)
        v = exp(y)
        return v ~ Normal()
    end
    @model function derived_outer(y)
        return a ~ to_submodel(derived_inner(y))
    end
    @test_logs (:warn, r"Variable a\.v.*derived from a model input") check_model(
        derived_outer(1.0)
    )

    ext = Base.get_extension(DynamicPPL, :DynamicPPLForwardDiffExt)
    acc = ext.InputDependencyAccumulator()
    vi = DynamicPPL.ThreadSafeVarInfo(OnlyAccsVarInfo((acc,)))
    dual = ForwardDiff.Dual{ext.InputDependencyTag}(1.0, 1.0)
    vi = DynamicPPL.check_input_dependency!!(vi, dual, @varname(x))
    @test isempty(vi.varinfo.accs[:InputDependency].vns)
    @test vi.accs_by_thread[Threads.threadid()][:InputDependency].vns == Set((@varname(x),))
    @test DynamicPPL.getacc(vi, Val(:InputDependency)).vns == Set((@varname(x),))
    @test isempty(vi.varinfo.accs[:InputDependency].vns)
end

end
