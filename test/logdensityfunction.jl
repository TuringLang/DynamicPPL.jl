module DynamicPPLLDFTests

using AbstractPPL: AbstractPPL
using Chairmarks
using DynamicPPL
using Distributions
using DistributionsAD: filldist
using ADTypes
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult, NoTest
using LinearAlgebra: I
using Test
using LogDensityProblems: LogDensityProblems

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

@testset "LogDensityFunction: Correctness" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        @testset "$islinked" for islinked in (false, true)
            unlinked_vi = DynamicPPL.VarInfo(m)
            vi = if islinked
                DynamicPPL.link!!(unlinked_vi, m)
            else
                unlinked_vi
            end
            ranges = DynamicPPL.get_ranges_and_linked(vi.values)
            params = [x for x in vi[:]]
            # Iterate over all variables
            for vn in keys(vi)
                # Check that `getindex_internal` returns the same thing as using the ranges
                # directly
                range_with_linked = ranges[vn]
                @test params[range_with_linked.range] ==
                    DynamicPPL.tovec(DynamicPPL.getindex_internal(vi, vn))
                # Check that the link status is correct
                @test range_with_linked.is_linked == islinked
            end
        end
    end

    @testset "Threaded observe" begin
        @model function threaded(y)
            x ~ Normal()
            Threads.@threads for i in eachindex(y)
                y[i] ~ Normal(x)
            end
        end
        N = 100
        model = setthreadsafe(threaded(zeros(N)), true)
        ldf = DynamicPPL.LogDensityFunction(model)

        xs = [1.0]
        @test LogDensityProblems.logdensity(ldf, xs) ≈
            logpdf(Normal(), xs[1]) + N * logpdf(Normal(xs[1]), 0.0)
    end
end

@testset "LogDensityFunction: interface" begin
    # miscellaneous parts of the LogDensityProblems interface
    @testset "dimensions" begin
        @model function m1()
            x ~ Normal()
            y ~ Normal()
            return nothing
        end
        model = m1()
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.dimension(ldf) == 2

        @model function m2()
            x ~ Dirichlet(ones(4))
            y ~ Categorical(x)
            return nothing
        end
        model = m2()
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.dimension(ldf) == 5
        linked_vi = DynamicPPL.link!!(VarInfo(model), model)
        ldf = DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vi)
        @test LogDensityProblems.dimension(ldf) == 4
    end

    @testset "capabilities" begin
        @model f() = x ~ Normal()
        model = f()
        # No adtype
        ldf = DynamicPPL.LogDensityFunction(model)
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{0}()
        # With adtype
        ldf = DynamicPPL.LogDensityFunction(model; adtype=AutoForwardDiff())
        @test LogDensityProblems.capabilities(typeof(ldf)) ==
            LogDensityProblems.LogDensityOrder{1}()
    end

    @testset "Callable struct as getlogdensity" begin
        @model function f()
            x ~ Normal()
            return 1.0 ~ Normal(x)
        end
        struct ScaledLogLike
            scale::Float64
        end
        function (sll::ScaledLogLike)(vi::AbstractVarInfo)
            return sll.scale * getloglikelihood(vi)
        end
        model = f()
        vi = VarInfo(model)
        sll = ScaledLogLike(2.0)
        ldf = DynamicPPL.LogDensityFunction(model, sll, vi)
        x = vi[:]
        @test LogDensityProblems.logdensity(ldf, x) == sll.scale * logpdf(Normal(x[1]), 1.0)
    end

    @testset "Custom accumulators" begin
        # Define an accumulator that always throws an error to test that custom
        # accumulators can be used with LogDensityFunction
        struct ErrorAccumulatorException <: Exception end
        struct ErrorAccumulator <: DynamicPPL.AbstractAccumulator end
        DynamicPPL.accumulator_name(::ErrorAccumulator) = :ERROR
        DynamicPPL.accumulate_assume!!(
            ::ErrorAccumulator, ::Any, ::Any, ::Any, ::VarName, ::Distribution, ::Any
        ) = throw(ErrorAccumulatorException())
        DynamicPPL.accumulate_observe!!(
            ::ErrorAccumulator, ::Distribution, ::Any, ::Union{VarName,Nothing}
        ) = throw(ErrorAccumulatorException())
        DynamicPPL.reset(ea::ErrorAccumulator) = ea
        Base.copy(ea::ErrorAccumulator) = ea
        # Construct an LDF
        @model function demo_error()
            return x ~ Normal()
        end
        model = demo_error()
        # check that passing accs as a tuple works
        ldf = LogDensityFunction(model, getlogjoint, VarInfo(model), (ErrorAccumulator(),))
        @test_throws ErrorAccumulatorException LogDensityProblems.logdensity(ldf, [0.0])
        # check that passing accs as AccumulatorTuple also works
        ldf = LogDensityFunction(
            model,
            getlogjoint,
            VarInfo(model),
            DynamicPPL.AccumulatorTuple(ErrorAccumulator()),
        )
        @test_throws ErrorAccumulatorException LogDensityProblems.logdensity(ldf, [0.0])
    end
end

@testset "Conversions to/from vectors" begin
    xdist, ydist = LogNormal(), Dirichlet(ones(3))
    @model function f()
        x ~ xdist
        y ~ ydist
        return nothing
    end
    model = f()

    xraw, yraw = 0.5, [0.2, 0.3, 0.5]
    raw_values = @vnt begin
        x := xraw
        y := yraw
    end

    function manual_make_vec(transform_strategy)
        # Manually construct the vector of values that we're interested in. This function
        # assumes that `x` will come before `y` in the LDF!
        xvec = if target_transform(transform_strategy, @varname(x)) isa DynamicLink
            DynamicPPL.to_linked_vec_transform(xdist)(xraw)
        else
            DynamicPPL.to_vec_transform(xdist)(xraw)
        end
        yvec = if target_transform(transform_strategy, @varname(y)) isa DynamicLink
            DynamicPPL.to_linked_vec_transform(ydist)(yraw)
        else
            DynamicPPL.to_vec_transform(ydist)(yraw)
        end
        return vcat(xvec, yvec)
    end

    @testset "$transform_strategy" for transform_strategy in (
        UnlinkAll(),
        LinkAll(),
        LinkSome((@varname(x),), UnlinkAll()),
        UnlinkSome((@varname(x),), LinkAll()),
    )
        accs = OnlyAccsVarInfo(VectorValueAccumulator())
        _, accs = init!!(model, accs, InitFromPrior(), transform_strategy)
        vvals = get_vector_values(accs)
        ldf = LogDensityFunction(model, getlogjoint_internal, vvals)

        @testset "InitFromVector -> raw values" begin
            # Test that initialising a model with `InitFromVector` will generate the correct
            # raw values.
            vec = manual_make_vec(transform_strategy)
            init_strategy = InitFromVector(vec, ldf)
            accs = OnlyAccsVarInfo(RawValueAccumulator(false))
            _, accs = init!!(model, accs, init_strategy, UnlinkAll())
            new_raw_values = get_raw_values(accs)
            @test new_raw_values[@varname(x)] ≈ xraw
            @test new_raw_values[@varname(y)] ≈ yraw

            @testset "Throws an error if vector has wrong length" begin
                @test_throws ArgumentError InitFromVector(randn(100), ldf)
                @test_throws ArgumentError InitFromVector(Float64[], ldf)
            end
        end

        @testset "Raw values -> vector" begin
            # Test that initialising a model with raw values allows us to generate the right
            # vector
            init_strategy = InitFromParams(raw_values)
            accs = OnlyAccsVarInfo(VectorValueAccumulator())
            _, accs = init!!(model, accs, init_strategy, transform_strategy)
            vecvals = get_vector_values(accs)
            vec = to_vector_input(vecvals, ldf)
            @test vec ≈ manual_make_vec(transform_strategy)

            @testset "Throws an error if transform strategy doesn't line up" begin
                if transform_strategy != UnlinkAll()
                    _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
                    vecvals = get_vector_values(accs)
                    @test_throws ArgumentError to_vector_input(vecvals, ldf)
                end
            end

            @testset "Throws an error if there are extra variables" begin
                @model function extra_var_model()
                    x ~ xdist
                    y ~ ydist
                    return z ~ Normal()
                end
                extra_model = extra_var_model()
                accs = OnlyAccsVarInfo(VectorValueAccumulator())
                _, accs = init!!(extra_model, accs, InitFromPrior(), transform_strategy)
                vecvals = get_vector_values(accs)
                @test_throws ArgumentError to_vector_input(vecvals, ldf)
            end

            @testset "Throws an error if there aren't enough variables" begin
                @model function fewer_var_model()
                    return x ~ xdist
                end
                fewer_model = fewer_var_model()
                accs = OnlyAccsVarInfo(VectorValueAccumulator())
                _, accs = init!!(fewer_model, accs, InitFromPrior(), transform_strategy)
                vecvals = get_vector_values(accs)
                @test_throws ArgumentError to_vector_input(vecvals, ldf)
            end

            @testset "Throws an error if the variable lengths aren't right" begin
                @model function different_var_model()
                    x ~ xdist
                    return y ~ Dirichlet(ones(4))  # as opposed to ones(3)
                end
                different_model = different_var_model()
                accs = OnlyAccsVarInfo(VectorValueAccumulator())
                _, accs = init!!(different_model, accs, InitFromPrior(), transform_strategy)
                vecvals = get_vector_values(accs)
                @test_throws ArgumentError to_vector_input(vecvals, ldf)
            end
        end

        @testset "Roundtrip" begin
            # Test that we can roundtrip from vector -> raw values -> vector
            vec = manual_make_vec(transform_strategy)
            init_strategy = InitFromVector(vec, ldf)
            accs = OnlyAccsVarInfo(VectorValueAccumulator())
            _, accs = init!!(model, accs, init_strategy, transform_strategy)
            new_vecvals = get_vector_values(accs)
            new_vec = to_vector_input(new_vecvals, ldf)
            @test new_vec ≈ vec
        end
    end
end

@testset "LogDensityFunction: Type stability" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        @testset "$islinked" for islinked in (false, true)
            unlinked_vi = DynamicPPL.VarInfo(m)
            vi = if islinked
                DynamicPPL.link!!(unlinked_vi, m)
            else
                unlinked_vi
            end
            ldf = DynamicPPL.LogDensityFunction(m, DynamicPPL.getlogjoint_internal, vi)
            x = vi[:]
            # The below type inference fails on v1.10.
            skip = (VERSION < v"1.11.0" && m.f === DynamicPPL.TestUtils.demo_nested_colons)
            @test begin
                @inferred LogDensityProblems.logdensity(ldf, x)
                true
            end skip = skip
        end
    end
end

@testset "LogDensityFunction: performance" begin
    # Evaluating these three models should not lead to any allocations (but only when
    # not using TSVI).
    @model function f()
        x ~ Normal()
        return 1.0 ~ Normal(x)
    end
    @model function submodel_inner()
        m ~ Normal(0, 1)
        s ~ Exponential()
        return (m=m, s=s)
    end
    # Note that for the allocation tests to work on this one, `inner` has
    # to be passed as an argument to `submodel_outer`, instead of just
    # being called inside the model function itself
    @model function submodel_outer(inner)
        params ~ to_submodel(inner)
        y ~ Normal(params.m, params.s)
        return 1.0 ~ Normal(y)
    end
    @testset for model in
                 (f(), submodel_inner() | (; s=0.0), submodel_outer(submodel_inner()))
        vi = VarInfo(model)
        ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi)
        x = vi[:]
        bench = median(@be LogDensityProblems.logdensity($ldf, $x))
        @test iszero(bench.allocs)
    end
end

@testset "AD with LogDensityFunction" begin
    # Used as the ground truth that others are compared against.
    ref_adtype = AutoForwardDiff()

    test_adtypes = [
        AutoReverseDiff(; compile=false),
        AutoReverseDiff(; compile=true),
        AutoMooncake(; config=nothing),
    ]

    @testset "Correctness" begin
        @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
            varinfo = VarInfo(m)
            linked_varinfo = DynamicPPL.link(varinfo, m)
            f = LogDensityFunction(m, getlogjoint_internal, linked_varinfo)
            x = [p for p in linked_varinfo[:]]

            # Calculate reference logp + gradient of logp using ForwardDiff
            ref_ad_result = run_ad(m, ref_adtype; varinfo=linked_varinfo, test=NoTest())
            ref_logp, ref_grad = ref_ad_result.value_actual, ref_ad_result.grad_actual

            @testset "$adtype" for adtype in test_adtypes
                @info "Testing AD on: $(m.f) - $adtype"

                @test run_ad(
                    m,
                    adtype;
                    varinfo=linked_varinfo,
                    test=WithExpectedResult(ref_logp, ref_grad),
                ) isa Any
            end
        end
    end

    @testset "logdensity_and_gradient with views" begin
        # This test ensures that you can call `logdensity_and_gradient` with an array
        # type that isn't the same as the one used in the gradient preparation.
        @model function f()
            x ~ Normal()
            return y ~ Normal()
        end
        @testset "$adtype" for adtype in test_adtypes
            x = randn(2)
            ldf = LogDensityFunction(f(); adtype)
            logp, grad = LogDensityProblems.logdensity_and_gradient(ldf, x)
            logp_view, grad_view = LogDensityProblems.logdensity_and_gradient(
                ldf, (@view x[:])
            )
            @test logp == logp_view
            @test grad == grad_view
        end
    end

    # Test that various different ways of specifying array types as arguments work with all
    # ADTypes.
    @testset "Array argument types" begin
        test_m = randn(2, 3)

        function eval_logp_and_grad(model, m, adtype)
            ldf = LogDensityFunction(model(); adtype=adtype)
            return LogDensityProblems.logdensity_and_gradient(ldf, m[:])
        end

        @model function scalar_matrix_model(::Type{T}=Float64) where {T<:Real}
            m = Matrix{T}(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        scalar_matrix_model_reference = eval_logp_and_grad(
            scalar_matrix_model, test_m, ref_adtype
        )

        @model function matrix_model(::Type{T}=Matrix{Float64}) where {T}
            m = T(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        matrix_model_reference = eval_logp_and_grad(matrix_model, test_m, ref_adtype)

        @model function scalar_array_model(::Type{T}=Float64) where {T<:Real}
            m = Array{T}(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        scalar_array_model_reference = eval_logp_and_grad(
            scalar_array_model, test_m, ref_adtype
        )

        @model function array_model(::Type{T}=Array{Float64}) where {T}
            m = T(undef, 2, 3)
            return m ~ filldist(MvNormal(zeros(2), I), 3)
        end

        array_model_reference = eval_logp_and_grad(array_model, test_m, ref_adtype)

        @testset "$adtype" for adtype in test_adtypes
            scalar_matrix_model_logp_and_grad = eval_logp_and_grad(
                scalar_matrix_model, test_m, adtype
            )
            @test scalar_matrix_model_logp_and_grad[1] ≈ scalar_matrix_model_reference[1]
            @test scalar_matrix_model_logp_and_grad[2] ≈ scalar_matrix_model_reference[2]
            matrix_model_logp_and_grad = eval_logp_and_grad(matrix_model, test_m, adtype)
            @test matrix_model_logp_and_grad[1] ≈ matrix_model_reference[1]
            @test matrix_model_logp_and_grad[2] ≈ matrix_model_reference[2]
            scalar_array_model_logp_and_grad = eval_logp_and_grad(
                scalar_array_model, test_m, adtype
            )
            @test scalar_array_model_logp_and_grad[1] ≈ scalar_array_model_reference[1]
            @test scalar_array_model_logp_and_grad[2] ≈ scalar_array_model_reference[2]
            array_model_logp_and_grad = eval_logp_and_grad(array_model, test_m, adtype)
            @test array_model_logp_and_grad[1] ≈ array_model_reference[1]
            @test array_model_logp_and_grad[2] ≈ array_model_reference[2]
        end
    end
end

end
