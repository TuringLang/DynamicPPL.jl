module DynamicPPLLDFTests

using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
using Chairmarks
using DynamicPPL
using Distributions
using ADTypes
using DynamicPPL.TestUtils.AD: run_ad, WithExpectedResult, NoTest
using LinearAlgebra: I
using Test
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro
using StableRNGs: StableRNG

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

@testset "LogDensityFunction: constructors" begin
    dist = Beta(2, 2)
    @model f() = x ~ dist
    expected_ral_unlinked = @vnt begin
        x := DynamicPPL.RangeAndLinked(1:1, false)
    end
    expected_ral_linked = @vnt begin
        x := DynamicPPL.RangeAndLinked(1:1, true)
    end
    oavi_unlinked = begin
        accs = OnlyAccsVarInfo(VectorValueAccumulator())
        _, accs = init!!(f(), accs, InitFromPrior(), UnlinkAll())
        accs
    end
    oavi_linked = begin
        accs = OnlyAccsVarInfo(VectorValueAccumulator())
        _, accs = init!!(f(), accs, InitFromPrior(), LinkAll())
        accs
    end

    # Check that you can construct an LDF from a VarInfo, a VNT of vector values,
    # or a transform strategy itself.
    for arg in (
        VarInfo(f()),
        VarInfo(f()).values,
        oavi_unlinked,
        get_vector_values(oavi_unlinked),
        UnlinkAll(),
    )
        for adtype in (nothing, AutoForwardDiff())
            ldf = LogDensityFunction(f(), getlogjoint_internal, arg; adtype=adtype)
            @test ldf._varname_ranges == expected_ral_unlinked
            @test ldf.transform_strategy == UnlinkAll()
            @test LogDensityProblems.logdensity(ldf, [0.5]) ≈ logpdf(Beta(2, 2), 0.5)
            if adtype === nothing
                @test ldf.adtype === nothing
            else
                @test ldf.adtype isa AutoForwardDiff
            end
        end
    end
    for arg in (
        link!!(VarInfo(f()), f()),
        link!!(VarInfo(f()), f()).values,
        oavi_linked,
        get_vector_values(oavi_linked),
        LinkAll(),
    )
        for adtype in (nothing, AutoForwardDiff())
            ldf = LogDensityFunction(f(), getlogjoint_internal, arg; adtype=adtype)
            @test ldf._varname_ranges == expected_ral_linked
            @test ldf.transform_strategy == LinkAll()
            y = [0.5]
            x, logjac = Bijectors.with_logabsdet_jacobian(
                Bijectors.VectorBijectors.from_linked_vec(dist), y
            )
            @test LogDensityProblems.logdensity(ldf, y) ≈ logpdf(Beta(2, 2), x) + logjac
            if adtype === nothing
                @test ldf.adtype === nothing
            else
                @test ldf.adtype isa AutoForwardDiff
            end
        end
    end
end

@testset "LogDensityFunction: Correctness" begin
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
        ldf = DynamicPPL.LogDensityFunction(model, getlogjoint_internal, UnlinkAll())
        @test LogDensityProblems.dimension(ldf) == 5
        ldf = DynamicPPL.LogDensityFunction(model, getlogjoint_internal, LinkAll())
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
        sll = ScaledLogLike(2.0)
        ldf = DynamicPPL.LogDensityFunction(model, sll)
        x = [0.5]
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
            ::ErrorAccumulator, ::Distribution, ::Any, ::Union{VarName,Nothing}, ::Any
        ) = throw(ErrorAccumulatorException())
        DynamicPPL.reset(ea::ErrorAccumulator) = ea
        Base.copy(ea::ErrorAccumulator) = ea
        # Construct an LDF
        @model function demo_error()
            return x ~ Normal()
        end
        model = demo_error()
        # check that passing accs as a tuple works
        ldf = LogDensityFunction(model, getlogjoint, UnlinkAll(), (ErrorAccumulator(),))
        @test_throws ErrorAccumulatorException LogDensityProblems.logdensity(ldf, [0.0])
        # check that passing accs as AccumulatorTuple also works
        ldf = LogDensityFunction(
            model, getlogjoint, UnlinkAll(), DynamicPPL.AccumulatorTuple(ErrorAccumulator())
        )
        @test_throws ErrorAccumulatorException LogDensityProblems.logdensity(ldf, [0.0])
    end
end

@testset "rand() on LogDensityFunction interface" begin
    # Check that we can call rand
    @model function f()
        return x ~ Normal()
    end

    isa_single_float_vector(r) = r isa Vector{Float64} && length(r) == 1

    # It's hard to really *test* the output of rand
    @testset for init_strategy in (InitFromPrior(), InitFromUniform())
        @testset for tfm_strategy in (UnlinkAll(), LinkAll())
            rng = StableRNG(468)
            model = f()
            ldf = LogDensityFunction(model, getlogjoint_internal, tfm_strategy)
            rands = [Base.rand(rng, ldf, init_strategy) for _ in 1:1000]
            @test all(isa_single_float_vector, rands)
            @test mean(stack(rands)) ≈ 0.0 atol = 0.1
        end
    end

    # Check function interface
    ldf = LogDensityFunction(f())
    @test isa_single_float_vector(rand(ldf))
    @test isa_single_float_vector(rand(ldf, InitFromPrior()))
    @test isa_single_float_vector(rand(Xoshiro(468), ldf))
    @test isa_single_float_vector(rand(Xoshiro(468), ldf, InitFromPrior()))
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
            Bijectors.VectorBijectors.to_linked_vec(xdist)(xraw)
        else
            Bijectors.VectorBijectors.to_vec(xdist)(xraw)
        end
        yvec = if target_transform(transform_strategy, @varname(y)) isa DynamicLink
            Bijectors.VectorBijectors.to_linked_vec(ydist)(yraw)
        else
            Bijectors.VectorBijectors.to_vec(ydist)(yraw)
        end
        return vcat(xvec, yvec)
    end

    @testset "$transform_strategy" for transform_strategy in (
        UnlinkAll(),
        LinkAll(),
        LinkSome(Set([@varname(x)]), UnlinkAll()),
        UnlinkSome(Set([@varname(x)]), LinkAll()),
    )
        ldf = LogDensityFunction(model, getlogjoint_internal, transform_strategy)

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
            # vector (either indirectly via VectorValueAccumulator and to_vector_params, or
            # directly via VectorParamAccumulator).
            init_strategy = InitFromParams(raw_values)
            accs = OnlyAccsVarInfo(VectorValueAccumulator(), VectorParamAccumulator(ldf))
            _, accs = init!!(model, accs, init_strategy, transform_strategy)

            vecvals = get_vector_values(accs)
            vec = to_vector_params(vecvals, ldf)
            @test vec ≈ manual_make_vec(transform_strategy)

            vecparams = get_vector_params(accs)
            @test vecparams ≈ manual_make_vec(transform_strategy)

            # This isn't really 'random' since the init strategy fully determines the
            # output, but we can check it
            @test Base.rand(ldf, init_strategy) ≈ manual_make_vec(transform_strategy)

            @testset "Throws an error if transform strategy doesn't line up" begin
                if transform_strategy != UnlinkAll()
                    accs = OnlyAccsVarInfo(VectorValueAccumulator())
                    _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
                    vecvals = get_vector_values(accs)
                    @test_throws ArgumentError to_vector_params(vecvals, ldf)

                    accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
                    @test_throws ArgumentError init!!(
                        model, accs, InitFromPrior(), UnlinkAll()
                    )
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
                @test_throws ArgumentError to_vector_params(vecvals, ldf)

                accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
                @test_throws ErrorException init!!(
                    extra_model, accs, InitFromPrior(), transform_strategy
                )
            end

            @testset "Throws an error if there aren't enough variables" begin
                @model function fewer_var_model()
                    return x ~ xdist
                end
                fewer_model = fewer_var_model()

                accs = OnlyAccsVarInfo(VectorValueAccumulator())
                _, accs = init!!(fewer_model, accs, InitFromPrior(), transform_strategy)
                vecvals = get_vector_values(accs)
                @test_throws ArgumentError to_vector_params(vecvals, ldf)

                accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
                _, accs = init!!(fewer_model, accs, InitFromPrior(), transform_strategy)
                @test_throws ArgumentError get_vector_params(accs)
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
                @test_throws ArgumentError to_vector_params(vecvals, ldf)

                accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
                @test_throws ArgumentError init!!(
                    different_model, accs, InitFromPrior(), transform_strategy
                )
            end
        end

        @testset "Roundtrip" begin
            # Test that we can roundtrip from vector -> raw values -> vector
            vec = manual_make_vec(transform_strategy)
            init_strategy = InitFromVector(vec, ldf)

            accs = OnlyAccsVarInfo(VectorValueAccumulator(), VectorParamAccumulator(ldf))
            _, accs = init!!(model, accs, init_strategy, transform_strategy)
            new_vecvals = get_vector_values(accs)
            new_vec = to_vector_params(new_vecvals, ldf)
            @test new_vec ≈ vec

            new_vec_params = get_vector_params(accs)
            @test new_vec_params ≈ vec
        end
    end
end

@testset "LogDensityFunction: Type stability" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        @testset "$tfm_strategy" for tfm_strategy in (UnlinkAll(), LinkAll())
            ldf = DynamicPPL.LogDensityFunction(
                m, DynamicPPL.getlogjoint_internal, tfm_strategy
            )
            x = rand(ldf)
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
        @testset for tfm_strategy in (UnlinkAll(), LinkAll())
            ldf = DynamicPPL.LogDensityFunction(model, getlogjoint_internal, tfm_strategy)
            x = rand(ldf)
            bench = median(@be LogDensityProblems.logdensity($ldf, $x))
            @test iszero(bench.allocs)
        end
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
            f = LogDensityFunction(m, getlogjoint_internal, LinkAll())
            x = rand(f)

            # Calculate reference logp + gradient of logp using ForwardDiff
            ref_ad_result = run_ad(m, ref_adtype; params=x, test=NoTest())
            ref_logp, ref_grad = ref_ad_result.value_actual, ref_ad_result.grad_actual

            @testset "$adtype" for adtype in test_adtypes
                @info "Testing AD on: $(m.f) - $adtype"

                @test run_ad(
                    m, adtype; params=x, test=WithExpectedResult(ref_logp, ref_grad)
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
