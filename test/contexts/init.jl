module DynamicPPLInitContextTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Bijectors: Bijectors
using Distributions
using DynamicPPL
using LinearAlgebra: I
using Random: Xoshiro
using StableRNGs: StableRNG
using Test

@testset "InitContext" begin
    @model function test_init_model()
        x ~ Normal()
        y ~ MvNormal(fill(x, 2), I)
        1.0 ~ Normal()
        return nothing
    end

    function test_generating_new_values(strategy::AbstractInitStrategy)
        @testset "generating new values: $(typeof(strategy))" begin
            # Check that init!! can generate values that weren't there previously.
            model = test_init_model()
            empty_vi = VarInfo()
            this_vi = deepcopy(empty_vi)
            _, vi = DynamicPPL.init!!(model, this_vi, strategy, UnlinkAll())
            @test Set(keys(vi)) == Set([@varname(x), @varname(y)])
            x, y = vi[@varname(x)], vi[@varname(y)]
            @test x isa Real
            @test y isa AbstractVector{<:Real}
            @test length(y) == 2
            (; logprior, loglikelihood) = getlogp(vi)
            @test logpdf(Normal(), x) + logpdf(MvNormal(fill(x, 2), I), y) == logprior
            @test logpdf(Normal(), 1.0) == loglikelihood
        end
    end

    function test_replacing_values(strategy::AbstractInitStrategy)
        @testset "replacing old values: $(typeof(strategy))" begin
            # Check that init!! can overwrite values that were already there.
            model = test_init_model()
            empty_vi = VarInfo()
            # start by generating some rubbish values
            vi = deepcopy(empty_vi)
            old_x, old_y = 100000.00, [300000.00, 500000.00]
            vi = DynamicPPL.setindex_with_dist!!(
                vi,
                UntransformedValue(old_x),
                Normal(),
                @varname(x),
                DynamicPPL.NoTemplate(),
            )
            vi = DynamicPPL.setindex_with_dist!!(
                vi,
                UntransformedValue(old_y),
                MvNormal(fill(old_x, 2), I),
                @varname(y),
                DynamicPPL.NoTemplate(),
            )
            # then overwrite it
            _, new_vi = DynamicPPL.init!!(model, vi, strategy, UnlinkAll())
            new_x, new_y = new_vi[@varname(x)], new_vi[@varname(y)]
            # check that the values are (presumably) different
            @test old_x != new_x
            @test old_y != new_y
        end
    end

    function test_rng_respected(strategy::AbstractInitStrategy)
        @testset "check that RNG is respected: $(typeof(strategy))" begin
            model = test_init_model()
            empty_vi = VarInfo()
            _, vi1 = DynamicPPL.init!!(Xoshiro(468), model, deepcopy(empty_vi), strategy)
            _, vi2 = DynamicPPL.init!!(Xoshiro(468), model, deepcopy(empty_vi), strategy)
            _, vi3 = DynamicPPL.init!!(Xoshiro(469), model, deepcopy(empty_vi), strategy)
            @test vi1[@varname(x)] == vi2[@varname(x)]
            @test vi1[@varname(y)] == vi2[@varname(y)]
            @test vi1[@varname(x)] != vi3[@varname(x)]
            @test vi1[@varname(y)] != vi3[@varname(y)]
        end
    end

    function test_transform_strategy_respected(strategy::AbstractInitStrategy)
        # We want to check that regardless of the InitStrategy, the transform strategy takes
        # precedence when determining whether variables are linked or not.
        @testset "check that transform strategy is respected: $(typeof(strategy))" begin
            dist = LogNormal()
            from_linked_vec = DynamicPPL.from_linked_vec_transform(dist)
            @model function logn()
                a ~ dist
                return b ~ dist
            end
            model = logn()

            @testset "$transform_strategy" for transform_strategy in (
                LinkAll(),
                UnlinkAll(),
                LinkSome((@varname(a),), UnlinkAll()),
                LinkSome((@varname(b),), UnlinkAll()),
                UnlinkSome((@varname(a),), LinkAll()),
                UnlinkSome((@varname(b),), LinkAll()),
            )
                # Generate a VarInfo with that strategy
                vi = last(
                    DynamicPPL.init!!(model, VarInfo(), InitFromPrior(), transform_strategy)
                )
                # Check that initialising with that strategy preserves the linking
                # status of the VarInfo
                _, vi = DynamicPPL.init!!(model, vi, strategy, transform_strategy)

                expected_logprior = 0.0
                expected_logjac = 0.0
                for vn in (@varname(a), @varname(b))
                    if DynamicPPL.target_transform(transform_strategy, vn) isa DynamicLink
                        @test DynamicPPL.is_transformed(vi, vn)
                        # The VarInfo should hold a LinkedVectorValue
                        lvv = vi.values[vn]
                        @test lvv isa LinkedVectorValue
                        linked_vec = DynamicPPL.get_internal_value(lvv)
                        val, inv_logjac = Bijectors.with_logabsdet_jacobian(
                            from_linked_vec, linked_vec
                        )
                        expected_logprior += logpdf(dist, val)
                        expected_logjac -= inv_logjac
                    else
                        @test !DynamicPPL.is_transformed(vi, vn)
                        # The VarInfo should hold a VectorValue
                        vv = vi.values[vn]
                        @test vv isa VectorValue
                        # it should wrap a single value
                        val = only(DynamicPPL.get_internal_value(vv))
                        expected_logprior += logpdf(dist, val)
                    end
                end
                @test DynamicPPL.getlogprior(vi) ≈ expected_logprior
                @test DynamicPPL.getlogjac(vi) ≈ expected_logjac
            end
        end
    end

    @testset "InitFromPrior" begin
        test_generating_new_values(InitFromPrior())
        test_replacing_values(InitFromPrior())
        test_rng_respected(InitFromPrior())
        test_transform_strategy_respected(InitFromPrior())

        @testset "check that values are within support" begin
            @model just_unif() = x ~ Uniform(0.0, 1e-7)
            for _ in 1:100
                _, vi = DynamicPPL.init!!(just_unif(), VarInfo(), InitFromPrior())
                @test vi[@varname(x)] isa Real
                @test 0.0 <= vi[@varname(x)] <= 1e-7
            end
        end

        @testset "check that values are being drawn from prior" begin
            @model normal_m() = return x ~ Normal()
            model = normal_m()
            # Sample lots of times...
            N = 1000
            rng = StableRNG(468)
            xs = Vector{Float64}(undef, N)
            for i in 1:N
                xs[i] = first(
                    DynamicPPL.init!!(
                        rng, model, OnlyAccsVarInfo(()), InitFromPrior(), UnlinkAll()
                    ),
                )
            end
            @test mean(xs) ≈ 0.0 atol = 0.05
        end
    end

    @testset "InitFromUniform" begin
        test_generating_new_values(InitFromUniform())
        test_replacing_values(InitFromUniform())
        test_rng_respected(InitFromUniform())
        test_transform_strategy_respected(InitFromUniform())

        @testset "check that InitFromUniform really draws uniformly" begin
            @model logn() = a ~ LogNormal()
            model = logn()
            # Sample lots of times...
            N = 1000
            rng = StableRNG(468)
            xs = Vector{Float64}(undef, N)
            for i in 1:N
                xs[i] = first(
                    DynamicPPL.init!!(
                        rng, model, OnlyAccsVarInfo(()), InitFromUniform(), UnlinkAll()
                    ),
                )
            end
            # Transform values back to linked space
            xs = log.(xs)
            # All the values should be between -2 and 2
            @test all(-2.0 .<= xs .<= 2.0)
            # The mean should be close to 0 since the distribution is symmetric around 0 in
            # the linked space.
            @test mean(xs) ≈ 0.0 atol = 0.05
        end
    end

    @testset "InitFromParams" begin
        # Once we've checked that NTs and Dicts are internally promoted to VNTs, the rest of
        # the tests only need to check that InitFromParams(::VNT) is handled correctly.
        @testset "NT promotion to VNT" begin
            nt = (x=1.0, y=[2.0, 3.0], z="zzz")
            ifp = InitFromParams(nt)
            vnt = @vnt begin
                x := 1.0
                y := [2.0, 3.0]
                z := "zzz"
            end
            @test ifp.params == vnt
        end
        @testset "Dict promotion to VNT" begin
            dict = OrderedDict(
                @varname(x) => 1.0, @varname(y) => [2.0, 3.0], @varname(z) => "zzz"
            )
            ifp = InitFromParams(dict)
            vnt = @vnt begin
                x := 1.0
                y := [2.0, 3.0]
                z := "zzz"
            end
            @test ifp.params == vnt
            @testset "throws warning if GrowableArray is used" begin
                dict = OrderedDict(@varname(x[1]) => 1.0)
                warnmsg = r"Creating a growable `Base.Array`"
                @test_logs (:warn, warnmsg) InitFromParams(dict)
            end
        end

        test_transform_strategy_respected(InitFromParams(VarNamedTuple(; a=1.0)))

        @testset "given full set of parameters" begin
            # test_init_model has x ~ Normal() and y ~ MvNormal(zeros(2), I)
            my_x, my_y = 1.0, [2.0, 3.0]
            vnt = @vnt begin
                x := my_x
                y := my_y
            end

            model = test_init_model()
            acc = RawValueAccumulator(false)
            empty_vi = OnlyAccsVarInfo((acc,))
            _, vi = DynamicPPL.init!!(model, empty_vi, InitFromParams(vnt), UnlinkAll())
            vals = DynamicPPL.getacc(vi, Val(DynamicPPL.accumulator_name(acc))).values
            @test vals[@varname(x)] == my_x
            @test vals[@varname(y)] == my_y
        end

        @testset "given only partial parameters" begin
            my_x = 1.0
            vnt = @vnt begin
                x := my_x
            end

            @testset "with InitFromPrior fallback" begin
                model = test_init_model()
                acc = RawValueAccumulator(false)
                empty_vi = OnlyAccsVarInfo((acc,))
                _, vi = DynamicPPL.init!!(
                    model, empty_vi, InitFromParams(vnt, InitFromPrior()), UnlinkAll()
                )
                vals = DynamicPPL.getacc(vi, Val(DynamicPPL.accumulator_name(acc))).values
                @test vals[@varname(x)] == my_x

                # If we rerun it, the value of `x` should still be the same, but `y` should
                # change
                old_y = vals[@varname(y)]
                _, vi = DynamicPPL.init!!(
                    model, vi, InitFromParams(vnt, InitFromPrior()), UnlinkAll()
                )
                vals = DynamicPPL.getacc(vi, Val(DynamicPPL.accumulator_name(acc))).values
                @test vals[@varname(x)] == my_x
                new_y = vals[@varname(y)]
                @test new_y != old_y
            end

            @testset "with no fallback" begin
                model = test_init_model()
                acc = RawValueAccumulator(false)
                empty_vi = OnlyAccsVarInfo((acc,))

                # When there's no entry for `y`
                @test_throws ErrorException DynamicPPL.init!!(
                    model, empty_vi, InitFromParams(vnt, nothing), UnlinkAll()
                )

                # We also explicitly test the case where `y = missing`.
                vnt_missing = @vnt begin
                    x := my_x
                    y := missing
                end
                @test_throws ErrorException DynamicPPL.init!!(
                    model, empty_vi, InitFromParams(vnt_missing, nothing), UnlinkAll()
                )
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
