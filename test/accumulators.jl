module AccumulatorTests

using Test
using Distributions
using DynamicPPL
using DynamicPPL:
    AccumulatorTuple,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    accumulate_assume!!,
    accumulate_observe!!,
    combine,
    convert_eltype,
    getacc,
    increment,
    map_accumulator,
    setacc!!,
    split

@testset "accumulators" begin
    @testset "individual accumulator types" begin
        @testset "constructors" begin
            @test LogPriorAccumulator(0.0) ==
                LogPriorAccumulator() ==
                LogPriorAccumulator{Float64}() ==
                LogPriorAccumulator{Float64}(0.0) ==
                zero(LogPriorAccumulator(1.0))
            @test LogLikelihoodAccumulator(0.0) ==
                LogLikelihoodAccumulator() ==
                LogLikelihoodAccumulator{Float64}() ==
                LogLikelihoodAccumulator{Float64}(0.0) ==
                zero(LogLikelihoodAccumulator(1.0))
        end

        @testset "addition and incrementation" begin
            @test acclogp(LogPriorAccumulator(1.0f0), 1.0f0) == LogPriorAccumulator(2.0f0)
            @test acclogp(LogPriorAccumulator(1.0), 1.0f0) == LogPriorAccumulator(2.0)
            @test acclogp(LogLikelihoodAccumulator(1.0f0), 1.0f0) ==
                LogLikelihoodAccumulator(2.0f0)
            @test acclogp(LogLikelihoodAccumulator(1.0), 1.0f0) ==
                LogLikelihoodAccumulator(2.0)
        end

        @testset "split and combine" begin
            for acc in [
                LogPriorAccumulator(1.0),
                LogLikelihoodAccumulator(1.0),
                LogPriorAccumulator(1.0f0),
                LogLikelihoodAccumulator(1.0f0),
            ]
                @test combine(acc, split(acc)) == acc
            end
        end

        @testset "conversions" begin
            @test convert(LogPriorAccumulator{Float32}, LogPriorAccumulator(1.0)) ==
                LogPriorAccumulator{Float32}(1.0f0)
            @test convert(
                LogLikelihoodAccumulator{Float32}, LogLikelihoodAccumulator(1.0)
            ) == LogLikelihoodAccumulator{Float32}(1.0f0)

            @test convert_eltype(Float32, LogPriorAccumulator(1.0)) ==
                LogPriorAccumulator{Float32}(1.0f0)
            @test convert_eltype(Float32, LogLikelihoodAccumulator(1.0)) ==
                LogLikelihoodAccumulator{Float32}(1.0f0)
        end

        @testset "accumulate_assume" begin
            val = 2.0
            logjac = pi
            vn = @varname(x)
            dist = Normal()
            @test accumulate_assume!!(LogPriorAccumulator(1.0), val, logjac, vn, dist) ==
                LogPriorAccumulator(1.0 + logpdf(dist, val))
            @test accumulate_assume!!(LogJacobianAccumulator(2.0), val, logjac, vn, dist) ==
                LogJacobianAccumulator(2.0 + logjac)
            @test accumulate_assume!!(
                LogLikelihoodAccumulator(1.0), val, logjac, vn, dist
            ) == LogLikelihoodAccumulator(1.0)
        end

        @testset "accumulate_observe" begin
            right = Normal()
            left = 2.0
            vn = @varname(x)
            @test accumulate_observe!!(LogPriorAccumulator(1.0), right, left, vn) ==
                LogPriorAccumulator(1.0)
            @test accumulate_observe!!(LogJacobianAccumulator(1.0), right, left, vn) ==
                LogJacobianAccumulator(1.0)
            @test accumulate_observe!!(LogLikelihoodAccumulator(1.0), right, left, vn) ==
                LogLikelihoodAccumulator(1.0 + logpdf(right, left))
        end
    end

    @testset "accumulator tuples" begin
        # Some accumulators we'll use for testing
        lp_f64 = LogPriorAccumulator(1.0)
        lp_f32 = LogPriorAccumulator(1.0f0)
        ll_f64 = LogLikelihoodAccumulator(1.0)
        ll_f32 = LogLikelihoodAccumulator(1.0f0)

        @testset "constructors" begin
            @test AccumulatorTuple(lp_f64, ll_f64) == AccumulatorTuple((lp_f64, ll_f64))
            # Names in NamedTuple arguments are ignored
            @test AccumulatorTuple((; a=lp_f64)) == AccumulatorTuple(lp_f64)

            # Can't have two accumulators of the same type.
            @test_throws "duplicate field name" AccumulatorTuple(lp_f64, lp_f64)
            # Not even if their element types differ.
            @test_throws "duplicate field name" AccumulatorTuple(lp_f64, lp_f32)
        end

        @testset "basic operations" begin
            at_all64 = AccumulatorTuple(lp_f64, ll_f64)

            @test at_all64[:LogPrior] == lp_f64
            @test at_all64[:LogLikelihood] == ll_f64

            @test haskey(AccumulatorTuple(lp_f64), Val(:LogPrior))
            @test ~haskey(AccumulatorTuple(lp_f64), Val(:LogLikelihood))
            @test length(AccumulatorTuple(lp_f64, ll_f64)) == 2
            @test keys(at_all64) == (:LogPrior, :LogLikelihood)
            @test collect(at_all64) == [lp_f64, ll_f64]

            # Replace the existing LogPriorAccumulator
            @test setacc!!(at_all64, lp_f32)[:LogPrior] == lp_f32
            # Check that setacc!! didn't modify the original
            @test at_all64 == AccumulatorTuple(lp_f64, ll_f64)
            # Add a new accumulator type.
            @test setacc!!(AccumulatorTuple(lp_f64), ll_f64) ==
                AccumulatorTuple(lp_f64, ll_f64)

            @test getacc(at_all64, Val(:LogPrior)) == lp_f64
        end

        @testset "map_accumulator(s)!!" begin
            # map over all accumulators
            accs = AccumulatorTuple(lp_f32, ll_f32)
            @test map(zero, accs) == AccumulatorTuple(
                LogPriorAccumulator(0.0f0), LogLikelihoodAccumulator(0.0f0)
            )
            # Test that the original wasn't modified.
            @test accs == AccumulatorTuple(lp_f32, ll_f32)

            # A map with a closure that changes the types of the accumulators.
            @test map(acc -> convert_eltype(Float64, acc), accs) ==
                AccumulatorTuple(LogPriorAccumulator(1.0), LogLikelihoodAccumulator(1.0))

            # only apply to a particular accumulator
            @test map_accumulator(zero, accs, Val(:LogLikelihood)) ==
                AccumulatorTuple(lp_f32, LogLikelihoodAccumulator(0.0f0))
            @test map_accumulator(
                acc -> convert_eltype(Float64, acc), accs, Val(:LogLikelihood)
            ) == AccumulatorTuple(lp_f32, LogLikelihoodAccumulator(1.0))
        end
    end
end

end
