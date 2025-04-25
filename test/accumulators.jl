module AccumulatorTests

using Test
using Distributions
using DynamicPPL
using DynamicPPL:
    AccumulatorTuple,
    LogLikelihood,
    LogPrior,
    NumProduce,
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
            @test LogPrior(0.0) ==
                LogPrior() ==
                LogPrior{Float64}() ==
                LogPrior{Float64}(0.0) ==
                zero(LogPrior(1.0))
            @test LogLikelihood(0.0) ==
                LogLikelihood() ==
                LogLikelihood{Float64}() ==
                LogLikelihood{Float64}(0.0) ==
                zero(LogLikelihood(1.0))
            @test NumProduce(0) ==
                NumProduce() ==
                NumProduce{Int}() ==
                NumProduce{Int}(0) ==
                zero(NumProduce(1))
        end

        @testset "addition and incrementation" begin
            @test LogPrior(1.0f0) + LogPrior(1.0f0) == LogPrior(2.0f0)
            @test LogPrior(1.0) + LogPrior(1.0f0) == LogPrior(2.0)
            @test LogLikelihood(1.0f0) + LogLikelihood(1.0f0) == LogLikelihood(2.0f0)
            @test LogLikelihood(1.0) + LogLikelihood(1.0f0) == LogLikelihood(2.0)
            @test increment(NumProduce()) == NumProduce(1)
            @test increment(NumProduce{UInt8}()) == NumProduce{UInt8}(1)
        end

        @testset "split and combine" begin
            for acc in [
                LogPrior(1.0),
                LogLikelihood(1.0),
                NumProduce(1),
                LogPrior(1.0f0),
                LogLikelihood(1.0f0),
                NumProduce(UInt8(1)),
            ]
                @test combine(acc, split(acc)) == acc
            end
        end

        @testset "conversions" begin
            @test convert(LogPrior{Float32}, LogPrior(1.0)) == LogPrior{Float32}(1.0f0)
            @test convert(LogLikelihood{Float32}, LogLikelihood(1.0)) ==
                LogLikelihood{Float32}(1.0f0)
            @test convert(NumProduce{UInt8}, NumProduce(1)) == NumProduce{UInt8}(1)

            @test convert_eltype(Float32, LogPrior(1.0)) == LogPrior{Float32}(1.0f0)
            @test convert_eltype(Float32, LogLikelihood(1.0)) ==
                LogLikelihood{Float32}(1.0f0)
        end

        @testset "accumulate_assume" begin
            val = 2.0
            logjac = pi
            vn = @varname(x)
            dist = Normal()
            @test accumulate_assume!!(LogPrior(1.0), val, logjac, vn, dist) ==
                LogPrior(1.0 + logjac + logpdf(dist, val))
            @test accumulate_assume!!(LogLikelihood(1.0), val, logjac, vn, dist) ==
                LogLikelihood(1.0)
            @test accumulate_assume!!(NumProduce(1), val, logjac, vn, dist) == NumProduce(1)
        end

        @testset "accumulate_observe" begin
            right = Normal()
            left = 2.0
            vn = @varname(x)
            @test accumulate_observe!!(LogPrior(1.0), right, left, vn) == LogPrior(1.0)
            @test accumulate_observe!!(LogLikelihood(1.0), right, left, vn) ==
                LogLikelihood(1.0 + logpdf(right, left))
            @test accumulate_observe!!(NumProduce(1), right, left, vn) == NumProduce(2)
        end
    end

    @testset "accumulator tuples" begin
        # Some accumulators we'll use for testing
        lp_f64 = LogPrior(1.0)
        lp_f32 = LogPrior(1.0f0)
        ll_f64 = LogLikelihood(1.0)
        ll_f32 = LogLikelihood(1.0f0)
        np_i64 = NumProduce(1)

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
            at_all64 = AccumulatorTuple(lp_f64, ll_f64, np_i64)

            @test at_all64[:LogPrior] == lp_f64
            @test at_all64[:LogLikelihood] == ll_f64
            @test at_all64[:NumProduce] == np_i64

            @test haskey(AccumulatorTuple(np_i64), Val(:NumProduce))
            @test ~haskey(AccumulatorTuple(np_i64), Val(:LogPrior))
            @test length(AccumulatorTuple(lp_f64, ll_f64, np_i64)) == 3
            @test keys(at_all64) == (:LogPrior, :LogLikelihood, :NumProduce)
            @test collect(at_all64) == [lp_f64, ll_f64, np_i64]

            # Replace the existing LogPrior
            @test setacc!!(at_all64, lp_f32)[:LogPrior] == lp_f32
            # Check that setacc!! didn't modify the original
            @test at_all64 == AccumulatorTuple(lp_f64, ll_f64, np_i64)
            # Add a new accumulator type.
            @test setacc!!(AccumulatorTuple(lp_f64), ll_f64) ==
                AccumulatorTuple(lp_f64, ll_f64)

            @test getacc(at_all64, Val(:LogPrior)) == lp_f64
        end

        @testset "map_accumulator(s)!!" begin
            # map over all accumulators
            accs = AccumulatorTuple(lp_f32, ll_f32)
            @test map(zero, accs) == AccumulatorTuple(LogPrior(0.0f0), LogLikelihood(0.0f0))
            # Test that the original wasn't modified.
            @test accs == AccumulatorTuple(lp_f32, ll_f32)

            # A map with a closure that changes the types of the accumulators.
            @test map(acc -> convert_eltype(Float64, acc), accs) ==
                AccumulatorTuple(LogPrior(1.0), LogLikelihood(1.0))

            # only apply to a particular accumulator
            @test map_accumulator(zero, accs, Val(:LogLikelihood)) ==
                AccumulatorTuple(lp_f32, LogLikelihood(0.0f0))
            @test map_accumulator(
                acc -> convert_eltype(Float64, acc), accs, Val(:LogLikelihood)
            ) == AccumulatorTuple(lp_f32, LogLikelihood(1.0))
        end
    end
end

end
