module AccumulatorTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

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
    promote_for_threadsafe_eval,
    getacc,
    map_accumulator,
    reset,
    setacc!!,
    split

@testset "accumulators" begin
    @testset "individual accumulator types" begin
        @testset "constructors" begin
            @test LogPriorAccumulator(0.0) ==
                LogPriorAccumulator() ==
                LogPriorAccumulator{Float64}() ==
                LogPriorAccumulator{Float64}(0.0) ==
                DynamicPPL.reset(LogPriorAccumulator(1.0))
            @test LogLikelihoodAccumulator(0.0) ==
                LogLikelihoodAccumulator() ==
                LogLikelihoodAccumulator{Float64}() ==
                LogLikelihoodAccumulator{Float64}(0.0) ==
                DynamicPPL.reset(LogLikelihoodAccumulator(1.0))
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

            @test promote_for_threadsafe_eval(LogPriorAccumulator(1.0), Float32) ==
                LogPriorAccumulator{Float32}(1.0f0)
            @test promote_for_threadsafe_eval(LogLikelihoodAccumulator(1.0), Float32) ==
                LogLikelihoodAccumulator{Float32}(1.0f0)
        end

        @testset "accumulate_assume" begin
            val = 2.0
            tval = DynamicPPL.UntransformedValue(nothing)
            logjac = pi
            vn = @varname(x)
            dist = Normal()
            template = nothing
            @test accumulate_assume!!(
                LogPriorAccumulator(1.0), val, tval, logjac, vn, dist, template
            ) == LogPriorAccumulator(1.0 + logpdf(dist, val))
            @test accumulate_assume!!(
                LogJacobianAccumulator(2.0), val, tval, logjac, vn, dist, template
            ) == LogJacobianAccumulator(2.0 + logjac)
            @test accumulate_assume!!(
                LogLikelihoodAccumulator(1.0), val, tval, logjac, vn, dist, template
            ) == LogLikelihoodAccumulator(1.0)
        end

        @testset "accumulate_observe" begin
            right = Normal()
            left = 2.0
            vn = @varname(x)
            template = nothing
            @test accumulate_observe!!(
                LogPriorAccumulator(1.0), right, left, vn, template
            ) == LogPriorAccumulator(1.0)
            @test accumulate_observe!!(
                LogJacobianAccumulator(1.0), right, left, vn, template
            ) == LogJacobianAccumulator(1.0)
            @test accumulate_observe!!(
                LogLikelihoodAccumulator(1.0), right, left, vn, template
            ) == LogLikelihoodAccumulator(1.0 + logpdf(right, left))
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
            @test !haskey(AccumulatorTuple(lp_f64), Val(:LogLikelihood))
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
            @test map(DynamicPPL.reset, accs) == AccumulatorTuple(
                LogPriorAccumulator(0.0f0), LogLikelihoodAccumulator(0.0f0)
            )
            # Test that the original wasn't modified.
            @test accs == AccumulatorTuple(lp_f32, ll_f32)

            # A map with a closure that changes the types of the accumulators.
            @test map(acc -> promote_for_threadsafe_eval(acc, Float64), accs) ==
                AccumulatorTuple(LogPriorAccumulator(1.0), LogLikelihoodAccumulator(1.0))

            # only apply to a particular accumulator
            @test map_accumulator(DynamicPPL.reset, accs, Val(:LogLikelihood)) ==
                AccumulatorTuple(lp_f32, LogLikelihoodAccumulator(0.0f0))
            @test map_accumulator(
                acc -> promote_for_threadsafe_eval(acc, Float64), accs, Val(:LogLikelihood)
            ) == AccumulatorTuple(lp_f32, LogLikelihoodAccumulator(1.0))
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
