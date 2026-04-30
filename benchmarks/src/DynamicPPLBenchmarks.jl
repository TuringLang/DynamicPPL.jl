module DynamicPPLBenchmarks

using DynamicPPL: VarInfo, VarName, LinkAll, UnlinkAll
using DynamicPPL: DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad, NoTest
using ADTypes: ADTypes
using LogDensityProblems: LogDensityProblems

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
using Enzyme: Enzyme
using StableRNGs: StableRNG

include("./Models.jl")
using .Models: Models
export Models, benchmark, model_dimension

"""
    model_dimension(model, islinked)

Return the dimension of `model`, accounting for linking, if any.
"""
function model_dimension(model, islinked)
    tfm_strategy = islinked ? DynamicPPL.LinkAll() : DynamicPPL.UnlinkAll()
    vi = last(
        DynamicPPL.init!!(
            StableRNG(23), model, VarInfo(), DynamicPPL.InitFromPrior(), tfm_strategy
        ),
    )
    return length(vi[:])
end

# Utility functions for representing AD backends using symbols.
# Copied from TuringBenchmarking.jl.
const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ADTypes.AutoForwardDiff(),
    :reversediff => ADTypes.AutoReverseDiff(; compile=false),
    :reversediff_compiled => ADTypes.AutoReverseDiff(; compile=true),
    :mooncake => ADTypes.AutoMooncake(; config=nothing),
    :enzyme => ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    ),
)

to_backend(x) = error("Unknown backend: $x")
to_backend(x::ADTypes.AbstractADType) = x
function to_backend(x::Union{AbstractString,Symbol})
    k = Symbol(lowercase(string(x)))
    haskey(SYMBOL_TO_BACKEND, k) || error("Unknown backend: $x")
    return SYMBOL_TO_BACKEND[k]
end

"""
    benchmark(model, adbackend::Symbol, islinked::Bool; seconds::Real=2)

Benchmark log-density evaluation and gradient calculation for `model` using the
selected AD backend.

`adbackend` is a Symbol key into `SYMBOL_TO_BACKEND` (e.g. `:forwarddiff`,
`:reversediff`, `:reversediff_compiled`, `:mooncake`, `:enzyme`).

`islinked` determines whether to link the VarInfo for evaluation.

`seconds` is the per-measurement time budget passed to Chairmarks; the default
doubles Chairmarks' own default to tighten the median estimate.
"""
function benchmark(model, adbackend::Symbol, islinked::Bool; seconds::Real=2)
    transform_strategy = islinked ? LinkAll() : UnlinkAll()
    return run_ad(
        model,
        to_backend(adbackend);
        rng=StableRNG(23),
        transform_strategy,
        benchmark=true,
        benchmark_seconds=seconds,
        test=NoTest(),
        verbose=false,
    )
end

end
