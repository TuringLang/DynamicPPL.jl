module DynamicPPLBenchmarks

using DynamicPPL: VarInfo, VarName
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
    vi = VarInfo()
    model(StableRNG(23), vi)
    if islinked
        vi = DynamicPPL.link(vi, model)
    end
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
    benchmark(model, varinfo_choice::Symbol, adbackend::Symbol, islinked::Bool)

Benchmark evaluation and gradient calculation for `model` using the selected varinfo type
and AD backend.

Available varinfo choices:
  • `:untyped`           → uses `DynamicPPL.untyped_varinfo(model)`
  • `:typed`             → uses `DynamicPPL.typed_varinfo(model)`

The AD backend should be specified as a Symbol (e.g. `:forwarddiff`, `:reversediff`, `:zygote`).

`islinked` determines whether to link the VarInfo for evaluation.
"""
function benchmark(model, varinfo_choice::Symbol, adbackend::Symbol, islinked::Bool)
    rng = StableRNG(23)

    adbackend = to_backend(adbackend)

    vi = if varinfo_choice == :untyped
        DynamicPPL.untyped_varinfo(rng, model)
    elseif varinfo_choice == :typed
        DynamicPPL.typed_varinfo(rng, model)
    elseif varinfo_choice == :typed_vector
        DynamicPPL.typed_vector_varinfo(rng, model)
    elseif varinfo_choice == :untyped_vector
        DynamicPPL.untyped_vector_varinfo(rng, model)
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    adbackend = to_backend(adbackend)

    if islinked
        vi = DynamicPPL.link(vi, model)
    end

    return run_ad(
        model, adbackend; varinfo=vi, benchmark=true, test=NoTest(), verbose=false
    )
end

end # module
