module DynamicPPLBenchmarks

using DynamicPPL: Model, VarInfo, SimpleVarInfo
using DynamicPPL: DynamicPPL
using ADTypes: ADTypes

using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG

include("./Models.jl")
using .Models: Models

export Models, to_backend, make_varinfo

# Utility functions for representing AD backends using symbols.
# Copied from TuringBenchmarking.jl.
const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ADTypes.AutoForwardDiff(),
    :reversediff => ADTypes.AutoReverseDiff(; compile=false),
    :reversediff_compiled => ADTypes.AutoReverseDiff(; compile=true),
    :mooncake => ADTypes.AutoMooncake(; config=nothing),
)

to_backend(x) = error("Unknown backend: $x")
to_backend(x::ADTypes.AbstractADType) = x
function to_backend(x::Union{AbstractString,Symbol})
    k = Symbol(lowercase(string(x)))
    haskey(SYMBOL_TO_BACKEND, k) || error("Unknown backend: $x")
    return SYMBOL_TO_BACKEND[k]
end

"""
    make_varinfo(model, varinfo_choice::Symbol)

Create a VarInfo for the given `model` using the selected varinfo type.
Available varinfo choices:
  • `:untyped`           → uses `DynamicPPL.untyped_varinfo(model)`
  • `:typed`             → uses `DynamicPPL.typed_varinfo(model)`
  • `:simple_namedtuple` → uses `SimpleVarInfo{Float64}(model())`
  • `:simple_dict`       → builds a `SimpleVarInfo{Float64}` from a Dict (pre-populated with the model’s outputs)

The VarInfo is always linked.
"""
function make_varinfo(model::Model, varinfo_choice::Symbol, adbackend::Symbol)
    rng = StableRNG(23)

    vi = if varinfo_choice == :untyped
        DynamicPPL.untyped_varinfo(rng, model)
    elseif varinfo_choice == :typed
        DynamicPPL.typed_varinfo(rng, model)
    elseif varinfo_choice == :simple_namedtuple
        SimpleVarInfo{Float64}(model(rng))
    elseif varinfo_choice == :simple_dict
        vi = DynamicPPL.typed_varinfo(rng, model)
        vals = DynamicPPL.values_as(vi, Dict)
        SimpleVarInfo{Float64}(vals)
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    return DynamicPPL.link!!(vi, model)
end

end # module
