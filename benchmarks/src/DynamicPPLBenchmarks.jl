module DynamicPPLBenchmarks

using DynamicPPL: VarInfo, SimpleVarInfo, VarName
using BenchmarkTools: BenchmarkGroup, @benchmarkable
using DynamicPPL: DynamicPPL
using ADTypes: ADTypes
using LogDensityProblems: LogDensityProblems

# Load some the default backends to trigger conditional loading.
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

include("./Models.jl")
using .Models: Models

export Models, make_suite

# Utility functions for representing AD backends using symbols.
# Copied from TuringBenchmarking.jl.
const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ADTypes.AutoForwardDiff(; chunksize=0),
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
    make_suite(model, varinfo_choice::Symbol, adbackend::Symbol, islinked::Bool)

Create a benchmark suite for `model` using the selected varinfo type and AD backend.
Available varinfo choices:
  • `:untyped`           → uses `VarInfo()`
  • `:typed`             → uses `VarInfo(model)`
  • `:simple_namedtuple` → uses `SimpleVarInfo{Float64}(model())`
  • `:simple_dict`       → builds a `SimpleVarInfo{Float64}` from a Dict (pre-populated with the model’s outputs)

The AD backend should be specified as a Symbol (e.g. `:forwarddiff`, `:reversediff`, `:zygote`).

`islinked` determines whether to link the VarInfo for evaluation.
"""
function make_suite(model, varinfo_choice::Symbol, adbackend::Symbol, islinked::Bool)
    suite = BenchmarkGroup()

    vi = if varinfo_choice == :untyped
        vi = VarInfo()
        model(vi)
        vi
    elseif varinfo_choice == :typed
        VarInfo(model)
    elseif varinfo_choice == :simple_namedtuple
        SimpleVarInfo{Float64}(model())
    elseif varinfo_choice == :simple_dict
        retvals = model()
        vns = [VarName{k}() for k in keys(retvals)]
        SimpleVarInfo{Float64}(Dict(zip(vns, values(retvals))))
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    adbackend = to_backend(adbackend)
    context = DynamicPPL.DefaultContext()

    if islinked
        vi = DynamicPPL.link(vi, model)
    end

    # We construct `LogDensityFunction` using different values
    # than the ones we're going to use for the test. Some of the AD backends
    # compile the tape upon `LogDensityFunction` construction, and we want to
    # evaluate using inputs different from those that the tape was compiled for.
    f = DynamicPPL.LogDensityFunction(model, vi, context; adtype=adbackend)

    # The parameters at which we evaluate f.
    θ = if islinked
        randn(length(vi[:]))
    else
        rand(Vector, model)
    end

    # Run once to trigger compilation.
    LogDensityProblems.logdensity_and_gradient(f, θ)
    suite["gradient"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)

    # Also benchmark just standard model evaluation because why not.
    suite["evaluation"] = @benchmarkable $(DynamicPPL.evaluate!!)($model, $vi, $context)

    return suite
end

end # module
