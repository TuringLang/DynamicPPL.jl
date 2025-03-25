module DynamicPPLBenchmarkToolsExt

using DynamicPPL:
    DynamicPPL, ADTypes, LogDensityProblems, Model, VarInfo, SimpleVarInfo, VarName
using BenchmarkTools: BenchmarkGroup, @benchmarkable
using Random: Random

"""
    make_benchmark_suite(
        [rng::Random.AbstractRNG,]
        model::Model,
        varinfo_choice::Symbol,
        adtype::ADTypes.AbstractADType,
        islinked::Bool
    )

Create a benchmark suite for `model` using the selected varinfo type and AD backend.
Available varinfo choices:
  • `:untyped`           → uses `VarInfo()`
  • `:typed`             → uses `VarInfo(model)`
  • `:simple_namedtuple` → uses `SimpleVarInfo{Float64}(model())`
  • `:simple_dict`       → builds a `SimpleVarInfo{Float64}` from a Dict (pre-populated with the model’s outputs)

`islinked` determines whether to link the VarInfo for evaluation.
"""
function make_benchmark_suite(
    rng::Random.AbstractRNG,
    model::Model,
    varinfo_choice::Symbol,
    adtype::ADTypes.AbstractADType,
    islinked::Bool,
)
    suite = BenchmarkGroup()

    vi = if varinfo_choice == :untyped
        vi = VarInfo()
        model(rng, vi)
        vi
    elseif varinfo_choice == :typed
        VarInfo(rng, model)
    elseif varinfo_choice == :simple_namedtuple
        SimpleVarInfo{Float64}(model(rng))
    elseif varinfo_choice == :simple_dict
        retvals = model(rng)
        vns = [VarName{k}() for k in keys(retvals)]
        SimpleVarInfo{Float64}(Dict(zip(vns, values(retvals))))
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    context = DynamicPPL.DefaultContext()

    if islinked
        vi = DynamicPPL.link(vi, model)
    end

    f = DynamicPPL.LogDensityFunction(model, vi, context; adtype=adtype)
    # The parameters at which we evaluate f.
    θ = vi[:]

    # Run once to trigger compilation.
    LogDensityProblems.logdensity_and_gradient(f, θ)
    suite["gradient"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)

    # Also benchmark just standard model evaluation because why not.
    suite["evaluation"] = @benchmarkable $(LogDensityProblems.logdensity)($f, $θ)

    return suite
end
function make_benchmark_suite(
    model::Model, varinfo_choice::Symbol, adtype::Symbol, islinked::Bool
)
    return make_benchmark_suite(
        Random.default_rng(), model, varinfo_choice, adtype, islinked
    )
end

end
