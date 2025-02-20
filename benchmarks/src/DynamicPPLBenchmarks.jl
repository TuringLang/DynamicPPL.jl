module DynamicPPLBenchmarks

using DynamicPPL
using BenchmarkTools
using TuringBenchmarking: make_turing_suite

export make_suite

"""
    make_suite(model, varinfo_choice::Symbol, adbackend::Symbol)

Create a benchmark suite for `model` using the selected varinfo type and AD backend.
Available varinfo choices:
  • `:untyped`           → uses `VarInfo()`
  • `:typed`             → uses `VarInfo(model)`
  • `:simple_namedtuple` → uses `SimpleVarInfo{Float64}(model())`
  • `:simple_dict`       → builds a `SimpleVarInfo{Float64}` from a Dict (pre-populated with the model’s outputs)

The AD backend should be specified as a Symbol (e.g. `:forwarddiff`, `:reversediff`, `:zygote`).
"""
function make_suite(model, varinfo_choice::Symbol, adbackend::Symbol)
    suite = BenchmarkGroup()
    context = DefaultContext()
    
    # Create the chosen varinfo.
    vi = nothing
    if varinfo_choice == :untyped
        vi = VarInfo()
        model(vi)
    elseif varinfo_choice == :typed
        vi = VarInfo(model)
    elseif varinfo_choice == :simple_namedtuple
        vi = SimpleVarInfo{Float64}(model())
    elseif varinfo_choice == :simple_dict
        retvals = model()
        vns = map(keys(retvals)) do k
            VarName{k}()
        end
        vi = SimpleVarInfo{Float64}(Dict(zip(vns, values(retvals))))
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    # Add the evaluation benchmark.
    suite["evaluation"] = @benchmarkable $model($vi, $context)
    
    # Add the AD benchmarking suite.
    suite["AD_Benchmarking"] = make_turing_suite(model; adbackends=[adbackend])
    
    return suite
end

end # module
