module DynamicPPLBenchmarks

using DynamicPPL: VarInfo, SimpleVarInfo, VarName
using BenchmarkTools: BenchmarkGroup
using TuringBenchmarking: make_turing_suite

export make_suite

"""
    make_suite(model, varinfo_choice::Symbol, adbackend::Symbol)

Create a benchmark suite for `model` using the selected varinfo type and AD backend.
Available varinfo choices:
  • `:untyped`           → uses `VarInfo()`
  • `:typed`             → uses `VarInfo(model)`
  • `:simple_namedtuple` → uses `SimpleVarInfo{Float64}(free_nt)`
  • `:simple_dict`       → builds a `SimpleVarInfo{Float64}` from a Dict (pre-populated with the model’s outputs)

The AD backend should be specified as a Symbol (e.g. `:forwarddiff`, `:reversediff`, `:zygote`).
"""
function make_suite(model, varinfo_choice::Symbol, adbackend::Symbol)
    suite = BenchmarkGroup()

    vi = if varinfo_choice == :untyped
        v = VarInfo()
        model(v)
        v
    elseif varinfo_choice == :typed
        VarInfo(model)
    elseif varinfo_choice == :simple_namedtuple
        free_nt = NamedTuple{(:m,)}(model())  # Extract only the free parameter(s)
        SimpleVarInfo{Float64}(free_nt)
    elseif varinfo_choice == :simple_dict
        retvals = model()
        vns = [VarName{k}() for k in keys(retvals)]
        SimpleVarInfo{Float64}(Dict(zip(vns, values(retvals))))
    else
        error("Unknown varinfo choice: $varinfo_choice")
    end

    # Add the AD benchmarking suite.
    suite["AD_Benchmarking"] = make_turing_suite(
        model;
        adbackends=[adbackend],
        varinfo=vi,
        check_grads=true,
        error_on_failed_backend=true,
    )

    return suite
end

end # module
