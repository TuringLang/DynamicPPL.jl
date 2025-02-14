using BenchmarkTools
using DynamicPPL
using Distributions
using DynamicPPLBenchmarks: time_model_def, make_suite
using PrettyTables
using Dates
using LibGit2

const RESULTS_DIR = "results"
const BENCHMARK_NAME = let
    repo = try
        LibGit2.GitRepo(joinpath(pkgdir(DynamicPPL), ".."))
    catch
        nothing
    end
    isnothing(repo) ? "benchmarks_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))" :
    "$(LibGit2.headname(repo))_$(string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))))[1:6])"
end

mkpath(joinpath(RESULTS_DIR, BENCHMARK_NAME))

@model function demo1(x)
    m ~ Normal()
    x ~ Normal(m, 1)
    return (m = m, x = x)
end

@model function demo2(y)
    p ~ Beta(1, 1)
    N = length(y)
    for n in 1:N
        y[n] ~ Bernoulli(p)
    end
    return (; p)
end

models = [
    (name = "demo1", model = demo1, data = (1.0,)),
    (name = "demo2", model = demo2, data = (rand(0:1, 10),))
]

results = []
for (model_name, model_def, data) in models
    println(">> Running benchmarks for model: $model_name")
    m = time_model_def(model_def, data...)
    println()
    suite = make_suite(m)
    bench_results = run(suite, seconds=10)
    
    output_path = joinpath(RESULTS_DIR, BENCHMARK_NAME, "$(model_name)_benchmarks.json")
    BenchmarkTools.save(output_path, bench_results)
    
    for (eval_type, trial) in bench_results
        push!(results, (
            Model = model_name,
            Evaluation = eval_type,
            Time = minimum(trial).time,
            Memory = trial.memory,
            Allocations = trial.allocs,
            Samples = length(trial.times)
        ))
    end
end

formatted = map(results) do r
    (Model = r.Model,
     Evaluation = replace(r.Evaluation, "_" => " "),
     Time = BenchmarkTools.prettytime(r.Time),
     Memory = BenchmarkTools.prettymemory(r.Memory),
     Allocations = string(r.Allocations),
     Samples = string(r.Samples))
end

md_output = """
## DynamicPPL Benchmark Results ($BENCHMARK_NAME)

### Execution Environment
- Julia version: $(VERSION)
- DynamicPPL version: $(pkgversion(DynamicPPL))
- Benchmark date: $(now())

$(pretty_table(String, formatted,
    tf = tf_markdown,
    header = ["Model", "Evaluation Type", "Time", "Memory", "Allocs", "Samples"],
    alignment = [:l, :l, :r, :r, :r, :r]
))
"""

println(md_output)
open(joinpath(RESULTS_DIR, BENCHMARK_NAME, "REPORT.md"), "w") do io
    write(io, md_output)
end

println("Benchmark results saved to: $RESULTS_DIR/$BENCHMARK_NAME")