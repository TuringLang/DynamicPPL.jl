using DynamicPPL
using DynamicPPLBenchmarks
using BenchmarkTools
using TuringBenchmarking
using Distributions
using PrettyTables

# Define models
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

demo1_data = randn()
demo2_data = rand(Bool, 10)

# Create model instances with the data
demo1_instance = demo1(demo1_data)
demo2_instance = demo2(demo2_data)

# Define available AD backends
available_ad_backends = Dict(
    :forwarddiff => :forwarddiff,
    :reversediff => :reversediff,
    :zygote      => :zygote
)

# Define available VarInfo types.
# Each entry is (Name, function to produce the VarInfo)
available_varinfo_types = Dict(
    :untyped           => ("UntypedVarInfo", VarInfo),
    :typed             => ("TypedVarInfo", m -> VarInfo(m)),
    :simple_namedtuple => ("SimpleVarInfo (NamedTuple)", m -> SimpleVarInfo{Float64}(m())),
    :simple_dict       => ("SimpleVarInfo (Dict)", m -> begin
        retvals = m()
        varnames = map(keys(retvals)) do k
            VarName{k}()
        end
        SimpleVarInfo{Float64}(Dict(zip(varnames, values(retvals))))
    end)
)

# Specify the combinations to test:
# (Model Name, model instance, VarInfo choice, AD backend)
chosen_combinations = [
    ("Demo1", demo1_instance, :typed,           :forwarddiff),
    ("Demo1", demo1_instance, :simple_namedtuple, :zygote),
    ("Demo2", demo2_instance, :untyped,           :reversediff),
    ("Demo2", demo2_instance, :simple_dict,       :forwarddiff)
]

# Store results as tuples: (Model, AD Backend, VarInfo Type, Eval Time, AD Eval Time)
results_table = Tuple{String, String, String, Float64, Float64}[]

for (model_name, model, varinfo_choice, adbackend) in chosen_combinations
    suite = make_suite(model, varinfo_choice, adbackend)
    results = run(suite)
    eval_time    = median(results["evaluation"]).time
    ad_eval_time = median(results["AD_Benchmarking"]["evaluation"]["standard"]).time
    push!(results_table, (model_name, string(adbackend), string(varinfo_choice), eval_time, ad_eval_time))
end

# Convert results to a 2D array for PrettyTables
function to_matrix(tuples::Vector{<:NTuple{5,Any}})
    n = length(tuples)
    data = Array{Any}(undef, n, 5)
    for i in 1:n
        for j in 1:5
            data[i, j] = tuples[i][j]
        end
    end
    return data
end

table_matrix = to_matrix(results_table)
header = ["Model", "AD Backend", "VarInfo Type", "Evaluation Time (ns)", "AD Eval Time (ns)"]
pretty_table(table_matrix; header=header, tf=PrettyTables.tf_markdown)
