using DynamicPPL: @model
using DynamicPPLBenchmarks: make_suite
using BenchmarkTools: median, run
using Distributions: Normal, Beta, Bernoulli
using PrettyTables: pretty_table, PrettyTables

# Define models
@model function demo1(x)
    m ~ Normal()
    x ~ Normal(m, 1)
    return (m=m, x=x)
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

# Specify the combinations to test:
# (Model Name, model instance, VarInfo choice, AD backend)
chosen_combinations = [
    ("Demo1", demo1_instance, :typed, :forwarddiff),
    ("Demo1", demo1_instance, :simple_namedtuple, :zygote),
    ("Demo2", demo2_instance, :untyped, :reversediff),
    ("Demo2", demo2_instance, :simple_dict, :forwarddiff),
]

results_table = Tuple{String,String,String,Float64,Float64}[]

for (model_name, model, varinfo_choice, adbackend) in chosen_combinations
    suite = make_suite(model, varinfo_choice, adbackend)
    results = run(suite)

    eval_time = median(results["AD_Benchmarking"]["evaluation"]["standard"]).time

    grad_group = results["AD_Benchmarking"]["gradient"]
    if isempty(grad_group)
        ad_eval_time = NaN
    else
        grad_backend_key = first(keys(grad_group))
        ad_eval_time = median(grad_group[grad_backend_key]["standard"]).time
    end

    push!(
        results_table,
        (model_name, string(adbackend), string(varinfo_choice), eval_time, ad_eval_time),
    )
end

table_matrix = hcat(Iterators.map(collect, zip(results_table...))...)
header = [
    "Model", "AD Backend", "VarInfo Type", "Evaluation Time (ns)", "AD Eval Time (ns)"
]
pretty_table(table_matrix; header=header, tf=PrettyTables.tf_markdown)
