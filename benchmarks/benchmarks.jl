using DynamicPPLBenchmarks: Models, make_suite
using BenchmarkTools: @benchmark, median, run
using PrettyTables: PrettyTables, ft_printf
using Random: seed!

seed!(23)

# Create DynamicPPL.Model instances to run benchmarks on.
smorgasbord_instance = Models.smorgasbord(randn(100), randn(100))
loop_univariate1k, multivariate1k = begin
    data_1k = randn(1_000)
    loop = Models.loop_univariate(length(data_1k)) | (; o=data_1k)
    multi = Models.multivariate(length(data_1k)) | (; o=data_1k)
    loop, multi
end
loop_univariate10k, multivariate10k = begin
    data_10k = randn(10_000)
    loop = Models.loop_univariate(length(data_10k)) | (; o=data_10k)
    multi = Models.multivariate(length(data_10k)) | (; o=data_10k)
    loop, multi
end
lda_instance = begin
    w = [1, 2, 3, 2, 1, 1]
    d = [1, 1, 1, 2, 2, 2]
    Models.lda(2, d, w)
end

# Specify the combinations to test:
# (Model Name, model instance, VarInfo choice, AD backend)
chosen_combinations = [
    ("Simple assume observe", Models.simple_assume_observe(randn()), :typed, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :typed, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :simple_namedtuple, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :untyped, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :simple_dict, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :typed, :reversediff),
    # TODO(mhauru) Add Mooncake once TuringBenchmarking.jl supports it. Consider changing
    # all the below :reversediffs to :mooncakes too.
    #("Smorgasbord", smorgasbord_instance, :typed, :mooncake),
    ("Loop univariate 1k", loop_univariate1k, :typed, :reversediff),
    ("Multivariate 1k", multivariate1k, :typed, :reversediff),
    ("Loop univariate 10k", loop_univariate10k, :typed, :reversediff),
    ("Multivariate 10k", multivariate10k, :typed, :reversediff),
    ("Dynamic", Models.dynamic(), :typed, :reversediff),
    ("Submodel", Models.parent(randn()), :typed, :reversediff),
    ("LDA", lda_instance, :typed, :reversediff),
]

# Time running a model-like function that does not use DynamicPPL, as a reference point.
# Eval timings will be relative to this.
reference_time = begin
    obs = randn()
    median(@benchmark Models.simple_assume_observe_non_model(obs)).time
end

results_table = Tuple{String,String,String,Float64,Float64}[]

for (model_name, model, varinfo_choice, adbackend) in chosen_combinations
    suite = make_suite(model, varinfo_choice, adbackend)
    results = run(suite)

    eval_time = median(results["evaluation"]["standard"]).time
    relative_eval_time = eval_time / reference_time

    grad_group = results["gradient"]
    if isempty(grad_group)
        relative_ad_eval_time = NaN
    else
        grad_backend_key = first(keys(grad_group))
        ad_eval_time = median(grad_group[grad_backend_key]["standard"]).time
        relative_ad_eval_time = ad_eval_time / eval_time
    end

    push!(
        results_table,
        (
            model_name,
            string(adbackend),
            string(varinfo_choice),
            relative_eval_time,
            relative_ad_eval_time,
        ),
    )
end

table_matrix = hcat(Iterators.map(collect, zip(results_table...))...)
header = [
    "Model",
    "AD Backend",
    "VarInfo Type",
    "Evaluation Time / Reference Time",
    "AD Time / Eval Time",
]
PrettyTables.pretty_table(
    table_matrix;
    header=header,
    tf=PrettyTables.tf_markdown,
    formatters=ft_printf("%.1f", [4, 5]),
)
