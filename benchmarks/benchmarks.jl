using DynamicPPLBenchmarks: Models, to_backend, make_varinfo
using DynamicPPL.TestUtils.AD: run_ad, NoTest
using Chairmarks: @be
using PrettyTables: PrettyTables, ft_printf
using StableRNGs: StableRNG
using Statistics: median

rng = StableRNG(23)

# Create DynamicPPL.Model instances to run benchmarks on.
smorgasbord_instance = Models.smorgasbord(randn(rng, 100), randn(rng, 100))
loop_univariate1k, multivariate1k = begin
    data_1k = randn(rng, 1_000)
    loop = Models.loop_univariate(length(data_1k)) | (; o=data_1k)
    multi = Models.multivariate(length(data_1k)) | (; o=data_1k)
    loop, multi
end
loop_univariate10k, multivariate10k = begin
    data_10k = randn(rng, 10_000)
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
# (Model Name, model instance, VarInfo choice, AD backend, linked)
chosen_combinations = [
    (
        "Simple assume observe",
        Models.simple_assume_observe(randn(rng)),
        :typed,
        :forwarddiff,
    ),
    ("Smorgasbord", smorgasbord_instance, :typed, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :simple_namedtuple, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :untyped, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :simple_dict, :forwarddiff),
    ("Smorgasbord", smorgasbord_instance, :typed, :reversediff),
    ("Smorgasbord", smorgasbord_instance, :typed, :mooncake),
    ("Loop univariate 1k", loop_univariate1k, :typed, :mooncake),
    ("Multivariate 1k", multivariate1k, :typed, :mooncake),
    ("Loop univariate 10k", loop_univariate10k, :typed, :mooncake),
    ("Multivariate 10k", multivariate10k, :typed, :mooncake),
    ("Dynamic", Models.dynamic(), :typed, :mooncake),
    ("Submodel", Models.parent(randn(rng)), :typed, :mooncake),
    ("LDA", lda_instance, :typed, :reversediff),
]

# Time running a model-like function that does not use DynamicPPL, as a reference point.
# Eval timings will be relative to this.
reference_time = begin
    obs = randn(rng)
    median(@be Models.simple_assume_observe_non_model(obs)).time
end

results_table = Tuple{String,Int,String,String,Float64,Float64}[]

for (model_name, model, varinfo_choice, adbackend) in chosen_combinations
    @info "Running benchmark for $model_name"
    adtype = to_backend(adbackend)
    varinfo = make_varinfo(model, varinfo_choice)
    ad_result = run_ad(model, adtype; test=NoTest(), benchmark=true, varinfo=varinfo)
    relative_eval_time = ad_result.primal_time / reference_time
    relative_ad_eval_time = ad_result.grad_time / ad_result.primal_time
    push!(
        results_table,
        (
            model_name,
            length(varinfo[:]),
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
    "Dimension",
    "AD Backend",
    "VarInfo Type",
    "Eval Time / Ref Time",
    "AD Time / Eval Time",
]
PrettyTables.pretty_table(
    table_matrix;
    header=header,
    tf=PrettyTables.tf_markdown,
    formatters=ft_printf("%.1f", [5, 6]),
    crop=:none,  # Always print the whole table, even if it doesn't fit in the terminal.
)
