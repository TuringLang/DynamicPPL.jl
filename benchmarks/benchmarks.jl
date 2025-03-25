using Pkg
# To ensure we benchmark the local version of DynamicPPL, dev the folder above.
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using DynamicPPL: DynamicPPL, make_benchmark_suite, VarInfo
using ADTypes
using BenchmarkTools: @benchmark, median, run
using PrettyTables: PrettyTables, ft_printf
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG

include("Models.jl")

"""
    model_dimension(model, islinked)

Return the dimension of `model`, accounting for linking, if any.
"""
function model_dimension(model, islinked)
    vi = VarInfo()
    model(vi)
    if islinked
        vi = DynamicPPL.link(vi, model)
    end
    return length(vi[:])
end

# Create DynamicPPL.Model instances to run benchmarks on.
smorgasbord_instance = Models.smorgasbord(
    randn(StableRNG(23), 100), randn(StableRNG(23), 100)
)
loop_univariate1k, multivariate1k = begin
    data_1k = randn(StableRNG(23), 1_000)
    loop = Models.loop_univariate(length(data_1k)) | (; o=data_1k)
    multi = Models.multivariate(length(data_1k)) | (; o=data_1k)
    loop, multi
end
loop_univariate10k, multivariate10k = begin
    data_10k = randn(StableRNG(23), 10_000)
    loop = Models.loop_univariate(length(data_10k)) | (; o=data_10k)
    multi = Models.multivariate(length(data_10k)) | (; o=data_10k)
    loop, multi
end
lda_instance = begin
    w = [1, 2, 3, 2, 1, 1]
    d = [1, 1, 1, 2, 2, 2]
    Models.lda(2, d, w)
end

# AD types setup
fd = AutoForwardDiff()
rd = AutoReverseDiff()
mc = AutoMooncake(; config=nothing)
"""
    get_adtype_shortname(adtype::ADTypes.AbstractADType)

Get the package name that corresponds to the the AD backend `adtype`. Only used
for pretty-printing.
"""
get_adtype_shortname(::AutoMooncake) = "Mooncake"
get_adtype_shortname(::AutoForwardDiff) = "ForwardDiff"
get_adtype_shortname(::AutoReverseDiff{false}) = "ReverseDiff"
get_adtype_shortname(::AutoReverseDiff{true}) = "ReverseDiff:Compiled"

# Specify the combinations to test:
# (Model Name, model instance, VarInfo choice, AD backend, linked)
chosen_combinations = [
    (
        "Simple assume observe",
        Models.simple_assume_observe(randn(StableRNG(23))),
        :typed,
        fd,
        false,
    ),
    ("Smorgasbord", smorgasbord_instance, :typed, fd, false),
    ("Smorgasbord", smorgasbord_instance, :simple_namedtuple, fd, true),
    ("Smorgasbord", smorgasbord_instance, :untyped, fd, true),
    ("Smorgasbord", smorgasbord_instance, :simple_dict, fd, true),
    ("Smorgasbord", smorgasbord_instance, :typed, rd, true),
    ("Smorgasbord", smorgasbord_instance, :typed, mc, true),
    ("Loop univariate 1k", loop_univariate1k, :typed, mc, true),
    ("Multivariate 1k", multivariate1k, :typed, mc, true),
    ("Loop univariate 10k", loop_univariate10k, :typed, mc, true),
    ("Multivariate 10k", multivariate10k, :typed, mc, true),
    ("Dynamic", Models.dynamic(), :typed, mc, true),
    ("Submodel", Models.parent(randn(StableRNG(23))), :typed, mc, true),
    ("LDA", lda_instance, :typed, rd, true),
]

# Time running a model-like function that does not use DynamicPPL, as a reference point.
# Eval timings will be relative to this.
reference_time = begin
    obs = randn(StableRNG(23))
    median(@benchmark Models.simple_assume_observe_non_model(obs)).time
end

results_table = Tuple{String,Int,String,String,Bool,Float64,Float64}[]

for (model_name, model, varinfo_choice, adbackend, islinked) in chosen_combinations
    @info "Running benchmark for $model_name / $varinfo_choice / $(get_adtype_shortname(adbackend))"
    suite = make_benchmark_suite(StableRNG(23), model, varinfo_choice, adbackend, islinked)
    results = run(suite)
    eval_time = median(results["evaluation"]).time
    relative_eval_time = eval_time / reference_time
    ad_eval_time = median(results["gradient"]).time
    relative_ad_eval_time = ad_eval_time / eval_time
    push!(
        results_table,
        (
            model_name,
            model_dimension(model, islinked),
            get_adtype_shortname(adbackend),
            string(varinfo_choice),
            islinked,
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
    "Linked",
    "Eval Time / Ref Time",
    "AD Time / Eval Time",
]
PrettyTables.pretty_table(
    table_matrix;
    header=header,
    tf=PrettyTables.tf_markdown,
    formatters=ft_printf("%.1f", [6, 7]),
)
