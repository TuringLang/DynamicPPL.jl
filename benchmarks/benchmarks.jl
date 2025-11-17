using Pkg

using Chairmarks: @be, median
using DynamicPPLBenchmarks: Models, benchmark, model_dimension
using JSON: JSON
using PrettyTables: pretty_table, fmt__printf, EmptyCells, MultiColumn, TextTableFormat
using Printf: @sprintf
using StableRNGs: StableRNG

rng = StableRNG(23)

head_filename = "benchmarks_result_head.json"
base_filename = "benchmarks_result_base.json"

colnames = [
    "Model", "Dim", "AD Backend", "VarInfo", "Linked", "t(eval)/t(ref)", "t(grad)/t(eval)"
]
function print_results(results_table; to_json=false)
    if to_json
        # Print to the given file as JSON
        results_array = [
            Dict(colnames[i] => results_table[j][i] for i in eachindex(colnames)) for
            j in eachindex(results_table)
        ]
        # do not use pretty=true, as GitHub Actions expects no linebreaks
        JSON.json(stdout, results_array)
    else
        # Pretty-print to terminal
        table_matrix = hcat(Iterators.map(collect, zip(results_table...))...)
        return pretty_table(
            table_matrix;
            column_labels=colnames,
            backend=:text,
            formatters=[fmt__printf("%.1f", [6, 7])],
            fit_table_in_display_horizontally=false,
            fit_table_in_display_vertically=false,
        )
    end
end

function run(; to_json=false)
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
            false,
        ),
        ("Smorgasbord", smorgasbord_instance, :typed, :forwarddiff, false),
        ("Smorgasbord", smorgasbord_instance, :simple_namedtuple, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :untyped, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :simple_dict, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :typed_vector, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :untyped_vector, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :typed, :reversediff, true),
        ("Smorgasbord", smorgasbord_instance, :typed, :mooncake, true),
        ("Smorgasbord", smorgasbord_instance, :typed, :enzyme, true),
        ("Loop univariate 1k", loop_univariate1k, :typed, :mooncake, true),
        ("Multivariate 1k", multivariate1k, :typed, :mooncake, true),
        ("Loop univariate 10k", loop_univariate10k, :typed, :mooncake, true),
        ("Multivariate 10k", multivariate10k, :typed, :mooncake, true),
        ("Dynamic", Models.dynamic(), :typed, :mooncake, true),
        ("Submodel", Models.parent(randn(rng)), :typed, :mooncake, true),
        ("LDA", lda_instance, :typed, :reversediff, true),
    ]

    # Time running a model-like function that does not use DynamicPPL, as a reference point.
    # Eval timings will be relative to this.
    reference_time = begin
        obs = randn(rng)
        median(@be Models.simple_assume_observe_non_model(obs)).time
    end
    @info "Reference evaluation time: $(reference_time) seconds"

    results_table = Tuple{
        String,Int,String,String,Bool,Union{Float64,Missing},Union{Float64,Missing}
    }[]

    for (model_name, model, varinfo_choice, adbackend, islinked) in chosen_combinations
        @info "Running benchmark for $model_name"
        relative_eval_time, relative_ad_eval_time = try
            results = benchmark(model, varinfo_choice, adbackend, islinked)
            (results.primal_time / reference_time),
            (results.grad_time / results.primal_time)
        catch e
            missing, missing
        end
        push!(
            results_table,
            (
                model_name,
                model_dimension(model, islinked),
                string(adbackend),
                string(varinfo_choice),
                islinked,
                relative_eval_time,
                relative_ad_eval_time,
            ),
        )
        print_results(results_table; to_json=to_json)
    end
    return print_results(results_table; to_json=to_json)
end

struct TestCase
    model_name::String
    dim::Integer
    ad_backend::String
    varinfo::String
    linked::Bool
    TestCase(d::Dict{String,Any}) = new((d[c] for c in colnames[1:5])...)
end
function combine()
    head_results = try
        JSON.parsefile(head_filename, Vector{Dict{String,Any}})
    catch
        Dict{String,Any}[]
    end
    base_results = try
        JSON.parsefile(base_filename, Vector{Dict{String,Any}})
    catch
        Dict{String,Any}[]
    end
    # Identify unique combinations of (Model, Dim, AD Backend, VarInfo, Linked)
    head_testcases = Dict(
        TestCase(d) => (d[colnames[6]], d[colnames[7]]) for d in head_results
    )
    base_testcases = Dict(
        TestCase(d) => (d[colnames[6]], d[colnames[7]]) for d in base_results
    )
    all_testcases = union(Set(keys(head_testcases)), Set(keys(base_testcases)))
    sorted_testcases = sort(
        collect(all_testcases); by=(c -> (c.model_name, c.ad_backend, c.varinfo, c.linked))
    )
    results_table = Tuple{
        String,Int,String,String,Bool,String,String,String,String,String,String
    }[]
    results_colnames = [
        [
            EmptyCells(5),
            MultiColumn(3, "t(eval) / t(ref)"),
            MultiColumn(3, "t(grad) / t(eval)"),
        ],
        [colnames[1:5]..., "base", "this PR", "speedup", "base", "this PR", "speedup"],
    ]
    sprint_float(x::Float64) = @sprintf("%.2f", x)
    sprint_float(m::Missing) = "err"
    for c in sorted_testcases
        head_eval, head_grad = get(head_testcases, c, (missing, missing))
        base_eval, base_grad = get(base_testcases, c, (missing, missing))
        speedup_eval = base_eval / head_eval
        speedup_grad = base_grad / head_grad
        push!(
            results_table,
            (
                c.model_name,
                c.dim,
                c.ad_backend,
                c.varinfo,
                c.linked,
                sprint_float(base_eval),
                sprint_float(head_eval),
                sprint_float(speedup_eval),
                sprint_float(base_grad),
                sprint_float(head_grad),
                sprint_float(speedup_grad),
            ),
        )
    end
    # Pretty-print to terminal
    table_matrix = hcat(Iterators.map(collect, zip(results_table...))...)
    return pretty_table(
        table_matrix;
        column_labels=results_colnames,
        backend=:text,
        fit_table_in_display_horizontally=false,
        fit_table_in_display_vertically=false,
        table_format=TextTableFormat(; horizontal_line_at_merged_column_labels=true),
    )
end

# The command-line arguments are used on CI purposes.
# Run with `julia --project=. benchmarks.jl [combine|json-head|json-base]`
if ARGS == ["combine"]
    combine()
elseif ARGS == ["json"]
    run(; to_json=true)
elseif ARGS == []
    # When running locally just omit the argument and it will just benchmark and print to
    # terminal.
    run()
end
