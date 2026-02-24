using Pkg

using Chairmarks: @be, median
using DynamicPPLBenchmarks: Models, benchmark, model_dimension
using JSON: JSON
using PrettyTables: pretty_table, fmt__printf, EmptyCells, MultiColumn, TextTableFormat
using Printf: @sprintf
using StableRNGs: StableRNG

rng = StableRNG(23)

colnames = ["Model", "Dim", "AD Backend", "Linked", "t(eval)/t(ref)", "t(grad)/t(eval)"]
function print_results(results_table; to_json=false)
    if to_json
        # Print to the given file as JSON
        results_array = [
            Dict(colnames[i] => results_table[j][i] for i in eachindex(colnames)) for
            j in eachindex(results_table)
        ]
        # do not use pretty=true, as GitHub Actions expects no linebreaks
        JSON.json(stdout, results_array)
        println()
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
    # (Model Name, model instance, AD backend, linked)
    chosen_combinations = [
        (
            "Simple assume observe",
            Models.simple_assume_observe(randn(rng)),
            :forwarddiff,
            false,
        ),
        ("Smorgasbord", smorgasbord_instance, :forwarddiff, false),
        ("Smorgasbord", smorgasbord_instance, :forwarddiff, true),
        ("Smorgasbord", smorgasbord_instance, :reversediff, true),
        ("Smorgasbord", smorgasbord_instance, :mooncake, true),
        ("Smorgasbord", smorgasbord_instance, :enzyme, true),
        ("Loop univariate 1k", loop_univariate1k, :mooncake, true),
        ("Multivariate 1k", multivariate1k, :mooncake, true),
        ("Loop univariate 10k", loop_univariate10k, :mooncake, true),
        ("Multivariate 10k", multivariate10k, :mooncake, true),
        ("Dynamic", Models.dynamic(), :mooncake, true),
        ("Submodel", Models.parent(randn(rng)), :mooncake, true),
        ("LDA", lda_instance, :reversediff, true),
    ]

    # Time running a model-like function that does not use DynamicPPL, as a reference point.
    # Eval timings will be relative to this.
    reference_time = begin
        obs = randn(rng)
        median(@be Models.simple_assume_observe_non_model(obs)).time
    end
    @info "Reference evaluation time: $(reference_time) seconds"

    results_table = Tuple{
        String,Int,String,Bool,Union{Float64,Missing},Union{Float64,Missing}
    }[]

    for (model_name, model, adbackend, islinked) in chosen_combinations
        @info "Running benchmark for $model_name, $adbackend, $islinked"
        relative_eval_time, relative_ad_eval_time = try
            results = benchmark(model, adbackend, islinked)
            @info " t(eval) = $(results.primal_time)"
            @info " t(grad) = $(results.grad_time)"
            (results.primal_time / reference_time),
            (results.grad_time / results.primal_time)
        catch e
            @info "benchmark errored: $e"
            missing, missing
        end
        push!(
            results_table,
            (
                model_name,
                model_dimension(model, islinked),
                string(adbackend),
                islinked,
                relative_eval_time,
                relative_ad_eval_time,
            ),
        )
        print_results(results_table; to_json=to_json)
    end
    print_results(results_table; to_json=to_json)
    return nothing
end

struct TestCase
    model_name::String
    dim::Integer
    ad_backend::String
    linked::Bool
    TestCase(d::Dict{String,Any}) = new((d[c] for c in colnames[1:4])...)
end
function combine(head_filename::String, base_filename::String)
    head_results = try
        JSON.parsefile(head_filename, Vector{Dict{String,Any}})
    catch
        Dict{String,Any}[]
    end
    @info "Loaded $(length(head_results)) results from $head_filename"
    base_results = try
        JSON.parsefile(base_filename, Vector{Dict{String,Any}})
    catch
        Dict{String,Any}[]
    end
    @info "Loaded $(length(base_results)) results from $base_filename"
    # Identify unique combinations of (Model, Dim, AD Backend, Linked)
    head_testcases = Dict(
        TestCase(d) => (d[colnames[5]], d[colnames[6]]) for d in head_results
    )
    base_testcases = Dict(
        TestCase(d) => (d[colnames[5]], d[colnames[6]]) for d in base_results
    )
    all_testcases = union(Set(keys(head_testcases)), Set(keys(base_testcases)))
    @info "$(length(all_testcases)) unique test cases found"
    sorted_testcases = sort(
        collect(all_testcases); by=(c -> (c.model_name, c.linked, c.ad_backend))
    )
    results_table = Tuple{
        String,
        Int,
        String,
        Bool,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    }[]
    sublabels = ["base", "this PR", "speedup"]
    results_colnames = [
        [
            EmptyCells(4),
            MultiColumn(3, "t(eval) / t(ref)"),
            MultiColumn(3, "t(grad) / t(eval)"),
            MultiColumn(3, "t(grad) / t(ref)"),
        ],
        [colnames[1:4]..., sublabels..., sublabels..., sublabels...],
    ]
    sprint_float(x::Float64) = @sprintf("%.2f", x)
    sprint_float(m::Missing) = "err"
    for c in sorted_testcases
        head_eval, head_grad = get(head_testcases, c, (missing, missing))
        base_eval, base_grad = get(base_testcases, c, (missing, missing))
        # If the benchmark errored, it will return `missing` in the `run()` function above.
        # The issue with this is that JSON serialisation converts it to `null`, and then
        # when reading back from JSON, it becomes `nothing` instead of `missing`!
        head_eval = head_eval === nothing ? missing : head_eval
        head_grad = head_grad === nothing ? missing : head_grad
        base_eval = base_eval === nothing ? missing : base_eval
        base_grad = base_grad === nothing ? missing : base_grad
        # Finally that lets us do this division safely
        speedup_eval = base_eval / head_eval
        speedup_grad = base_grad / head_grad
        # As well as this multiplication, which is t(grad) / t(ref)
        head_grad_vs_ref = head_grad * head_eval
        base_grad_vs_ref = base_grad * base_eval
        speedup_grad_vs_ref = base_grad_vs_ref / head_grad_vs_ref
        push!(
            results_table,
            (
                c.model_name,
                c.dim,
                c.ad_backend,
                c.linked,
                sprint_float(base_eval),
                sprint_float(head_eval),
                sprint_float(speedup_eval),
                sprint_float(base_grad),
                sprint_float(head_grad),
                sprint_float(speedup_grad),
                sprint_float(base_grad_vs_ref),
                sprint_float(head_grad_vs_ref),
                sprint_float(speedup_grad_vs_ref),
            ),
        )
    end
    # Pretty-print to terminal
    if isempty(results_table)
        println("No benchmark results obtained.")
    else
        table_matrix = hcat(Iterators.map(collect, zip(results_table...))...)
        println("```")
        pretty_table(
            table_matrix;
            column_labels=results_colnames,
            backend=:text,
            fit_table_in_display_horizontally=false,
            fit_table_in_display_vertically=false,
            table_format=TextTableFormat(;
                horizontal_line_at_merged_column_labels=true,
                horizontal_lines_at_data_rows=collect(3:3:length(results_table)),
            ),
        )
        println("```")
    end
end

# The command-line arguments are used on CI purposes.
# Run with `julia --project=. benchmarks.jl json` to run benchmarks and output JSON to
# stdout
# Run with `julia --project=. benchmarks.jl combine head.json base.json` to combine two JSON
# files
if length(ARGS) == 3 && ARGS[1] == "combine"
    combine(ARGS[2], ARGS[3])
elseif ARGS == ["json"]
    run(; to_json=true)
elseif ARGS == []
    # When running locally just omit the argument and it will just benchmark and print to
    # terminal.
    run()
else
    error("invalid arguments: $(ARGS)")
end
