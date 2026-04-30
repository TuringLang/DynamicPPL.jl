using Chairmarks: median
using DynamicPPLBenchmarks: Models, benchmark, model_dimension
using PrettyTables: pretty_table
using Printf: @sprintf
using StableRNGs: StableRNG

rng = StableRNG(23)

# Schema follows Mooncake's bench output: absolute log-density time plus the
# gradient/log-density ratio. We deliberately do not compare against the base
# branch — readers eyeball regressions across the PR-comment history instead.
# Cf. https://github.com/chalk-lab/Mooncake.jl/blob/main/bench/run_benchmarks.jl
const COLNAMES = [
    "Model", "Dim", "AD Backend", "Linked", "t(logdensity)", "t(grad)/t(logdensity)"
]

# Adapted from Mooncake's bench harness.
fix_sig_fig(t) = string(round(t; sigdigits=3))
function format_time(t::Float64)
    t < 1e-6 && return fix_sig_fig(t * 1e9) * " ns"
    t < 1e-3 && return fix_sig_fig(t * 1e6) * " μs"
    t < 1 && return fix_sig_fig(t * 1e3) * " ms"
    return fix_sig_fig(t) * " s"
end
format_time(::Missing) = "err"

format_ratio(x::Float64) = @sprintf("%.2f", x)
format_ratio(::Missing) = "err"

function print_results(results_table)
    isempty(results_table) && return println("No benchmark results obtained.")
    display_rows = map(results_table) do row
        (row[1], row[2], row[3], row[4], format_time(row[5]), format_ratio(row[6]))
    end
    table_matrix = hcat(Iterators.map(collect, zip(display_rows...))...)
    return pretty_table(
        table_matrix;
        column_labels=COLNAMES,
        backend=:text,
        fit_table_in_display_horizontally=false,
        fit_table_in_display_vertically=false,
    )
end

function run(; markdown::Bool=false)
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

    results_table = Tuple{
        String,Int,String,Bool,Union{Float64,Missing},Union{Float64,Missing}
    }[]

    for (model_name, model, adbackend, islinked) in chosen_combinations
        @info "Running benchmark for $model_name, $adbackend, $islinked"
        logdensity_time, grad_over_logdensity = try
            results = benchmark(model, adbackend, islinked)
            @info " t(logdensity) = $(results.primal_time)"
            @info " t(grad)       = $(results.grad_time)"
            (results.primal_time, results.grad_time / results.primal_time)
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
                logdensity_time,
                grad_over_logdensity,
            ),
        )
    end

    # Markdown mode wraps the text table in a fenced block so it renders
    # monospaced when posted as a PR comment.
    markdown && println("```")
    print_results(results_table)
    markdown && println("```")
    return nothing
end

# Run with `julia --project=. benchmarks.jl markdown` to emit a fenced text
# table to stdout, suitable for pasting into a PR comment. Run with no
# arguments to pretty-print to the terminal.
if ARGS == ["markdown"]
    run(; markdown=true)
elseif ARGS == []
    run()
else
    error("invalid arguments: $(ARGS)")
end
