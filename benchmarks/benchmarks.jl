<<<<<<< HEAD
using Pkg

using Chairmarks: @be
using DynamicPPLBenchmarks: Models, benchmark, model_dimension
using JSON: JSON
using PrettyTables: pretty_table, fmt__printf, EmptyCells, MultiColumn, TextTableFormat
=======
using ADTypes: ADTypes
using Distributions:
    Categorical,
    Dirichlet,
    Exponential,
    Gamma,
    InverseWishart,
    LKJCholesky,
    Normal,
    product_distribution,
    truncated
using DynamicPPL: DynamicPPL, @model, to_submodel, VarInfo, LinkAll, UnlinkAll
using DynamicPPL.TestUtils.AD: run_ad, NoTest
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using LinearAlgebra: cholesky
using Mooncake: Mooncake
using PrettyTables: pretty_table
>>>>>>> upstream/main
using Printf: @sprintf
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG

#
#  Models
#

<<<<<<< HEAD
colnames = ["Model", "Dim", "AD Backend", "Linked", "t(eval)/ns", "t(grad)/t(eval)"]
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
=======
"One scalar assumption, one scalar observation."
@model function simple_assume_observe(obs)
    x ~ Normal()
    obs ~ Normal(x, 1)
    return (; x=x)
end

"""
Covers many DynamicPPL features: scalar/vector/multivariate variables, `~`,
`.~`, loops, allocated vectors, and observations as both arguments and literals.
"""
@model function smorgasbord(x, y, ::Type{TV}=Vector{Float64}) where {TV}
    @assert length(x) == length(y)
    m ~ truncated(Normal(); lower=0)
    means ~ product_distribution(fill(Exponential(m), length(x)))
    stds = TV(undef, length(x))
    stds .~ Gamma(1, 1)
    for i in 1:length(x)
        x[i] ~ Normal(means[i], stds[i])
    end
    y ~ product_distribution(map((mean, std) -> Normal(mean, std), means, stds))
    0.0 ~ Normal(sum(y), 1)
    return (; m=m, means=means, stds=stds)
end

"`num_dims` univariate normals via a loop. Condition on `o` after instantiation."
@model function loop_univariate(num_dims, ::Type{TV}=Vector{Float64}) where {TV}
    a = TV(undef, num_dims)
    o = TV(undef, num_dims)
    for i in 1:num_dims
        a[i] ~ Normal(0, 1)
    end
    m = sum(a)
    for i in 1:num_dims
        o[i] ~ Normal(m, 1)
    end
    return (; a=a)
end

"As `loop_univariate`, but using `product_distribution` instead of loops."
@model function multivariate(num_dims, ::Type{TV}=Vector{Float64}) where {TV}
    a = TV(undef, num_dims)
    o = TV(undef, num_dims)
    a ~ product_distribution(fill(Normal(0, 1), num_dims))
    m = sum(a)
    o ~ product_distribution(fill(Normal(m, 1), num_dims))
    return (; a=a)
end

@model function _sub()
    x ~ Normal()
    return x
end

"As `simple_assume_observe`, but with the assumed RV inside a submodel."
@model function parent(obs)
    x ~ to_submodel(_sub())
    obs ~ Normal(x, 1)
    return (; x=x)
end

"Variables whose support varies under linking, or otherwise nontrivial bijectors."
@model function dynamic(::Type{T}=Vector{Float64}) where {T}
    eta ~ truncated(Normal(); lower=0.0, upper=0.1)
    mat1 ~ LKJCholesky(4, eta)
    mat2 ~ InverseWishart(3.2, cholesky([1.0 0.5; 0.5 1.0]))
    return (; eta=eta, mat1=mat1, mat2=mat2)
end

"Linear Discriminant Analysis."
@model function lda(K, d, w)
    V = length(unique(w))
    D = length(unique(d))
    N = length(d)
    @assert length(w) == N

    ϕ = Vector{Vector{Real}}(undef, K)
    for i in 1:K
        ϕ[i] ~ Dirichlet(ones(V) / V)
    end

    θ = Vector{Vector{Real}}(undef, D)
    for i in 1:D
        θ[i] ~ Dirichlet(ones(K) / K)
    end

    z = zeros(Int, N)
    for i in 1:N
        z[i] ~ Categorical(θ[d[i]])
        w[i] ~ Categorical(ϕ[d[i]])
    end
    return (; ϕ=ϕ, θ=θ, z=z)
end

#
#  Benchmark harness
#

# Copied from TuringBenchmarking.jl.
const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ADTypes.AutoForwardDiff(),
    :reversediff => ADTypes.AutoReverseDiff(; compile=false),
    :reversediff_compiled => ADTypes.AutoReverseDiff(; compile=true),
    :mooncake => ADTypes.AutoMooncake(; config=nothing),
    :enzyme => ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    ),
)

transform_strategy(islinked) = islinked ? LinkAll() : UnlinkAll()

"Dimension of `model`, accounting for linking. Used as a fallback when `benchmark` errors."
function model_dimension(model, islinked)
    return try
        vi = last(
            DynamicPPL.init!!(
                StableRNG(23),
                model,
                VarInfo(),
                DynamicPPL.InitFromPrior(),
                transform_strategy(islinked),
            ),
>>>>>>> upstream/main
        )
        length(vi[:])
    catch
        missing
    end
end

"""
    benchmark(model, adbackend, islinked; seconds=2)

Time log-density and gradient evaluation for `model` with the given AD backend.
`seconds` is Chairmarks' per-measurement budget (doubled from its default to
tighten the median estimate).
"""
function benchmark(model, adbackend::Symbol, islinked::Bool; seconds::Real=2)
    return run_ad(
        model,
        SYMBOL_TO_BACKEND[adbackend];
        rng=StableRNG(23),
        transform_strategy=transform_strategy(islinked),
        benchmark=true,
        benchmark_seconds=seconds,
        test=NoTest(),
        verbose=false,
    )
end

#
#  Reporting
#

# https://github.com/chalk-lab/Mooncake.jl/blob/main/bench/run_benchmarks.jl
const COLNAMES = [
    "Model", "Dim", "AD Backend", "Linked", "t(logdensity)", "t(grad)/t(logdensity)"
]

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

format_dim(d::Integer) = string(d)
format_dim(::Missing) = "err"

function print_results(results)
    isempty(results) && return println("No benchmark results obtained.")
    rows = map(results) do r
        (
<<<<<<< HEAD
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
        relative_eval_time, relative_ad_eval_time = try
            results = benchmark(model, adbackend, islinked)
            @info " t(eval) = $(results.primal_time)"
            @info " t(grad) = $(results.grad_time)"
            (results.primal_time * 1e9), (results.grad_time / results.primal_time)
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
=======
            r.name,
            format_dim(r.dim),
            r.adbackend,
            r.islinked,
            format_time(r.t_logd),
            format_ratio(r.ratio),
>>>>>>> upstream/main
        )
    end
    matrix = hcat(Iterators.map(collect, zip(rows...))...)
    return pretty_table(
        matrix;
        column_labels=COLNAMES,
        backend=:text,
        fit_table_in_display_horizontally=false,
        fit_table_in_display_vertically=false,
    )
end

#
#  Main
#

# Backends compared on every model. `:reversediff_compiled` is excluded because
# compiled tapes are input-dependent and silently produce wrong gradients on
# models with parameter-dependent control flow (see CLAUDE.md).
const BACKENDS = (:forwarddiff, :reversediff, :mooncake, :enzyme)

function build_combinations(rng)
    smorg = smorgasbord(randn(rng, 100), randn(rng, 100))
    models = Tuple{String,DynamicPPL.Model}[
        ("Simple assume observe", simple_assume_observe(randn(rng))), ("Smorgasbord", smorg)
    ]
    for n in (1_000, 10_000)
        data = randn(rng, n)
        push!(models, ("Loop univariate $(n ÷ 1_000)k", loop_univariate(n) | (; o=data)))
        push!(models, ("Multivariate $(n ÷ 1_000)k", multivariate(n) | (; o=data)))
    end
    push!(models, ("Dynamic", dynamic()))
    push!(models, ("Submodel", parent(randn(rng))))
    push!(models, ("LDA", lda(2, [1, 1, 1, 2, 2, 2], [1, 2, 3, 2, 1, 1])))

    # Order: model → linked → backend, so each model's eight rows are adjacent
    # and inspecting one model side-by-side across backends/links is trivial.
    combos = Tuple{String,DynamicPPL.Model,Symbol,Bool}[]
    for (name, model) in models, islinked in (false, true), backend in BACKENDS
        # LDA's discrete Categorical RVs make `linked = false` ill-defined for
        # gradient-based AD (every backend errors), so the row is omitted.
        name == "LDA" && !islinked && continue
        push!(combos, (name, model, backend, islinked))
    end
    return combos
end

# Representative model whose 8 rows are surfaced as the at-a-glance "gist"
# in markdown mode. `Smorgasbord` covers the broadest set of DPPL features
# (scalar/vector/multivariate variables, `~`, `.~`, loops, observations as
# both arguments and literals), so it is the most informative single row band.
const GIST_MODEL = "Smorgasbord"

function run(; markdown::Bool=false)
    combinations = build_combinations(StableRNG(23))
    total = length(combinations)
    results = []
    for (i, (name, model, adbackend, islinked)) in enumerate(combinations)
        # Mooncake-style header: index/total, then model + config, then backend.
        @info "$i / $total", name, (; linked=islinked)
        @info adbackend
        dim, t_logd, ratio = try
            r = benchmark(model, adbackend, islinked)
            @info "  t(logdensity) = $(format_time(r.primal_time))"
            @info "  t(grad)       = $(format_time(r.grad_time))"
            (length(r.params), r.primal_time, r.grad_time / r.primal_time)
        catch e
            @info "  errored: $(sprint(showerror, e))"
            (model_dimension(model, islinked), missing, missing)
        end
        push!(results, (; name, dim, adbackend=string(adbackend), islinked, t_logd, ratio))
    end
    if markdown
        gist = filter(r -> r.name == GIST_MODEL, results)
        if !isempty(gist)
            println("### Gist: ", GIST_MODEL)
            println()
            println("```")
            print_results(gist)
            println("```")
            println()
        end
        println("<details>")
        println("<summary>Full table (", length(results), " rows)</summary>")
        println()
        println("```")
        print_results(results)
        println("```")
        println()
        println("</details>")
    else
        print_results(results)
    end
    return nothing
end

<<<<<<< HEAD
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
            MultiColumn(3, "t(eval)/ns"),
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
=======
if abspath(PROGRAM_FILE) == @__FILE__
    if ARGS == ["markdown"]
        run(; markdown=true)
    elseif ARGS == []
        run()
>>>>>>> upstream/main
    else
        error("invalid arguments: $(ARGS)")
    end
end
