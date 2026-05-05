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
using DifferentiationInterface: DifferentiationInterface
using DynamicPPL: DynamicPPL, @model, to_submodel, VarInfo, LinkAll, UnlinkAll
using DynamicPPL.TestUtils.AD: run_ad, NoTest
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using LinearAlgebra: cholesky
using Mooncake: Mooncake
using Printf: @sprintf
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG

#
#  Models
#

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

# Pivot so each (Model, Dim, Linked) row spans all backends. A long-form table
# (one row per (model, linked, backend)) reads as four near-duplicate rows
# differing only in the backend column; pivoting puts the backends side-by-side
# where the ratios are actually compared. `t(logdensity)` does not depend on
# the AD backend (it is the primal model evaluation), so the four primal
# samples per group are noise around a common value — take the minimum, which
# is the most stable estimate (see `run_ad`'s benchmark docstring).
function pivot(results, backends)
    keys_in_order = Tuple{String,Bool}[]
    seen = Set{Tuple{String,Bool}}()
    for r in results
        k = (r.name, r.islinked)
        if !(k in seen)
            push!(seen, k)
            push!(keys_in_order, k)
        end
    end
    return map(keys_in_order) do (name, islinked)
        rows = filter(r -> r.name == name && r.islinked == islinked, results)
        primals = collect(skipmissing(r.t_logd for r in rows))
        primal = isempty(primals) ? missing : minimum(primals)
        ratios = Dict{String,Union{Float64,Missing}}(string(b) => missing for b in backends)
        for r in rows
            ratios[r.adbackend] = r.ratio
        end
        (; name, dim=first(rows).dim, islinked, primal, ratios)
    end
end

function print_results(results)
    isempty(results) && return println("No benchmark results obtained.")
    pivoted = pivot(results, BACKENDS)
    backend_info = [
        (key="forwarddiff", label="FwdDiff"),
        (key="reversediff", label="RvsDiff"),
        (key="mooncake", label="Mooncake"),
        (key="enzyme", label="Enzyme"),
    ]

    rows = map(pivoted) do g
        ratios = [format_ratio(g.ratios[b.key]) for b in backend_info]
        (
            name=g.name,
            dim=format_dim(g.dim),
            linked=string(g.islinked),
            primal=format_time(g.primal),
            ratios,
        )
    end

    name_w = max(length("Model"), maximum(textwidth(r.name) for r in rows)) + 1
    dim_w = max(length("dim"), maximum(textwidth(r.dim) for r in rows)) + 2
    linked_w = max(length("linked"), maximum(textwidth(r.linked) for r in rows)) + 2
    primal_w = max(length("primal"), maximum(textwidth(r.primal) for r in rows)) + 2
    ratio_ws = [
        max(length(b.label), maximum(textwidth(r.ratios[i]) for r in rows)) + 2 for
        (i, b) in enumerate(backend_info)
    ]

    gap = "  "
    gap_w = textwidth(gap)
    stub_w = name_w + dim_w + linked_w + 2 * gap_w
    eval_w = primal_w
    grad_w = sum(ratio_ws) + gap_w * (length(ratio_ws) - 1)
    total_w = stub_w + gap_w + eval_w + gap_w + grad_w

    center(s, w) = lpad(rpad(s, div(w + textwidth(s), 2)), w)
    println(repeat("=", total_w))
    println(
        rpad("", stub_w) * gap * center("eval", eval_w) * gap * center("gradient", grad_w)
    )
    println(rpad("", stub_w) * gap * repeat("-", eval_w) * gap * repeat("-", grad_w))

    header =
        rpad("Model", name_w) *
        gap *
        lpad("dim", dim_w) *
        gap *
        lpad("linked", linked_w) *
        gap *
        lpad("primal", primal_w) *
        gap *
        join((lpad(b.label, w) for (b, w) in zip(backend_info, ratio_ws)), gap)
    println(header)
    println(repeat("-", total_w))

    for r in rows
        row =
            rpad(r.name, name_w) *
            gap *
            lpad(r.dim, dim_w) *
            gap *
            lpad(r.linked, linked_w) *
            gap *
            lpad(r.primal, primal_w) *
            gap *
            join((lpad(x, w) for (x, w) in zip(r.ratios, ratio_ws)), gap)
        println(row)
    end
    println(repeat("=", total_w))
    return nothing
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
        println("```")
        print_results(results)
        println("```")
    else
        print_results(results)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if ARGS == ["markdown"]
        run(; markdown=true)
    elseif ARGS == []
        run()
    else
        error("invalid arguments: $(ARGS)")
    end
end
