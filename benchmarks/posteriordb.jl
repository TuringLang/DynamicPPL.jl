#!/usr/bin/env julia
#
# Benchmark logp + gradient evaluation for DynamicPPL vs Stan on PosteriorDB models.
#
# Usage:
#   julia --project=. posteriordb.jl                                      # all models
#   julia --project=. posteriordb.jl eight_schools-eight_schools_centered # one model
#   julia --project=. posteriordb.jl --eval-only                          # skip gradients
#   julia --project=. posteriordb.jl --verify eight_schools-eight_schools_centered
#
# Stan and DynamicPPL are benchmarked at the same constrained parameter realization.
#
# From the REPL:
#   include("posteriordb.jl")
#   ARGS = ["sblri-blr"]; include("posteriordb.jl")

using Printf, Random, LinearAlgebra
using Chairmarks: @be
using Statistics: median
using ADTypes: AutoForwardDiff, AutoEnzyme, AutoMooncake
using Enzyme: Reverse, set_runtime_activity
using DynamicPPL: LogDensityFunction, getlogjoint_internal
import BridgeStan
import DynamicPPL
import ForwardDiff
import Mooncake
import PosteriorDB

if haskey(ENV, "BRIDGESTAN_PATH")
    BridgeStan.set_bridgestan_path!(ENV["BRIDGESTAN_PATH"])
end

# ── PosteriorDB setup ──

const PDB = PosteriorDB.database()

# ── Helpers ──

function fmt_time(t)
    t === nothing && return "err"
    isnan(t) && return "err"
    isinf(t) && return "err"
    if t < 1e-6
        @sprintf("%.1f ns", t * 1e9)
    elseif t < 1e-3
        @sprintf("%.1f μs", t * 1e6)
    elseif t < 1.0
        @sprintf("%.2f ms", t * 1e3)
    else
        @sprintf("%.3f s", t)
    end
end

function fmt_ratio(numerator, denominator)
    ratio = numerator / denominator
    return isfinite(ratio) ? @sprintf("%.2f×", ratio) : "err"
end

function cpu_name()
    if Sys.islinux()
        names = try
            unique(
                strip(line) for
                line in eachline(`lscpu --parse=MODELNAME`) if !startswith(line, '#')
            )
        catch
            String[]
        end
        isempty(names) || return join(names, " + ")
    end
    if Sys.islinux() && isfile("/proc/cpuinfo")
        info = read("/proc/cpuinfo", String)
        names = unique(
            strip(only(match.captures)) for
            match in eachmatch(r"(?m)^model name\s*:\s*(.+)$", info)
        )
        isempty(names) || return join(names, " + ")
    end
    return Sys.CPU_NAME
end

struct RunMetadata
    julia_version::String
    system::String
    cpu::String
    cpu_threads::Int
    memory_gib::Float64
    julia_threads::Int
    blas_vendor::String
    blas_threads::Int
    dynamicppl_version::String
    mooncake_version::String
    bridgestan_version::String
    stan_version::String
    eval_only::Bool
    all_ad::Bool
    shard_count::Int
    shard_index::Int
    seed::Int
end

function run_metadata(
    stan_version; eval_only, all_ad, shard_count=1, shard_index=1, seed=468
)
    return RunMetadata(
        string(VERSION),
        "$(Sys.KERNEL) $(Sys.ARCH)",
        cpu_name(),
        Sys.CPU_THREADS,
        Sys.total_memory() / 2.0^30,
        Threads.nthreads(),
        string(LinearAlgebra.BLAS.vendor()),
        LinearAlgebra.BLAS.get_num_threads(),
        string(pkgversion(DynamicPPL)),
        string(pkgversion(Mooncake)),
        string(pkgversion(BridgeStan)),
        stan_version,
        eval_only,
        all_ad,
        shard_count,
        shard_index,
        seed,
    )
end

function write_environment(io, metadata::RunMetadata)
    println(io, "- Julia: $(metadata.julia_version)")
    println(io, "- System: $(metadata.system)")
    println(io, "- CPU: $(metadata.cpu) ($(metadata.cpu_threads) logical threads)")
    println(io, "- Memory: $(@sprintf("%.1f", metadata.memory_gib)) GiB")
    println(
        io,
        "- Benchmark parallelism: $(metadata.shard_count) Julia process(es), $(metadata.julia_threads) Julia thread(s) each",
    )
    println(
        io,
        "- BLAS: $(metadata.blas_vendor), $(metadata.blas_threads) thread(s)",
    )
    return println(
        io,
        "- Packages: DynamicPPL $(metadata.dynamicppl_version), Mooncake $(metadata.mooncake_version), BridgeStan $(metadata.bridgestan_version), Stan $(metadata.stan_version)",
    )
end

median_time(bench) = median(bench).time

function abbreviate(name, prefix_length=1)
    shorten(part) = begin
        words = Base.split(part, '_')
        length(words) == 1 ? part : join(first(word, min(prefix_length, length(word))) for word in words)
    end
    return join(shorten.(Base.split(name, '-')), "-")
end

function unique_model_labels(names)
    prefix_lengths = ones(Int, length(names))
    for _ in 1:maximum(length, names)
        labels = [
            abbreviate(name, prefix_lengths[index]) for (index, name) in pairs(names)
        ]
        allunique(labels) && return labels
        groups = Dict{String,Vector{Int}}()
        for (index, label) in pairs(labels)
            push!(get!(groups, label, Int[]), index)
        end
        for indices in values(groups)
            length(indices) == 1 && continue
            prefix_lengths[indices] .+= 1
        end
    end
    return collect(names)
end

const OTHER_AD_BACKENDS = (
    ("FD", AutoForwardDiff()),
    ("Ez", AutoEnzyme(; mode=set_runtime_activity(Reverse))),
)

function stable_ldf(model; adtype=nothing, logdensity=getlogjoint_internal)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, vi = DynamicPPL.init!!(
        model, vi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll()
    )
    fixed_vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _, fixed_vi = DynamicPPL.init!!(
        model, fixed_vi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll()
    )
    transforms = DynamicPPL.get_fixed_transforms(fixed_vi)
    vecvals = DynamicPPL.getacc(vi, Val(:VectorValue)).values
    vecvals = DynamicPPL.update_transforms!!(vecvals, transforms)
    try
        return LogDensityFunction(model, logdensity, vecvals; adtype, fix_transforms=false)
    catch error
        @info "using runtime transforms in benchmark" exception=typeof(error)
        return LogDensityFunction(model, logdensity, vi; adtype, fix_transforms=false)
    end
end

function bench_dynamicppl(
    dynamicppl_model, model_name, params; eval_only=false, all_ad=false
)
    ldf = stable_ldf(dynamicppl_model; adtype=eval_only ? nothing : AutoMooncake())
    length(params) == DynamicPPL.LogDensityProblems.dimension(ldf) ||
        error("parameter dimension mismatch for $model_name")
    DynamicPPL.LogDensityProblems.logdensity(ldf, params)
    primal_time = median_time(@be DynamicPPL.LogDensityProblems.logdensity($ldf, $params))
    grad_times = Dict{String,Float64}()
    if !eval_only
        DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, params)
        grad_times["Mc"] = median_time(
            @be DynamicPPL.LogDensityProblems.logdensity_and_gradient($ldf, $params)
        )
    end

    if !eval_only && all_ad
        for (label, adtype) in OTHER_AD_BACKENDS
            if label == "FD" && length(params) > 2_000
                grad_times[label] = Inf
                continue
            end
            try
                other_ldf = stable_ldf(dynamicppl_model; adtype)
                DynamicPPL.LogDensityProblems.logdensity_and_gradient(other_ldf, params)
                grad_times[label] = median_time(
                    @be DynamicPPL.LogDensityProblems.logdensity_and_gradient(
                        $other_ldf, $params
                    )
                )
            catch e
                @warn "$model_name: $label failed — $(typeof(e))"
                grad_times[label] = Inf
            end
        end
    end

    return (primal_time=primal_time, grad_times=grad_times, dim=length(params))
end

# ── Discover models ──

include(joinpath(@__DIR__, "posteriordb_models.jl"))
include(joinpath(@__DIR__, "posteriordb_coordinates.jl"))

# ── Results ──

struct Result
    name::String
    dim::Int
    dynamicppl_primal::Float64
    stan_primal::Float64
    dynamicppl_fd_grad::Float64
    dynamicppl_enzyme_grad::Float64
    dynamicppl_mooncake_grad::Float64
    stan_grad::Float64
end

const RESULT_COLUMNS = (
    "model",
    "dim",
    "dynamicppl_primal_s",
    "stan_primal_s",
    "forwarddiff_gradient_s",
    "enzyme_gradient_s",
    "mooncake_gradient_s",
    "stan_gradient_s",
)
const CHECKPOINT_FORMAT = "1"

function write_result(io, result)
    return println(
        io,
        join(
            (
                result.name,
                result.dim,
                result.dynamicppl_primal,
                result.stan_primal,
                result.dynamicppl_fd_grad,
                result.dynamicppl_enzyme_grad,
                result.dynamicppl_mooncake_grad,
                result.stan_grad,
            ),
            '\t',
        ),
    )
end

function write_checkpoint_header(io, metadata::RunMetadata)
    println(io, "# posteriordb_format\t$CHECKPOINT_FORMAT")
    for name in fieldnames(RunMetadata)
        println(io, "# $name\t$(getfield(metadata, name))")
    end
    return println(io, join(RESULT_COLUMNS, '\t'))
end

parse_metadata(::Type{String}, value) = value
parse_metadata(::Type{T}, value) where {T<:Union{Int,Float64,Bool}} = parse(T, value)

function read_checkpoint(path)
    metadata_values = Dict{Symbol,String}()
    results = Result[]
    found_header = false
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "# ")
            key, value = Base.split(line[3:end], '\t'; limit=2)
            key == "posteriordb_format" && begin
                value == CHECKPOINT_FORMAT ||
                    error("unsupported checkpoint format $value in $path")
                continue
            end
            metadata_values[Symbol(key)] = value
        elseif !found_header
            Base.split(line, '\t') == collect(RESULT_COLUMNS) ||
                error("unexpected result columns in $path")
            found_header = true
        else
            fields = Base.split(line, '\t')
            length(fields) == length(RESULT_COLUMNS) ||
                error("malformed result row in $path")
            push!(
                results,
                Result(fields[1], parse(Int, fields[2]), parse.(Float64, fields[3:end])...),
            )
        end
    end
    found_header || error("missing result header in $path")
    missing_fields = setdiff(fieldnames(RunMetadata), keys(metadata_values))
    isempty(missing_fields) || error("missing metadata $(join(missing_fields, ", ")) in $path")
    values = ntuple(fieldcount(RunMetadata)) do index
        name = fieldname(RunMetadata, index)
        parse_metadata(fieldtype(RunMetadata, index), metadata_values[name])
    end
    return RunMetadata(values...), results
end

function metadata_compatible(left::RunMetadata, right::RunMetadata)
    return all(fieldnames(RunMetadata)) do name
        name === :shard_index || getfield(left, name) == getfield(right, name)
    end
end

function merge_checkpoints(paths)
    checkpoints = read_checkpoint.(paths)
    isempty(checkpoints) && error("pass at least one checkpoint")
    reference = first(checkpoints)[1]
    all(metadata_compatible(reference, metadata) for (metadata, _) in checkpoints) ||
        error("checkpoint metadata differ")
    indices = sort!([metadata.shard_index for (metadata, _) in checkpoints])
    indices == collect(1:reference.shard_count) || error(
        "expected shard indices 1:$(reference.shard_count), received $(join(indices, ", "))",
    )

    results = reduce(vcat, (rows for (_, rows) in checkpoints); init=Result[])
    names = [result.name for result in results]
    length(unique(names)) == length(names) || error("duplicate models in checkpoints")
    Set(names) == Set(POSTERIOR_NAMES) || error(
        "checkpoint coverage differs from POSTERIOR_NAMES: " *
        "missing=$(join(setdiff(POSTERIOR_NAMES, names), ",")); " *
        "unexpected=$(join(setdiff(names, POSTERIOR_NAMES), ","))",
    )
    order = Dict(name => index for (index, name) in pairs(POSTERIOR_NAMES))
    sort!(results; by=result -> order[result.name])
    merged_metadata = RunMetadata((
        name === :shard_index ? 0 : getfield(reference, name) for
        name in fieldnames(RunMetadata)
    )...)
    return results, merged_metadata
end

function stan_version(sm)
    version_match = match(r"(?m)^Stan version:\s*(\S+)", BridgeStan.model_info(sm))
    return version_match === nothing ? "unknown" : only(version_match.captures)
end

const ODE_POSTERIORS = Set((
    "hudson_lynx_hare-lotka_volterra",
    "one_comp_mm_elim_abs-one_comp_mm_elim_abs",
    "sir-sir",
    "soil_carbon-soil_incubation",
))

function bridge_model(post, seed)
    stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
    stan_path = PosteriorDB.path(stan)
    data_json = PosteriorDB.load(PosteriorDB.dataset(post), String)
    stan_library = first(splitext(stan_path)) * "_model.so"
    return BridgeStan.StanModel(
        isfile(stan_library) ? stan_library : stan_path, data_json, seed
    )
end

function draw_center(posterior_name, dimension)
    center = if posterior_name == "hudson_lynx_hare-lotka_volterra"
        [0.0, log(0.05), 0.0, log(0.05), log(30.0), log(4.0), -1.0, -1.0]
    elseif posterior_name == "sir-sir"
        [-2.0, -2.0, 4.0, -2.0]
    elseif posterior_name == "soil_carbon-soil_incubation"
        [-2.3, -2.3, -2.3, -2.3, 2.0, -1.6]
    else
        zeros(dimension)
    end
    length(center) == dimension || error("invalid draw center for $posterior_name")
    return center
end

function random_valid_points(stan_model, rng, draws, scale; center)
    points = Vector{Vector{Float64}}()
    dimension = Int(BridgeStan.param_unc_num(stan_model))
    attempts = 0
    while length(points) < draws
        attempts += 1
        attempts <= max(100, 100draws) ||
            error("could not draw $draws valid unconstrained parameter vectors")
        q = center .+ scale .* randn(rng, dimension)
        valid = try
            isfinite(BridgeStan.log_density(stan_model, q; propto=false, jacobian=false))
        catch
            false
        end
        valid && push!(points, q)
    end
    return points
end

"""
    verify_against_stan(posterior_name; draws=30, seed=468, scale=0.2)

Verify relative log density and the complete gradient for random draws in Stan's
unconstrained parameter space. Log densities are centered at the first draw so
that parameter-independent normalization constants are allowed.
"""
function verify_against_stan(posterior_name; draws=30, seed=468, scale=0.2)
    draws >= 1 || error("draws must be positive")
    post = PosteriorDB.posterior(PDB, posterior_name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    dynamicppl_model = make_model(Val(Symbol(posterior_name)), data)
    stan_model = bridge_model(post, seed)
    model_name = PosteriorDB.name(PosteriorDB.model(post))

    map_ldf = stable_ldf(dynamicppl_model; logdensity=DynamicPPL.getlogjoint)
    ldf = stable_ldf(
        dynamicppl_model; adtype=AutoMooncake(), logdensity=DynamicPPL.getlogjoint
    )
    coordinate_map = CoordinateMap(
        stan_model,
        BridgeStan.param_names(stan_model; include_tp=false, include_gq=false),
        model_name,
        DynamicPPL.get_all_ranges_and_transforms(map_ldf),
    )

    dimension = Int(BridgeStan.param_unc_num(stan_model))
    rng = Xoshiro(seed + sum(codeunits(posterior_name)))
    center = draw_center(posterior_name, dimension)
    all_points = random_valid_points(stan_model, rng, max(draws, 3), scale; center)
    groups = classify_coordinate_groups(coordinate_map, all_points[1:3])
    points = all_points[1:draws]

    dynamicppl_logp = zeros(draws)
    stan_logp = zeros(draws)
    max_gradient_error = 0.0
    max_gradient_relative_error = 0.0
    finite_difference_checks = 0
    gradient_rtol = posterior_name in ODE_POSTERIORS ? 2e-3 : 5e-6

    for (draw, q) in enumerate(points)
        qt = coordinate_map(q)
        dynamicppl_logp[draw], dynamicppl_gradient = DynamicPPL.LogDensityProblems.logdensity_and_gradient(
            ldf, qt
        )
        stan_logp[draw], stan_gradient = BridgeStan.log_density_gradient(
            stan_model, q; propto=false, jacobian=false
        )
        mapped_gradient = gradient_in_stan_coordinates(
            coordinate_map, groups, q, dynamicppl_gradient
        )

        reference_gradient = stan_gradient
        gradient_scale = max(
            1.0, maximum(abs, stan_gradient), maximum(abs, mapped_gradient)
        )
        if posterior_name == "synthetic_grid_RBF_kernels-kronecker_gp"
            mismatches = findall(
                abs.(mapped_gradient .- stan_gradient) .> gradient_rtol * gradient_scale
            )
            if !isempty(mismatches)
                reference_gradient = copy(stan_gradient)
                for i in mismatches
                    h = 1e-5 * max(1.0, abs(q[i]))
                    qplus, qminus = copy(q), copy(q)
                    qplus[i] += h
                    qminus[i] -= h
                    reference_gradient[i] =
                        (
                            BridgeStan.log_density(
                                stan_model, qplus; propto=false, jacobian=false
                            ) - BridgeStan.log_density(
                                stan_model, qminus; propto=false, jacobian=false
                            )
                        ) / (2h)
                end
                finite_difference_checks += length(mismatches)
            end
        end

        gradient_scale = max(
            1.0, maximum(abs, reference_gradient), maximum(abs, mapped_gradient)
        )
        gradient_error = maximum(abs, mapped_gradient .- reference_gradient)
        max_gradient_error = max(max_gradient_error, gradient_error)
        max_gradient_relative_error = max(
            max_gradient_relative_error, gradient_error / gradient_scale
        )
        gradient_error <= gradient_rtol * gradient_scale || error(
            "$posterior_name draw $draw gradient mismatch: " *
            "scaled error $(gradient_error / gradient_scale)",
        )
    end

    dynamicppl_delta = dynamicppl_logp .- first(dynamicppl_logp)
    stan_delta = stan_logp .- first(stan_logp)
    density_scale = max(1.0, maximum(abs, dynamicppl_delta), maximum(abs, stan_delta))
    density_error = maximum(abs, dynamicppl_delta .- stan_delta)
    density_rtol = posterior_name in ODE_POSTERIORS ? 5e-4 : 1e-7
    density_error <= density_rtol * density_scale || error(
        "$posterior_name log-density mismatch: " *
        "scaled error $(density_error / density_scale)",
    )

    return (;
        logp_offset=first(dynamicppl_logp) - first(stan_logp),
        max_density_error=density_error,
        max_density_relative_error=density_error / density_scale,
        max_gradient_error,
        max_gradient_relative_error,
        stan_finite_difference_checks=finite_difference_checks,
        nonlinear_maps=groups.nonlinear,
    )
end

function benchmark_one(model_name; eval_only=false, all_ad=true, seed=468)
    post = PosteriorDB.posterior(PDB, model_name)
    pdb_dataset = PosteriorDB.dataset(post)
    data = PosteriorDB.load(pdb_dataset)
    dynamicppl_model = make_model(Val(Symbol(model_name)), data)
    pdb_model = PosteriorDB.model(post)
    stan_model = bridge_model(post, seed)
    dimension = Int(BridgeStan.param_unc_num(stan_model))
    q = only(
        random_valid_points(
            stan_model,
            Xoshiro(seed),
            1,
            0.2;
            center=draw_center(model_name, dimension),
        )
    )

    mapping_ldf = stable_ldf(dynamicppl_model)
    coordinate_map = CoordinateMap(
        stan_model,
        BridgeStan.param_names(stan_model; include_tp=false, include_gq=false),
        PosteriorDB.name(pdb_model),
        DynamicPPL.get_all_ranges_and_transforms(mapping_ldf),
    )
    dynamicppl_params = coordinate_map(q)
    dynamicppl = bench_dynamicppl(
        dynamicppl_model, model_name, dynamicppl_params; eval_only, all_ad
    )

    BridgeStan.log_density(stan_model, q; propto=false)
    stan_primal = median_time(
        @be BridgeStan.log_density($stan_model, $q; propto=false)
    )

    stan_gradient = NaN
    if !eval_only
        BridgeStan.log_density_gradient(stan_model, q; propto=false)
        stan_gradient = median_time(
            @be BridgeStan.log_density_gradient($stan_model, $q; propto=false)
        )
    end

    return (
        Result(
            model_name,
            dynamicppl.dim,
            dynamicppl.primal_time,
            stan_primal,
            get(dynamicppl.grad_times, "FD", NaN),
            get(dynamicppl.grad_times, "Ez", NaN),
            get(dynamicppl.grad_times, "Mc", NaN),
            stan_gradient,
        ),
        stan_version(stan_model),
    )
end

function benchmark_models(
    models;
    eval_only=false,
    all_ad=true,
    seed=468,
    shard_count=1,
    shard_index=1,
    checkpoint_path="",
)
    isempty(models) && error("no models selected")
    results = Result[]
    metadata = nothing
    for model_name in models
        println(stderr, "Running: $model_name ...")
        result, version = benchmark_one(model_name; eval_only, all_ad, seed)
        if metadata === nothing
            metadata = run_metadata(
                version; eval_only, all_ad, shard_count, shard_index, seed
            )
            if !isempty(checkpoint_path)
                open(checkpoint_path, "w") do io
                    write_checkpoint_header(io, metadata)
                end
            end
        elseif version != metadata.stan_version
            error("Stan version changed from $(metadata.stan_version) to $version")
        end
        push!(results, result)
        if !isempty(checkpoint_path)
            open(checkpoint_path, "a") do io
                write_result(io, result)
            end
        end
    end
    return results, metadata::RunMetadata
end

function verify_models(models; draws=30, seed=468, scale=0.2)
    passed = true
    for posterior_name in models
        println(stderr, "Verifying: $posterior_name ...")
        try
            verification = verify_against_stan(posterior_name; draws, seed, scale)
            println(
                stderr,
                "  density=$(verification.max_density_relative_error) scaled, " *
                "gradient=$(verification.max_gradient_relative_error) scaled",
            )
        catch error
            passed = false
            @error "verification failed" posterior_name exception=(error, catch_backtrace())
        end
    end
    return passed
end

# ── Table ──

function write_table(io, results; eval_only=false)
    gap = "  "
    labels = unique_model_labels([result.name for result in results])
    headers = if eval_only
        ["Model", "dim", "Turing", "Stan"]
    else
        [
            "Model",
            "dim",
            "Turing",
            "Stan",
            "FwdDiff",
            "EzyDiff",
            "McRvs",
            "Stan",
            "McRvs / Stan",
        ]
    end
    rows = [
        if eval_only
            [
                label,
                string(result.dim),
                fmt_time(result.dynamicppl_primal),
                fmt_time(result.stan_primal),
            ]
        else
            [
                label,
                string(result.dim),
                fmt_time(result.dynamicppl_primal),
                fmt_time(result.stan_primal),
                fmt_time(result.dynamicppl_fd_grad),
                fmt_time(result.dynamicppl_enzyme_grad),
                fmt_time(result.dynamicppl_mooncake_grad),
                fmt_time(result.stan_grad),
                fmt_ratio(result.dynamicppl_mooncake_grad, result.stan_grad),
            ]
        end for (result, label) in zip(results, labels)
    ]
    widths = [
        maximum(length(row[column]) for row in Iterators.flatten(([headers], rows))) for
        column in eachindex(headers)
    ]
    widths[1] = max(widths[1], 16)
    total_w = sum(widths) + length(gap) * (length(widths) - 1)

    render(row) = join(
        (
            column == 1 ? rpad(value, widths[column]) : lpad(value, widths[column]) for
            (column, value) in pairs(row)
        ),
        gap,
    )
    center(value, width) = begin
        left = div(width - length(value), 2)
        lpad(rpad(value, width - left), width)
    end

    pre_w = widths[1] + length(gap) + widths[2]
    eval_w = widths[3] + length(gap) + widths[4]
    grad_w = eval_only ? 0 : sum(widths[5:end]) + length(gap) * (length(widths) - 5)

    println(io)
    println(io, "=" ^ total_w)
    println(
        io,
        " " ^ (pre_w + length(gap)) *
        center("eval", eval_w) *
        (eval_only ? "" : gap * center("gradient", grad_w)),
    )
    println(
        io,
        " " ^ (pre_w + length(gap)) * "-" ^ eval_w * (eval_only ? "" : gap * "-" ^ grad_w),
    )
    println(io, render(headers))
    println(io, "-" ^ total_w)
    for row in rows
        println(io, render(row))
    end
    return println(io, "=" ^ total_w)
end

function write_markdown(path, results, metadata::RunMetadata)
    open(path, "w") do io
        println(io, "# PosteriorDB benchmark results\n")
        println(
            io, "Generated by `julia --project=benchmarks benchmarks/posteriordb.jl`.\n"
        )
        write_environment(io, metadata)
        println(io)
        println(
            io,
            "*Table 1. Median log-density and gradient evaluation times at a matched " *
            "parameter realization. FwdDiff: ForwardDiff; EzyDiff: Enzyme; McRvs: " *
            "Mooncake reverse mode; McRvs / Stan: McRvs time divided by Stan gradient " *
            "time (lower is better).*\n",
        )
        println(io, "```text")
        write_table(io, results; eval_only=metadata.eval_only)
        return println(io, "```")
    end
end

function parse_options(args)
    known = Set(("--eval-only", "--mooncake-only", "--merge", "--verify"))
    options = Set(argument for argument in args if startswith(argument, "--"))
    unknown = setdiff(options, known)
    isempty(unknown) || error("unknown option(s): $(join(sort!(collect(unknown)), ", "))")
    positional = [argument for argument in args if !startswith(argument, "--")]
    merge = "--merge" in options
    verify = "--verify" in options
    merge && verify && error("--merge and --verify are mutually exclusive")
    (merge || verify) && "--eval-only" in options &&
        error("--eval-only is only valid when benchmarking")
    merge && "--mooncake-only" in options &&
        error("--mooncake-only is only valid when benchmarking")
    return (;
        positional,
        merge,
        verify,
        eval_only="--eval-only" in options,
        all_ad=!("--mooncake-only" in options),
    )
end

function shard_configuration()
    count = parse(Int, get(ENV, "PDB_BENCH_SHARDS", "1"))
    index = parse(Int, get(ENV, "PDB_BENCH_SHARD", "1"))
    1 <= index <= count || error("PDB_BENCH_SHARD must be in 1:PDB_BENCH_SHARDS")
    return count, index
end

function main(args=ARGS)
    options = parse_options(args)
    if options.merge
        isempty(options.positional) && error("pass checkpoint files after --merge")
        results, metadata = merge_checkpoints(options.positional)
        write_table(stdout, results; eval_only=metadata.eval_only)
        path = get(ENV, "PDB_BENCH_MARKDOWN", joinpath(@__DIR__, "posteriordb.md"))
        write_markdown(path, results, metadata)
        return 0
    end

    shard_count, shard_index = shard_configuration()
    models = isempty(options.positional) ?
        POSTERIOR_NAMES[shard_index:shard_count:end] : options.positional
    println(stderr, "Selected $(length(models)) model(s).")

    if options.verify
        draws = parse(Int, get(ENV, "PDB_TEST_DRAWS", "30"))
        scale = parse(Float64, get(ENV, "PDB_TEST_SCALE", "0.2"))
        seed = parse(Int, get(ENV, "PDB_TEST_SEED", "468"))
        return verify_models(models; draws, seed, scale) ? 0 : 1
    end

    checkpoint_path = get(ENV, "PDB_BENCH_OUTPUT", "")
    shard_count > 1 && isempty(checkpoint_path) &&
        error("sharded benchmarks require a distinct PDB_BENCH_OUTPUT per shard")
    seed = parse(Int, get(ENV, "PDB_BENCH_SEED", "468"))
    results, metadata = benchmark_models(
        models;
        eval_only=options.eval_only,
        all_ad=options.all_ad && !options.eval_only,
        seed,
        shard_count,
        shard_index,
        checkpoint_path,
    )
    write_table(stdout, results; eval_only=metadata.eval_only)

    full_catalog = isempty(options.positional) && shard_count == 1
    markdown_path = get(
        ENV,
        "PDB_BENCH_MARKDOWN",
        full_catalog ? joinpath(@__DIR__, "posteriordb.md") : "",
    )
    isempty(markdown_path) || write_markdown(markdown_path, results, metadata)
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
