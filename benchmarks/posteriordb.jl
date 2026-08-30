#!/usr/bin/env julia
#
# Benchmark logp + gradient evaluation for DynamicPPL vs Stan on PosteriorDB models.
#
# Usage:
#   julia --project=. posteriordb.jl                                      # all models
#   julia --project=. posteriordb.jl eight_schools-eight_schools_centered # one model
#   julia --project=. posteriordb.jl --logdensity-only                    # skip gradients
#   julia --project=. posteriordb.jl --stan-only                          # skip DynamicPPL
#   julia --project=. posteriordb.jl --turing-only                        # skip Stan timing
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
    logdensity_only::Bool
    stan_only::Bool
    turing_only::Bool
    all_ad::Bool
    shard_count::Int
    shard_index::Int
    seed::Int
end

function run_metadata(
    stan_version;
    logdensity_only,
    stan_only,
    turing_only,
    all_ad,
    shard_count=1,
    shard_index=1,
    seed=468,
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
        logdensity_only,
        stan_only,
        turing_only,
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
    println(io, "- BLAS: $(metadata.blas_vendor), $(metadata.blas_threads) thread(s)")
    return println(
        io,
        "- Packages: DynamicPPL $(metadata.dynamicppl_version), Mooncake $(metadata.mooncake_version), BridgeStan $(metadata.bridgestan_version), Stan $(metadata.stan_version)",
    )
end

median_time(bench) = median(bench).time

function abbreviate(name, prefix_length=1)
    function shorten(part)
        words = Base.split(part, '_')
        if length(words) == 1
            part
        else
            join(first(word, min(prefix_length, length(word))) for word in words)
        end
    end
    return join(shorten.(Base.split(name, '-')), "-")
end

function unique_model_labels(names)
    prefix_lengths = ones(Int, length(names))
    for _ in 1:maximum(length, names)
        labels = [abbreviate(name, prefix_lengths[index]) for (index, name) in pairs(names)]
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
    ("FD", AutoForwardDiff()), ("Ez", AutoEnzyme(; mode=set_runtime_activity(Reverse)))
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
        @info "using runtime transforms in benchmark" exception = typeof(error)
        return LogDensityFunction(model, logdensity, vi; adtype, fix_transforms=false)
    end
end

function bench_dynamicppl(
    dynamicppl_model, model_name, params; logdensity_only=false, all_ad=false
)
    ldf = stable_ldf(dynamicppl_model; adtype=logdensity_only ? nothing : AutoMooncake())
    length(params) == DynamicPPL.LogDensityProblems.dimension(ldf) ||
        error("parameter dimension mismatch for $model_name")
    DynamicPPL.LogDensityProblems.logdensity(ldf, params)
    primal_time = median_time(@be DynamicPPL.LogDensityProblems.logdensity($ldf, $params))
    grad_times = Dict{String,Float64}()
    if !logdensity_only
        DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, params)
        grad_times["Mc"] = median_time(
            @be DynamicPPL.LogDensityProblems.logdensity_and_gradient($ldf, $params)
        )
    end

    if !logdensity_only && all_ad
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

# Stan and DynamicPPL use different unconstrained coordinates. These helpers map
# matched parameter values and gradients without constructing dense Jacobians.

function constrained_named_tuple(names, values)
    groups = Dict{String,Vector{Tuple{Vector{Int},Float64}}}()
    for (name, value) in zip(names, values)
        parts = Base.split(name, '.')
        item = (parse.(Int, parts[2:end]), value)
        push!(get!(groups, parts[1], Tuple{Vector{Int},Float64}[]), item)
    end

    result = Dict{Symbol,Any}()
    for (base, items) in groups
        indices = first(items)[1]
        if isempty(indices)
            result[Symbol(base)] = first(items)[2]
            continue
        end
        dims = ntuple(i -> maximum(item[1][i] for item in items), length(indices))
        array = zeros(dims)
        for (index, value) in items
            array[index...] = value
        end
        result[Symbol(base)] = array
    end
    return result
end

function adapt_dynamicppl_values!(values, model_name)
    # Product distributions store repeated multivariate draws in columns.
    if model_name in ("ldaK2", "ldaK5")
        values[:theta] = permutedims(values[:theta])
        values[:phi] = permutedims(values[:phi])
    elseif model_name == "hmm_gaussian"
        values[:A] = permutedims(values[:A])
    elseif model_name == "hierarchical_gp"
        values[:GP_region_std] = vec(values[:GP_region_std])
        values[:GP_state_std] = vec(values[:GP_state_std])
    elseif model_name in ("nn_rbm1bJ10", "nn_rbm1bJ100")
        values[:alpha] = vec(values[:alpha])
        values[:beta] = vec(values[:beta])
    end

    for key in (:L, :L_Omega, :L_logit_ab)
        haskey(values, key) || continue
        values[key] = Cholesky(LowerTriangular(values[key]), 'L', 0)
    end
    return (; values...)
end

struct CoordinateMap{S,N,R}
    stan::S
    constrained_names::N
    model_name::String
    ranges::R
end

function fixed_bijector(range_and_transform)
    transform = DynamicPPL.get_transform(range_and_transform)
    transform isa DynamicPPL.FixedTransform ||
        error("coordinate mapping requires fixed transforms")
    return transform.transform
end

function (map::CoordinateMap)(q)
    constrained = BridgeStan.param_constrain(
        map.stan, q; include_tp=false, include_gq=false
    )
    parameters = constrained_named_tuple(map.constrained_names, constrained)
    parameters = adapt_dynamicppl_values!(parameters, map.model_name)
    dimension = mapreduce(item -> length(item.range), +, Base.values(map.ranges); init=0)
    qt = similar(q, dimension)
    for (vn, range_and_transform) in pairs(map.ranges)
        raw = getproperty(parameters, DynamicPPL.getsym(vn))
        qt[range_and_transform.range] .= Bijectors.inverse(
            fixed_bijector(range_and_transform)
        )(
            raw
        )
    end
    return qt
end

function coordinate_groups(map::CoordinateMap)
    stan_groups = Dict{String,Vector{Int}}()
    for (i, name) in enumerate(BridgeStan.param_unc_names(map.stan))
        push!(get!(stan_groups, first(Base.split(name, '.')), Int[]), i)
    end
    dynamicppl_groups = Dict(
        String(DynamicPPL.getsym(vn)) => collect(range_and_transform.range) for
        (vn, range_and_transform) in pairs(map.ranges)
    )
    keys(stan_groups) == keys(dynamicppl_groups) ||
        error("Stan and DynamicPPL parameter groups differ")
    return stan_groups, dynamicppl_groups
end

"""Detect a scaled permutation `y = offset + scale * x[permutation]`."""
function scaled_permutation(x1, x2, y0, y1, y2)
    length(x1) == length(y1) || return nothing
    dxnorm = norm(x1)
    dynorm = norm(y1 .- y0)
    (dxnorm == 0 || dynorm == 0) && return nothing
    scale = dynorm / dxnorm

    source_order = sortperm(x1)
    target_order = sortperm((y1 .- y0) ./ scale)
    isapprox(x1[source_order], ((y1 .- y0) ./ scale)[target_order]; atol=2e-9, rtol=2e-9) ||
        return nothing

    permutation = similar(target_order)
    for k in eachindex(source_order)
        permutation[target_order[k]] = source_order[k]
    end
    predicted = y0 .+ scale .* x2[permutation]
    isapprox(y2, predicted; atol=2e-8, rtol=2e-8) || return nothing
    return (; permutation, scale)
end

function stan_to_dynamicppl_simplex(x)
    K = length(x) + 1
    z = zeros(eltype(x), K)
    sum_w = zero(eltype(x))
    for i in (K - 1):-1:1
        w = x[i] / sqrt(i * (i + 1))
        sum_w += w
        z[i] += sum_w
        z[i + 1] -= i * w
    end
    z .-= maximum(z)
    simplex = exp.(z)
    simplex ./= sum(simplex)
    return Bijectors.transform(Bijectors.SimplexBijector(), simplex)
end

is_simplex_transform(::Any) = false
is_simplex_transform(::Bijectors.SimplexBijector) = true
is_simplex_transform(transform::Bijectors.Inverse) = is_simplex_transform(transform.orig)

function uses_simplex_transform(map, name)
    for (vn, range_and_transform) in pairs(map.ranges)
        if String(DynamicPPL.getsym(vn)) == name
            return is_simplex_transform(fixed_bijector(range_and_transform))
        end
    end
    return false
end

function simplex_source_blocks(map, indices)
    names = BridgeStan.param_unc_names(map.stan)[indices]
    suffixes = [Base.split(name, '.')[2:end] for name in names]
    maximum(length, suffixes) == 1 && return [indices]

    block_length = length(unique(last.(suffixes)))
    length(indices) % block_length == 0 || error("invalid simplex coordinate blocks")
    return [
        indices[i:(i + block_length - 1)] for
        i in firstindex(indices):block_length:lastindex(indices)
    ]
end

function classify_coordinate_groups(map, probes)
    stan_groups, dynamicppl_groups = coordinate_groups(map)
    q0, q1, q2 = probes
    t0, t1, t2 = map(q0), map(q1), map(q2)
    affine = Dict{String,Any}()
    nonlinear = String[]
    simplex = Dict{String,Any}()

    for name in keys(stan_groups)
        stan_indices = stan_groups[name]
        dynamicppl_indices = dynamicppl_groups[name]
        relation = scaled_permutation(
            q1[stan_indices] .- q0[stan_indices],
            q2[stan_indices] .- q0[stan_indices],
            t0[dynamicppl_indices],
            t1[dynamicppl_indices],
            t2[dynamicppl_indices],
        )
        if relation === nothing && uses_simplex_transform(map, name)
            source_blocks = simplex_source_blocks(map, stan_indices)
            target_cursor = 1
            block_relations = Any[]
            for source_block in source_blocks
                block_length = length(source_block)
                target_block = dynamicppl_indices[target_cursor:(target_cursor + block_length - 1)]
                target_cursor += block_length
                local_value = stan_to_dynamicppl_simplex(q1[source_block])
                target_value = t1[target_block]
                source_order = sortperm(local_value)
                target_order = sortperm(target_value)
                permutation = similar(target_order)
                for k in eachindex(source_order)
                    permutation[target_order[k]] = source_order[k]
                end
                isapprox(target_value, local_value[permutation]; atol=2e-8, rtol=2e-8) ||
                    error("simplex coordinate mismatch for $name")
                push!(block_relations, (; source_block, target_block, permutation))
            end
            target_cursor == length(dynamicppl_indices) + 1 ||
                error("incomplete simplex coordinate mapping")
            simplex[name] = block_relations
        elseif relation === nothing
            push!(nonlinear, name)
        else
            affine[name] = relation
        end
    end

    dependencies = Dict{String,Vector{Int}}()
    for target in nonlinear
        target_indices = dynamicppl_groups[target]
        source_indices = Int[]
        for source in keys(stan_groups)
            probe = copy(q0)
            indices = stan_groups[source]
            probe[indices] .= q1[indices]
            if !isapprox(
                map(probe)[target_indices], t0[target_indices]; atol=2e-10, rtol=2e-10
            )
                append!(source_indices, indices)
            end
        end
        dependencies[target] = sort!(unique!(source_indices))
    end
    return (; stan_groups, dynamicppl_groups, affine, simplex, nonlinear, dependencies)
end

function gradient_in_stan_coordinates(map, groups, q, dynamicppl_gradient)
    result = zeros(length(q))
    for (name, relation) in groups.affine
        stan_indices = groups.stan_groups[name]
        dynamicppl_indices = groups.dynamicppl_groups[name]
        for j in eachindex(dynamicppl_indices)
            result[stan_indices[relation.permutation[j]]] +=
                relation.scale * dynamicppl_gradient[dynamicppl_indices[j]]
        end
    end

    for block_relations in values(groups.simplex)
        for relation in block_relations
            source_block = relation.source_block
            target_block = relation.target_block
            local_gradient = zeros(length(source_block))
            for j in eachindex(target_block)
                local_gradient[relation.permutation[j]] += dynamicppl_gradient[target_block[j]]
            end
            jacobian = ForwardDiff.jacobian(stan_to_dynamicppl_simplex, q[source_block])
            result[source_block] .+= jacobian' * local_gradient
        end
    end

    for name in groups.nonlinear
        target_indices = groups.dynamicppl_groups[name]
        for i in groups.dependencies[name]
            h = 1e-5 * max(1.0, abs(q[i]))
            qplus, qminus = copy(q), copy(q)
            qplus[i] += h
            qminus[i] -= h
            tplus = map(qplus)
            tminus = map(qminus)
            result[i] +=
                dot(
                    dynamicppl_gradient[target_indices],
                    tplus[target_indices] .- tminus[target_indices],
                ) / (2h)
        end
    end
    return result
end

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
const CHECKPOINT_FORMAT = "2"

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
    isempty(missing_fields) ||
        error("missing metadata $(join(missing_fields, ", ")) in $path")
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
    indices == collect(1:(reference.shard_count)) || error(
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
    merged_metadata = RunMetadata(
        (
            name === :shard_index ? 0 : getfield(reference, name) for
            name in fieldnames(RunMetadata)
        )...,
    )
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
    draws >= 2 || error("draws must be at least 2")
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

function benchmark_one(
    model_name;
    logdensity_only=false,
    stan_only=false,
    turing_only=false,
    all_ad=true,
    seed=468,
)
    post = PosteriorDB.posterior(PDB, model_name)
    stan_model = bridge_model(post, seed)
    dimension = Int(BridgeStan.param_unc_num(stan_model))
    q = only(
        random_valid_points(
            stan_model, Xoshiro(seed), 1, 0.2; center=draw_center(model_name, dimension)
        ),
    )

    dynamicppl = if stan_only
        (primal_time=NaN, grad_times=Dict{String,Float64}(), dim=dimension)
    else
        data = PosteriorDB.load(PosteriorDB.dataset(post))
        dynamicppl_model = make_model(Val(Symbol(model_name)), data)
        mapping_ldf = stable_ldf(dynamicppl_model)
        coordinate_map = CoordinateMap(
            stan_model,
            BridgeStan.param_names(stan_model; include_tp=false, include_gq=false),
            PosteriorDB.name(PosteriorDB.model(post)),
            DynamicPPL.get_all_ranges_and_transforms(mapping_ldf),
        )
        dynamicppl_params = coordinate_map(q)
        bench_dynamicppl(
            dynamicppl_model, model_name, dynamicppl_params; logdensity_only, all_ad
        )
    end

    stan_primal = NaN
    if !turing_only
        BridgeStan.log_density(stan_model, q; propto=false)
        stan_primal = median_time(@be BridgeStan.log_density($stan_model, $q; propto=false))
    end

    stan_gradient = NaN
    if !logdensity_only && !turing_only
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
    logdensity_only=false,
    stan_only=false,
    turing_only=false,
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
        result, version = benchmark_one(
            model_name; logdensity_only, stan_only, turing_only, all_ad, seed
        )
        if metadata === nothing
            metadata = run_metadata(
                version;
                logdensity_only,
                stan_only,
                turing_only,
                all_ad,
                shard_count,
                shard_index,
                seed,
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
            @error "verification failed" posterior_name exception = (
                error, catch_backtrace()
            )
        end
    end
    return passed
end

# ── Summary and table ──

function time_comparison(results, backend_field, stan_field)
    ratios = Float64[]
    for result in results
        backend_time = getfield(result, backend_field)
        stan_time = getfield(result, stan_field)
        if isfinite(backend_time) &&
            backend_time > 0 &&
            isfinite(stan_time) &&
            stan_time > 0
            push!(ratios, backend_time / stan_time)
        end
    end
    isempty(ratios) && return (successful=0, geometric_mean=NaN, median=NaN, faster=0)
    return (;
        successful=length(ratios),
        geometric_mean=exp(sum(log, ratios) / length(ratios)),
        median=median(ratios),
        faster=count(<(1), ratios),
    )
end

fmt_summary_ratio(ratio) = isfinite(ratio) ? @sprintf("%.2f×", ratio) : "n/a"

function write_summary(
    io, results; logdensity_only=false, stan_only=false, turing_only=false, all_ad=true
)
    total = length(results)
    posterior = total == 1 ? "posterior" : "posteriors"
    println(io, "## Summary\n")
    println(io, "- **$total PosteriorDB $posterior** benchmarked.")
    (stan_only || turing_only) && return println(io)

    println(io)
    println(
        io,
        "Ratios are backend time divided by the corresponding Stan time; lower is " *
        "better. Aggregates exclude failed measurements.\n",
    )
    println(
        io,
        "| Workload | Successful measurements | Geometric mean vs Stan | Median vs Stan | Faster than Stan |",
    )
    println(io, "|:--|--:|--:|--:|--:|")
    workloads = [("Turing log density", :dynamicppl_primal, :stan_primal)]
    if !logdensity_only
        push!(workloads, ("Mooncake gradient", :dynamicppl_mooncake_grad, :stan_grad))
        all_ad && push!(workloads, ("Enzyme gradient", :dynamicppl_enzyme_grad, :stan_grad))
    end
    for (name, backend_field, stan_field) in workloads
        comparison = time_comparison(results, backend_field, stan_field)
        println(
            io,
            "| $name | $(comparison.successful) / $total | " *
            "$(fmt_summary_ratio(comparison.geometric_mean)) | " *
            "$(fmt_summary_ratio(comparison.median)) | " *
            "$(comparison.faster) / $(comparison.successful) |",
        )
    end
    return println(io)
end

function write_table(io, results; logdensity_only=false, stan_only=false, turing_only=false)
    gap = "  "
    labels = unique_model_labels([result.name for result in results])
    headers = if stan_only
        logdensity_only ? ["Model", "dim", "Stan"] : ["Model", "dim", "Stan", "Stan"]
    elseif turing_only
        if logdensity_only
            ["Model", "dim", "Turing"]
        else
            ["Model", "dim", "Turing", "FwdDiff", "EzyRvs", "McRvs"]
        end
    elseif logdensity_only
        ["Model", "dim", "Turing", "Stan"]
    else
        ["Model", "dim", "Turing", "Stan", "FwdDiff", "EzyRvs", "McRvs", "Stan", "McRvs / Stan"]
    end
    rows = [
        if stan_only
            if logdensity_only
                [label, string(result.dim), fmt_time(result.stan_primal)]
            else
                [
                    label,
                    string(result.dim),
                    fmt_time(result.stan_primal),
                    fmt_time(result.stan_grad),
                ]
            end
        elseif turing_only
            if logdensity_only
                [label, string(result.dim), fmt_time(result.dynamicppl_primal)]
            else
                [
                    label,
                    string(result.dim),
                    fmt_time(result.dynamicppl_primal),
                    fmt_time(result.dynamicppl_fd_grad),
                    fmt_time(result.dynamicppl_enzyme_grad),
                    fmt_time(result.dynamicppl_mooncake_grad),
                ]
            end
        elseif logdensity_only
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

    function render(row)
        return join(
            (
                column == 1 ? rpad(value, widths[column]) : lpad(value, widths[column]) for
                (column, value) in pairs(row)
            ),
            gap,
        )
    end
    center(value, width) = begin
        left = div(width - length(value), 2)
        lpad(rpad(value, width - left), width)
    end

    eval_columns = stan_only || turing_only ? 1 : 2
    gradient_columns = logdensity_only ? 0 : length(headers) - 2 - eval_columns
    pre_w = widths[1] + length(gap) + widths[2]
    eval_w = sum(widths[3:(2 + eval_columns)]) + length(gap) * (eval_columns - 1)
    gradient_w = if iszero(gradient_columns)
        0
    else
        first_gradient = 3 + eval_columns
        sum(widths[first_gradient:end]) + length(gap) * (gradient_columns - 1)
    end

    println(io)
    println(io, "="^total_w)
    println(
        io,
        rstrip(
            " "^(pre_w + length(gap)) *
            center("eval", eval_w) *
            (iszero(gradient_columns) ? "" : gap * center("gradient", gradient_w)),
        ),
    )
    println(
        io,
        " "^(pre_w + length(gap)) *
        "-"^eval_w *
        (iszero(gradient_columns) ? "" : gap * "-"^gradient_w),
    )
    println(io, render(headers))
    println(io, "-"^total_w)
    for row in rows
        println(io, render(row))
    end
    return println(io, "="^total_w)
end

function write_markdown(path, results, metadata::RunMetadata)
    open(path, "w") do io
        println(io, "# PosteriorDB benchmark results\n")
        println(
            io, "Generated by `julia --project=benchmarks benchmarks/posteriordb.jl`.\n"
        )
        write_summary(
            io,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            turing_only=metadata.turing_only,
            all_ad=metadata.all_ad,
        )
        println(io, "## Environment\n")
        write_environment(io, metadata)
        println(io)
        caption = if metadata.stan_only
            if metadata.logdensity_only
                "*Table 1. Median Stan log-density evaluation times.*\n"
            else
                "*Table 1. Median Stan log-density and gradient evaluation times.*\n"
            end
        elseif metadata.turing_only
            if metadata.logdensity_only
                "*Table 1. Median Turing log-density evaluation times.*\n"
            else
                "*Table 1. Median Turing log-density and gradient evaluation times. " *
                "FwdDiff: ForwardDiff; EzyRvs: Enzyme; McRvs: Mooncake reverse mode.*\n"
            end
        elseif metadata.logdensity_only
            "*Table 1. Median Turing and Stan log-density evaluation times at a " *
            "matched parameter realization.*\n"
        else
            "*Table 1. Median log-density and gradient evaluation times at a matched " *
            "parameter realization. FwdDiff: ForwardDiff; EzyRvs: Enzyme; McRvs: " *
            "Mooncake reverse mode; McRvs / Stan: McRvs time divided by Stan gradient " *
            "time (lower is better).*\n"
        end
        println(io, caption)
        println(io, "```text")
        write_table(
            io,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            turing_only=metadata.turing_only,
        )
        return println(io, "```")
    end
end

function parse_options(args)
    known = Set((
        "--logdensity-only", # Skip gradients.
        "--stan-only",       # Time only Stan.
        "--turing-only",     # Time only Turing.
        "--mooncake-only",   # Skip ForwardDiff and Enzyme.
        "--merge",           # Merge shard checkpoints.
        "--verify",          # Compare log density and gradients with Stan.
    ))
    options = Set(argument for argument in args if startswith(argument, "--"))
    unknown = setdiff(options, known)
    isempty(unknown) || error("unknown option(s): $(join(sort!(collect(unknown)), ", "))")
    positional = [argument for argument in args if !startswith(argument, "--")]
    merge = "--merge" in options
    verify = "--verify" in options
    merge && verify && error("--merge and --verify are mutually exclusive")
    for option in ("--logdensity-only", "--stan-only", "--turing-only", "--mooncake-only")
        (merge || verify) &&
            option in options &&
            error("$option is only valid when benchmarking")
    end
    "--stan-only" in options &&
        "--mooncake-only" in options &&
        error("--stan-only and --mooncake-only are mutually exclusive")
    "--stan-only" in options &&
        "--turing-only" in options &&
        error("--stan-only and --turing-only are mutually exclusive")
    return (;
        positional,
        merge,
        verify,
        logdensity_only="--logdensity-only" in options,
        stan_only="--stan-only" in options,
        turing_only="--turing-only" in options,
        all_ad=!("--mooncake-only" in options || "--stan-only" in options),
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
        write_table(
            stdout,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            turing_only=metadata.turing_only,
        )
        path = get(ENV, "PDB_BENCH_MARKDOWN", joinpath(@__DIR__, "posteriordb.md"))
        write_markdown(path, results, metadata)
        return 0
    end

    shard_count, shard_index = shard_configuration()
    models = if isempty(options.positional)
        POSTERIOR_NAMES[shard_index:shard_count:end]
    else
        options.positional
    end
    println(stderr, "Selected $(length(models)) model(s).")

    if options.verify
        draws = parse(Int, get(ENV, "PDB_TEST_DRAWS", "30"))
        scale = parse(Float64, get(ENV, "PDB_TEST_SCALE", "0.2"))
        seed = parse(Int, get(ENV, "PDB_TEST_SEED", "468"))
        return verify_models(models; draws, seed, scale) ? 0 : 1
    end

    checkpoint_path = get(ENV, "PDB_BENCH_OUTPUT", "")
    shard_count > 1 &&
        isempty(checkpoint_path) &&
        error("sharded benchmarks require a distinct PDB_BENCH_OUTPUT per shard")
    seed = parse(Int, get(ENV, "PDB_BENCH_SEED", "468"))
    results, metadata = benchmark_models(
        models;
        logdensity_only=options.logdensity_only,
        stan_only=options.stan_only,
        turing_only=options.turing_only,
        all_ad=options.all_ad && !options.logdensity_only,
        seed,
        shard_count,
        shard_index,
        checkpoint_path,
    )
    write_table(
        stdout,
        results;
        logdensity_only=metadata.logdensity_only,
        stan_only=metadata.stan_only,
        turing_only=metadata.turing_only,
    )

    full_catalog = isempty(options.positional) && shard_count == 1
    markdown_path = get(
        ENV, "PDB_BENCH_MARKDOWN", full_catalog ? joinpath(@__DIR__, "posteriordb.md") : ""
    )
    isempty(markdown_path) || write_markdown(markdown_path, results, metadata)
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
