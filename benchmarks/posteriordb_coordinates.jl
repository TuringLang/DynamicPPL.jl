# Stan and DynamicPPL use different unconstrained coordinates. This file maps
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
        map.stan, q; include_tp=false, include_gq=false,
    )
    parameters = constrained_named_tuple(map.constrained_names, constrained)
    parameters = adapt_dynamicppl_values!(parameters, map.model_name)
    dimension = mapreduce(item -> length(item.range), +, Base.values(map.ranges); init=0)
    qt = similar(q, dimension)
    for (vn, range_and_transform) in pairs(map.ranges)
        raw = getproperty(parameters, DynamicPPL.getsym(vn))
        qt[range_and_transform.range] .= Bijectors.inverse(
            fixed_bijector(range_and_transform)
        )(raw)
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
    isapprox(
        x1[source_order],
        ((y1 .- y0) ./ scale)[target_order];
        atol=2e-9,
        rtol=2e-9,
    ) || return nothing

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
                target_block = dynamicppl_indices[
                    target_cursor:(target_cursor + block_length - 1)
                ]
                target_cursor += block_length
                local_value = stan_to_dynamicppl_simplex(q1[source_block])
                target_value = t1[target_block]
                source_order = sortperm(local_value)
                target_order = sortperm(target_value)
                permutation = similar(target_order)
                for k in eachindex(source_order)
                    permutation[target_order[k]] = source_order[k]
                end
                isapprox(
                    target_value,
                    local_value[permutation];
                    atol=2e-8,
                    rtol=2e-8,
                ) || error("simplex coordinate mismatch for $name")
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
            if !isapprox(map(probe)[target_indices], t0[target_indices]; atol=2e-10, rtol=2e-10)
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
                local_gradient[relation.permutation[j]] +=
                    dynamicppl_gradient[target_block[j]]
            end
            jacobian = ForwardDiff.jacobian(
                stan_to_dynamicppl_simplex, q[source_block]
            )
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
            result[i] += dot(
                dynamicppl_gradient[target_indices],
                tplus[target_indices] .- tminus[target_indices],
            ) / (2h)
        end
    end
    return result
end
