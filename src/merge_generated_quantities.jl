"""
    get_generated_quantities(model::Model, chains::AbstractChains)

Get the generated_quantities from the `chains` exluding any internal parameters.
See https://discourse.julialang.org/t/turing-jl-warnings-when-running-generated-quantities/64698/2

"""
function get_generated_quantities(model::Model, chains::AbstractChains)
    chains_params = MCMCChains.get_sections(chains, :parameters)
    return generated_quantities(model, chains_params)
end


function get_generated_quantities(dict::Dict)
    return get_generated_quantities(dict[:model], dict[:chains])
end


"""
    get_K(dict::Dict, variable::Union{Symbol,String})

Get the number of dimensions (`K`) for the specific `variable`.
"""
function get_K(dict::Dict, variable::Union{Symbol,String})
    K = length(first(dict[:generated_quantities])[variable])
    return K
end


function get_variables(dict::Dict)
    return dict[:generated_quantities] |> first |> keys
end


function get_N_samples(dict::Dict)
    return length(dict[:chains])
end


function get_N_chains(dict::Dict)
    return length(MCMCChains.chains(dict[:chains]))
end


"""
    generated_quantities_to_chain(dict::Dict, variable::Union{Symbol,String})

Convert the array of values returned by the generated quantities to a
MCMCChains chain for a single `variable`.
"""
function generated_quantities_to_chain(dict::Dict, variable::Union{Symbol,String})

    K = get_K(dict, variable)

    matrix = zeros(dict[:N_samples], K, dict[:N_chains])
    for chain = 1:dict[:N_chains]
        for (i, xi) in enumerate(dict[:generated_quantities][:, chain])
            matrix[i, :, chain] .= xi[variable]
        end
    end

    if K == 1
        chain_names = [Symbol("$variable")]
    else
        chain_names = [Symbol("$variable[$i]") for i = 1:K]
    end
    generated_chain = MCMCChains.Chains(matrix, chain_names, info = dict[:chains].info)

    return generated_chain

end


"""
    generated_quantities_to_chains(dict::Dict)

Convert the array of values returned by the generated quantities to a
MCMCChains chain by concatenating the individual variables' chains.
"""
function generated_quantities_to_chains(dict::Dict)
    return hcat(
        [generated_quantities_to_chain(dict, variable) for variable in dict[:variables]]...,
    )
end

"""
    merge_generated_chains(dict::Dict)

Merge the `generated_chains` with the original chains and use the same range.
"""
function merge_generated_chains(dict::Dict)
    return hcat(dict[:chains], setrange(dict[:generated_chains], range(dict[:chains])))
end




"""
    get_merged_chains(model::Model, chains::AbstractChains)

Execute `model` for each of the samples in `chain` and return original chain
merged with the values returned by the `model` for each sample.

# Examples
## General
Often you might have additional quantities computed inside the model that you want to
inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()
    f = interesting_quantity(θ, x)
    return (; f)
end
m = demo(data)
chain = sample(m, alg, n)
# To merge the `interesting_quantity(θ, x)`, where `θ` is replaced by samples
# from the posterior/`chain`, into the chains:
get_merged_chains(m, chain) # <= results in new `Chain` that includes `f`.
```
"""
function get_merged_chains(model::Model, chains::AbstractChains)

    dict = Dict{Symbol,Any}(:model => model, :chains => chains)

    dict[:generated_quantities] = get_generated_quantities(dict)
    dict[:variables] = get_variables(dict)
    dict[:N_samples] = get_N_samples(dict)
    dict[:N_chains] = get_N_chains(dict)

    dict[:generated_chains] = generated_quantities_to_chains(dict)
    return merge_generated_chains(dict)

end


