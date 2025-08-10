# default model
@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
const gdemo_default = gdemo_d()

"""
    short_varinfo_name(vi::AbstractVarInfo)

Return string representing a short description of `vi`.
"""
function short_varinfo_name(vi::DynamicPPL.ThreadSafeVarInfo)
    return "threadsafe($(short_varinfo_name(vi.varinfo)))"
end
function short_varinfo_name(vi::DynamicPPL.NTVarInfo)
    return if DynamicPPL.has_varnamedvector(vi)
        "TypedVectorVarInfo"
    else
        "TypedVarInfo"
    end
end
short_varinfo_name(::DynamicPPL.UntypedVarInfo) = "UntypedVarInfo"
short_varinfo_name(::DynamicPPL.UntypedVectorVarInfo) = "UntypedVectorVarInfo"
function short_varinfo_name(::SimpleVarInfo{<:NamedTuple,<:Ref})
    return "SimpleVarInfo{<:NamedTuple,<:Ref}"
end
function short_varinfo_name(::SimpleVarInfo{<:OrderedDict,<:Ref})
    return "SimpleVarInfo{<:OrderedDict,<:Ref}"
end
function short_varinfo_name(::SimpleVarInfo{<:DynamicPPL.VarNamedVector,<:Ref})
    return "SimpleVarInfo{<:VarNamedVector,<:Ref}"
end
short_varinfo_name(::SimpleVarInfo{<:NamedTuple}) = "SimpleVarInfo{<:NamedTuple}"
short_varinfo_name(::SimpleVarInfo{<:OrderedDict}) = "SimpleVarInfo{<:OrderedDict}"
function short_varinfo_name(::SimpleVarInfo{<:DynamicPPL.VarNamedVector})
    return "SimpleVarInfo{<:VarNamedVector}"
end

# convenient functions for testing model.jl
# function to modify the representation of values based on their length
function modify_value_representation(nt::NamedTuple)
    modified_nt = NamedTuple()
    for (key, value) in zip(keys(nt), values(nt))
        if length(value) == 1  # Scalar value
            modified_value = value[1]
        else  # Non-scalar value
            modified_value = value
        end
        modified_nt = merge(modified_nt, (key => modified_value,))
    end
    return modified_nt
end

"""
    make_chain_from_prior([rng,] model, n_iters)

Construct an MCMCChains.Chains object by sampling from the prior of `model` for
`n_iters` iterations.
"""
function make_chain_from_prior(rng::Random.AbstractRNG, model::Model, n_iters::Int)
    # Sample from the prior
    varinfos = [VarInfo(rng, model) for _ in 1:n_iters]
    # Extract all varnames found in any dictionary. Doing it this way guards
    # against the possibility of having different varnames in different
    # dictionaries, e.g. for models that have dynamic variables / array sizes
    varnames = OrderedSet{VarName}()
    # Convert each varinfo into an OrderedDict of vns => params.
    # We have to use varname_and_value_leaves so that each parameter is a scalar
    dicts = map(varinfos) do t
        vals = DynamicPPL.values_as(t, OrderedDict)
        iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
        tuples = mapreduce(collect, vcat, iters)
        # The following loop is a replacement for:
        #     push!(varnames, map(first, tuples)...)
        # which causes a stack overflow if `map(first, tuples)` is too large.
        # Unfortunately there isn't a union() function for OrderedSet.
        for vn in map(first, tuples)
            push!(varnames, vn)
        end
        OrderedDict(tuples)
    end
    # Convert back to list
    varnames = collect(varnames)
    # Construct matrix of values
    vals = [get(dict, vn, missing) for dict in dicts, vn in varnames]
    # Construct and return the Chains object
    return Chains(vals, varnames)
end
function make_chain_from_prior(model::Model, n_iters::Int)
    return make_chain_from_prior(Random.default_rng(), model, n_iters)
end
