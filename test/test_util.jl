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
    vi = VarInfo(model)
    vi = DynamicPPL.setaccs!!(vi, (DynamicPPL.ValuesAsInModelAccumulator(false),))
    ps = [
        ParamsWithStats(last(DynamicPPL.init!!(rng, model, vi)), nothing) for _ in 1:n_iters
    ]
    return DynamicPPL.to_chains(MCMCChains.Chains, ps)
end
function make_chain_from_prior(model::Model, n_iters::Int)
    return make_chain_from_prior(Random.default_rng(), model, n_iters)
end
