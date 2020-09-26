# Improved implementation
struct LikelihoodSampler{T} <: AbstractSampler
    loglikelihoods::T
end

LikelihoodSampler() = LikelihoodSampler(Dict{String, Vector{Float64}}())
getspace(::LikelihoodSampler) = ()
has_eval_num(::LikelihoodSampler) = false

init(rng, dist, ::LikelihoodSampler) = rand(rng, dist)
init(rng, dist, ::LikelihoodSampler, n::Int) = rand(rng, dist, n)

function getindex(vi::VarInfo, spl::LikelihoodSampler)
    return getindex(vi, SampleFromPrior())
end

function assume(
    rng,
    spl::LikelihoodSampler,
    dist::Distribution,
    vn::VarName,
    vi
)
    return assume(rng, SampleFromPrior(), dist, vn, vi)
end

function dot_assume(
    rng,
    spl::LikelihoodSampler,
    dists::Any,
    vns::AbstractArray{<:VarName},
    var::Any,
    vi::Any,
)
    return dot_assume(rng, SampleFromPrior(), dists, vns, var, vi)
end

function tilde_observe(
    ctx, sampler::LikelihoodSampler, right, left, vname, vinds, vi
)
    logp = tilde(ctx, SampleFromPrior(), right, left, vi)
    acclogp!(vi, logp)

    # Add `logp` to the corresponding entry in `likelihoods`
    parent_name = string(getsym(vname))
    name = string(vname)

    # lookup = if isempty(getindexing(vname))
    #     # No "parent" variable
    #     sampler.loglikelihoods
    # else
    #     parent_name = string(getsym(vname))
    #     get!(sampler.loglikelihoods, parent_name, Dict{String, Vector{Float64}}())
    # end
    lookup = sampler.loglikelihoods
    ℓ = get!(lookup, string(vname), Float64[])
    push!(ℓ, logp)
    
    return left
end

function loglikelihoods(model::Model, chain)
    # Get the data by executing the model once
    spl = LikelihoodSampler()
    vi = Turing.VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Clear previous values
        empty!(vi)

        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        model(vi, spl)
    end
    return spl.loglikelihoods
end
