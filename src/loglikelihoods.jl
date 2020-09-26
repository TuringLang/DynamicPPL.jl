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

"""
    loglikelihoods(model::Model, chain::Chains)

Runs `model` on each sample in `chain` returning an array of arrays with
the i-th element inner arrays corresponding to the the likelihood of the i-th
observation for that particular sample in `chain`.

# Notes
Say `y` is a `Vector` of `n` i.i.d. `Normal(μ, σ)` variables, with `μ` and `σ`
both being `<:Real`. Then the observe statements can be implemented in two ways:
```julia
for i in eachindex(y)
    y[i] ~ Normal(μ, σ)
end
```
or
```julia
y ~ MvNormal(fill(μ, n), fill(σ, n))
```
Unfortunately, just by looking at the latter statement, it's impossible to tell whether or
not this is one *single* observation which is `n` dimensional OR if we have *multiple*
1-dimensional observations. Therefore, `loglikelihoods` will only work with the first
example.

# Examples
```julia-repl
julia> using DynamicPPL, Turing

julia> @model function demo(xs, y)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end

           y ~ Normal(m, √s)
       end
demo (generic function with 1 method)

julia> model = demo(randn(3), randn());

julia> chain = sample(model, MH(), 10);

julia> DynamicPPL.loglikelihoods(model, chain)
Dict{String,Array{Float64,1}} with 4 entries:
  "xs[3]" => [-1.02616, -1.26931, -1.05003, -5.05458, -1.33825, -1.02904, -1.23761, -1.30128, -1.04872, -2.03716]
  "xs[1]" => [-2.08205, -2.51387, -3.03175, -2.5981, -2.31322, -2.62284, -2.70874, -1.18617, -1.36281, -4.39839]
  "xs[2]" => [-2.20604, -2.63495, -3.22802, -2.48785, -2.40941, -2.78791, -2.85013, -1.24081, -1.46019, -4.59025]
  "y"     => [-1.36627, -1.21964, -1.03342, -7.46617, -1.3234, -1.14536, -1.14781, -2.48912, -2.23705, -1.26267]
```
"""
function loglikelihoods(model::Model, chain)
    # Get the data by executing the model once
    spl = LikelihoodSampler()
    ctx = LikelihoodContext()
    vi = VarInfo(model, ctx)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        model(vi, spl, ctx)
    end
    return spl.loglikelihoods
end
