# Context version
struct PointwiseLikelihoodContext{A,Ctx} <: AbstractContext
    loglikelihoods::A
    context::Ctx
end

function PointwiseLikelihoodContext(
    likelihoods=Dict{VarName,Vector{Float64}}(),
    context::AbstractContext=LikelihoodContext(),
)
    return PointwiseLikelihoodContext{typeof(likelihoods),typeof(context)}(
        likelihoods, context
    )
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{VarName,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{VarName,Float64}}, vn::VarName, logp::Real
)
    return context.loglikelihoods[vn] = logp
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{String,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, string(vn), Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{String,Float64}}, vn::VarName, logp::Real
)
    return context.loglikelihoods[string(vn)] = logp
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{String,Vector{Float64}}},
    vn::String,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{Dict{String,Float64}}, vn::String, logp::Real
)
    return context.loglikelihoods[vn] = logp
end

function tilde_assume(context::PointwiseLikelihoodContext, right, vn, inds, vi)
    return tilde_assume(context.context, right, vn, inds, vi)
end

function dot_tilde_assume(context::PointwiseLikelihoodContext, right, left, vn, inds, vi)
    return dot_tilde_assume(context.context, right, left, vn, inds, vi)
end

function tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return tilde_observe!!(context.context, right, left, vi)
end
function tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vn, vinds, vi)
    # Need the `logp` value, so we cannot defer `acclogp!` to child-context, i.e.
    # we have to intercept the call to `tilde_observe!!`.
    logp = tilde_observe(context.context, right, left, vi)

    # Track loglikelihood value.
    push!(context, vn, logp)

    return left, acclogp!(vi, logp)
end

function dot_tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return dot_tilde_observe!!(context.context, right, left, vi)
end
function dot_tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vn, inds, vi)
    # Need the `logp` value, so we cannot defer `acclogp!` to child-context, i.e.
    # we have to intercept the call to `dot_tilde_observe!!`.
    logp = dot_tilde_observe(context.context, right, left, vi)

    # Track loglikelihood value.
    push!(context, vn, logp)

    return left, acclogp!(vi, logp)
end

"""
    pointwise_loglikelihoods(model::Model, chain::Chains, keytype = String)

Runs `model` on each sample in `chain` returning a `Dict{String, Matrix{Float64}}`
with keys corresponding to symbols of the observations, and values being matrices
of shape `(num_chains, num_samples)`.

`keytype` specifies what the type of the keys used in the returned `Dict` are.
Currently, only `String` and `VarName` are supported.

# Notes
Say `y` is a `Vector` of `n` i.i.d. `Normal(μ, σ)` variables, with `μ` and `σ`
both being `<:Real`. Then the *observe* (i.e. when the left-hand side is an
*observation*) statements can be implemented in two ways:
```julia
for i in eachindex(y)
    y[i] ~ Normal(μ, σ)
end
```
or
```julia
y ~ MvNormal(fill(μ, n), fill(σ, n))
```
Unfortunately, just by looking at the latter statement, it's impossible to tell 
whether or not this is one *single* observation which is `n` dimensional OR if we
have *multiple* 1-dimensional observations. Therefore, `loglikelihoods` will only
work with the first example.

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

julia> pointwise_loglikelihoods(model, chain)
Dict{String,Array{Float64,2}} with 4 entries:
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_loglikelihoods(model, chain, String)
Dict{String,Array{Float64,2}} with 4 entries:
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_loglikelihoods(model, chain, VarName)
Dict{VarName,Array{Float64,2}} with 4 entries:
  xs[2] => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  y     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]
  xs[1] => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  xs[3] => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
```
"""
function pointwise_loglikelihoods(model::Model, chain, keytype::Type{T}=String) where {T}
    # Get the data by executing the model once
    vi = VarInfo(model)
    context = PointwiseLikelihoodContext(Dict{T,Vector{Float64}}())

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        model(vi, context)
    end

    niters = size(chain, 1)
    nchains = size(chain, 3)
    loglikelihoods = Dict(
        varname => reshape(logliks, niters, nchains) for
        (varname, logliks) in context.loglikelihoods
    )
    return loglikelihoods
end

function pointwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo)
    context = PointwiseLikelihoodContext(Dict{VarName,Vector{Float64}}())
    model(varinfo, context)
    return context.loglikelihoods
end
