# Context version
struct ElementwiseLikelihoodContext{A, Ctx} <: AbstractContext
    loglikelihoods::A
    ctx::Ctx
end

function ElementwiseLikelihoodContext(
    likelihoods = Dict{VarName, Vector{Float64}}(),
    ctx::AbstractContext = LikelihoodContext()
)
    return ElementwiseLikelihoodContext{typeof(likelihoods),typeof(ctx)}(likelihoods, ctx)
end

function Base.push!(
    ctx::ElementwiseLikelihoodContext{Dict{VarName, Vector{Float64}}},
    vn::VarName,
    logp::Real
)
    lookup = ctx.loglikelihoods
    ℓ = get!(lookup, vn, Float64[])
    push!(ℓ, logp)
end

function Base.push!(
    ctx::ElementwiseLikelihoodContext{Dict{VarName, Float64}},
    vn::VarName,
    logp::Real
)
    ctx.loglikelihoods[vn] = logp
end


function tilde_assume(rng, ctx::ElementwiseLikelihoodContext, sampler, right, vn, inds, vi)
    return tilde_assume(rng, ctx.ctx, sampler, right, vn, inds, vi)
end

function dot_tilde_assume(rng, ctx::ElementwiseLikelihoodContext, sampler, right, left, vn, inds, vi)
    value, logp = dot_tilde(rng, ctx.ctx, sampler, right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end


function tilde_observe(ctx::ElementwiseLikelihoodContext, sampler, right, left, vname, vinds, vi)
    # This is slightly unfortunate since it is not completely generic...
    # Ideally we would call `tilde_observe` recursively but then we don't get the
    # loglikelihood value.
    logp = tilde(ctx.ctx, sampler, right, left, vi)
    acclogp!(vi, logp)

    # track loglikelihood value
    push!(ctx, vname, logp)

    return left
end


"""
    elementwise_loglikelihoods(model::Model, chain::Chains)

Runs `model` on each sample in `chain` returning an array of arrays with
the i-th element inner arrays corresponding to the the likelihood of the i-th
observation for that particular sample in `chain`.

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

julia> DynamicPPL.elementwise_loglikelihoods(model, chain)
Dict{String,Array{Float64,1}} with 4 entries:
  "xs[3]" => [-1.02616, -1.26931, -1.05003, -5.05458, -1.33825, -1.02904, -1.23761, -1.30128, -1.04872, -2.03716]
  "xs[1]" => [-2.08205, -2.51387, -3.03175, -2.5981, -2.31322, -2.62284, -2.70874, -1.18617, -1.36281, -4.39839]
  "xs[2]" => [-2.20604, -2.63495, -3.22802, -2.48785, -2.40941, -2.78791, -2.85013, -1.24081, -1.46019, -4.59025]
  "y"     => [-1.36627, -1.21964, -1.03342, -7.46617, -1.3234, -1.14536, -1.14781, -2.48912, -2.23705, -1.26267]
```
"""
function elementwise_loglikelihoods(model::Model, chain)
    # Get the data by executing the model once
    spl = SampleFromPrior()
    vi = VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    loglikelihoods = map(1:size(chain, 3)) do chain_idx
        ctx = ElementwiseLikelihoodContext()
        for sample_idx = 1:size(chain, 1)
            # Update the values
            setval!(vi, chain, sample_idx, chain_idx)

            # Execute model
            model(vi, spl, ctx)
        end
        return ctx.loglikelihoods
    end

    K = keytype(loglikelihoods[1])
    T = valtype(loglikelihoods[1])
    res = Dict{K, Vector{T}}()
    for ℓ in loglikelihoods
        for (k, v) in ℓ
            container = get!(res, k, T[])
            push!(container, v)
        end
    end

    return res
end

function elementwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo)
    ctx = ElementwiseLikelihoodContext(Dict{VarName, Float64}())
    model(varinfo, SampleFromPrior(), ctx)
    return ctx.loglikelihoods
end
