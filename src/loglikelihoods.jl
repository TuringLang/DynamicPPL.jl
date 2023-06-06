# Context version
struct PointwiseLikelihoodContext{A,Ctx} <: AbstractContext
    loglikelihoods::A
    context::Ctx
end

function PointwiseLikelihoodContext(
    likelihoods=OrderedDict{VarName,Vector{Float64}}(),
    context::AbstractContext=LikelihoodContext(),
)
    return PointwiseLikelihoodContext{typeof(likelihoods),typeof(context)}(
        likelihoods, context
    )
end

NodeTrait(::PointwiseLikelihoodContext) = IsParent()
childcontext(context::PointwiseLikelihoodContext) = context.context
function setchildcontext(context::PointwiseLikelihoodContext, child)
    return PointwiseLikelihoodContext(context.loglikelihoods, child)
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{VarName,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{VarName,Float64}},
    vn::VarName,
    logp::Real,
)
    return context.loglikelihoods[vn] = logp
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{String,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, string(vn), Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{String,Float64}},
    vn::VarName,
    logp::Real,
)
    return context.loglikelihoods[string(vn)] = logp
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{String,Vector{Float64}}},
    vn::String,
    logp::Real,
)
    lookup = context.loglikelihoods
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLikelihoodContext{<:AbstractDict{String,Float64}},
    vn::String,
    logp::Real,
)
    return context.loglikelihoods[vn] = logp
end

function tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return tilde_observe!!(context.context, right, left, vi)
end
function tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vn, vi)
    # Need the `logp` value, so we cannot defer `acclogp!` to child-context, i.e.
    # we have to intercept the call to `tilde_observe!`.
    logp, vi = tilde_observe(context.context, right, left, vi)

    # Track loglikelihood value.
    push!(context, vn, logp)

    return left, acclogp!!(vi, logp)
end

function dot_tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return dot_tilde_observe!!(context.context, right, left, vi)
end
function dot_tilde_observe!!(context::PointwiseLikelihoodContext, right, left, vn, vi)
    # Need the `logp` value, so we cannot defer `acclogp!` to child-context, i.e.
    # we have to intercept the call to `dot_tilde_observe!`.

    # We want to treat `.~` as a collection of independent observations,
    # hence we need the `logp` for each of them. Broadcasting the univariate
    # `tilde_obseve` does exactly this.
    logps = _pointwise_tilde_observe(context.context, right, left, vi)

    # Need to unwrap the `vn`, i.e. get one `VarName` for each entry in `left`.
    _, _, vns = unwrap_right_left_vns(right, left, vn)
    for (vn, logp) in zip(vns, logps)
        # Track loglikelihood value.
        push!(context, vn, logp)
    end

    return left, acclogp!!(vi, sum(logps))
end

# FIXME: This is really not a good approach since it needs to stay in sync with
# the `dot_assume` implementations, but as things are _right now_ this is the best we can do.
function _pointwise_tilde_observe(context, right, left, vi)
    # We need to drop the `vi` returned.
    return broadcast(right, left) do r, l
        return first(tilde_observe(context, r, l, vi))
    end
end

function _pointwise_tilde_observe(
    context, right::MultivariateDistribution, left::AbstractMatrix, vi::AbstractVarInfo
)
    # We need to drop the `vi` returned.
    return map(eachcol(left)) do l
        return first(tilde_observe(context, right, l, vi))
    end
end

"""
    pointwise_loglikelihoods(model::Model, chain::Chains, keytype = String)

Runs `model` on each sample in `chain` returning a `OrderedDict{String, Matrix{Float64}}`
with keys corresponding to symbols of the observations, and values being matrices
of shape `(num_chains, num_samples)`.

`keytype` specifies what the type of the keys used in the returned `OrderedDict` are.
Currently, only `String` and `VarName` are supported.

# Notes
Say `y` is a `Vector` of `n` i.i.d. `Normal(μ, σ)` variables, with `μ` and `σ`
both being `<:Real`. Then the *observe* (i.e. when the left-hand side is an
*observation*) statements can be implemented in three ways:
1. using a `for` loop:
```julia
for i in eachindex(y)
    y[i] ~ Normal(μ, σ)
end
```
2. using `.~`:
```julia
y .~ Normal(μ, σ)
```
3. using `MvNormal`:
```julia
y ~ MvNormal(fill(μ, n), σ^2 * I)
```

In (1) and (2), `y` will be treated as a collection of `n` i.i.d. 1-dimensional variables,
while in (3) `y` will be treated as a _single_ n-dimensional observation.

This is important to keep in mind, in particular if the computation is used
for downstream computations.

# Examples
## From chain
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
OrderedDict{String,Array{Float64,2}} with 4 entries:
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_loglikelihoods(model, chain, String)
OrderedDict{String,Array{Float64,2}} with 4 entries:
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_loglikelihoods(model, chain, VarName)
OrderedDict{VarName,Array{Float64,2}} with 4 entries:
  xs[1] => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  xs[2] => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  xs[3] => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  y     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]
```

## Broadcasting
Note that `x .~ Dist()` will treat `x` as a collection of
_independent_ observations rather than as a single observation.

```jldoctest; setup = :(using Distributions)
julia> @model function demo(x)
           x .~ Normal()
       end;

julia> m = demo([1.0, ]);

julia> ℓ = pointwise_loglikelihoods(m, VarInfo(m)); first(ℓ[@varname(x[1])])
-1.4189385332046727

julia> m = demo([1.0; 1.0]);

julia> ℓ = pointwise_loglikelihoods(m, VarInfo(m)); first.((ℓ[@varname(x[1])], ℓ[@varname(x[2])]))
(-1.4189385332046727, -1.4189385332046727)
```

"""
function pointwise_loglikelihoods(model::Model, chain, keytype::Type{T}=String) where {T}
    # Get the data by executing the model once
    vi = VarInfo(model)
    context = PointwiseLikelihoodContext(OrderedDict{T,Vector{Float64}}())

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        model(vi, context)
    end

    niters = size(chain, 1)
    nchains = size(chain, 3)
    loglikelihoods = OrderedDict(
        varname => reshape(logliks, niters, nchains) for
        (varname, logliks) in context.loglikelihoods
    )
    return loglikelihoods
end

function pointwise_loglikelihoods(model::Model, varinfo::AbstractVarInfo)
    context = PointwiseLikelihoodContext(OrderedDict{VarName,Vector{Float64}}())
    model(varinfo, context)
    return context.loglikelihoods
end
