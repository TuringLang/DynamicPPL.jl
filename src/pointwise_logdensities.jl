# Context version
struct PointwiseLogdensityContext{A,Ctx} <: AbstractContext
    logdensities::A
    context::Ctx
end

function PointwiseLogdensityContext(
    likelihoods=OrderedDict{VarName,Vector{Float64}}(),
    context::AbstractContext=DefaultContext(),
)
    return PointwiseLogdensityContext{typeof(likelihoods),typeof(context)}(
        likelihoods, context
    )
end

NodeTrait(::PointwiseLogdensityContext) = IsParent()
childcontext(context::PointwiseLogdensityContext) = context.context
function setchildcontext(context::PointwiseLogdensityContext, child)
    return PointwiseLogdensityContext(context.logdensities, child)
end

function _include_prior(context::PointwiseLogdensityContext)
    return leafcontext(context) isa Union{PriorContext,DefaultContext}
end
function _include_likelihood(context::PointwiseLogdensityContext)
    return leafcontext(context) isa Union{LikelihoodContext,DefaultContext}
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{VarName,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.logdensities
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{VarName,Float64}},
    vn::VarName,
    logp::Real,
)
    return context.logdensities[vn] = logp
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{String,Vector{Float64}}},
    vn::VarName,
    logp::Real,
)
    lookup = context.logdensities
    ℓ = get!(lookup, string(vn), Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{String,Float64}},
    vn::VarName,
    logp::Real,
)
    return context.logdensities[string(vn)] = logp
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{String,Vector{Float64}}},
    vn::String,
    logp::Real,
)
    lookup = context.logdensities
    ℓ = get!(lookup, vn, Float64[])
    return push!(ℓ, logp)
end

function Base.push!(
    context::PointwiseLogdensityContext{<:AbstractDict{String,Float64}},
    vn::String,
    logp::Real,
)
    return context.logdensities[vn] = logp
end

function tilde_observe!!(context::PointwiseLogdensityContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return tilde_observe!!(context.context, right, left, vi)
end
function tilde_observe!!(context::PointwiseLogdensityContext, right, left, vn, vi)
    # Completely defer to child context if we are not tracking likelihoods.
    if !(_include_likelihood(context))
        return tilde_observe!!(context.context, right, left, vn, vi)
    end

    # Need the `logp` value, so we cannot defer `acclogp!` to child-context, i.e.
    # we have to intercept the call to `tilde_observe!`.
    logp, vi = tilde_observe(context.context, right, left, vi)

    # Track loglikelihood value.
    push!(context, vn, logp)

    return left, acclogp!!(vi, logp)
end

function dot_tilde_observe!!(context::PointwiseLogdensityContext, right, left, vi)
    # Defer literal `observe` to child-context.
    return dot_tilde_observe!!(context.context, right, left, vi)
end
function dot_tilde_observe!!(context::PointwiseLogdensityContext, right, left, vn, vi)
    # Completely defer to child context if we are not tracking likelihoods.
    if !(_include_likelihood(context))
        return dot_tilde_observe!!(context.context, right, left, vn, vi)
    end

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

function tilde_assume!!(context::PointwiseLogdensityContext, right, vn, vi)
    # Completely defer to child context if we are not tracking prior densities.
    _include_prior(context) || return tilde_assume!!(context.context, right, vn, vi)

    # Otherwise, capture the return values.
    value, logp, vi = tilde_assume(context.context, right, vn, vi)
    # Track loglikelihood value.
    push!(context, vn, logp)

    return value, acclogp!!(vi, logp)
end

function dot_tilde_assume!!(context::PointwiseLogdensityContext, right, left, vns, vi)
    # Completely defer to child context if we are not tracking prior densities.
    if !(_include_prior(context))
        return dot_tilde_assume!!(context.context, right, left, vns, vi)
    end

    value, logps = _pointwise_tilde_assume(context, right, left, vns, vi)
    # Track loglikelihood values.
    for (vn, logp) in zip(vns, logps)
        push!(context, vn, logp)
    end
    return value, acclogp!!(vi, sum(logps))
end

function _pointwise_tilde_assume(context, right, left, vns, vi)
    # We need to drop the `vi` returned.
    values_and_logps = broadcast(right, left, vns) do r, l, vn
        # HACK(torfjelde): This drops the `vi` returned, which means the `vi` is not updated
        # in case of immutable varinfos. But a) atm we're only using mutable varinfos for this,
        # and b) even if the variables aren't stored in the vi correctly, we're not going to use
        # this vi for anything downstream anyways, i.e. I don't see a case where this would matter
        # for this particular use case.
        val, logp, _ = tilde_assume(context, r, vn, vi)
        return val, logp
    end
    return map(first, values_and_logps), map(last, values_and_logps)
end

function _pointwise_tilde_assume(
    context, right::MultivariateDistribution, left::AbstractMatrix, vns, vi
)
    # We need to drop the `vi` returned.
    values_and_logps = map(eachcol(left), vns) do l, vn
        val, logp, _ = tilde_assume(context, right, vn, vi)
        return val, logp
    end
    # HACK(torfjelde): Due to the way we handle `.~`, we should use `recombine` to stay consistent.
    # But this also means that we need to first flatten the entire `values` component before recombining.
    values = recombine(right, mapreduce(vec ∘ first, vcat, values_and_logps), length(vns))
    return values, map(last, values_and_logps)
end

"""
    pointwise_logdensities(model::Model, chain::Chains[, keytype::Type, context::AbstractContext])

Runs `model` on each sample in `chain` returning a `OrderedDict{String, Matrix{Float64}}`
with keys corresponding to symbols of the variables, and values being matrices
of shape `(num_chains, num_samples)`.

# Arguments
- `model`: the `Model` to run.
- `chain`: the `Chains` to run the model on.
- `keytype`: the type of the keys used in the returned `OrderedDict` are.
  Currently, only `String` and `VarName` are supported.
- `context`: the context to use when running the model. Default: `DefaultContext`.
  The [`leafcontext`](@ref) is used to decide which variables to include.

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

julia> pointwise_logdensities(model, chain)
OrderedDict{String,Array{Float64,2}} with 4 entries:
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_logdensities(model, chain, String)
OrderedDict{String,Array{Float64,2}} with 4 entries:
  "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
  "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
  "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
  "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

julia> pointwise_logdensities(model, chain, VarName)
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

julia> ℓ = pointwise_logdensities(m, VarInfo(m)); first(ℓ[@varname(x[1])])
-1.4189385332046727

julia> m = demo([1.0; 1.0]);

julia> ℓ = pointwise_logdensities(m, VarInfo(m)); first.((ℓ[@varname(x[1])], ℓ[@varname(x[2])]))
(-1.4189385332046727, -1.4189385332046727)
```

"""
function pointwise_logdensities(
    model::Model, chain, keytype::Type{T}=String, context::AbstractContext=DefaultContext()
) where {T}
    # Get the data by executing the model once
    vi = VarInfo(model)
    point_context = PointwiseLogdensityContext(OrderedDict{T,Vector{Float64}}(), context)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        model(vi, point_context)
    end

    niters = size(chain, 1)
    nchains = size(chain, 3)
    logdensities = OrderedDict(
        varname => reshape(logliks, niters, nchains) for
        (varname, logliks) in point_context.logdensities
    )
    return logdensities
end

function pointwise_logdensities(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext=DefaultContext()
)
    point_context = PointwiseLogdensityContext(
        OrderedDict{VarName,Vector{Float64}}(), context
    )
    model(varinfo, point_context)
    return point_context.logdensities
end

"""
    pointwise_loglikelihoods(model, chain[, keytype, context])

Compute the pointwise log-likelihoods of the model given the chain.

This is the same as `pointwise_logdensities(model, chain, context)`, but only
including the likelihood terms.

See also: [`pointwise_logdensities`](@ref).
"""
function pointwise_loglikelihoods(
    model::Model,
    chain,
    keytype::Type{T}=String,
    context::AbstractContext=LikelihoodContext(),
) where {T}
    if !(leafcontext(context) isa LikelihoodContext)
        throw(ArgumentError("Leaf context should be a LikelihoodContext"))
    end

    return pointwise_logdensities(model, chain, T, context)
end

function pointwise_loglikelihoods(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext=LikelihoodContext()
)
    if !(leafcontext(context) isa LikelihoodContext)
        throw(ArgumentError("Leaf context should be a LikelihoodContext"))
    end

    return pointwise_logdensities(model, varinfo, context)
end

"""
    pointwise_prior_logdensities(model, chain[, keytype, context])

Compute the pointwise log-prior-densities of the model given the chain.

This is the same as `pointwise_logdensities(model, chain, context)`, but only
including the prior terms.

See also: [`pointwise_logdensities`](@ref).
"""
function pointwise_prior_logdensities(
    model::Model, chain, keytype::Type{T}=String, context::AbstractContext=PriorContext()
) where {T}
    if !(leafcontext(context) isa PriorContext)
        throw(ArgumentError("Leaf context should be a PriorContext"))
    end

    return pointwise_logdensities(model, chain, T, context)
end

function pointwise_prior_logdensities(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext=PriorContext()
)
    if !(leafcontext(context) isa PriorContext)
        throw(ArgumentError("Leaf context should be a PriorContext"))
    end

    return pointwise_logdensities(model, varinfo, context)
end
