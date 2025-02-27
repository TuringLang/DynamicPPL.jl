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

function _include_prior(context::PointwiseLogdensityContext)
    return leafcontext(context) isa Union{PriorContext,DefaultContext}
end
function _include_likelihood(context::PointwiseLogdensityContext)
    return leafcontext(context) isa Union{LikelihoodContext,DefaultContext}
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

# Note on submodels (penelopeysm)
#
# We don't need to overload tilde_observe!! for Sampleables (yet), because it
# is currently not possible to evaluate a model with a Sampleable on the RHS
# of an observe statement.
#
# Note that calling tilde_assume!! on a Sampleable does not necessarily imply
# that there are no observe statements inside the Sampleable. There could well
# be likelihood terms in there, which must be included in the returned logp.
# See e.g. the `demo_dot_assume_observe_submodel` demo model.
#
# This is handled by passing the same context to rand_like!!, which figures out
# which terms to include using the context, and also mutates the context and vi
# appropriately. Thus, we don't need to check against _include_prior(context)
# here.
function tilde_assume!!(context::PointwiseLogdensityContext, right::Sampleable, vn, vi)
    value, vi = DynamicPPL.rand_like!!(right, context, vi)
    return value, vi
end

function tilde_assume!!(context::PointwiseLogdensityContext, right, vn, vi)
    !_include_prior(context) && return (tilde_assume!!(context.context, right, vn, vi))
    value, logp, vi = tilde_assume(context.context, right, vn, vi)
    # Track loglikelihood value.
    push!(context, vn, logp)
    return value, acclogp!!(vi, logp)
end

"""
    pointwise_logdensities(model::Model, chain::Chains, keytype = String)

Runs `model` on each sample in `chain` returning a `OrderedDict{String, Matrix{Float64}}`
with keys corresponding to symbols of the variables, and values being matrices
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

In (1) `y` will be treated as a collection of `n` i.i.d. 1-dimensional variables,
while in (2) and (3) `y` will be treated as a _single_ n-dimensional observation.

This is important to keep in mind, in particular if the computation is used
for downstream computations.

# Examples
## From chain
```jldoctest pointwise-logdensities-chains; setup=:(using Distributions)
julia> using MCMCChains

julia> @model function demo(xs, y)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end
           y ~ Normal(m, √s)
       end
demo (generic function with 2 methods)

julia> # Example observations.
       model = demo([1.0, 2.0, 3.0], [4.0]);

julia> # A chain with 3 iterations.
       chain = Chains(
           reshape(1.:6., 3, 2),
           [:s, :m]
       );

julia> pointwise_logdensities(model, chain)
OrderedDict{String, Matrix{Float64}} with 6 entries:
  "s"     => [-0.802775; -1.38222; -2.09861;;]
  "m"     => [-8.91894; -7.51551; -7.46824;;]
  "xs[1]" => [-5.41894; -5.26551; -5.63491;;]
  "xs[2]" => [-2.91894; -3.51551; -4.13491;;]
  "xs[3]" => [-1.41894; -2.26551; -2.96824;;]
  "y"     => [-0.918939; -1.51551; -2.13491;;]

julia> pointwise_logdensities(model, chain, String)
OrderedDict{String, Matrix{Float64}} with 6 entries:
  "s"     => [-0.802775; -1.38222; -2.09861;;]
  "m"     => [-8.91894; -7.51551; -7.46824;;]
  "xs[1]" => [-5.41894; -5.26551; -5.63491;;]
  "xs[2]" => [-2.91894; -3.51551; -4.13491;;]
  "xs[3]" => [-1.41894; -2.26551; -2.96824;;]
  "y"     => [-0.918939; -1.51551; -2.13491;;]

julia> pointwise_logdensities(model, chain, VarName)
OrderedDict{VarName, Matrix{Float64}} with 6 entries:
  s     => [-0.802775; -1.38222; -2.09861;;]
  m     => [-8.91894; -7.51551; -7.46824;;]
  xs[1] => [-5.41894; -5.26551; -5.63491;;]
  xs[2] => [-2.91894; -3.51551; -4.13491;;]
  xs[3] => [-1.41894; -2.26551; -2.96824;;]
  y     => [-0.918939; -1.51551; -2.13491;;]
```

## Broadcasting
Note that `x .~ Dist()` will treat `x` as a single multivariate observation.

```jldoctest; setup = :(using Distributions)
julia> @model function demo(x)
           x .~ Normal()
       end;

julia> m = demo([1.0; 1.0]);

julia> ℓ = pointwise_logdensities(m, VarInfo(m)); first(ℓ[@varname(x)])
-2.8378770664093453
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
