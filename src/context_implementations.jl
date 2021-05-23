using Distributions:
    UnivariateDistribution, MultivariateDistribution, MatrixDistribution, Distribution

const AMBIGUITY_MSG =
    "Ambiguous `LHS .~ RHS` or `@. LHS ~ RHS` syntax. The broadcasting " *
    "can either be column-wise following the convention of Distributions.jl or " *
    "element-wise following Julia's general broadcasting semantics. Please make sure " *
    "that the element type of `LHS` is not a supertype of the support type of " *
    "`AbstractVector` to eliminate ambiguity."

alg_str(spl::Sampler) = string(nameof(typeof(spl.alg)))

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

_getindex(x, inds::Tuple) = _getindex(x[first(inds)...], Base.tail(inds))
_getindex(x, inds::Tuple{}) = x

include("context_implementations/prior.jl")
include("context_implementations/likelihood.jl")
include("context_implementations/minibatch.jl")
include("context_implementations/prefix.jl")

# assume
function tilde(
    rng,
    ctx::Union{SamplingContext,EvaluationContext},
    sampler,
    right,
    left,
    vn::VarName,
    _,
    vi,
)
    return tilde_primitive(rng, ctx, sampler, right, left, vn, vi)
end

"""
    tilde_assume(rng, ctx, sampler, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value.

Falls back to `tilde(rng, ctx, sampler, right, vn, inds, vi)`.
"""
function tilde_assume(rng, ctx, sampler, right, vn, inds, vi)
    value, logp = tilde(rng, ctx, sampler, right, nothing, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function tilde_primitive(rng, ctx::SamplingContext, sampler, right, left, vn::VarName, vi)
    return assume(rng, sampler, right, nothing, vn, vi)
end
function tilde_primitive(
    rng, ctx::EvaluationContext, sampler, right, left::Nothing, vn::VarName, vi
)
    return assume(sampler, right, vi[vn], vn, vi)
end
function tilde_primitive(rng, ctx::EvaluationContext, sampler, right, left, vn::VarName, vi)
    return assume(sampler, right, left, vn, vi)
end

# observe
function tilde(ctx::Union{SamplingContext,EvaluationContext}, sampler, right, left, vi)
    return tilde_primitive(sampler, right, left, vi)
end

"""
    tilde_observe(ctx, sampler, right, left, vname, vinds, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `tilde(ctx, sampler, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe(ctx, sampler, right, left, vname, vinds, vi)
    logp = tilde(ctx, sampler, right, left, vi)
    acclogp!(vi, logp)
    return left
end

"""
    tilde_observe(ctx, sampler, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.

Falls back to `tilde(ctx, sampler, right, left, vi)`.
"""
function tilde_observe(ctx, sampler, right, left, vi)
    logp = tilde(ctx, sampler, right, left, vi)
    acclogp!(vi, logp)
    return left
end

tilde_primitive(sampler, right, left, vi) = observe(sampler, right, left, vi)

function assume(rng, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    return error("DynamicPPL.observe: unmanaged inference algorithm: $(typeof(spl))")
end

function assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    left::Nothing,
    vn::VarName,
    vi,
)
    r = init(rng, dist, spl)
    if haskey(vi, vn)
        vi[vn] = vectorize(dist, r)
        setorder!(vi, vn, get_num_produce(vi))
    else
        push!(vi, vn, r, dist, spl)
    end
    settrans!(vi, false, vn)
    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(
    spl::Union{SampleFromPrior,SampleFromUniform}, dist::Distribution, left, vn::VarName, vi
)
    return left, Bijectors.logpdf_with_trans(dist, left, istrans(vi, vn))
end

function observe(
    spl::Union{SampleFromPrior,SampleFromUniform}, dist::Distribution, left, vi
)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(dist, left)
end

# .~ functions

# assume
function dot_tilde(
    rng, ctx::Union{SamplingContext,EvaluationContext}, sampler, right, left, vn, _, vi
)
    return dot_tilde_primitive(rng, ctx, sampler, right, left, vn, vi)
end

"""
    dot_tilde_assume(rng, ctx, sampler, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value.

Falls back to `dot_tilde(rng, ctx, sampler, right, left, vn, inds, vi)`.
"""
function dot_tilde_assume(rng, ctx, sampler, right, left, vn, inds, vi)
    value, logp = dot_tilde(rng, ctx, sampler, right, nothing, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function dot_tilde_primitive(rng, ctx::SamplingContext, sampler, right, left, vns, vi)
    return dot_assume(rng, sampler, right, vns, nothing, vi)
end

function dot_tilde_primitive(rng, ctx::EvaluationContext, sampler, right, left, vns, vi)
    return dot_assume(sampler, right, vns, left, vi)
end

function dot_tilde_primitive(
    rng, ctx::EvaluationContext, sampler, right, left::Nothing, vns, vi
)
    return dot_assume(sampler, right, vns, vi[vns], vi)
end

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function dot_tilde_primitive(
    rng,
    ctx,
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution,AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vn::AbstractVector{<:VarName},
    vi,
)
    return throw(DimensionMismatch(AMBIGUITY_MSG))
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::Nothing,
    vi,
)
    @assert length(dist) == size(var, 1)
    r = get_and_set_val!(rng, vi, vns, dist, spl)
    lp = sum(Bijectors.logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    return r, lp
end

function dot_assume(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi,
)
    @assert length(dist) == size(var, 1)
    lp = sum(Bijectors.logpdf_with_trans(dist, var, istrans(vi, vns[1])))
    return var, lp
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::Nothing,
    vi,
)
    r = get_and_set_val!(rng, vi, vns, dists, spl)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    return r, lp
end

function dot_assume(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi,
)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, var, istrans(vi, vns[1])))
    return var, lp
end

function dot_assume(rng, spl::Sampler, ::Any, ::AbstractArray{<:VarName}, ::Any, ::Any)
    return error(
        "[DynamicPPL] $(alg_str(spl)) doesn't support vectorizing assume statement"
    )
end

function get_and_set_val!(
    rng,
    vi,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    n = length(vns)
    if haskey(vi, vns[1])
        r = init(rng, dist, spl, n)
        for i in 1:n
            vn = vns[i]
            vi[vn] = vectorize(dist, r[:, i])
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        end
    else
        r = init(rng, dist, spl, n)
        for i in 1:n
            vn = vns[i]
            push!(vi, vn, r[:, i], dist, spl)
            settrans!(vi, false, vn)
        end
    end
    return r
end

function get_and_set_val!(
    rng,
    vi,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    if haskey(vi, vns[1])
        f = (vn, dist) -> init(rng, dist, spl)
        r = f.(vns, dists)
        for i in eachindex(vns)
            vn = vns[i]
            dist = dists isa AbstractArray ? dists[i] : dists
            vi[vn] = vectorize(dist, r[i])
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        end
    else
        f = (vn, dist) -> init(rng, dist, spl)
        r = f.(vns, dists)
        push!.(Ref(vi), vns, r, dists, Ref(spl))
        settrans!.(Ref(vi), false, vns)
    end
    return r
end

function set_val!(
    vi, vns::AbstractVector{<:VarName}, dist::MultivariateDistribution, val::AbstractMatrix
)
    @assert size(val, 2) == length(vns)
    foreach(enumerate(vns)) do (i, vn)
        vi[vn] = val[:, i]
    end
    return val
end
function set_val!(
    vi,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    val::AbstractArray,
)
    @assert size(val) == size(vns)
    foreach(CartesianIndices(val)) do ind
        dist = dists isa AbstractArray ? dists[ind] : dists
        vi[vns[ind]] = vectorize(dist, val[ind])
    end
    return val
end

# observe
function dot_tilde(ctx::Union{SamplingContext,EvaluationContext}, sampler, right, left, vi)
    return dot_tilde_primitive(sampler, right, left, vi)
end

"""
    dot_tilde_observe(ctx, sampler, right, left, vname, vinds, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `dot_tilde(ctx, sampler, right, left, vi)` ignoring the information about variable
name and indices; if needed, these can be accessed through this function, though.
"""
function dot_tilde_observe(ctx, sampler, right, left, vn, inds, vi)
    logp = dot_tilde(ctx, sampler, right, left, vi)
    acclogp!(vi, logp)
    return left
end

"""
    dot_tilde_observe(ctx, sampler, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value.

Falls back to `dot_tilde(ctx, sampler, right, left, vi)`.
"""
function dot_tilde_observe(ctx, sampler, right, left, vi)
    logp = dot_tilde(ctx, sampler, right, left, vi)
    acclogp!(vi, logp)
    return left
end

function dot_tilde_primitive(sampler, right, left::AbstractArray, vi)
    return dot_observe(sampler, right, left, vi)
end
# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function dot_tilde_primitive(
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution,AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vi,
)
    return throw(DimensionMismatch(AMBIGUITY_MSG))
end

function dot_observe(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    value::AbstractMatrix,
    vi,
)
    increment_num_produce!(vi)
    @debug "dist = $dist"
    @debug "value = $value"
    return Distributions.loglikelihood(dist, value)
end
function dot_observe(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Distribution,
    value::AbstractArray,
    vi,
)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return Distributions.loglikelihood(dists, value)
end
function dot_observe(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::AbstractArray{<:Distribution},
    value::AbstractArray,
    vi,
)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return sum(Distributions.loglikelihood.(dists, value))
end
function dot_observe(spl::Sampler, ::Any, ::Any, ::Any)
    return error(
        "[DynamicPPL] $(alg_str(spl)) doesn't support vectorizing observe statement"
    )
end
