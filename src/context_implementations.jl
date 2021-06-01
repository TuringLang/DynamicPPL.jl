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

# assume
function tilde_assume(ctx::SamplingContext, right, vn, inds, vi)
    return assume(ctx.rng, ctx.sampler, right, vn, inds, vi)
end
tilde_assume(ctx::EvaluationContext, right, vn, inds, vi) = assume(right, vn, inds, vi)
function tilde_assume(ctx::PriorContext, right, vn, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(childcontext(ctx), right, vn, inds, vi)
end
function tilde_assume(ctx::LikelihoodContext, right, vn, inds, vi)
    if ctx.vars isa NamedTuple && haskey(ctx.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(childcontext(ctx), NoDist(right), vn, inds, vi)
end
function tilde_assume(ctx::MiniBatchContext, right, left, inds, vi)
    return tilde_assume(childcontext(ctx), right, left, inds, vi)
end
function tilde_assume(ctx::PrefixContext, right, vn, inds, vi)
    return tilde_assume(childcontext(ctx), right, prefix(ctx, vn), inds, vi)
end

"""
    tilde_assume!(ctx, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value.

Falls back to `tilde_assume!(ctx, right, vn, inds, vi)`.
"""
function tilde_assume!(ctx, right, vn, inds, vi)
    value, logp = tilde_assume(ctx, right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# observe
function tilde_observe(ctx::Union{SamplingContext,EvaluationContext}, right, left, vi)
    return observe(right, left, vi)
end
tilde_observe(ctx::PriorContext, right, left, vi) = 0
function tilde_observe(ctx::LikelihoodContext, right, left, vi)
    return tilde_observe(childcontext(ctx), right, left, vi)
end
function tilde_observe(ctx::MiniBatchContext, right, left, vi)
    return ctx.loglike_scalar * tilde_observe(childcontext(ctx), right, left, vi)
end
function tilde_observe(ctx::PrefixContext, right, left, vi)
    return tilde_observe(childcontext(ctx), right, left, vi)
end

"""
    tilde_observe!(ctx, right, left, vname, vinds, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `tilde_observe(ctx, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!(ctx, right, left, vname, vinds, vi)
    logp = tilde_observe(ctx, right, left, vi)
    acclogp!(vi, logp)
    return left
end

"""
    tilde_observe(ctx, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.

Falls back to `tilde(ctx, right, left, vi)`.
"""
function tilde_observe!(ctx, right, left, vi)
    logp = tilde_observe(ctx, right, left, vi)
    acclogp!(vi, logp)
    return left
end

function assume(rng, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    return error("DynamicPPL.observe: unmanaged inference algorithm: $(typeof(spl))")
end

# fallback without sampler
function assume(dist::Distribution, vn::VarName, inds, vi)
    if !haskey(vi, vn)
        error("variable $vn does not exist")
    end
    r = vi[vn]
    return r, Bijectors.logpdf_with_trans(dist, vi[vn], istrans(vi, vn))
end

# SampleFromPrior and SampleFromUniform
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    inds,
    vi,
)
    # Always overwrite the parameters with new ones.
    r = init(rng, dist, sampler)
    if haskey(vi, vn)
        vi[vn] = vectorize(dist, r)
        setorder!(vi, vn, get_num_produce(vi))
    else
        push!(vi, vn, r, dist, sampler)
    end
    settrans!(vi, false, vn)
    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
function observe(right::Distribution, left, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(right, left)
end

# .~ functions

# assume
function dot_tilde_assume(ctx::SamplingContext, right, left, vns, _, vi)
    return dot_assume(ctx.rng, ctx.sampler, right, vns, left, vi)
end
function dot_tilde_assume(ctx::EvaluationContext, right, left, vns, inds, vi)
    return dot_assume(right, vns, left, inds, vi)
end
function dot_tilde_assume(ctx::LikelihoodContext, right, left, vns, inds, vi)
    sym = getsym(vns)
    if ctx.vars isa NamedTuple && haskey(ctx.vars, sym)
        var = _getindex(getfield(ctx.vars, sym), inds)
        set_val!(vi, vns, right, var)
        settrans!.(Ref(vi), false, vns)
    end
    return dot_tilde_assume(childcontext(ctx), NoDist.(right), vns, left, vi)
end
function dot_tilde_assume(ctx::MiniBatchContext, right, left, vns, inds, vi)
    return dot_tilde_assume(childcontext(ctx), right, left, vns, inds, vi)
end
function dot_tilde_assume(ctx::PriorContext, right, left, vns, inds, vi)
    sym = getsym(vns)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, sym), inds)
        set_val!(vi, vns, right, var)
        settrans!.(Ref(vi), false, vns)
    end
    return dot_tilde_assume(childcontext(ctx), right, vns, left, vi)
end

"""
    dot_tilde_assume!(ctx, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value.

Falls back to `dot_tilde_assume(ctx, right, left, vn, inds, vi)`.
"""
function dot_tilde_assume!(ctx, right, left, vn, inds, vi)
    value, logp = dot_tilde_assume(ctx, right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# `dot_assume`
function dot_assume(
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    inds,
    vi,
)
    @assert length(dist) == size(var, 1)
    lp = sum(zip(vns, eachcol(var))) do vn, ri
        return Bijectors.logpdf_with_trans(dist, ri, istrans(vi, vn))
    end
    return var, lp
end
function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi,
)
    @assert length(dist) == size(var, 1)
    r = get_and_set_val!(rng, vi, vns, dist, spl)
    lp = sum(Bijectors.logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end

function dot_assume(
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    inds,
    vi,
)
    # Make sure `var` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, var, istrans(vi, vns[1])))
    return var, lp
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi,
)
    r = get_and_set_val!(rng, vi, vns, dists, spl)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    var .= r
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
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
            r = init(rng, dist, spl, n)
            for i in 1:n
                vn = vns[i]
                vi[vn] = vectorize(dist, r[:, i])
                settrans!(vi, false, vn)
                setorder!(vi, vn, get_num_produce(vi))
            end
        else
            r = vi[vns]
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
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
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
            r = reshape(vi[vec(vns)], size(vns))
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
function dot_tilde_observe(ctx::Union{SamplingContext,EvaluationContext}, right, left, vi)
    return dot_observe(right, left, vi)
end
dot_tilde_observe(ctx::PriorContext, right, left, vi) = 0
function dot_tilde_observe(ctx::LikelihoodContext, right, left, vi)
    return dot_observe(childcontext(ctx), right, left, vi)
end
function dot_tilde_observe(ctx::MiniBatchContext, right, left, vi)
    return ctx.loglike_scalar * dot_tilde_observe(childcontext(ctx), right, left, vi)
end

"""
    dot_tilde_observe!(ctx, right, left, vname, vinds, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `dot_tilde_observe(ctx, right, left, vi)` ignoring the information about variable
name and indices; if needed, these can be accessed through this function, though.
"""
function dot_tilde_observe!(ctx, right, left, vn, inds, vi)
    logp = dot_tilde_observe(ctx, right, left, vi)
    acclogp!(vi, logp)
    return left
end

"""
    dot_tilde_observe!(ctx, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value.

Falls back to `dot_tilde_observe(ctx, right, left, vi)`.
"""
function dot_tilde_observe!(ctx, right, left, vi)
    logp = dot_tilde_observe(ctx, right, left, vi)
    acclogp!(vi, logp)
    return left
end

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function dot_observe(dist::MultivariateDistribution, value::AbstractMatrix, vi)
    increment_num_produce!(vi)
    @debug "dist = $dist"
    @debug "value = $value"
    return Distributions.loglikelihood(dist, value)
end
function dot_observe(dists::Distribution, value::AbstractArray, vi)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return Distributions.loglikelihood(dists, value)
end
function dot_observe(dists::AbstractArray{<:Distribution}, value::AbstractArray, vi)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return sum(Distributions.loglikelihood.(dists, value))
end
