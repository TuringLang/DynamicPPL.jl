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

_getindex(x, inds::Tuple) = _getindex(view(x, first(inds)...), Base.tail(inds))
_getindex(x, inds::Tuple{}) = x

# assume
"""
    tilde_assume(context::SamplingContext, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value with a context associated
with a sampler.

Falls back to
```julia
tilde_assume(context.rng, context.context, context.sampler, right, vn, inds, vi)
```
"""
function tilde_assume(context::SamplingContext, right, vn, inds, vi)
    return tilde_assume(context.rng, context.context, context.sampler, right, vn, inds, vi)
end

# Leaf contexts
function tilde_assume(context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), context, args...)
end
function tilde_assume(::IsLeaf, context::AbstractContext, right, vn, vinds, vi)
    return assume(right, vn, vi)
end
function tilde_assume(::IsParent, context::AbstractContext, args...)
    return tilde_assume(childcontext(context), args...)
end

function tilde_assume(rng, context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), rng, context, args...)
end
function tilde_assume(
    ::IsLeaf, rng, context::AbstractContext, sampler, right, vn, vinds, vi
)
    return assume(rng, sampler, right, vn, vi)
end
function tilde_assume(::IsParent, rng, context::AbstractContext, args...)
    return tilde_assume(rng, childcontext(context), args...)
end

function tilde_assume(context::PriorContext{<:NamedTuple}, right, vn, inds, vi)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(PriorContext(), right, vn, inds, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG,
    context::PriorContext{<:NamedTuple},
    sampler,
    right,
    vn,
    inds,
    vi,
)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(rng, PriorContext(), sampler, right, vn, inds, vi)
end

function tilde_assume(context::LikelihoodContext{<:NamedTuple}, right, vn, inds, vi)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(LikelihoodContext(), right, vn, inds, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG,
    context::LikelihoodContext{<:NamedTuple},
    sampler,
    right,
    vn,
    inds,
    vi,
)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(rng, LikelihoodContext(), sampler, right, vn, inds, vi)
end
function tilde_assume(::LikelihoodContext, right, vn, inds, vi)
    return assume(NoDist(right), vn, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG, ::LikelihoodContext, sampler, right, vn, inds, vi
)
    return assume(rng, sampler, NoDist(right), vn, vi)
end

function tilde_assume(context::PrefixContext, right, vn, inds, vi)
    return tilde_assume(context.context, right, prefix(context, vn), inds, vi)
end
function tilde_assume(rng, context::PrefixContext, sampler, right, vn, inds, vi)
    return tilde_assume(rng, context.context, sampler, right, prefix(context, vn), inds, vi)
end

"""
    tilde_assume!(context, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value.

By default, calls `tilde_assume(context, right, vn, inds, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_assume!(context, right, vn, inds, vi)
    value, logp = tilde_assume(context, right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# observe
"""
    tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)

Handle observed variables with a `context` associated with a sampler.

Falls back to
```julia
tilde_observe(context.rng, context.context, context.sampler, right, left, vname, vinds, vi)
```
"""
function tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)
    return tilde_observe(
        context.rng, context.context, context.sampler, right, left, vname, vinds, vi
    )
end

"""
    tilde_observe(context::SamplingContext, right, left, vi)

Handle observed constants with a `context` associated with a sampler.

Falls back to `tilde_observe(context.context, context.sampler, right, left, vi)`.
"""
function tilde_observe(context::SamplingContext, right, left, vi)
    return tilde_observe(context.context, context.sampler, right, left, vi)
end

# Leaf contexts
function tilde_observe(context::AbstractContext, args...)
    return tilde_observe(NodeTrait(tilde_observe, context), context, args...)
end
tilde_observe(::IsLeaf, context::AbstractContext, args...) = observe(args...)
function tilde_observe(::IsParent, context::AbstractContext, args...)
    return tilde_observe(childcontext(context), args...)
end

tilde_observe(::PriorContext, right, left, vi) = 0
tilde_observe(::PriorContext, sampler, right, left, vi) = 0

# `MiniBatchContext`
function tilde_observe(context::MiniBatchContext, right, left, vi)
    return context.loglike_scalar * tilde_observe(context.context, right, left, vi)
end
function tilde_observe(context::MiniBatchContext, right, left, vname, vi)
    return context.loglike_scalar * tilde_observe(context.context, right, left, vname, vi)
end

# `PrefixContext`
function tilde_observe(context::PrefixContext, right, left, vname, vi)
    return tilde_observe(context.context, right, left, prefix(context, vname), vi)
end

"""
    tilde_observe!(context, right, left, vname, vinds, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `tilde_observe!(context, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!(context, right, left, vname, vinds, vi)
    return tilde_observe!(context, right, left, vi)
end

"""
    tilde_observe(context, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.

By default, calls `tilde_observe(context, right, left, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_observe!(context, right, left, vi)
    logp = tilde_observe(context, right, left, vi)
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
function assume(dist::Distribution, vn::VarName, vi)
    r = vi[vn]
    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end

# SampleFromPrior and SampleFromUniform
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi,
)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if sampler isa SampleFromUniform || is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = init(rng, dist, sampler)
            vi[vn] = vectorize(dist, r)
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            r = vi[vn]
        end
    else
        r = init(rng, dist, sampler)
        push!(vi, vn, r, dist, sampler)
        settrans!(vi, false, vn)
    end

    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
observe(sampler::AbstractSampler, right, left, vi) = observe(right, left, vi)
function observe(right::Distribution, left, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(right, left)
end

# .~ functions

# assume
"""
    dot_tilde_assume(context::SamplingContext, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value for a context
associated with a sampler.

Falls back to
```julia
dot_tilde_assume(context.rng, context.context, context.sampler, right, left, vn, inds, vi)
```
"""
function dot_tilde_assume(context::SamplingContext, right, left, vn, inds, vi)
    return dot_tilde_assume(
        context.rng, context.context, context.sampler, right, left, vn, inds, vi
    )
end

# `DefaultContext`
function dot_tilde_assume(context::AbstractContext, args...)
    return dot_tilde_assume(NodeTrait(dot_tilde_assume, context), context, args...)
end
function dot_tilde_assume(rng, context::AbstractContext, args...)
    return dot_tilde_assume(rng, NodeTrait(dot_tilde_assume, context), context, args...)
end

function dot_tilde_assume(::IsLeaf, ::AbstractContext, right, left, vns, inds, vi)
    return dot_assume(right, left, vns, vi)
end
function dot_tilde_assume(
    ::IsLeaf, rng, ::AbstractContext, sampler, right, left, vns, inds, vi
)
    return dot_assume(rng, sampler, right, vns, left, vi)
end

function dot_tilde_assume(::IsParent, context::AbstractContext, args...)
    return dot_tilde_assume(childcontext(context), args...)
end
function dot_tilde_assume(rng, ::IsParent, context::AbstractContext, args...)
    return dot_tilde_assume(rng, childcontext(context), args...)
end

function dot_tilde_assume(rng, ::DefaultContext, sampler, right, left, vns, inds, vi)
    return dot_assume(rng, sampler, right, vns, left, vi)
end

# `LikelihoodContext`
function dot_tilde_assume(
    context::LikelihoodContext{<:NamedTuple}, right, left, vn, inds, vi
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(LikelihoodContext(), _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(LikelihoodContext(), right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(
    rng::Random.AbstractRNG,
    context::LikelihoodContext{<:NamedTuple},
    sampler,
    right,
    left,
    vn,
    inds,
    vi,
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(rng, LikelihoodContext(), sampler, _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(rng, LikelihoodContext(), sampler, right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(context::LikelihoodContext, right, left, vn, inds, vi)
    return dot_assume(NoDist.(right), left, vn, vi)
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, context::LikelihoodContext, sampler, right, left, vn, inds, vi
)
    return dot_assume(rng, sampler, NoDist.(right), vn, left, vi)
end

# `PriorContext`
function dot_tilde_assume(context::PriorContext{<:NamedTuple}, right, left, vn, inds, vi)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(PriorContext(), _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(PriorContext(), right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(
    rng::Random.AbstractRNG,
    context::PriorContext{<:NamedTuple},
    sampler,
    right,
    left,
    vn,
    inds,
    vi,
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(rng, PriorContext(), sampler, _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(rng, PriorContext(), sampler, right, left, vn, inds, vi)
    end
end

# `PrefixContext`
function dot_tilde_assume(context::PrefixContext, right, left, vn, inds, vi)
    return dot_tilde_assume(context.context, right, prefix.(Ref(context), vn), inds, vi)
end

function dot_tilde_assume(rng, context::PrefixContext, sampler, right, left, vn, inds, vi)
    return dot_tilde_assume(
        rng, context.context, sampler, right, prefix.(Ref(context), vn), inds, vi
    )
end

"""
    dot_tilde_assume!(context, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value.

Falls back to `dot_tilde_assume(context, right, left, vn, inds, vi)`.
"""
function dot_tilde_assume!(context, right, left, vn, inds, vi)
    value, logp = dot_tilde_assume(context, right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# `dot_assume`
function dot_assume(
    dist::MultivariateDistribution, var::AbstractMatrix, vns::AbstractVector{<:VarName}, vi
)
    @assert length(dist) == size(var, 1)
    # NOTE: We cannot work with `var` here because we might have a model of the form
    #
    #     m = Vector{Float64}(undef, n)
    #     m .~ Normal()
    #
    # in which case `var` will have `undef` elements, even if `m` is present in `vi`.
    r = vi[vns]
    lp = sum(zip(vns, eachcol(r))) do vn, ri
        return Bijectors.logpdf_with_trans(dist, ri, istrans(vi, vn))
    end
    return r, lp
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
    return r, lp
end

function dot_assume(
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    vi,
)
    # NOTE: We cannot work with `var` here because we might have a model of the form
    #
    #     m = Vector{Float64}(undef, n)
    #     m .~ Normal()
    #
    # in which case `var` will have `undef` elements, even if `m` is present in `vi`.
    r = reshape(vi[vec(vns)], size(vns))
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    return r, lp
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
    return r, lp
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
"""
    dot_tilde_observe(context::SamplingContext, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value for a context associated with a sampler.

Falls back to `dot_tilde_observe(context.context, context.sampler, right, left, vi)`.
"""
function dot_tilde_observe(context::SamplingContext, right, left, vi)
    return dot_tilde_observe(context.context, context.sampler, right, left, vi)
end

# Leaf contexts
function dot_tilde_observe(context::AbstractContext, args...)
    return dot_tilde_observe(NodeTrait(tilde_observe, context), context, args...)
end
dot_tilde_observe(::IsLeaf, ::AbstractContext, args...) = dot_observe(args...)
function dot_tilde_observe(::IsParent, context::AbstractContext, args...)
    return dot_tilde_observe(childcontext(context), args...)
end

dot_tilde_observe(::PriorContext, right, left, vi) = 0
dot_tilde_observe(::PriorContext, sampler, right, left, vi) = 0

# `MiniBatchContext`
function dot_tilde_observe(context::MiniBatchContext, right, left, vi)
    return context.loglike_scalar * dot_tilde_observe(context.context, right, left, vi)
end

# `PrefixContext`
function dot_tilde_observe(context::PrefixContext, right, left, vi)
    return dot_tilde_observe(context.context, right, left, vi)
end

"""
    dot_tilde_observe!(context, right, left, vname, vinds, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `dot_tilde_observe!(context, right, left, vi)` ignoring the information about variable
name and indices; if needed, these can be accessed through this function, though.
"""
function dot_tilde_observe!(context, right, left, vn, inds, vi)
    return dot_tilde_observe!(context, right, left, vi)
end

"""
    dot_tilde_observe!(context, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value.

Falls back to `dot_tilde_observe(context, right, left, vi)`.
"""
function dot_tilde_observe!(context, right, left, vi)
    logp = dot_tilde_observe(context, right, left, vi)
    acclogp!(vi, logp)
    return left
end

# Falls back to non-sampler definition.
function dot_observe(::AbstractSampler, dist, value, vi)
    return dot_observe(dist, value, vi)
end
function dot_observe(dist::MultivariateDistribution, value::AbstractMatrix, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(dist, value)
end
function dot_observe(dists::Distribution, value::AbstractArray, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(dists, value)
end
function dot_observe(dists::AbstractArray{<:Distribution}, value::AbstractArray, vi)
    increment_num_produce!(vi)
    return sum(Distributions.loglikelihood.(dists, value))
end
