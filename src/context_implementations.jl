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

# Allows samplers, etc. to hook into the final logp accumulation in the tilde-pipeline.
function acclogp_assume!!(context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_assume!!(NodeTrait(acclogp_assume!!, context), context, vi, logp)
end
function acclogp_assume!!(::IsParent, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_assume!!(childcontext(context), vi, logp)
end
function acclogp_assume!!(::IsLeaf, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp!!(context, vi, logp)
end

function acclogp_observe!!(context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_observe!!(NodeTrait(acclogp_observe!!, context), context, vi, logp)
end
function acclogp_observe!!(::IsParent, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_observe!!(childcontext(context), vi, logp)
end
function acclogp_observe!!(::IsLeaf, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp!!(context, vi, logp)
end

# assume
"""
    tilde_assume(context::SamplingContext, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value with a context associated
with a sampler.

Falls back to
```julia
tilde_assume(context.rng, context.context, context.sampler, right, vn, vi)
```
"""
function tilde_assume(context::SamplingContext, right, vn, vi)
    return tilde_assume(context.rng, context.context, context.sampler, right, vn, vi)
end

# Leaf contexts
function tilde_assume(context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), context, args...)
end
function tilde_assume(::IsLeaf, context::AbstractContext, right, vn, vi)
    return assume(right, vn, vi)
end
function tilde_assume(::IsParent, context::AbstractContext, args...)
    return tilde_assume(childcontext(context), args...)
end

function tilde_assume(rng::Random.AbstractRNG, context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), rng, context, args...)
end
function tilde_assume(
    ::IsLeaf, rng::Random.AbstractRNG, context::AbstractContext, sampler, right, vn, vi
)
    return assume(rng, sampler, right, vn, vi)
end
function tilde_assume(
    ::IsParent, rng::Random.AbstractRNG, context::AbstractContext, args...
)
    return tilde_assume(rng, childcontext(context), args...)
end

function tilde_assume(::LikelihoodContext, right, vn, vi)
    return assume(nodist(right), vn, vi)
end
function tilde_assume(rng::Random.AbstractRNG, ::LikelihoodContext, sampler, right, vn, vi)
    return assume(rng, sampler, nodist(right), vn, vi)
end

function tilde_assume(context::PrefixContext, right, vn, vi)
    return tilde_assume(context.context, right, prefix(context, vn), vi)
end
function tilde_assume(
    rng::Random.AbstractRNG, context::PrefixContext, sampler, right, vn, vi
)
    return tilde_assume(rng, context.context, sampler, right, prefix(context, vn), vi)
end

"""
    tilde_assume!!(context, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value and updated `vi`.

By default, calls `tilde_assume(context, right, vn, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_assume!!(context, right, vn, vi)
    return if is_rhs_model(right)
        # Prefix the variables using the `vn`.
        rand_like!!(
            right,
            should_auto_prefix(right) ? PrefixContext{Symbol(vn)}(context) : context,
            vi,
        )
    else
        value, logp, vi = tilde_assume(context, right, vn, vi)
        value, acclogp_assume!!(context, vi, logp)
    end
end

# observe
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

tilde_observe(::PriorContext, right, left, vi) = 0, vi
tilde_observe(::PriorContext, sampler, right, left, vi) = 0, vi

# `MiniBatchContext`
function tilde_observe(context::MiniBatchContext, right, left, vi)
    logp, vi = tilde_observe(context.context, right, left, vi)
    return context.loglike_scalar * logp, vi
end
function tilde_observe(context::MiniBatchContext, sampler, right, left, vi)
    logp, vi = tilde_observe(context.context, sampler, right, left, vi)
    return context.loglike_scalar * logp, vi
end

# `PrefixContext`
function tilde_observe(context::PrefixContext, right, left, vi)
    return tilde_observe(context.context, right, left, vi)
end
function tilde_observe(context::PrefixContext, sampler, right, left, vi)
    return tilde_observe(context.context, sampler, right, left, vi)
end

"""
    tilde_observe!!(context, right, left, vname, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value and updated `vi`.

Falls back to `tilde_observe!!(context, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!!(context, right, left, vname, vi)
    is_rhs_model(right) && throw(
        ArgumentError(
            "`~` with a model on the right-hand side of an observe statement is not supported",
        ),
    )
    return tilde_observe!!(context, right, left, vi)
end

"""
    tilde_observe(context, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.

By default, calls `tilde_observe(context, right, left, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_observe!!(context, right, left, vi)
    is_rhs_model(right) && throw(
        ArgumentError(
            "`~` with a model on the right-hand side of an observe statement is not supported",
        ),
    )
    logp, vi = tilde_observe(context, right, left, vi)
    return left, acclogp_observe!!(context, vi, logp)
end

function assume(rng::Random.AbstractRNG, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    return error("DynamicPPL.observe: unmanaged inference algorithm: $(typeof(spl))")
end

# fallback without sampler
function assume(dist::Distribution, vn::VarName, vi)
    r, logp = invlink_with_logpdf(vi, vn, dist)
    return r, logp, vi
end

# TODO: Remove this thing.
# SampleFromPrior and SampleFromUniform
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::VarInfoOrThreadSafeVarInfo,
)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if sampler isa SampleFromUniform || is_flagged(vi, vn, "del")
            # TODO(mhauru) Is it important to unset the flag here? The `true` allows us
            # to ignore the fact that for VarNamedVector this does nothing, but I'm unsure
            # if that's okay.
            unset_flag!(vi, vn, "del", true)
            r = init(rng, dist, sampler)
            f = to_maybe_linked_internal_transform(vi, vn, dist)
            # TODO(mhauru) This should probably be call a function called setindex_internal!
            # Also, if we use !! we shouldn't ignore the return value.
            BangBang.setindex!!(vi, f(r), vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            # Otherwise we just extract it.
            r = vi[vn, dist]
        end
    else
        r = init(rng, dist, sampler)
        if istrans(vi)
            f = to_linked_internal_transform(vi, vn, dist)
            push!!(vi, vn, f(r), dist)
            # By default `push!!` sets the transformed flag to `false`.
            settrans!!(vi, true, vn)
        else
            push!!(vi, vn, r, dist)
        end
    end

    # HACK: The above code might involve an `invlink` somewhere, etc. so we need to correct.
    logjac = logabsdetjac(istrans(vi, vn) ? link_transform(dist) : identity, r)
    return r, logpdf(dist, r) - logjac, vi
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
observe(sampler::AbstractSampler, right, left, vi) = observe(right, left, vi)
function observe(right::Distribution, left, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(right, left), vi
end
