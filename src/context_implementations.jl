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

function tilde_assume(context::AbstractContext, args...)
    return tilde_assume(childcontext(context), args...)
end
function tilde_assume(::DefaultContext, right, vn, vi)
    return assume(right, vn, vi)
end

function tilde_assume(rng::Random.AbstractRNG, context::AbstractContext, args...)
    return tilde_assume(rng, childcontext(context), args...)
end
function tilde_assume(rng::Random.AbstractRNG, ::DefaultContext, sampler, right, vn, vi)
    return assume(rng, sampler, right, vn, vi)
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
        value, vi = tilde_assume(context, right, vn, vi)
        return value, vi
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

function tilde_observe(context::AbstractContext, args...)
    return tilde_observe(childcontext(context), args...)
end

function tilde_observe(::DefaultContext, args...)
    return observe(args...)
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
    vi = tilde_observe(context, right, left, vi)
    return left, vi
end

function assume(rng::Random.AbstractRNG, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    return error("DynamicPPL.observe: unmanaged inference algorithm: $(typeof(spl))")
end

# fallback without sampler
function assume(dist::Distribution, vn::VarName, vi)
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, dist)
    x, logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, logjac, vn, dist)
    return x, vi
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
            vi = BangBang.setindex!!(vi, f(r), vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            # Otherwise we just extract it.
            r = vi[vn, dist]
        end
    else
        r = init(rng, dist, sampler)
        if istrans(vi)
            f = to_linked_internal_transform(vi, vn, dist)
            vi = push!!(vi, vn, f(r), dist)
            # By default `push!!` sets the transformed flag to `false`.
            vi = settrans!!(vi, true, vn)
        else
            vi = push!!(vi, vn, r, dist)
        end
    end

    # HACK: The above code might involve an `invlink` somewhere, etc. so we need to correct.
    logjac = logabsdetjac(istrans(vi, vn) ? link_transform(dist) : identity, r)
    vi = accumulate_assume!!(vi, r, -logjac, vn, dist)
    return r, vi
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
observe(sampler::AbstractSampler, right, left, vi) = observe(right, left, vi)

function observe(right::Distribution, left, vi)
    return accumulate_observe!!(vi, left, right)
end
