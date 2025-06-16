using Distributions:
    UnivariateDistribution, MultivariateDistribution, MatrixDistribution, Distribution

alg_str(spl::Sampler) = string(nameof(typeof(spl.alg)))

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
    # no rng nor sampler
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
    # rng and sampler
    return assume(rng, sampler, right, vn, vi)
end
function tilde_assume(::IsLeaf, context::AbstractContext, sampler, right, vn, vi)
    # sampler but no rng
    return assume(Random.default_rng(), sampler, right, vn, vi)
end
function tilde_assume(
    ::IsParent, rng::Random.AbstractRNG, context::AbstractContext, args...
)
    # rng but no sampler
    return tilde_assume(rng, childcontext(context), args...)
end

function tilde_assume(::LikelihoodContext, right, vn, vi)
    return assume(nodist(right), vn, vi)
end
function tilde_assume(rng::Random.AbstractRNG, ::LikelihoodContext, sampler, right, vn, vi)
    return assume(rng, sampler, nodist(right), vn, vi)
end

function tilde_assume(context::PrefixContext, right, vn, vi)
    # Note that we can't use something like this here:
    #     new_vn = prefix(context, vn)
    #     return tilde_assume(childcontext(context), right, new_vn, vi)
    # This is because `prefix` applies _all_ prefixes in a given context to a
    # variable name. Thus, if we had two levels of nested prefixes e.g.
    # `PrefixContext{:a}(PrefixContext{:b}(DefaultContext()))`, then the
    # first call would apply the prefix `a.b._`, and the recursive call
    # would apply the prefix `b._`, resulting in `b.a.b._`.
    # This is why we need a special function, `prefix_and_strip_contexts`.
    new_vn, new_context = prefix_and_strip_contexts(context, vn)
    return tilde_assume(new_context, right, new_vn, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG, context::PrefixContext, sampler, right, vn, vi
)
    new_vn, new_context = prefix_and_strip_contexts(context, vn)
    return tilde_assume(rng, new_context, sampler, right, new_vn, vi)
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
        # Here, we apply the PrefixContext _not_ to the parent `context`, but
        # to the context of the submodel being evaluated. This means that later=
        # on in `make_evaluate_args_and_kwargs`, the context stack will be
        # correctly arranged such that it goes like this:
        #  parent_context[1] -> parent_context[2] -> ... -> PrefixContext ->
        #    submodel_context[1] -> submodel_context[2] -> ... -> leafcontext
        # See the docstring of `make_evaluate_args_and_kwargs`, and the internal
        # DynamicPPL documentation on submodel conditioning, for more details.
        #
        # NOTE: This relies on the existence of `right.model.model`. Right now,
        # the only thing that can return true for `is_rhs_model` is something
        # (a `Sampleable`) that has a `model` field that itself (a
        # `ReturnedModelWrapper`) has a `model` field. This may or may not
        # change in the future.
        if should_auto_prefix(right)
            dppl_model = right.model.model # This isa DynamicPPL.Model
            prefixed_submodel_context = PrefixContext(vn, dppl_model.context)
            new_dppl_model = contextualize(dppl_model, prefixed_submodel_context)
            right = to_submodel(new_dppl_model, true)
        end
        rand_like!!(right, context, vi)
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
