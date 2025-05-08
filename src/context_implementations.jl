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
        value, vi = tilde_assume(context, right, vn, vi)
        return value, vi
    end
end

# observe
"""
    tilde_observe!!(context::SamplingContext, right, left, vi)

Handle observed constants with a `context` associated with a sampler.

Falls back to `tilde_observe!!(context.context, right, left, vi)`.
"""
function tilde_observe!!(context::SamplingContext, right, left, vn, vi)
    return tilde_observe!!(context.context, right, left, vn, vi)
end

function tilde_observe!!(context::AbstractContext, right, left, vn, vi)
    return tilde_observe!!(childcontext(context), right, left, vn, vi)
end

# `PrefixContext`
function tilde_observe!!(context::PrefixContext, right, left, vn, vi)
    # In the observe case, unlike assume, `vn` may be `nothing` if the LHS is a literal
    # value. For the need for prefix_and_strip_contexts rather than just prefix, see the
    # comment in `tilde_assume!!`.
    new_vn, new_context = if vn !== nothing
        prefix_and_strip_contexts(context, vn)
    else
        vn, childcontext(context)
    end
    return tilde_observe!!(new_context, right, left, new_vn, vi)
end

"""
    tilde_observe!!(context, right, left, vn, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value and updated `vi`.

Falls back to `tilde_observe!!(context, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!!(context::DefaultContext, right, left, vn, vi)
    is_rhs_model(right) && throw(
        ArgumentError(
            "`~` with a model on the right-hand side of an observe statement is not supported",
        ),
    )
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end

function assume(rng::Random.AbstractRNG, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
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
