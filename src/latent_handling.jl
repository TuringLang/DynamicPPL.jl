struct LatentHandlingContext{Ctx<:AbstractContext} <: AbstractContext
    context::Ctx
end

LatentHandlingContext() = LatentHandlingContext(DefaultContext())

NodeTrait(context::LatentHandlingContext) = IsParent()
childcontext(context::LatentHandlingContext) = context.context
function setchildcontext(context::LatentHandlingContext, child::AbstractContext)
    return LatentHandlingContext(child)
end

"""
    latent(dist)

Return a distribution for the latent parameters of `dist`.
"""
function latent end

"""
    conditional(dist, latents)

Return the distribution of emissions with the latent parameters of `dist` set to `latents`.
"""
function conditional end

"""
    marginalize(dist)

Return the `dist` with the latent parameters marginalized out.
"""
function marginalize end

"""
    has_latents(dist)

Return `true` if the distribution `dist` has latent parameters, otherwise `false`.

Note that if `has_latents(dist) = true`, then `dist` is assumed to implement the following methods:
1. `latent(dist)`: Return the latent parameters of the distribution.
2. `conditional(dist, latents)`: Return a new distribution with the latent parameters set to `latents`.
3. `marginalize(dist)`: Return a new distribution with the latent parameters marginalized out.
"""
has_latents(dist) = false

# Overload the tilde-statements to handle latent parameters.
function suffix_varname(vn::VarName{sym}, ::Val{suffix}) where {sym,suffix}
    return VarName{Symbol(sym, ".", suffix)}(vn.optic)
end

# Cand dispatch on `dist` to choose different suffixes for latent variables.
suffix_latent_varname(dist, vn) = suffix_varname(vn, Val{:latent}())

# `tilde_assume`
function tilde_assume(context::LatentHandlingContext, right, vn, vi)
    has_latents(right) || return tilde_assume(childcontext(context), right, vn, vi)
    # Execute `tilde_assume` for the latent variables first.
    right_latent = latent(right)
    value_latent, logp_marginal, vi = tilde_assume(
        childcontext(context), right_latent, suffix_latent_varname(right, vn), vi
    )
    # Now execute the conditional on the latent variables.
    right_conditional = conditional(right, value_latent)
    value_conditional, logp_conditional, vi = tilde_assume(
        childcontext(context), right_conditional, vn, vi
    )
    # Return as usual.
    return value_conditional, logp_marginal + logp_conditional, vi
end
function tilde_assume(
    rng::Random.AbstractRNG, context::LatentHandlingContext, sampler, right, vn, vi
)
    if !has_latents(right)
        return tilde_assume(rng, childcontext(context), sampler, right, vn, vi)
    end
    # Execute `tilde_assume` for the latent variables first.
    right_latent = latent(right)
    value_latent, logp_marginal, vi = tilde_assume(
        rng,
        childcontext(context),
        sampler,
        right_latent,
        suffix_latent_varname(right, vn),
        vi,
    )
    # Now execute the conditional on the latent variables.
    right_conditional = conditional(right, value_latent)
    value_conditional, logp_conditional, vi = tilde_assume(
        rng, childcontext(context), sampler, right_conditional, vn, vi
    )
    # Return as usual.
    return value_conditional, logp_marginal + logp_conditional, vi
end
# `tilde_observe`
function tilde_observe(context::LatentHandlingContext, right, left, vi)
    has_latents(right) || return tilde_observe(childcontext(context), right, left, vi)
    # When used as `observe`, we want to use the marginalized version.
    right_marginal = marginalize(right)
    return tilde_observe(childcontext(context), right_marginal, left, vi)
end
function tilde_observe(context::LatentHandlingContext, sampler, right, left, vi)
    if !has_latents(right)
        return tilde_observe(childcontext(context), sampler, right, left, vi)
    end
    # When used as `observe`, we want to use the marginalized version.
    right_marginal = marginalize(right)
    return tilde_observe(childcontext(context), sampler, right_marginal, left, vi)
end
