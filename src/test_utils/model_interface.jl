# model_interface.jl
# ------------------
#
# This file contains the functions that the models inside models.jl should
# implement.

"""
    logprior_true(model, args...)

Return the `logprior` of `model` for `args`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`loglikelihood_true`](@ref).
"""
function logprior_true end

"""
    loglikelihood_true(model, args...)

Return the `loglikelihood` of `model` for `args`.

This should generally be implemented by hand for every specific `model`.

See also: [`logjoint_true`](@ref), [`logprior_true`](@ref).
"""
function loglikelihood_true end

"""
    logjoint_true(model, args...)

Return the `logjoint` of `model` for `args`.

Defaults to `logprior_true(model, args...) + loglikelihood_true(model, args..)`.

This should generally be implemented by hand for every specific `model`
so that the returned value can be used as a ground-truth for testing things like:

1. Validity of evaluation of `model` using a particular implementation of `AbstractVarInfo`.
2. Validity of a sampler when combined with DynamicPPL by running the sampler twice: once targeting ground-truth functions, e.g. `logjoint_true`, and once targeting `model`.

And more.

See also: [`logprior_true`](@ref), [`loglikelihood_true`](@ref).
"""
function logjoint_true(model::Model, args...)
    return logprior_true(model, args...) + loglikelihood_true(model, args...)
end

"""
    logjoint_true_with_logabsdet_jacobian(model::Model, args...)

Return a tuple `(args_unconstrained, logjoint)` of `model` for `args`.

Unlike [`logjoint_true`](@ref), the returned logjoint computation includes the
log-absdet-jacobian adjustment, thus computing logjoint for the unconstrained variables.

Note that `args` are assumed be in the support of `model`, while `args_unconstrained`
will be unconstrained.

This should generally not be implemented directly, instead one should implement
[`logprior_true_with_logabsdet_jacobian`](@ref) for a given `model`.

See also: [`logjoint_true`](@ref), [`logprior_true_with_logabsdet_jacobian`](@ref).
"""
function logjoint_true_with_logabsdet_jacobian(model::Model, args...)
    args_unconstrained, lp = logprior_true_with_logabsdet_jacobian(model, args...)
    return args_unconstrained, lp + loglikelihood_true(model, args...)
end

"""
    logprior_true_with_logabsdet_jacobian(model::Model, args...)

Return a tuple `(args_unconstrained, logprior_unconstrained)` of `model` for `args...`.

Unlike [`logprior_true`](@ref), the returned logprior computation includes the
log-absdet-jacobian adjustment, thus computing logprior for the unconstrained variables.

Note that `args` are assumed be in the support of `model`, while `args_unconstrained`
will be unconstrained.

See also: [`logprior_true`](@ref).
"""
function logprior_true_with_logabsdet_jacobian end

"""
    varnames(model::Model)

Return a collection of `VarName` as they are expected to appear in the model.

Even though it is recommended to implement this by hand for a particular `Model`,
a default implementation using [`SimpleVarInfo{<:Dict}`](@ref) is provided.
"""
function varnames(model::Model)
    return collect(
        keys(last(DynamicPPL.evaluate!!(model, SimpleVarInfo(Dict()), SamplingContext())))
    )
end

"""
    posterior_mean(model::Model)

Return a `NamedTuple` compatible with `varnames(model)` where the values represent
the posterior mean under `model`.

"Compatible" means that a `varname` from `varnames(model)` can be used to extract the
corresponding value using `get`, e.g. `get(posterior_mean(model), varname)`.
"""
function posterior_mean end

"""
    rand_prior_true([rng::AbstractRNG, ]model::DynamicPPL.Model)

Return a `NamedTuple` of realizations from the prior of `model` compatible with `varnames(model)`.
"""
function rand_prior_true(model::DynamicPPL.Model)
    return rand_prior_true(Random.default_rng(), model)
end
