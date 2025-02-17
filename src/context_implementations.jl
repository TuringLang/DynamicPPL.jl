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
function tilde_assume!!(context, dist_or_model, vn, vi, has_right)
    if dist_or_model isa DynamicPPL.Model
        # Forbid things like x.a ~ submodel or x[i] ~ submodel
        # TODO(penelopeysm): This restriction is not really necessary and could
        # be hurtful (say if someone wants to evaluate a submodel in a loop).
        # It is not very difficult to lift this restriction, we just have to
        # let `prefix` and `unprefix` handle cases with both a sym + optic,
        # instead of just the sym as it is right now.
        getoptic(vn) !== identity &&
            error("cannot use e.g. x.a ~ submodel, lhs must be a single identifier")
        # Evaluate the inner model with the appropriate context
        # NOTE: usage of _evaluate!! instead of evaluate!! is intentional. The 
        # version without the underscore resets logp before evaluation.
        retval, vi = DynamicPPL._evaluate!!(
            dist_or_model, vi, PrefixContext{getsym(vn)}(context)
        )

        #=
        NOTE(penelopeysm): Why do we use OrderedDict as the output type here?
        Didn't we want to use NamedTuple?

        Well, it turns out that values_as(vi, NamedTuple) has one annoying problem.

        Consider the following model:

            using DynamicPPL, Distributions
            @model function inner()
                x = (a=1, b=2)
                x.a ~ Normal()
                x.b ~ Normal()
            end
            values_as(VarInfo(inner()), NamedTuple)

        Now, the varinfo contains the varnames `@varname(x.a)` and `@varname(x.b)`
        (with the correct representation, i.e. it knows that `a` is a field of `x`
        and `b` is a field of `x`). So, you might expect to get this from values_as():

            (x = (a = f1, b = f2),)

        where `f1` and `f2` are the values sampled for `x.a` and `x.b`, respectively.
        If this were the case, it would then be quite easy to insert some code into
        the compiler that looked like

            retval = values_as(VarInfo(inner()), NamedTuple)
            x = retval.x

        Unfortunately, that's not how values_as works. We actually get this:

            (var"x.a" = f1, var"x.b" = f2)

        The fundamental reason for this is because the varinfo does not store any
        information about the full structure of `x`. For example, it doesn't know if
        `x` is a NamedTuple or a struct, and it doesn't know what other fields/keys `x`
        might possibly have. So, it doesn't attempt to reconstruct the full structure
        of `x` when converting to a NamedTuple. Instead, it just converts the varnames
        into strings that can be used as keys in the NamedTuple.

        This inability to reproduce the correct structure of internal variables needs
        to be fixed before we can consider using NamedTuple as the output type. I have
        opened an issue here: https://github.com/TuringLang/DynamicPPL.jl/issues/814 

        However, my suspicion is that it cannot be fixed. The only way to be completely
        safe is to stick to using a dictionary structure (it doesn't necessarily have
        to be OrderedDict, but the rest of DynamicPPL uses it so we may as well stick
        with it).

        The good news about OrderedDict is that it is a perfectly natural way to
        represent the result of a model. In particular, we have the following parallels:

            UnivariateDistribution    ==> Float
            MultivariateDistribution  ==> Vector{Float}
            MatrixDistribution        ==> Matrix{Float}
            Model                     ==> OrderedDict(VarName => Any)

        where the right-hand side type represents the value obtained by sampling from
        something on the left-hand side. Furthermore, in much the same way we can
        calculate

            logpdf(Normal(), 1.0)

        we already have all the machinery needed to calculate

            logpdf(model, dict),

        and thus the implementation of submodel `observe` should not be very onerous.
        In fact, I think it basically boils down to wrapping `dict` in a
        ConditionContext and calling exactly the same code as we do here.

        Note that the same cannot be said of NamedTuple: we cannot, in general,
        calculate

            logpdf(model, nt)

        or condition on a NamedTuple, because of the reasons described above. In
        fact, writing this makes me think that we should really just get rid of all
        NamedTuple stuff internally. It would substantially reduce the number of
        headaches we get about models with non-trivial variable structures.
        =#

        # Get all the keys that have the correct symbol
        new_keys = collect(filter(k -> getsym(k) == getsym(vn), keys(vi)))
        new_values_prefixed = values_as(subset(vi, new_keys), OrderedDict)
        # Get rid of the prefix
        # TODO(penelopeysm): Note that this does not yet work correctly for
        # nested submodels (see the failing tests). To deal with that
        # correctly, we have to also take into account any prefixes that have
        # been applied in the _current_ parent context.
        new_values_unprefixed = OrderedDict((
            unprefix_outer_layer(vn) => val for (vn, val) in new_values_prefixed
        ))
        return if has_right
            (new_values_unprefixed, retval), vi
        else
            new_values_unprefixed, vi
        end
    elseif is_rhs_model(dist_or_model)
        # Prefix the variables using the `vn`.
        return rand_like!!(
            dist_or_model,
            if should_auto_prefix(dist_or_model)
                PrefixContext{Symbol(vn)}(context)
            else
                context
            end,
            vi,
        )
    else
        value, logp, vi = tilde_assume(context, dist_or_model, vn, vi)
        return value, acclogp_assume!!(context, vi, logp)
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

function assume(rng, spl::Sampler, dist)
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
            push!!(vi, vn, f(r), dist, sampler)
            # By default `push!!` sets the transformed flag to `false`.
            settrans!!(vi, true, vn)
        else
            push!!(vi, vn, r, dist, sampler)
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
