"""
    unwrap_childcontext(context::AbstractContext)

Return a tuple of the child context of a `context`, or `nothing` if the context does
not wrap any other context, and a function `f(c::AbstractContext)` that constructs
an instance of `context` in which the child context is replaced with `c`.

Falls back to `(nothing, _ -> context)`.
"""
function unwrap_childcontext(context::AbstractContext)
    reconstruct_context(@nospecialize(x)) = context
    return nothing, reconstruct_context
end

"""
    SamplingContext(rng, sampler, context)

Create a context that allows you to sample parameters with the `sampler` when running the model.
The `context` determines how the returned log density is computed when running the model.

See also: [`JointContext`](@ref), [`LoglikelihoodContext`](@ref), [`PriorContext`](@ref)
"""
struct SamplingContext{S<:AbstractSampler,C<:AbstractContext,R} <: AbstractContext
    rng::R
    sampler::S
    context::C
end

function unwrap_childcontext(context::SamplingContext)
    child = context.context
    function reconstruct_samplingcontext(c::AbstractContext)
        return SamplingContext(context.rng, context.sampler, c)
    end
    return child, reconstruct_samplingcontext
end

"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute log the joint probability of the data 
and parameters when running the model.
"""
struct DefaultContext <: AbstractContext end

"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars} <: AbstractContext
    vars::Tvars
end
PriorContext() = PriorContext(nothing)

"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` enables the computation of the log likelihood of the parameters when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars} <: AbstractContext
    vars::Tvars
end
LikelihoodContext() = LikelihoodContext(nothing)

"""
    struct MiniBatchContext{Tctx, T} <: AbstractContext
        ctx::Tctx
        loglike_scalar::T
    end

The `MiniBatchContext` enables the computation of 
`log(prior) + s * log(likelihood of a batch)` when running the model, where `s` is the 
`loglike_scalar` field, typically equal to `the number of data points / batch size`. 
This is useful in batch-based stochastic gradient descent algorithms to be optimizing 
`log(prior) + log(likelihood of all the data points)` in the expectation.
"""
struct MiniBatchContext{Tctx,T} <: AbstractContext
    ctx::Tctx
    loglike_scalar::T
end
function MiniBatchContext(ctx=DefaultContext(); batch_size, npoints)
    return MiniBatchContext(ctx, npoints / batch_size)
end

function unwrap_childcontext(context::MiniBatchContext)
    child = context.context
    function reconstruct_minibatchcontext(c::AbstractContext)
        return MiniBatchContext(c, context.loglike_scalar)
    end
    return child, reconstruct_minibatchcontext
end

"""
    PrefixContext{Prefix}(context)

Create a context that allows you to use the wrapped `context` when running the model and
adds the `Prefix` to all parameters.

This context is useful in nested models to ensure that the names of the parameters are
unique.

See also: [`@submodel`](@ref)
"""
struct PrefixContext{Prefix,C} <: AbstractContext
    ctx::C
end
function PrefixContext{Prefix}(ctx::AbstractContext) where {Prefix}
    return PrefixContext{Prefix,typeof(ctx)}(ctx)
end

const PREFIX_SEPARATOR = Symbol(".")

function PrefixContext{PrefixInner}(
    ctx::PrefixContext{PrefixOuter}
) where {PrefixInner,PrefixOuter}
    if @generated
        :(PrefixContext{$(QuoteNode(Symbol(PrefixOuter, _prefix_seperator, PrefixInner)))}(
            ctx.ctx
        ))
    else
        PrefixContext{Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)}(ctx.ctx)
    end
end

function prefix(::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    if @generated
        return :(VarName{$(QuoteNode(Symbol(Prefix, _prefix_seperator, Sym)))}(vn.indexing))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(vn.indexing)
    end
end

function unwrap_childcontext(context::PrefixContext{P}) where {P}
    child = context.context
    function reconstruct_prefixcontext(c::AbstractContext)
        return PrefixContext{P}(c)
    end
    return child, reconstruct_prefixcontext
end
