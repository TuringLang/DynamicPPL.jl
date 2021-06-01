abstract type PrimitiveContext <: AbstractContext end
struct EvaluationContext{S<:AbstractSampler} <: PrimitiveContext
    # TODO: do we even need the sampler these days?
    sampler::S
end
EvaluationContext() = EvaluationContext(SampleFromPrior())

struct SamplingContext{R<:Random.AbstractRNG,S<:AbstractSampler} <: PrimitiveContext
    rng::R
    sampler::S
end
SamplingContext(sampler=SampleFromPrior()) = SamplingContext(Random.GLOBAL_RNG, sampler)

########################
### Wrapped contexts ###
########################
abstract type WrappedContext{LeafCtx<:PrimitiveContext} <: AbstractContext end

"""
    childcontext(ctx)

Returns the child-context of `ctx`.

Returns `nothing` if `ctx` is not a `WrappedContext`.
"""
childcontext(ctx::WrappedContext) = ctx.ctx
childcontext(ctx::AbstractContext) = nothing

"""
    unwrap(ctx::AbstractContext)

Returns the unwrapped context from `ctx`.
"""
unwrap(ctx::WrappedContext) = unwrap(ctx.ctx)
unwrap(ctx::AbstractContext) = ctx

"""
    unwrappedtype(ctx::AbstractContext)

Returns the type of the unwrapped context from `ctx`.
"""
unwrappedtype(ctx::AbstractContext) = typeof(ctx)
unwrappedtype(ctx::WrappedContext{LeafCtx}) where {LeafCtx} = LeafCtx

"""
    rewrap(parent::WrappedContext, leaf::PrimitiveContext)

Rewraps `leaf` in `parent`. Supports nested `WrappedContext`.
"""
rewrap(::AbstractContext, leaf::PrimitiveContext) = leaf

"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars,Ctx,LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::Ctx

    PriorContext(vars, ctx) = new{typeof(vars),typeof(ctx),unwrappedtype(ctx)}(vars, ctx)
end
PriorContext(vars=nothing) = PriorContext(vars, EvaluationContext())
PriorContext(ctx::AbstractContext) = PriorContext(nothing, ctx)

function rewrap(parent::PriorContext, leaf::PrimitiveContext)
    return PriorContext(parent.vars, rewrap(childcontext(parent), leaf))
end

"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` enables the computation of the log likelihood of the parameters when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars,Ctx,LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::Ctx

    function LikelihoodContext(vars, ctx)
        return new{typeof(vars),typeof(ctx),unwrappedtype(ctx)}(vars, ctx)
    end
end
LikelihoodContext(vars=nothing) = LikelihoodContext(vars, EvaluationContext())
LikelihoodContext(ctx::AbstractContext) = LikelihoodContext(nothing, ctx)

function rewrap(parent::LikelihoodContext, leaf::PrimitiveContext)
    return LikelihoodContext(parent.vars, rewrap(childcontext(parent), leaf))
end

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
struct MiniBatchContext{T,Ctx,LeafCtx} <: WrappedContext{LeafCtx}
    loglike_scalar::T
    ctx::Ctx

    function MiniBatchContext(loglike_scalar, ctx::AbstractContext)
        return new{typeof(loglike_scalar),typeof(ctx),unwrappedtype(ctx)}(
            loglike_scalar, ctx
        )
    end
end

MiniBatchContext(loglike_scalar) = MiniBatchContext(loglike_scalar, EvaluationContext())
function MiniBatchContext(ctx::AbstractContext=EvaluationContext(); batch_size, npoints)
    return MiniBatchContext(npoints / batch_size, ctx)
end

function rewrap(parent::MiniBatchContext, leaf::PrimitiveContext)
    return MiniBatchContext(parent.loglike_scalar, rewrap(childcontext(parent), leaf))
end

"""
    PrefixContext{Prefix}(context)

Create a context that allows you to use the wrapped `context` when running the model and
adds the `Prefix` to all parameters.

This context is useful in nested models to ensure that the names of the parameters are
unique.

See also: [`@submodel`](@ref)
"""
struct PrefixContext{Prefix,C,LeafCtx} <: WrappedContext{LeafCtx}
    ctx::C

    function PrefixContext{Prefix}(ctx::AbstractContext) where {Prefix}
        return new{Prefix,typeof(ctx),unwrappedtype(ctx)}(ctx)
    end
end
PrefixContext{Prefix}() where {Prefix} = PrefixContext{Prefix}(EvaluationContext())

function rewrap(parent::PrefixContext{Prefix}, leaf::PrimitiveContext) where {Prefix}
    return PrefixContext{Prefix}(rewrap(childcontext(parent), leaf))
end

const PREFIX_SEPARATOR = Symbol(".")

function PrefixContext{PrefixInner}(
    ctx::PrefixContext{PrefixOuter}
) where {PrefixInner,PrefixOuter}
    if @generated
        :(PrefixContext{$(QuoteNode(Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)))}(
            ctx.ctx
        ))
    else
        PrefixContext{Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)}(ctx.ctx)
    end
end

function prefix(::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    if @generated
        return :(VarName{$(QuoteNode(Symbol(Prefix, PREFIX_SEPARATOR, Sym)))}(vn.indexing))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(vn.indexing)
    end
end
