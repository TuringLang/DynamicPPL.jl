"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute log the joint probability of the data 
and parameters when running the model.
"""
struct DefaultContext <: AbstractContext end

abstract type PrimitiveContext <: AbstractContext end
struct EvaluateContext <: PrimitiveContext end
struct SampleContext <: PrimitiveContext end

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
    rewrap(parent::WrappedContext, leaf::AbstractContext)

Rewraps `leaf` in `parent`. Supports nested `WrappedContext`.
"""
rewrap(::AbstractContext, leaf::AbstractContext) = leaf

"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars, LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::LeafCtx
end
PriorContext(vars=nothing, ctx=EvaluateContext()) = PriorContext{typeof(vars), typeof(ctx)}(vars, ctx)

"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` enables the computation of the log likelihood of the parameters when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars, LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::LeafCtx
end
LikelihoodContext(vars=nothing, ctx=EvaluateContext()) = LikelihoodContext{typeof(vars), typeof(ctx)}(vars, ctx)

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
        return new{typeof(loglike_scalar),typeof(ctx),typeof(ctx)}(loglike_scalar, ctx)
    end

    function MiniBatchContext(loglike_scalar, ctx::WrappedContext{LeafCtx}) where {LeafCtx}
        return new{typeof(loglike_scalar),typeof(ctx),LeafCtx}(loglike_scalar, ctx)
    end
end
function MiniBatchContext(ctx=DefaultContext(); batch_size, npoints)
    return MiniBatchContext(npoints / batch_size, ctx)
end

function rewrap(parent::MiniBatchContext, leaf::AbstractContext)
    return MiniBatchContext(parent.loglike_scalar, rewrap(childcontext(parent), leaf))
end

struct PrefixContext{Prefix,C,LeafCtx} <: WrappedContext{LeafCtx}
    ctx::C

    function PrefixContext{Prefix}(ctx::AbstractContext) where {Prefix}
        return new{Prefix,typeof(ctx),typeof(ctx)}(ctx)
    end
    function PrefixContext{Prefix}(ctx::WrappedContext{LeafCtx}) where {Prefix,LeafCtx}
        return new{Prefix,typeof(ctx),LeafCtx}(ctx)
    end
end
PrefixContext{Prefix}() where {Prefix} = PrefixContext{Prefix}(DefaultContext())

function rewrap(parent::PrefixContext{Prefix}, leaf::AbstractContext) where {Prefix}
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
