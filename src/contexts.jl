"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute log the joint probability of the data 
and parameters when running the model.
"""
abstract type PrimitiveContext <: AbstractContext end
struct EvaluationContext <: PrimitiveContext end
struct SamplingContext <: PrimitiveContext end

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

include("contexts/prior.jl")
include("contexts/likelihood.jl")
include("contexts/minibatch.jl")
include("contexts/prefix.jl")
