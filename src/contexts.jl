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
    childcontext(context)

Returns the child-context of `context`.

Returns `nothing` if `context` is not a `WrappedContext`.
"""
childcontext(context::WrappedContext) = context.context
childcontext(context::AbstractContext) = nothing

"""
    unwrap(context::AbstractContext)

Returns the unwrapped context from `context`.
"""
unwrap(context::WrappedContext) = unwrap(context.context)
unwrap(context::AbstractContext) = context

"""
    unwrappedtype(context::AbstractContext)

Returns the type of the unwrapped context from `context`.
"""
unwrappedtype(context::AbstractContext) = typeof(context)
unwrappedtype(context::WrappedContext{LeafCtx}) where {LeafCtx} = LeafCtx

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
    context::Ctx

    function PriorContext(vars, context)
        return new{typeof(vars),typeof(context),unwrappedtype(context)}(vars, context)
    end
end
PriorContext(vars=nothing) = PriorContext(vars, EvaluationContext())
PriorContext(context::AbstractContext) = PriorContext(nothing, context)

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
    context::Ctx

    function LikelihoodContext(vars, context)
        return new{typeof(vars),typeof(context),unwrappedtype(context)}(vars, context)
    end
end
LikelihoodContext(vars=nothing) = LikelihoodContext(vars, EvaluationContext())
LikelihoodContext(context::AbstractContext) = LikelihoodContext(nothing, context)

function rewrap(parent::LikelihoodContext, leaf::PrimitiveContext)
    return LikelihoodContext(parent.vars, rewrap(childcontext(parent), leaf))
end

"""
    struct MiniBatchContext{Tctx, T} <: AbstractContext
        context::Tctx
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
    context::Ctx

    function MiniBatchContext(loglike_scalar, context::AbstractContext)
        return new{typeof(loglike_scalar),typeof(context),unwrappedtype(context)}(
            loglike_scalar, context
        )
    end
end

MiniBatchContext(loglike_scalar) = MiniBatchContext(loglike_scalar, EvaluationContext())
function MiniBatchContext(context::AbstractContext=EvaluationContext(); batch_size, npoints)
    return MiniBatchContext(npoints / batch_size, context)
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
    context::C

    function PrefixContext{Prefix}(context::AbstractContext) where {Prefix}
        return new{Prefix,typeof(context),unwrappedtype(context)}(context)
    end
end
PrefixContext{Prefix}() where {Prefix} = PrefixContext{Prefix}(EvaluationContext())

function rewrap(parent::PrefixContext{Prefix}, leaf::PrimitiveContext) where {Prefix}
    return PrefixContext{Prefix}(rewrap(childcontext(parent), leaf))
end

const PREFIX_SEPARATOR = Symbol(".")

function PrefixContext{PrefixInner}(
    context::PrefixContext{PrefixOuter}
) where {PrefixInner,PrefixOuter}
    if @generated
        :(PrefixContext{$(QuoteNode(Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)))}(
            context.context
        ))
    else
        PrefixContext{Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)}(context.context)
    end
end

function prefix(::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    if @generated
        return :(VarName{$(QuoteNode(Symbol(Prefix, PREFIX_SEPARATOR, Sym)))}(vn.indexing))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(vn.indexing)
    end
end
