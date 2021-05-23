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

    function MiniBatchContext(loglike_scalar, ctx::AbstractContext=EvaluationContext())
        return new{typeof(loglike_scalar),typeof(ctx),typeof(ctx)}(loglike_scalar, ctx)
    end

    function MiniBatchContext(loglike_scalar, ctx::WrappedContext{LeafCtx}=EvaluationContext()) where {LeafCtx}
        return new{typeof(loglike_scalar),typeof(ctx),LeafCtx}(loglike_scalar, ctx)
    end
end
function MiniBatchContext(ctx=EvaluationContext(); batch_size, npoints)
    return MiniBatchContext(npoints / batch_size, ctx)
end

function rewrap(parent::MiniBatchContext, leaf::PrimitiveContext)
    return MiniBatchContext(parent.loglike_scalar, rewrap(childcontext(parent), leaf))
end
