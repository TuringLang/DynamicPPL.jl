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
