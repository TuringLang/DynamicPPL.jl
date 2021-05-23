"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars,LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::LeafCtx
end
PriorContext(vars=nothing) = PriorContext(vars, EvaluationContext())
PriorContext(ctx::AbstractContext) = PriorContext(nothing, ctx)
