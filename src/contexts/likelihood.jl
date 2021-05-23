"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` enables the computation of the log likelihood of the parameters when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars,LeafCtx} <: WrappedContext{LeafCtx}
    vars::Tvars
    ctx::LeafCtx
end
LikelihoodContext(vars=nothing) = LikelihoodContext(vars, EvaluationContext())
LikelihoodContext(ctx::AbstractContext) = LikelihoodContext(nothing, ctx)
