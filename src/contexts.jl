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
    propogate_context(context::AbstractContext)

Wrap `context` in its child-context using [`unwrap_childcontext`](@ref), effectively
swapping the order of the two contexts.
"""
function propogate_context(context::AbstractContext)
    c, reconstruct_context = unwrap_childcontext(context)
    child_of_c, reconstruct_c = unwrap_childcontext(c)
    return reconstruct_c(reconstruct_context(child_of_c))
end

"""
    SamplingContext(rng=Random.GLOBAL_RNG, sampler=SampleFromPrior(), context=nothing)

Create a context that allows you to sample parameters with the `sampler` when running the model.
The `context` determines how the returned log density is computed when running the model.

See also: [`EvaluationContext.`](@ref)
"""
struct SamplingContext{S<:AbstractSampler,C,R} <: AbstractContext
    rng::R
    sampler::S
    context::C
end

function SamplingContext(
    rng::Random.AbstractRNG, sampler::AbstractSampler=SampleFromPrior()
)
    return SamplingContext(rng, sampler, nothing)
end
function SamplingContext(sampler::AbstractSampler=SampleFromPrior())
    return SamplingContext(Random.GLOBAL_RNG, sampler)
end

function unwrap_childcontext(context::SamplingContext)
    function reconstruct_samplingcontext(c::Union{AbstractContext,Nothing})
        return SamplingContext(context.rng, context.sampler, c)
    end
    return context.context, reconstruct_samplingcontext
end

"""
    EvaluationContext(context=nothing)

Create a context that allows you to evaluate the model without performing any sampling.
The `context` determines how the returned log density is computed when running the model.

See also: [`SamplingContext`](@ref)
"""
struct EvaluationContext{Ctx} <: AbstractContext
    context::Ctx
end

EvaluationContext() = EvaluationContext(nothing)

function unwrap_childcontext(context::EvaluationContext)
    function reconstruct_evaluationcontext(c::Union{AbstractContext,Nothing})
        return EvaluationContext(c)
    end
    return context.context, reconstruct_evaluationcontext
end

"""
    struct PriorContext{Tvars,Ctx} <: AbstractContext
        vars::Tvars
        context::Ctx
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars,Ctx} <: AbstractContext
    vars::Tvars
    context::Ctx
end
PriorContext(vars=nothing) = PriorContext(vars, EvaluationContext())
PriorContext(context::AbstractContext) = PriorContext(nothing, context)

"""
    struct LikelihoodContext{Tvars,Ctx} <: AbstractContext
        vars::Tvars
        context::Ctx
    end

The `LikelihoodContext` enables the computation of the log likelihood of the parameters when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars,Ctx} <: AbstractContext
    vars::Tvars
    context::Ctx
end
LikelihoodContext(vars=nothing) = LikelihoodContext(vars, EvaluationContext())
LikelihoodContext(context::AbstractContext) = LikelihoodContext(nothing, context)

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
struct MiniBatchContext{Tctx,T} <: AbstractContext
    context::Tctx
    loglike_scalar::T
end
function MiniBatchContext(context=EvaluationContext(); batch_size, npoints)
    return MiniBatchContext(context, npoints / batch_size)
end

function unwrap_childcontext(context::MiniBatchContext)
    function reconstruct_minibatchcontext(c::AbstractContext)
        return MiniBatchContext(c, context.loglike_scalar)
    end
    return context.context, reconstruct_minibatchcontext
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
    context::C
end
function PrefixContext{Prefix}(context::AbstractContext) where {Prefix}
    return PrefixContext{Prefix,typeof(context)}(context)
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

function unwrap_childcontext(context::PrefixContext{P}) where {P}
    function reconstruct_prefixcontext(c::AbstractContext)
        return PrefixContext{P}(c)
    end
    return context.context, reconstruct_prefixcontext
end
