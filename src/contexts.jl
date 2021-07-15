"""
    SamplingContext(rng, sampler, context)

Create a context that allows you to sample parameters with the `sampler` when running the model.
The `context` determines how the returned log density is computed when running the model.

See also: [`DefaultContext`](@ref), [`LikelihoodContext`](@ref), [`PriorContext`](@ref)
"""
struct SamplingContext{S<:AbstractSampler,C<:AbstractContext,R} <: AbstractContext
    rng::R
    sampler::S
    context::C
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
function MiniBatchContext(context=DefaultContext(); batch_size, npoints)
    return MiniBatchContext(context, npoints / batch_size)
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

struct ConditionContext{Vars,Values,Ctx<:AbstractContext} <: AbstractContext
    values::Values
    context::Ctx

    function ConditionContext{Values}(
        values::Values, context::AbstractContext
    ) where {names,Values<:NamedTuple{names}}
        return new{names,typeof(values),typeof(context)}(values, context)
    end
end

@generated function drop_missings(nt::NamedTuple{names,values}) where {names,values}
    names_expr = Expr(:tuple)
    values_expr = Expr(:tuple)

    for (n, v) in zip(names, values.parameters)
        if !(v <: Missing)
            push!(names_expr.args, QuoteNode(n))
            push!(values_expr.args, :(nt.$n))
        end
    end

    return :(NamedTuple{$names_expr}($values_expr))
end

function ConditionContext(context::ConditionContext, child_context::AbstractContext)
    return ConditionContext(context.values, child_context)
end
function ConditionContext(values::NamedTuple)
    return ConditionContext(values, DefaultContext())
end

function ConditionContext(values::NamedTuple, context::AbstractContext)
    values_wo_missing = drop_missings(values)
    return ConditionContext{typeof(values_wo_missing)}(values_wo_missing, context)
end

# Try to avoid nested `ConditionContext`.
function ConditionContext(values::NamedTuple{Vars}, context::ConditionContext) where {Vars}
    # Note that this potentially overrides values from `context`, thus giving
    # precedence to the outmost `ConditionContext`.
    return ConditionContext(merge(context.values, values), context.context)
end

function Base.haskey(context::ConditionContext{vars}, vn::VarName{sym}) where {vars,sym}
    # TODO: Add possibility of indexed variables, e.g. `x[1]`, etc.
    return sym in vars
end

# TODO: Can we maybe do this in a better way?
# When no second argument is given, we remove _all_ conditioned variables.
# TODO: Should we remove this and just return `context.context`?
# That will work better if `Model` becomes like `ContextualModel`.
decondition(context::ConditionContext) = ConditionContext(NamedTuple(), context.context)
function decondition(context::ConditionContext, sym)
    return ConditionContext(BangBang.delete!!(context.values, sym), context.context)
end
function decondition(context::ConditionContext, sym, syms...)
    return decondition(
        ConditionContext(BangBang.delete!!(context.values, sym), context.context), syms...
    )
end
