# Fallback traits
# TODO: Should this instead be `NoChildren()`, `HasChild()`, etc. so we allow plural too, e.g. `HasChildren()`?
struct IsLeaf end
struct IsParent end

"""
    NodeTrait(context)
    NodeTrait(f, context)

Specifies the role of `context` in the context-tree.

The officially supported traits are:
- `IsLeaf`: `context` does not have any decendants.
- `IsParent`: `context` has a child context to which we often defer.
  Expects the following methods to be implemented:
  - [`childcontext`](@ref)
  - [`rewrap`](@ref)
"""
NodeTrait(_, context) = NodeTrait(context)

"""
    childcontext(context)

Return the descendant context of `context`.
"""
childcontext

"""
    rewrap(parent::AbstractContext, child::AbstractContext)

Reconstruct `parent` but now using `child` is its [`childcontext`](@ref),
effectively updating the child context.

# Examples
```jldoctest
julia> ctx = SamplingContext();

julia> DynamicPPL.childcontext(ctx)
DefaultContext()

julia> ctx_prior = DynamicPPL.rewrap(ctx, PriorContext()); # only compute the logprior

julia> DynamicPPL.childcontext(ctx_prior)
PriorContext{Nothing}(nothing)
```
"""
rewrap

# Contexts
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
SamplingContext(sampler, context) = SamplingContext(Random.GLOBAL_RNG, sampler, context)
SamplingContext(context::AbstractContext) = SamplingContext(SampleFromPrior(), context)
SamplingContext(sampler::AbstractSampler) = SamplingContext(sampler, DefaultContext())
SamplingContext() = SamplingContext(SampleFromPrior())

NodeTrait(context::SamplingContext) = IsParent()
childcontext(context::SamplingContext) = context.context
rewrap(parent::SamplingContext, child) = SamplingContext(parent.rng, parent.sampler, child)

"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute log the joint probability of the data 
and parameters when running the model.
"""
struct DefaultContext <: AbstractContext end
NodeTrait(context::DefaultContext) = IsLeaf()

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
NodeTrait(context::PriorContext) = IsLeaf()

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
NodeTrait(context::LikelihoodContext) = IsLeaf()

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
NodeTrait(context::MiniBatchContext) = IsParent()
childcontext(context::MiniBatchContext) = context.context
rewrap(parent::MiniBatchContext, child) = MiniBatchContext(child, parent.loglike_scalar)

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

NodeTrait(context::PrefixContext) = IsParent()
childcontext(context::PrefixContext) = context.context
rewrap(parent::PrefixContext{Prefix}, child) where {Prefix} = PrefixContext{Prefix}(child)

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

struct ConditionContext{Names,Values,Ctx<:AbstractContext} <: AbstractContext
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

ConditionContext(; values...) = ConditionContext((; values...))
function ConditionContext(values::NamedTuple)
    return ConditionContext(values, DefaultContext())
end

function ConditionContext(values::NamedTuple, context::AbstractContext)
    values_wo_missing = drop_missings(values)
    return ConditionContext{typeof(values_wo_missing)}(values_wo_missing, context)
end

# Try to avoid nested `ConditionContext`.
function ConditionContext(
    values::NamedTuple{Names}, context::ConditionContext
) where {Names}
    # Note that this potentially overrides values from `context`, thus giving
    # precedence to the outmost `ConditionContext`.
    return ConditionContext(merge(context.values, values), childcontext(context))
end

NodeTrait(context::ConditionContext) = IsParent()
childcontext(context::ConditionContext) = context.context
rewrap(parent::ConditionContext, child) = ConditionContext(parent.values, child)

"""
    getvalue(context, vn)

Return the value of the parameter corresponding to `vn` from `context`.
If `context` does not contain the value for `vn`, then `nothing` is returned,
e.g. [`DefaultContext`](@ref) will always return `nothing`.
"""
getvalue(::IsLeaf, context, vn) = nothing
getvalue(::IsParent, context, vn) = getvalue(childcontext(context), vn)
getvalue(context::AbstractContext, vn) = getvalue(NodeTrait(getvalue, context), context, vn)
getvalue(context::PrefixContext, vn) = getvalue(childcontext(context), prefix(context, vn))
function getvalue(context::ConditionContext, vn)
    return if haskey(context, vn)
        _getvalue(context.values, vn)
    else
        getvalue(childcontext(context), vn)
    end
end

# General implementations of `haskey`.
Base.haskey(::IsLeaf, context, vn) = false
Base.haskey(::IsParent, context, vn) = Base.haskey(childcontext(context), vn)
Base.haskey(context::AbstractContext, vn) = Base.haskey(NodeTrait(context), context, vn)

# Specific to `ConditionContext`.
function Base.haskey(context::ConditionContext{vars}, vn::VarName{sym}) where {vars,sym}
    # TODO: Add possibility of indexed variables, e.g. `x[1]`, etc.
    return sym in vars
end

function Base.haskey(
    context::ConditionContext{vars}, vn::AbstractArray{<:VarName{sym}}
) where {vars,sym}
    # TODO: Add possibility of indexed variables, e.g. `x[1]`, etc.
    return sym in vars
end

# Recursively `decondition` the context.
decondition(::IsLeaf, context, args...) = context
function decondition(::IsParent, context, args...)
    return rewrap(context, decondition(childcontext(context), args...))
end
decondition(context, args...) = decondition(NodeTrait(context), context, args...)
function decondition(context::ConditionContext)
    return ConditionContext(NamedTuple(), decondition(childcontext(context)))
end
function decondition(context::ConditionContext, sym)
    return ConditionContext(
        BangBang.delete!!(context.values, sym), childcontext(context, sym)
    )
end
function decondition(context::ConditionContext, sym, syms...)
    return decondition(
        ConditionContext(
            BangBang.delete!!(context.values, sym),
            decondition(childcontext(context), syms...),
        ),
        syms...,
    )
end
