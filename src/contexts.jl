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
  - [`setchildcontext`](@ref)
"""
NodeTrait(_, context) = NodeTrait(context)

"""
    childcontext(context)

Return the descendant context of `context`.
"""
childcontext

"""
    setchildcontext(parent::AbstractContext, child::AbstractContext)

Reconstruct `parent` but now using `child` is its [`childcontext`](@ref),
effectively updating the child context.

# Examples
```jldoctest
julia> ctx = SamplingContext();

julia> DynamicPPL.childcontext(ctx)
DefaultContext()

julia> ctx_prior = DynamicPPL.setchildcontext(ctx, PriorContext()); # only compute the logprior

julia> DynamicPPL.childcontext(ctx_prior)
PriorContext{Nothing}(nothing)
```
"""
setchildcontext

"""
    leafcontext(context)

Return the leaf of `context`, i.e. the first descendant context that `IsLeaf`.
"""
leafcontext(context) = leafcontext(NodeTrait(leafcontext, context), context)
leafcontext(::IsLeaf, context) = context
leafcontext(::IsParent, context) = leafcontext(childcontext(context))

"""
    setleafcontext(left, right)

Return `left` but now with its leaf context replaced by `right`.

Note that this also works even if `right` is not a leaf context,
in which case effectively append `right` to `left`, dropping the
original leaf context of `left`.

# Examples
```jldoctest
julia> using DynamicPPL: leafcontext, setleafcontext, childcontext, setchildcontext

julia> struct ParentContext{C}
           context::C
       end

julia> DynamicPPL.NodeTrait(::ParentContext) = DynamicPPL.IsParent()

julia> DynamicPPL.childcontext(context::ParentContext) = context.context

julia> DynamicPPL.setchildcontext(::ParentContext, child) = ParentContext(child)

julia> Base.show(io::IO, c::ParentContext) = print(io, "ParentContext(", childcontext(c), ")")

julia> ctx = ParentContext(ParentContext(DefaultContext()))
ParentContext(ParentContext(DefaultContext()))

julia> # Replace the leaf context with another leaf.
       leafcontext(setleafcontext(ctx, PriorContext()))
PriorContext{Nothing}(nothing)

julia> # Append another parent context.
       setleafcontext(ctx, ParentContext(DefaultContext()))
ParentContext(ParentContext(ParentContext(DefaultContext())))
```
"""
function setleafcontext(left, right)
    return setleafcontext(
        NodeTrait(setleafcontext, left),
        NodeTrait(setleafcontext, right),
        left,
        right
    )
end
function setleafcontext(::IsParent, ::IsParent, left, right)
    return setchildcontext(left, setleafcontext(childcontext(left), right))
end
function setleafcontext(::IsParent, ::IsLeaf, left, right)
    return setchildcontext(left, setleafcontext(childcontext(left), right))
end
setleafcontext(::IsLeaf, ::IsParent, left, right) = right
setleafcontext(::IsLeaf, ::IsLeaf, left, right) = right

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
function setchildcontext(parent::SamplingContext, child)
    return SamplingContext(parent.rng, parent.sampler, child)
end

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
function setchildcontext(parent::MiniBatchContext, child)
    return MiniBatchContext(child, parent.loglike_scalar)
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

NodeTrait(context::PrefixContext) = IsParent()
childcontext(context::PrefixContext) = context.context
function setchildcontext(parent::PrefixContext{Prefix}, child) where {Prefix}
    return PrefixContext{Prefix}(child)
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

struct ConditionContext{Names,Values,Ctx<:AbstractContext} <: AbstractContext
    values::Values
    context::Ctx

    function ConditionContext{Values}(
        values::Values, context::AbstractContext
    ) where {names,Values<:NamedTuple{names}}
        return new{names,typeof(values),typeof(context)}(values, context)
    end
end

function ConditionContext(context::ConditionContext, child_context::AbstractContext)
    return ConditionContext(context.values, child_context)
end
function ConditionContext(values::NamedTuple)
    return ConditionContext(values, DefaultContext())
end
function ConditionContext(values::NamedTuple, context::AbstractContext)
    return ConditionContext{typeof(values)}(values, context)
end

# Try to avoid nested `ConditionContext`.
function ConditionContext(
    values::NamedTuple{Names}, context::ConditionContext
) where {Names}
    # Note that this potentially overrides values from `context`, thus giving
    # precedence to the outmost `ConditionContext`.
    return ConditionContext(merge(context.values, values), childcontext(context))
end

function Base.show(io::IO, context::ConditionContext)
    return print(io, "ConditionContext($(context.values), $(childcontext(context)))")
end

NodeTrait(context::ConditionContext) = IsParent()
childcontext(context::ConditionContext) = context.context
setchildcontext(parent::ConditionContext, child) = ConditionContext(parent.values, child)

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
    return if hasvalue(context, vn)
        _getvalue(context.values, vn)
    else
        getvalue(childcontext(context), vn)
    end
end

# General implementations of `haskey`.
hasvalue(::IsLeaf, context, vn) = false
hasvalue(::IsParent, context, vn) = hasvalue(childcontext(context), vn)
hasvalue(context::AbstractContext, vn) = hasvalue(NodeTrait(context), context, vn)

# Specific to `ConditionContext`.
function hasvalue(context::ConditionContext{vars}, vn::VarName{sym}) where {vars,sym}
    # TODO: Add possibility of indexed variables, e.g. `x[1]`, etc.
    return sym in vars
end

function hasvalue(
    context::ConditionContext{vars}, vn::AbstractArray{<:VarName{sym}}
) where {vars,sym}
    # TODO: Add possibility of indexed variables, e.g. `x[1]`, etc.
    return sym in vars
end

"""
    context([context::AbstractContext,] values::NamedTuple)
    context([context::AbstractContext]; values...)

Return `ConditionContext` with `values` and `context` if `values` is non-empty,
otherwise return `context` which is [`DefaultContext`](@ref) by default.

See also: [`decondition`](@ref)
"""
condition() = decondition(ConditionContext())
condition(values::NamedTuple) = condition(DefaultContext(), values)
condition(context::AbstractContext, values::NamedTuple{()}) = context
condition(context::AbstractContext, values::NamedTuple) = ConditionContext(values, context)
function condition(context::AbstractContext; values...)
    return condition(context, (; values...))
end

"""
    decondition(context::AbstractContext, syms...)

Return `context` but with `syms` no longer conditioned on.

Note that this recursively traverses contexts, deconditioning all along the way.

See also: [`condition`](@ref)

# Examples
```jldoctest
julia> ctx = DefaultContext();

julia> decondition(ctx) === ctx # this is a no-op
true

julia> ctx = ConditionContext(x = 1.0);

julia> decondition(ctx)
DefaultContext()

julia> ctx_nested = ConditionContext(SamplingContext(ConditionContext(y=2.0)), x=1.0);

julia> decondition(ctx_nested)
SamplingContext{SampleFromPrior, DefaultContext, Random._GLOBAL_RNG}(Random._GLOBAL_RNG(), SampleFromPrior(), DefaultContext())
```
"""
decondition(::IsLeaf, context, args...) = context
function decondition(::IsParent, context, args...)
    return setchildcontext(context, decondition(childcontext(context), args...))
end
decondition(context, args...) = decondition(NodeTrait(context), context, args...)
function decondition(context::ConditionContext)
    return decondition(childcontext(context))
end
function decondition(context::ConditionContext, sym)
    return condition(
        decondition(childcontext(context), sym), BangBang.delete!!(context.values, sym)
    )
end
function decondition(context::ConditionContext, sym, syms...)
    return decondition(
        condition(
            decondition(childcontext(context), syms...),
            BangBang.delete!!(context.values, sym),
        ),
        syms...,
    )
end
