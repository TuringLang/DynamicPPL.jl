# Fallback traits
# TODO: Should this instead be `NoChildren()`, `HasChild()`, etc. so we allow plural too, e.g. `HasChildren()`?

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
abstract type NodeTrait end
NodeTrait(_, context) = NodeTrait(context)

"""
    IsLeaf

Specifies that the context is a leaf in the context-tree.
"""
struct IsLeaf <: NodeTrait end
"""
    IsParent

Specifies that the context is a parent in the context-tree.
"""
struct IsParent <: NodeTrait end

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
julia> using DynamicPPL: leafcontext, setleafcontext, childcontext, setchildcontext, AbstractContext

julia> struct ParentContext{C} <: AbstractContext
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
        NodeTrait(setleafcontext, left), NodeTrait(setleafcontext, right), left, right
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
    SamplingContext(
            rng=Random.GLOBAL_RNG, 
            sampler::AbstractSampler=SampleFromPrior(), 
            context::AbstractContext=DefaultContext(),
    )

Create a context that allows you to sample parameters with the `sampler` when running the model.
The `context` determines how the returned log density is computed when running the model.

See also: [`DefaultContext`](@ref), [`LikelihoodContext`](@ref), [`PriorContext`](@ref)
"""
struct SamplingContext{S<:AbstractSampler,C<:AbstractContext,R} <: AbstractContext
    rng::R
    sampler::S
    context::C
end

function SamplingContext(
    rng::Random.AbstractRNG=Random.GLOBAL_RNG, sampler::AbstractSampler=SampleFromPrior()
)
    return SamplingContext(rng, sampler, DefaultContext())
end

function SamplingContext(
    sampler::AbstractSampler, context::AbstractContext=DefaultContext()
)
    return SamplingContext(Random.GLOBAL_RNG, sampler, context)
end

function SamplingContext(rng::Random.AbstractRNG, context::AbstractContext)
    return SamplingContext(rng, SampleFromPrior(), context)
end

function SamplingContext(context::AbstractContext)
    return SamplingContext(Random.GLOBAL_RNG, SampleFromPrior(), context)
end

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
        return :(VarName{$(QuoteNode(Symbol(Prefix, PREFIX_SEPARATOR, Sym)))}(getlens(vn)))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(getlens(vn))
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
    hasvalue(context, vn)

Return `true` if `vn` is found in `context`.
"""
hasvalue(context, vn) = false

function hasvalue(context::ConditionContext{vars}, vn::VarName{sym}) where {vars,sym}
    return sym in vars
end
function hasvalue(
    context::ConditionContext{vars}, vn::AbstractArray{<:VarName{sym}}
) where {vars,sym}
    return sym in vars
end

"""
    getvalue(context, vn)

Return value of `vn` in `context`.
"""
function getvalue(context::AbstractContext, vn)
    return error("context $(context) does not contain value for $vn")
end
getvalue(context::ConditionContext, vn) = get(context.values, vn)

"""
    hasvalue_nested(context, vn)

Return `true` if `vn` is found in `context` or any of its descendants.

This is contrast to [`hasvalue`](@ref) which only checks for `vn` in `context`,
not recursively checking if `vn` is in any of its descendants.
"""
function hasvalue_nested(context::AbstractContext, vn)
    return hasvalue_nested(NodeTrait(hasvalue_nested, context), context, vn)
end
hasvalue_nested(::IsLeaf, context, vn) = hasvalue(context, vn)
function hasvalue_nested(::IsParent, context, vn)
    return hasvalue(context, vn) || hasvalue_nested(childcontext(context), vn)
end
function hasvalue_nested(context::PrefixContext, vn)
    return hasvalue_nested(childcontext(context), prefix(context, vn))
end

"""
    getvalue_nested(context, vn)

Return the value of the parameter corresponding to `vn` from `context` or its descendants.

This is contrast to [`getvalue`](@ref) which only returns the value `vn` in `context`,
not recursively looking into its descendants.
"""
function getvalue_nested(context::AbstractContext, vn)
    return getvalue_nested(NodeTrait(getvalue_nested, context), context, vn)
end
function getvalue_nested(::IsLeaf, context, vn)
    return error("context $(context) does not contain value for $vn")
end
function getvalue_nested(context::PrefixContext, vn)
    return getvalue_nested(childcontext(context), prefix(context, vn))
end
function getvalue_nested(::IsParent, context, vn)
    return if hasvalue(context, vn)
        getvalue(context, vn)
    else
        getvalue_nested(childcontext(context), vn)
    end
end

"""
    condition([context::AbstractContext,] values::NamedTuple)
    condition([context::AbstractContext]; values...)

Return `ConditionContext` with `values` and `context` if `values` is non-empty,
otherwise return `context` which is [`DefaultContext`](@ref) by default.

See also: [`decondition`](@ref)
"""
AbstractPPL.condition(; values...) = condition(DefaultContext(), NamedTuple(values))
AbstractPPL.condition(values::NamedTuple) = condition(DefaultContext(), values)
AbstractPPL.condition(context::AbstractContext, values::NamedTuple{()}) = context
function AbstractPPL.condition(context::AbstractContext, values::NamedTuple)
    return ConditionContext(values, context)
end
function AbstractPPL.condition(context::AbstractContext; values...)
    return condition(context, NamedTuple(values))
end

"""
    decondition(context::AbstractContext, syms...)

Return `context` but with `syms` no longer conditioned on.

Note that this recursively traverses contexts, deconditioning all along the way.

See also: [`condition`](@ref)
"""
AbstractPPL.decondition(::IsLeaf, context, args...) = context
function AbstractPPL.decondition(::IsParent, context, args...)
    return setchildcontext(context, decondition(childcontext(context), args...))
end
function AbstractPPL.decondition(context, args...)
    return decondition(NodeTrait(context), context, args...)
end
function AbstractPPL.decondition(context::ConditionContext)
    return decondition(childcontext(context))
end
function AbstractPPL.decondition(context::ConditionContext, sym)
    return condition(
        decondition(childcontext(context), sym), BangBang.delete!!(context.values, sym)
    )
end
function AbstractPPL.decondition(context::ConditionContext, sym, syms...)
    return decondition(
        condition(
            decondition(childcontext(context), syms...),
            BangBang.delete!!(context.values, sym),
        ),
        syms...,
    )
end

"""
    conditioned(context::AbstractContext)

Return `NamedTuple` of values that are conditioned on under context`.

Note that this will recursively traverse the context stack and return
a merged version of the condition values.
"""
function conditioned(context::AbstractContext)
    return conditioned(NodeTrait(conditioned, context), context)
end
conditioned(::IsLeaf, context) = ()
conditioned(::IsParent, context) = conditioned(childcontext(context))
function conditioned(context::ConditionContext)
    # Note the order of arguments to `merge`. The behavior of the rest of DPPL
    # is that the outermost `context` takes precendence, hence when resolving
    # the `conditioned` variables we need to ensure that `context.values` takes
    # precedence over decendants of `context`.
    return merge(context.values, conditioned(childcontext(context)))
end
