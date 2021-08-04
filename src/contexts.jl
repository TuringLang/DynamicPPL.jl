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
condition(; values...) = condition(DefaultContext(), (; values...))
condition(values::NamedTuple) = condition(DefaultContext(), values)
condition(context::AbstractContext, values::NamedTuple{()}) = context
condition(context::AbstractContext, values::NamedTuple) = ConditionContext(values, context)
condition(context::AbstractContext; values...) = condition(context, (; values...))
"""
    decondition(context::AbstractContext, syms...)

Return `context` but with `syms` no longer conditioned on.

Note that this recursively traverses contexts, deconditioning all along the way.

See also: [`condition`](@ref)

# Examples
```jldoctest
julia> using DynamicPPL: AbstractContext, leafcontext, setleafcontext, childcontext, setchildcontext

julia> struct ParentContext{C} <: AbstractContext
           context::C
       end

julia> DynamicPPL.NodeTrait(::ParentContext) = DynamicPPL.IsParent()

julia> DynamicPPL.childcontext(context::ParentContext) = context.context

julia> DynamicPPL.setchildcontext(::ParentContext, child) = ParentContext(child)

julia> Base.show(io::IO, c::ParentContext) = print(io, "ParentContext(", childcontext(c), ")")

julia> ctx = DefaultContext()
DefaultContext()

julia> decondition(ctx) === ctx # this is a no-op
true

julia> ctx = condition(x = 1.0) # default "constructor" for `ConditionContext`
ConditionContext((x = 1.0,), DefaultContext())

julia> decondition(ctx) # `decondition` without arguments drops all conditioning
DefaultContext()

julia> # Nested conditioning is supported.
       ctx_nested = condition(ParentContext(condition(y=2.0)), x=1.0)
ConditionContext((x = 1.0,), ParentContext(ConditionContext((y = 2.0,), DefaultContext())))

julia> # We can also specify which variables to drop.
       decondition(ctx_nested, :x)
ParentContext(ConditionContext((y = 2.0,), DefaultContext()))

julia> # No matter the nested level.
       decondition(ctx_nested, :y)
ConditionContext((x = 1.0,), ParentContext(DefaultContext()))

julia> # Or specify multiple at in one call.
       decondition(ctx_nested, :x, :y)
ParentContext(DefaultContext())

julia> decondition(ctx_nested)
ParentContext(DefaultContext())

julia> # `Val` is also supported.
       decondition(ctx_nested, Val(:x))
ParentContext(ConditionContext((y = 2.0,), DefaultContext()))

julia> decondition(ctx_nested, Val(:y))
ConditionContext((x = 1.0,), ParentContext(DefaultContext()))

julia> decondition(ctx_nested, Val(:x), Val(:y))
ParentContext(DefaultContext())
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

"""
    conditioned(model::Model)
    conditioned(context::AbstractContext)

Return `NamedTuple` of values that are conditioned on under `model`/`context`.

# Examples
```jldoctest
julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 1 methods)

julia> m = demo();

julia> # Returns all the variables we have conditioned on + their values.
       conditioned(condition(m, x=100.0, m=1.0))
(x = 100.0, m = 1.0)

julia> # Nested ones also work (note that `PrefixContext` does nothing to the result).
       cm = condition(contextualize(m, PrefixContext{:a}(condition(m=1.0))), x=100.0);

julia> conditioned(cm)
(x = 100.0, m = 1.0)

julia> # Since we conditioned on `m`, not `a.m` as it will appear after prefixed,
       # `a.m` is treated as a random variable.
       keys(VarInfo(cm))
1-element Vector{VarName{Symbol("a.m"), Tuple{}}}:
 a.m

julia> # If we instead condition on `a.m`, `m` in the model will be considered an observation.
       cm = condition(contextualize(m, PrefixContext{:a}(condition(var"a.m"=1.0))), x=100.0);

julia> conditioned(cm)
(x = 100.0, a.m = 1.0)

julia> keys(VarInfo(cm)) # <= no variables are sampled
Any[]
```
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
