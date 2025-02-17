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
PriorContext()
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
PriorContext()

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
            [rng::Random.AbstractRNG=Random.default_rng()],
            [sampler::AbstractSampler=SampleFromPrior()],
            [context::AbstractContext=DefaultContext()],
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
    rng::Random.AbstractRNG=Random.default_rng(), sampler::AbstractSampler=SampleFromPrior()
)
    return SamplingContext(rng, sampler, DefaultContext())
end

function SamplingContext(
    sampler::AbstractSampler, context::AbstractContext=DefaultContext()
)
    return SamplingContext(Random.default_rng(), sampler, context)
end

function SamplingContext(rng::Random.AbstractRNG, context::AbstractContext)
    return SamplingContext(rng, SampleFromPrior(), context)
end

function SamplingContext(context::AbstractContext)
    return SamplingContext(Random.default_rng(), SampleFromPrior(), context)
end

NodeTrait(context::SamplingContext) = IsParent()
childcontext(context::SamplingContext) = context.context
function setchildcontext(parent::SamplingContext, child)
    return SamplingContext(parent.rng, parent.sampler, child)
end

"""
    hassampler(context)

Return `true` if `context` has a sampler.
"""
hassampler(::SamplingContext) = true
hassampler(context::AbstractContext) = hassampler(NodeTrait(context), context)
hassampler(::IsLeaf, context::AbstractContext) = false
hassampler(::IsParent, context::AbstractContext) = hassampler(childcontext(context))

"""
    getsampler(context)

Return the sampler of the context `context`.

This will traverse the context tree until it reaches the first [`SamplingContext`](@ref),
at which point it will return the sampler of that context.
"""
getsampler(context::SamplingContext) = context.sampler
getsampler(context::AbstractContext) = getsampler(NodeTrait(context), context)
getsampler(::IsParent, context::AbstractContext) = getsampler(childcontext(context))

"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute the log joint probability of the data
and parameters when running the model.
"""
struct DefaultContext <: AbstractContext end
NodeTrait(context::DefaultContext) = IsLeaf()

"""
    PriorContext <: AbstractContext

A leaf context resulting in the exclusion of likelihood terms when running the model.
"""
struct PriorContext <: AbstractContext end
NodeTrait(context::PriorContext) = IsLeaf()

"""
    LikelihoodContext <: AbstractContext

A leaf context resulting in the exclusion of prior terms when running the model.
"""
struct LikelihoodContext <: AbstractContext end
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

See also: [`to_submodel`](@ref)
"""
struct PrefixContext{Prefix,C} <: AbstractContext
    context::C
end
function PrefixContext{Prefix}(context::AbstractContext) where {Prefix}
    return PrefixContext{Prefix,typeof(context)}(context)
end

NodeTrait(::PrefixContext) = IsParent()
childcontext(context::PrefixContext) = context.context
function setchildcontext(::PrefixContext{Prefix}, child) where {Prefix}
    return PrefixContext{Prefix}(child)
end

function optic_to_vn(::Accessors.PropertyLens{sym}) where {sym}
    return VarName{sym}()
end
function optic_to_vn(o::Base.ComposedFunction{Outer,typeof(identity)}) where {Outer}
    return optic_to_vn(o.outer)
end
function optic_to_vn(
    o::Base.ComposedFunction{Outer,Accessors.PropertyLens{sym}}
) where {Outer,sym}
    return VarName{sym}(o.outer)
end
function optic_to_vn(@nospecialize(o))
    return error("optic_to_vn failed with optic $o")
end

function unprefix_outer_layer(vn::VarName{sym}) where {sym}
    return optic_to_vn(getoptic(vn))
end

function prefix(ctx::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    optic = getoptic(vn)
    new_optic = if optic === identity
        Accessors.PropertyLens{Sym}()
    else
        Base.ComposedFunction(optic, Accessors.PropertyLens{Sym}())
    end
    return VarName{Symbol(Prefix)}(new_optic)
end
prefix(ctx::AbstractContext, vn::VarName) = prefix(NodeTrait(ctx), ctx, vn)
prefix(::IsLeaf, ::AbstractContext, vn::VarName) = vn
prefix(::IsParent, ctx::AbstractContext, vn::VarName) = prefix(childcontext(ctx), vn)

"""
    prefix(model::Model, x)

Return `model` but with all random variables prefixed by `x`.

If `x` is known at compile-time, use `Val{x}()` to avoid runtime overheads for prefixing.

# Examples

```jldoctest
julia> using DynamicPPL: prefix

julia> @model demo() = x ~ Dirac(1)
demo (generic function with 2 methods)

julia> rand(prefix(demo(), :my_prefix))
(var"my_prefix.x" = 1,)

julia> # One can also use `Val` to avoid runtime overheads.
       rand(prefix(demo(), Val(:my_prefix)))
(var"my_prefix.x" = 1,)
```
"""
prefix(model::Model, x) = contextualize(model, PrefixContext{Symbol(x)}(model.context))
function prefix(model::Model, ::Val{x}) where {x}
    return contextualize(model, PrefixContext{Symbol(x)}(model.context))
end

"""

    ConditionContext{Values<:Union{NamedTuple,AbstractDict},Ctx<:AbstractContext}

Model context that contains values that are to be conditioned on. The values
can either be a NamedTuple mapping symbols to values, such as `(a=1, b=2)`, or
an AbstractDict mapping varnames to values (e.g. `Dict(@varname(a) => 1,
@varname(b) => 2)`). The former is more performant, but the latter must be used
when there are varnames that cannot be represented as symbols, e.g.
`@varname(x[1])`.
"""
struct ConditionContext{
    Values<:Union{NamedTuple,AbstractDict{<:VarName}},Ctx<:AbstractContext
} <: AbstractContext
    values::Values
    context::Ctx
end

const NamedConditionContext{Names} = ConditionContext{<:NamedTuple{Names}}
const DictConditionContext = ConditionContext{<:AbstractDict}

# Use DefaultContext as the default base context
function ConditionContext(values::Union{NamedTuple,AbstractDict})
    return ConditionContext(values, DefaultContext())
end
# Optimisation when there are no values to condition on
ConditionContext(::NamedTuple{()}, context::AbstractContext) = context
# Collapse consecutive levels of `ConditionContext`. Note that this overrides
# values inside the child context, thus giving precedence to the outermost
# `ConditionContext`.
function ConditionContext(values::NamedTuple, context::NamedConditionContext)
    return ConditionContext(merge(context.values, values), childcontext(context))
end
function ConditionContext(values::AbstractDict{<:VarName}, context::DictConditionContext)
    return ConditionContext(merge(context.values, values), childcontext(context))
end

function Base.show(io::IO, context::ConditionContext)
    return print(io, "ConditionContext($(context.values), $(childcontext(context)))")
end

NodeTrait(::ConditionContext) = IsParent()
childcontext(context::ConditionContext) = context.context
setchildcontext(parent::ConditionContext, child) = ConditionContext(parent.values, child)

"""
    hasconditioned(context::AbstractContext, vn::VarName)

Return `true` if `vn` is found in `context`.
"""
hasconditioned(context::AbstractContext, vn::VarName) = false
hasconditioned(context::ConditionContext, vn::VarName) = hasvalue(context.values, vn)
function hasconditioned(context::ConditionContext, vns::AbstractArray{<:VarName})
    return all(Base.Fix1(hasvalue, context.values), vns)
end

"""
    getconditioned(context::AbstractContext, vn::VarName)

Return value of `vn` in `context`.
"""
function getconditioned(context::AbstractContext, vn::VarName)
    return error("context $(context) does not contain value for $vn")
end
getconditioned(context::ConditionContext, vn::VarName) = getvalue(context.values, vn)

"""
    hasconditioned_nested(context, vn)

Return `true` if `vn` is found in `context` or any of its descendants.

This is contrast to [`hasconditioned(::AbstractContext, ::VarName)`](@ref) which only checks
for `vn` in `context`, not recursively checking if `vn` is in any of its descendants.
"""
function hasconditioned_nested(context::AbstractContext, vn)
    return hasconditioned_nested(NodeTrait(hasconditioned_nested, context), context, vn)
end
hasconditioned_nested(::IsLeaf, context, vn) = hasconditioned(context, vn)
function hasconditioned_nested(::IsParent, context, vn)
    return hasconditioned(context, vn) || hasconditioned_nested(childcontext(context), vn)
end
function hasconditioned_nested(context::PrefixContext, vn)
    return hasconditioned_nested(childcontext(context), prefix(context, vn))
end

"""
    getconditioned_nested(context, vn)

Return the value of the parameter corresponding to `vn` from `context` or its descendants.

This is contrast to [`getconditioned`](@ref) which only returns the value `vn` in `context`,
not recursively looking into its descendants.
"""
function getconditioned_nested(context::AbstractContext, vn)
    return getconditioned_nested(NodeTrait(getconditioned_nested, context), context, vn)
end
function getconditioned_nested(::IsLeaf, context, vn)
    return error("context $(context) does not contain value for $vn")
end
function getconditioned_nested(context::PrefixContext, vn)
    return getconditioned_nested(childcontext(context), prefix(context, vn))
end
function getconditioned_nested(::IsParent, context, vn)
    return if hasconditioned(context, vn)
        getconditioned(context, vn)
    else
        getconditioned_nested(childcontext(context), vn)
    end
end

"""
    decondition(context::AbstractContext, syms...)

Return `context` but with `syms` no longer conditioned on.

Note that this recursively traverses contexts, deconditioning all along the way.

See also: [`condition`](@ref)
"""
decondition_context(::IsLeaf, context, args...) = context
function decondition_context(::IsParent, context, args...)
    return setchildcontext(context, decondition_context(childcontext(context), args...))
end
function decondition_context(context, args...)
    return decondition_context(NodeTrait(context), context, args...)
end
function decondition_context(context::ConditionContext)
    return decondition_context(childcontext(context))
end
function decondition_context(context::ConditionContext, sym, syms...)
    new_values = deepcopy(context.values)
    for s in (sym, syms...)
        new_values = BangBang.delete!!(new_values, s)
    end
    return if length(new_values) == 0
        # No more values left, can unwrap
        decondition_context(childcontext(context), syms...)
    else
        ConditionContext(
            new_values, decondition_context(childcontext(context), sym, syms...)
        )
    end
end
function decondition_context(context::NamedConditionContext, vn::VarName{sym}) where {sym}
    return ConditionContext(
        BangBang.delete!!(context.values, sym),
        decondition_context(childcontext(context), vn),
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
conditioned(::IsLeaf, context) = NamedTuple()
conditioned(::IsParent, context) = conditioned(childcontext(context))
function conditioned(context::ConditionContext)
    # Note the order of arguments to `merge`. The behavior of the rest of DPPL
    # is that the outermost `context` takes precendence, hence when resolving
    # the `conditioned` variables we need to ensure that `context.values` takes
    # precedence over decendants of `context`.
    return _merge(context.values, conditioned(childcontext(context)))
end

struct FixedContext{Values,Ctx<:AbstractContext} <: AbstractContext
    values::Values
    context::Ctx
end

const NamedFixedContext{Names} = FixedContext{<:NamedTuple{Names}}
const DictFixedContext = FixedContext{<:AbstractDict}

FixedContext(values) = FixedContext(values, DefaultContext())

# Try to avoid nested `FixedContext`.
function FixedContext(values::NamedTuple, context::NamedFixedContext)
    # Note that this potentially overrides values from `context`, thus giving
    # precedence to the outmost `FixedContext`.
    return FixedContext(merge(context.values, values), childcontext(context))
end

function Base.show(io::IO, context::FixedContext)
    return print(io, "FixedContext($(context.values), $(childcontext(context)))")
end

NodeTrait(::FixedContext) = IsParent()
childcontext(context::FixedContext) = context.context
setchildcontext(parent::FixedContext, child) = FixedContext(parent.values, child)

"""
    hasfixed(context::AbstractContext, vn::VarName)

Return `true` if a fixed value for `vn` is found in `context`.
"""
hasfixed(context::AbstractContext, vn::VarName) = false
hasfixed(context::FixedContext, vn::VarName) = hasvalue(context.values, vn)
function hasfixed(context::FixedContext, vns::AbstractArray{<:VarName})
    return all(Base.Fix1(hasvalue, context.values), vns)
end

"""
    getfixed(context::AbstractContext, vn::VarName)

Return the fixed value of `vn` in `context`.
"""
function getfixed(context::AbstractContext, vn::VarName)
    return error("context $(context) does not contain value for $vn")
end
getfixed(context::FixedContext, vn::VarName) = getvalue(context.values, vn)

"""
    hasfixed_nested(context, vn)

Return `true` if a fixed value for `vn` is found in `context` or any of its descendants.

This is contrast to [`hasfixed(::AbstractContext, ::VarName)`](@ref) which only checks
for `vn` in `context`, not recursively checking if `vn` is in any of its descendants.
"""
function hasfixed_nested(context::AbstractContext, vn)
    return hasfixed_nested(NodeTrait(hasfixed_nested, context), context, vn)
end
hasfixed_nested(::IsLeaf, context, vn) = hasfixed(context, vn)
function hasfixed_nested(::IsParent, context, vn)
    return hasfixed(context, vn) || hasfixed_nested(childcontext(context), vn)
end
function hasfixed_nested(context::PrefixContext, vn)
    return hasfixed_nested(childcontext(context), prefix(context, vn))
end

"""
    getfixed_nested(context, vn)

Return the fixed value of the parameter corresponding to `vn` from `context` or its descendants.

This is contrast to [`getfixed`](@ref) which only returns the value `vn` in `context`,
not recursively looking into its descendants.
"""
function getfixed_nested(context::AbstractContext, vn)
    return getfixed_nested(NodeTrait(getfixed_nested, context), context, vn)
end
function getfixed_nested(::IsLeaf, context, vn)
    return error("context $(context) does not contain value for $vn")
end
function getfixed_nested(context::PrefixContext, vn)
    return getfixed_nested(childcontext(context), prefix(context, vn))
end
function getfixed_nested(::IsParent, context, vn)
    return if hasfixed(context, vn)
        getfixed(context, vn)
    else
        getfixed_nested(childcontext(context), vn)
    end
end

"""
    fix([context::AbstractContext,] values::NamedTuple)
    fix([context::AbstractContext]; values...)

Return `FixedContext` with `values` and `context` if `values` is non-empty,
otherwise return `context` which is [`DefaultContext`](@ref) by default.

See also: [`unfix`](@ref)
"""
fix(; values...) = fix(NamedTuple(values))
fix(values::NamedTuple) = fix(DefaultContext(), values)
function fix(value::Pair{<:VarName}, values::Pair{<:VarName}...)
    return fix((value, values...))
end
function fix(values::NTuple{<:Any,<:Pair{<:VarName}})
    return fix(DefaultContext(), values)
end
fix(context::AbstractContext, values::NamedTuple{()}) = context
function fix(context::AbstractContext, values::Union{AbstractDict,NamedTuple})
    return FixedContext(values, context)
end
function fix(context::AbstractContext; values...)
    return fix(context, NamedTuple(values))
end
function fix(context::AbstractContext, value::Pair{<:VarName}, values::Pair{<:VarName}...)
    return fix(context, (value, values...))
end
function fix(context::AbstractContext, values::NTuple{<:Any,Pair{<:VarName}})
    return fix(context, Dict(values))
end

"""
    unfix(context::AbstractContext, syms...)

Return `context` but with `syms` no longer fixed.

Note that this recursively traverses contexts, unfixing all along the way.

See also: [`fix`](@ref)
"""
unfix(::IsLeaf, context, args...) = context
function unfix(::IsParent, context, args...)
    return setchildcontext(context, unfix(childcontext(context), args...))
end
function unfix(context, args...)
    return unfix(NodeTrait(context), context, args...)
end
function unfix(context::FixedContext)
    return unfix(childcontext(context))
end
function unfix(context::FixedContext, sym)
    return fix(unfix(childcontext(context), sym), BangBang.delete!!(context.values, sym))
end
function unfix(context::FixedContext, sym, syms...)
    return unfix(
        fix(unfix(childcontext(context), syms...), BangBang.delete!!(context.values, sym)),
        syms...,
    )
end

function unfix(context::NamedFixedContext, vn::VarName{sym}) where {sym}
    return fix(unfix(childcontext(context), vn), BangBang.delete!!(context.values, sym))
end
function unfix(context::FixedContext, vn::VarName)
    return fix(unfix(childcontext(context), vn), BangBang.delete!!(context.values, vn))
end

"""
    fixed(context::AbstractContext)

Return the values that are fixed under `context`.

Note that this will recursively traverse the context stack and return
a merged version of the fix values.
"""
fixed(context::AbstractContext) = fixed(NodeTrait(fixed, context), context)
fixed(::IsLeaf, context) = NamedTuple()
fixed(::IsParent, context) = fixed(childcontext(context))
function fixed(context::FixedContext)
    # Note the order of arguments to `merge`. The behavior of the rest of DPPL
    # is that the outermost `context` takes precendence, hence when resolving
    # the `fixed` variables we need to ensure that `context.values` takes
    # precedence over decendants of `context`.
    return _merge(context.values, fixed(childcontext(context)))
end
