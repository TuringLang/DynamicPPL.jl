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
julia> using DynamicPPL: DynamicTransformationContext

julia> ctx = ConditionContext((; a = 1);

julia> DynamicPPL.childcontext(ctx)
DefaultContext()

julia> ctx_prior = DynamicPPL.setchildcontext(ctx, DynamicTransformationContext{true}());

julia> DynamicPPL.childcontext(ctx_prior)
DynamicTransformationContext{true}()
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
julia> using DynamicPPL: leafcontext, setleafcontext, childcontext, setchildcontext, AbstractContext, DynamicTransformationContext

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
       leafcontext(setleafcontext(ctx, DynamicTransformationContext{true}()))
DynamicTransformationContext{true}()

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
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to accumulate values like the log joint probability
when running the model.
"""
struct DefaultContext <: AbstractContext end
NodeTrait(::DefaultContext) = IsLeaf()

"""
    PrefixContext(vn::VarName[, context::AbstractContext])
    PrefixContext(vn::Val{sym}[, context::AbstractContext]) where {sym}

Create a context that allows you to use the wrapped `context` when running the model and
prefixes all parameters with the VarName `vn`.

`PrefixContext(Val(:a), context)` is equivalent to `PrefixContext(@varname(a), context)`.
If `context` is not provided, it defaults to `DefaultContext()`.

This context is useful in nested models to ensure that the names of the parameters are
unique.

See also: [`to_submodel`](@ref)
"""
struct PrefixContext{Tvn<:VarName,C<:AbstractContext} <: AbstractContext
    vn_prefix::Tvn
    context::C
end
PrefixContext(vn::VarName) = PrefixContext(vn, DefaultContext())
function PrefixContext(::Val{sym}, context::AbstractContext) where {sym}
    return PrefixContext(VarName{sym}(), context)
end
PrefixContext(::Val{sym}) where {sym} = PrefixContext(VarName{sym}())

NodeTrait(::PrefixContext) = IsParent()
childcontext(context::PrefixContext) = context.context
function setchildcontext(ctx::PrefixContext, child::AbstractContext)
    return PrefixContext(ctx.vn_prefix, child)
end

"""
    prefix(ctx::AbstractContext, vn::VarName)

Apply the prefixes in the context `ctx` to the variable name `vn`.
"""
function prefix(ctx::PrefixContext, vn::VarName)
    return AbstractPPL.prefix(prefix(childcontext(ctx), vn), ctx.vn_prefix)
end
function prefix(ctx::AbstractContext, vn::VarName)
    return prefix(NodeTrait(ctx), ctx, vn)
end
prefix(::IsLeaf, ::AbstractContext, vn::VarName) = vn
function prefix(::IsParent, ctx::AbstractContext, vn::VarName)
    return prefix(childcontext(ctx), vn)
end

"""
    prefix_and_strip_contexts(ctx::PrefixContext, vn::VarName)

Same as `prefix`, but additionally returns a new context stack that has all the
PrefixContexts removed.

NOTE: This does _not_ modify any variables in any `ConditionContext` and
`FixedContext` that may be present in the context stack. This is because this
function is only used in `tilde_assume`, which is lower in the tilde-pipeline
than `contextual_isassumption` and `contextual_isfixed` (the functions which
actually use the `ConditionContext` and `FixedContext` values). Thus, by this
time, any `ConditionContext`s and `FixedContext`s present have already served
their purpose.

If you call this function, you must therefore be careful to ensure that you _do
not_ need to modify any inner `ConditionContext`s and `FixedContext`s. If you
_do_ need to modify them, then you may need to use
`prefix_cond_and_fixed_variables` instead.
"""
function prefix_and_strip_contexts(ctx::PrefixContext, vn::VarName)
    child_context = childcontext(ctx)
    # vn_prefixed contains the prefixes from all lower levels
    vn_prefixed, child_context_without_prefixes = prefix_and_strip_contexts(
        child_context, vn
    )
    return AbstractPPL.prefix(vn_prefixed, ctx.vn_prefix), child_context_without_prefixes
end
function prefix_and_strip_contexts(ctx::AbstractContext, vn::VarName)
    return prefix_and_strip_contexts(NodeTrait(ctx), ctx, vn)
end
prefix_and_strip_contexts(::IsLeaf, ctx::AbstractContext, vn::VarName) = (vn, ctx)
function prefix_and_strip_contexts(::IsParent, ctx::AbstractContext, vn::VarName)
    vn, new_ctx = prefix_and_strip_contexts(childcontext(ctx), vn)
    return vn, setchildcontext(ctx, new_ctx)
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
# Same as above, and avoids method ambiguity with below
ConditionContext(::NamedTuple{()}, context::NamedConditionContext) = context
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
function getconditioned(context::ConditionContext, vn::VarName)
    return getvalue(context.values, vn)
end

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
    return hasconditioned_nested(collapse_prefix_stack(context), vn)
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
    return getconditioned_nested(collapse_prefix_stack(context), vn)
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
function conditioned(context::PrefixContext)
    return conditioned(collapse_prefix_stack(context))
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
    return hasfixed_nested(collapse_prefix_stack(context), vn)
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
    return getfixed_nested(collapse_prefix_stack(context), vn)
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
function fixed(context::PrefixContext)
    return fixed(collapse_prefix_stack(context))
end

"""
    collapse_prefix_stack(context::AbstractContext)

Apply `PrefixContext`s to any conditioned or fixed values inside them, and remove
the `PrefixContext`s from the context stack.

!!! note
    If you are reading this docstring, you might probably be interested in a more
thorough explanation of how PrefixContext and ConditionContext / FixedContext
interact with one another, especially in the context of submodels.
    The DynamicPPL documentation contains [a separate page on this
topic](https://turinglang.org/DynamicPPL.jl/previews/PR892/internals/submodel_condition/)
which explains this in much more detail.

```jldoctest
julia> using DynamicPPL: collapse_prefix_stack

julia> c1 = PrefixContext(@varname(a), ConditionContext((x=1, )));

julia> collapse_prefix_stack(c1)
ConditionContext(Dict(a.x => 1), DefaultContext())

julia> # Here, `x` gets prefixed only with `a`, whereas `y` is prefixed with both.
       c2 = PrefixContext(@varname(a), ConditionContext((x=1, ), PrefixContext(@varname(b), ConditionContext((y=2,)))));

julia> collapsed = collapse_prefix_stack(c2);

julia> # `collapsed` really looks something like this:
       # ConditionContext(Dict{VarName{:a}, Int64}(a.b.y => 2, a.x => 1), DefaultContext())
       # To avoid fragility arising from the order of the keys in the doctest, we test
       # this indirectly:
       collapsed.values[@varname(a.x)], collapsed.values[@varname(a.b.y)]
(1, 2)
```
"""
function collapse_prefix_stack(context::PrefixContext)
    # Collapse the child context (thus applying any inner prefixes first)
    collapsed = collapse_prefix_stack(childcontext(context))
    # Prefix any conditioned variables with the current prefix
    # Note: prefix_conditioned_variables is O(N) in the depth of the context stack.
    # So is this function. In the worst case scenario, this is O(N^2) in the
    # depth of the context stack.
    return prefix_cond_and_fixed_variables(collapsed, context.vn_prefix)
end
function collapse_prefix_stack(context::AbstractContext)
    return collapse_prefix_stack(NodeTrait(collapse_prefix_stack, context), context)
end
collapse_prefix_stack(::IsLeaf, context) = context
function collapse_prefix_stack(::IsParent, context)
    new_child_context = collapse_prefix_stack(childcontext(context))
    return setchildcontext(context, new_child_context)
end

"""
    prefix_cond_and_fixed_variables(context::AbstractContext, prefix::VarName)

Prefix all the conditioned and fixed variables in a given context with a single
`prefix`.

```jldoctest
julia> using DynamicPPL: prefix_cond_and_fixed_variables, ConditionContext

julia> c1 = ConditionContext((a=1, ))
ConditionContext((a = 1,), DefaultContext())

julia> prefix_cond_and_fixed_variables(c1, @varname(y))
ConditionContext(Dict(y.a => 1), DefaultContext())
```
"""
function prefix_cond_and_fixed_variables(ctx::ConditionContext, prefix::VarName)
    # Replace the prefix of the conditioned variables
    vn_dict = to_varname_dict(ctx.values)
    prefixed_vn_dict = Dict(
        AbstractPPL.prefix(vn, prefix) => value for (vn, value) in vn_dict
    )
    # Prefix the child context as well
    prefixed_child_ctx = prefix_cond_and_fixed_variables(childcontext(ctx), prefix)
    return ConditionContext(prefixed_vn_dict, prefixed_child_ctx)
end
function prefix_cond_and_fixed_variables(ctx::FixedContext, prefix::VarName)
    # Replace the prefix of the conditioned variables
    vn_dict = to_varname_dict(ctx.values)
    prefixed_vn_dict = Dict(
        AbstractPPL.prefix(vn, prefix) => value for (vn, value) in vn_dict
    )
    # Prefix the child context as well
    prefixed_child_ctx = prefix_cond_and_fixed_variables(childcontext(ctx), prefix)
    return FixedContext(prefixed_vn_dict, prefixed_child_ctx)
end
function prefix_cond_and_fixed_variables(c::AbstractContext, prefix::VarName)
    return prefix_cond_and_fixed_variables(
        NodeTrait(prefix_cond_and_fixed_variables, c), c, prefix
    )
end
function prefix_cond_and_fixed_variables(
    ::IsLeaf, context::AbstractContext, prefix::VarName
)
    return context
end
function prefix_cond_and_fixed_variables(
    ::IsParent, context::AbstractContext, prefix::VarName
)
    return setchildcontext(
        context, prefix_cond_and_fixed_variables(childcontext(context), prefix)
    )
end
