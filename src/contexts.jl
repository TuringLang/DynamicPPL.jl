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

julia> ctx = ConditionContext((; a = 1));

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

"""
    DynamicPPL.tilde_assume!!(
        context::AbstractContext,
        right::Distribution,
        vn::VarName,
        vi::AbstractVarInfo
    )

Handle assumed variables, i.e. anything which is not observed (see
[`tilde_observe!!`](@ref)). Accumulate the associated log probability, and return the
sampled value and updated `vi`.

`vn` is the VarName on the left-hand side of the tilde statement.

This function should return a tuple `(x, vi)`, where `x` is the sampled value (which
must be in unlinked space!) and `vi` is the updated VarInfo.
"""
function tilde_assume!!(
    context::AbstractContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    return tilde_assume!!(childcontext(context), right, vn, vi)
end

"""
    DynamicPPL.tilde_observe!!(
        context::AbstractContext,
        right::Distribution,
        left,
        vn::Union{VarName, Nothing},
        vi::AbstractVarInfo
    )

This function handles observed variables, which may be:

- literals on the left-hand side, e.g., `3.0 ~ Normal()`
- a model input, e.g. `x ~ Normal()` in a model `@model f(x) ... end`
- a conditioned or fixed variable, e.g. `x ~ Normal()` in a model `model | (; x = 3.0)`.

The relevant log-probability associated with the observation is computed and accumulated in
the VarInfo object `vi` (except for fixed variables, which do not contribute to the
log-probability).

`left` is the actual value that the left-hand side evaluates to. `vn` is the VarName on the
left-hand side, or `nothing` if the left-hand side is a literal value.

Observations of submodels are not yet supported in DynamicPPL.

This function should return a tuple `(left, vi)`, where `left` is the same as the input, and
`vi` is the updated VarInfo.
"""
function tilde_observe!!(
    context::AbstractContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    return tilde_observe!!(childcontext(context), right, left, vn, vi)
end
