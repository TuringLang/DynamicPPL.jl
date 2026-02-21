"""
    AbstractParentContext

An abstract context that has a child context.

Subtypes of `AbstractParentContext` must implement the following interface:

- `DynamicPPL.childcontext(context::AbstractParentContext)`: Return the child context.
- `DynamicPPL.setchildcontext(parent::AbstractParentContext, child::AbstractContext)`: Reconstruct
  `parent` but now using `child` as its child context.
"""
abstract type AbstractParentContext <: AbstractContext end

"""
    childcontext(context::AbstractParentContext)

Return the descendant context of `context`.
"""
childcontext

"""
    setchildcontext(parent::AbstractParentContext, child::AbstractContext)

Reconstruct `parent` but now using `child` is its [`childcontext`](@ref),
effectively updating the child context.

# Examples
```jldoctest; setup=:(using Random)
julia> using DynamicPPL: InitContext, CondFixContext, Condition

julia> ctx = CondFixContext{Condition}(VarNamedTuple(; a = 1));

julia> DynamicPPL.childcontext(ctx)
DefaultContext()

julia> ctx_prior = DynamicPPL.setchildcontext(ctx, InitContext(MersenneTwister(23), InitFromPrior(), UnlinkAll()));

julia> DynamicPPL.childcontext(ctx_prior)
InitContext{MersenneTwister, InitFromPrior, UnlinkAll}(MersenneTwister(23), InitFromPrior(), UnlinkAll())
```
"""
setchildcontext

"""
    leafcontext(context::AbstractContext)

Return the leaf of `context`, i.e. the first descendant context that is not an
`AbstractParentContext`.
"""
leafcontext(context::AbstractContext) = context
leafcontext(context::AbstractParentContext) = leafcontext(childcontext(context))

"""
    setleafcontext(left::AbstractContext, right::AbstractContext)

Return `left` but now with its leaf context replaced by `right`.

Note that this also works even if `right` is not a leaf context,
in which case effectively append `right` to `left`, dropping the
original leaf context of `left`.

# Examples
```jldoctest; setup=:(using Random)
julia> using DynamicPPL: leafcontext, setleafcontext, childcontext, setchildcontext, AbstractContext, InitContext

julia> struct ParentContext{C} <: AbstractParentContext
           context::C
       end

julia> DynamicPPL.childcontext(context::ParentContext) = context.context

julia> DynamicPPL.setchildcontext(::ParentContext, child) = ParentContext(child)

julia> Base.show(io::IO, c::ParentContext) = print(io, "ParentContext(", childcontext(c), ")")

julia> ctx = ParentContext(ParentContext(DefaultContext()))
ParentContext(ParentContext(DefaultContext()))

julia> # Replace the leaf context with another leaf.
       leafcontext(setleafcontext(ctx, InitContext(MersenneTwister(23), InitFromPrior(), UnlinkAll())))
InitContext{MersenneTwister, InitFromPrior, UnlinkAll}(MersenneTwister(23), InitFromPrior(), UnlinkAll())

julia> # Append another parent context.
       setleafcontext(ctx, ParentContext(DefaultContext()))
ParentContext(ParentContext(ParentContext(DefaultContext())))
```
"""
function setleafcontext(left::AbstractParentContext, right::AbstractContext)
    return setchildcontext(left, setleafcontext(childcontext(left), right))
end
setleafcontext(::AbstractContext, right::AbstractContext) = right

"""
    DynamicPPL.tilde_assume!!(
        context::AbstractContext,
        right::Distribution,
        vn::VarName,
        template::Any,
        vi::AbstractVarInfo
    )::Tuple{Any,AbstractVarInfo}

Handle assumed variables, i.e. anything which is not observed (see
[`tilde_observe!!`](@ref)). Accumulate the associated log probability, and return the
sampled value and updated `vi`.

`vn` is the VarName on the left-hand side of the tilde statement.

`template` is the value of the top-level symbol in `vn`.

This function should return a tuple `(x, vi)`, where `x` is the sampled value (which must be
untransformed, i.e., `insupport(right, x)` must be true!) and `vi` is the updated VarInfo.
"""
function tilde_assume!!(
    context::AbstractParentContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    return tilde_assume!!(childcontext(context), right, vn, template, vi)
end
function tilde_assume!!(
    context::AbstractContext, ::Distribution, ::VarName, ::Any, ::AbstractVarInfo
)
    return error("tilde_assume!! not implemented for context of type $(typeof(context))")
end

"""
    DynamicPPL.tilde_observe!!(
        context::AbstractContext,
        right::Distribution,
        left,
        vn::Union{VarName, Nothing},
        template::Any,
        vi::AbstractVarInfo
    )::Tuple{Any,AbstractVarInfo}

This function handles observed variables, which may be:

- literals on the left-hand side, e.g., `3.0 ~ Normal()`
- a model input, e.g. `x ~ Normal()` in a model `@model f(x) ... end`
- a conditioned or fixed variable, e.g. `x ~ Normal()` in a model `model | (; x = 3.0)`.

The relevant log-probability associated with the observation is computed and accumulated in
the VarInfo object `vi` (except for fixed variables, which do not contribute to the
log-probability).

`left` is the actual value that the left-hand side evaluates to. `vn` is the VarName on the
left-hand side, or `nothing` if the left-hand side is a literal value. `template` is the
value of the top-level symbol in `vn`; if `vn` is `nothing`, then `template` will be
`NoTemplate()`.

This function should return a tuple `(left, vi)`, where `left` is the same as the input, and
`vi` is the updated VarInfo.
"""
function tilde_observe!!(
    context::AbstractParentContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template::Any,
    vi::AbstractVarInfo,
)
    return tilde_observe!!(childcontext(context), right, left, vn, template, vi)
end
function tilde_observe!!(
    context::AbstractContext,
    ::Distribution,
    ::Any,
    ::Union{VarName,Nothing},
    ::Any,
    ::AbstractVarInfo,
)
    return error("tilde_observe!! not implemented for context of type $(typeof(context))")
end

"""
    DynamicPPL.store_coloneq_value!!(
        context::AbstractContext,
        left::VarName,
        right::Any,
        template::Any,
        vi::AbstractVarInfo
    )::AbstractVarInfo

Handle storing the value assigned by a statement `left := right`.

`left` is the VarName on the left-hand side of the `:=` statement. `right` is the value on
the right-hand side, and `template` is the value of the top-level symbol in `left`.

This function should return only the updated VarInfo (not a tuple).

!!! note
    This function is not part of DynamicPPL's public API as the only case where this
    function has any effect is when using `RawValueAccumulator`, which is itself fully
    contained within DynamicPPL. There should be no need for users to directly call or
    overload this function.
"""
function store_coloneq_value!!(
    context::AbstractParentContext,
    left::VarName,
    right::Any,
    template::Any,
    vi::AbstractVarInfo,
)
    return store_coloneq_value!!(childcontext(context), left, right, template, vi)
end
function store_coloneq_value!!(
    ::AbstractContext, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    # This is the method that will be hit for leaf contexts. Importantly, if there are any
    # PrefixContexts in the context stack, both `vn` and `template` will have been appropriately
    # prefixed (PrefixContext overloads store_coloneq_value!!). That allows us to not fuss over
    # prefixes here.
    return DynamicPPL.map_accumulator!!(
        acc -> store_colon_eq!!(acc, vn, right, template),
        vi,
        Val(DynamicPPL.RAW_VALUE_ACCNAME),
    )
end
