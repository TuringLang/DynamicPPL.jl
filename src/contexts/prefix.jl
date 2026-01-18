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
struct PrefixContext{Tvn<:VarName,C<:AbstractContext} <: AbstractParentContext
    vn_prefix::Tvn
    context::C
end
PrefixContext(vn::VarName) = PrefixContext(vn, DefaultContext())
function PrefixContext(::Val{sym}, context::AbstractContext) where {sym}
    return PrefixContext(VarName{sym}(), context)
end
PrefixContext(::Val{sym}) where {sym} = PrefixContext(VarName{sym}())

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
prefix(::AbstractContext, vn::VarName) = vn
function prefix(ctx::AbstractParentContext, vn::VarName)
    return prefix(childcontext(ctx), vn)
end

"""
    prefix_and_strip_contexts(ctx::PrefixContext, vn::VarName)

Same as `prefix`, but additionally returns a new context stack that has all the
PrefixContexts removed.

NOTE: This does _not_ modify any variables in any `ConditionContext` and
`FixedContext` that may be present in the context stack. This is because this
function is only used in `tilde_assume!!`, which is lower in the tilde-pipeline
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
prefix_and_strip_contexts(ctx::AbstractContext, vn::VarName) = (vn, ctx)
function prefix_and_strip_contexts(ctx::AbstractParentContext, vn::VarName)
    vn, new_ctx = prefix_and_strip_contexts(childcontext(ctx), vn)
    return vn, setchildcontext(ctx, new_ctx)
end

function tilde_assume!!(
    context::PrefixContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    # Note that we can't use something like this here:
    #     new_vn = prefix(context, vn)
    #     return tilde_assume!!(childcontext(context), right, new_vn, vi)
    # This is because `prefix` applies _all_ prefixes in a given context to a
    # variable name. Thus, if we had two levels of nested prefixes e.g.
    # `PrefixContext{:a}(PrefixContext{:b}(DefaultContext()))`, then the
    # first call would apply the prefix `a.b._`, and the recursive call
    # would apply the prefix `b._`, resulting in `b.a.b._`.
    # This is why we need a special function, `prefix_and_strip_contexts`.
    new_vn, new_context = prefix_and_strip_contexts(context, vn)
    return tilde_assume!!(new_context, right, new_vn, template, vi)
end

function tilde_observe!!(
    context::PrefixContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    # In the observe case, unlike assume, `vn` may be `nothing` if the LHS is a literal
    # value. For the need for prefix_and_strip_contexts rather than just prefix, see the
    # comment in `tilde_assume!!`.
    new_vn, new_context = if vn !== nothing
        prefix_and_strip_contexts(context, vn)
    else
        vn, childcontext(context)
    end
    return tilde_observe!!(new_context, right, left, new_vn, vi)
end
