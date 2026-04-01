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
    optic_skip_length(optic::AbstractPPL.Optic)

Determine the number of optics that would be added to a VarName by prefixing the VarName
with this optic.

This is needed when providing templates when setting values in a VarNamedTuple, _within_ a
submodel. That is, inside a submodel, suppose we have `a[1] ~ Normal()`. To call
`templated_setindex!!` for this variable correctly, we need to provide the shape of `a`. Of
course, we can do this, because `a` should be a top-level variable in the model function.
The problem is that `templated_setindex!!` is called from `tilde_assume!!`, which does _not_
see the variable `a[1]`, but rather `x.a[1]` where `x` is the prefix added by the
`PrefixContext`.

Thus, to correctly communicate the template, we need to wrap the value for `a` inside N
levels of `SkipTemplate`, which says that "the template is actually for the variable N
levels down". This function computes that N.

Note that this only counts the prefixes added by `PrefixContext`s; it does not count any
child contexts. That is because each child context will add its own prefixes.
"""
optic_skip_length(::AbstractPPL.Iden) = 0
optic_skip_length(c::AbstractPPL.Index) = 1 + optic_skip_length(c.child)
optic_skip_length(c::AbstractPPL.Property) = 1 + optic_skip_length(c.child)

"""
    prefix_and_strip_contexts(ctx::PrefixContext, vn::VarName)

Same as `prefix`, but additionally returns a new context stack that has all the
`PrefixContext`s removed.

NOTE: This does _not_ modify any variables in any `CondFixContext`s that may be present in
the context stack. This is because this function is only used in `tilde_assume!!`, which is
lower in the tilde-pipeline than `contextual_isassumption` and `contextual_isfixed` (the
functions which actually use the `CondFixContext`'s values). Thus, by this time, any
`CondFixContext`s present have already served their purpose.

If you call this function, you must therefore be careful to ensure that you _do not_ need to
modify any inner `CondFixContext`s. If you _do_ need to modify them, then you may need to
use `prefix_cond_and_fixed_variables` instead.
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

"""
    DynamicPPL.extract_prefixes(ctx::AbstractContext)

Recursively process the context stack `ctx` to remove all levels of `PrefixContext`, and
aggregate their prefixes into a single VarName which can be used to apply all the prefixes
at once.

Returns a tuple of:

- `ctx_without_prefixes`: the context stack with all `PrefixContext`s removed
- `vn`: the VarName that contains all the aggregated prefixes that were removed, or
  `nothing` if there were no `PrefixContext`s in the stack.

# Example

```jldoctest
julia> using DynamicPPL: extract_prefixes, PrefixContext, AbstractContext, @varname

julia> struct MyContext <: AbstractContext end

julia> ctx = PrefixContext(@varname(b), PrefixContext(@varname(a), MyContext()));

julia> extract_prefixes(ctx)
(MyContext(), b.a)
```

!!! warning

    This does _not_ modify any variables in any `CondFixContext`s that may be present in the
    context stack. This is because this function is only used in `tilde_assume!!`, which is
    lower in the tilde-pipeline than `contextual_isassumption` and `contextual_isfixed` (the
    functions which actually use the `CondFixContext`'s values). Thus, by this time, any
    `CondFixContext`s present have already served their purpose.

    If you call this function, you must therefore be careful to ensure that you _do not_
    need to modify any inner `CondFixContext`s. If you _do_ need to modify them, then you
    may need to use `prefix_cond_and_fixed_variables` instead.
"""
function extract_prefixes(ctx::AbstractContext)
    new_ctx, new_optic = _extract_prefixes(ctx, AbstractPPL.Iden())
    return if new_optic === AbstractPPL.Iden()
        new_ctx, nothing
    else
        new_ctx, AbstractPPL.optic_to_varname(new_optic)
    end
end
function _extract_prefixes(ctx::PrefixContext, optic::AbstractPPL.AbstractOptic)
    new_child_context, child_optic = _extract_prefixes(childcontext(ctx), optic)
    this_optic = AbstractPPL.varname_to_optic(ctx.vn_prefix)
    return (new_child_context, Base.cat(this_optic, child_optic))
end
function _extract_prefixes(ctx::AbstractParentContext, optic::AbstractPPL.AbstractOptic)
    # ctx is not a PrefixContext, so just need to recurse on the child context
    new_child_context, optic = _extract_prefixes(childcontext(ctx), optic)
    return setchildcontext(ctx, new_child_context), optic
end
function _extract_prefixes(ctx::AbstractContext, optic::AbstractPPL.AbstractOptic)
    return ctx, optic
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
    # This is why we need a special function, `extract_prefixes`.
    new_context, aggregated_prefix_vn = extract_prefixes(context)
    new_vn = AbstractPPL.prefix(vn, aggregated_prefix_vn)
    # Figure out how many levels of templating to skip for this variable (and add 1 for the
    # top-level symbol in the VarName).
    # NOTE(penelopeysm): I tried to move this computation into an inner constructor of
    # PrefixContext, so that it could be reused here and in store_coloneq_value!!, and also
    # just because it makes sense to tie this information to the PrefixContext. But that
    # caused nonzero allocations on the LogDensityFunction submodel test, for reasons that
    # are rather unclear! Be careful if you think of doing that.
    n = optic_skip_length(AbstractPPL.getoptic(aggregated_prefix_vn)) + 1
    return tilde_assume!!(new_context, right, new_vn, SkipTemplate{n}(template), vi)
end

function tilde_observe!!(
    context::PrefixContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template::Any,
    vi::AbstractVarInfo,
)
    # In the observe case, unlike assume, `vn` may be `nothing` if the LHS is a literal
    # value. For the need for prefix_and_strip_contexts rather than just prefix, see the
    # comment in `tilde_assume!!`.
    new_vn, new_context, new_template = if vn !== nothing
        new_context, aggregated_prefix_vn = extract_prefixes(context)
        n = optic_skip_length(AbstractPPL.getoptic(aggregated_prefix_vn)) + 1
        prefixed_vn = AbstractPPL.prefix(vn, aggregated_prefix_vn)
        prefixed_vn, new_context, SkipTemplate{n}(template)
    else
        # We don't need a template at all in this branch, so may as well avoid trying to
        # choose one.
        vn, childcontext(context), NoTemplate()
    end
    return tilde_observe!!(new_context, right, left, new_vn, new_template, vi)
end

function store_coloneq_value!!(
    context::PrefixContext, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    new_context, aggregated_prefix_vn = extract_prefixes(context)
    new_vn = AbstractPPL.prefix(vn, aggregated_prefix_vn)
    n = optic_skip_length(AbstractPPL.getoptic(aggregated_prefix_vn)) + 1
    return store_coloneq_value!!(new_context, new_vn, right, SkipTemplate{n}(template), vi)
end
