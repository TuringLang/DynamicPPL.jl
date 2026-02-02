abstract type ConditionOrFix end
struct Condition <: ConditionOrFix end
struct Fix <: ConditionOrFix end

"""
    CondFixContext{CF<:ConditionOrFix,Values<:VarNamedTuple,Ctx<:AbstractContext}

Model context that contains values that are to be either conditioned on or fixed.

If `CF` is `Condition`, the values are to be conditioned on; if `CF` is `Fix`, the values
are to be fixed.

The values are stored as a `VarNamedTuple`.
"""
struct CondFixContext{CF<:ConditionOrFix,Values<:VarNamedTuple,Ctx<:AbstractContext} <:
       AbstractParentContext
    values::Values
    context::Ctx

    function CondFixContext{CF}(
        values::VarNamedTuple, context::AbstractContext=DefaultContext()
    ) where {CF<:ConditionOrFix}
        return new{CF,typeof(values),typeof(context)}(values, context)
    end
end
function CondFixContext{CF}(
    ::VarNamedTuple{()}, context::AbstractContext=DefaultContext()
) where {CF<:ConditionOrFix}
    # If there are no values, just return the child context.
    return context
end
function CondFixContext{CF}(
    values::VarNamedTuple, context::CondFixContext{CF}
) where {CF<:ConditionOrFix}
    # Note that `values` takes precedence over `context.values`, i.e., the outermost
    # `CondFixContext`'s values override those of the inner `CondFixContext`.
    merged_values = merge(context.values, values)
    return CondFixContext{CF}(merged_values, childcontext(context))
end
# For method ambiguity resolution
function CondFixContext{CF}(
    ::VarNamedTuple{()}, context::CondFixContext{CF}
) where {CF<:ConditionOrFix}
    return context
end

function Base.show(io::IO, context::CondFixContext{CF}) where {CF<:ConditionOrFix}
    return print(io, "CondFixContext{$CF}($(context.values), $(childcontext(context)))")
end
function Base.:(==)(
    c1::CondFixContext{CF}, c2::CondFixContext{CF}
) where {CF<:ConditionOrFix}
    return (c1.values == c2.values) & (childcontext(c1) == childcontext(c2))
end
function Base.isequal(
    c1::CondFixContext{CF}, c2::CondFixContext{CF}
) where {CF<:ConditionOrFix}
    return isequal(c1.values, c2.values) && isequal(childcontext(c1), childcontext(c2))
end

childcontext(context::CondFixContext) = context.context
function setchildcontext(parent::CondFixContext{CF}, child::AbstractContext) where {CF}
    return CondFixContext{CF}(parent.values, child)
end

"""
    has_cf_value(::Type{CF}, context::AbstractContext, vn::VarName)

Return `true` if `vn` is found as a conditioned or fixed value in `context`. The first
argument `CF` specifies whether to check for conditioned or fixed values.

Note that this only checks the top-level `context`, and does not recursively check child
contexts.
"""
function has_cf_value(
    ::Type{CF}, context::AbstractContext, ::VarName
) where {CF<:ConditionOrFix}
    return false
end
function has_cf_value(
    ::Type{CF}, context::CondFixContext{CF}, vn::VarName
) where {CF<:ConditionOrFix}
    return hasvalue(context.values, vn)
end
function has_cf_value(
    ::Type{CF}, context::CondFixContext{CF}, vns::AbstractArray{<:VarName}
) where {CF<:ConditionOrFix}
    return all(Base.Fix1(hasvalue, context.values), vns)
end

"""
    get_cf_value(::Type{CF}, context::AbstractContext, vn::VarName)

Return the value of `vn` in `context`. The first argument `CF` specifies whether to
get conditioned or fixed values.

Note that this only checks the top-level `context`, and does not recursively check child
contexts.
"""
function get_cf_value(
    ::Type{CF}, context::AbstractContext, vn::VarName
) where {CF<:ConditionOrFix}
    return error("context $(context) does not contain value for $vn")
end
function get_cf_value(
    ::Type{CF}, context::CondFixContext{CF}, vn::VarName
) where {CF<:ConditionOrFix}
    return getvalue(context.values, vn)
end

"""
    has_cf_value_nested(::Type{CF}, context, vn)

Same as `has_cf_value(::Type{CF}, ::AbstractContext, ::VarName)` but recursively checks
child contexts.
"""
has_cf_value_nested(
    ::Type{CF}, context::AbstractContext, vn::VarName
) where {CF<:ConditionOrFix} = has_cf_value(CF, context, vn)
function has_cf_value_nested(
    ::Type{CF}, context::AbstractParentContext, vn::VarName
) where {CF<:ConditionOrFix}
    return has_cf_value(CF, context, vn) ||
           has_cf_value_nested(CF, childcontext(context), vn)
end
function has_cf_value_nested(
    ::Type{CF}, context::PrefixContext, vn::VarName
) where {CF<:ConditionOrFix}
    return has_cf_value_nested(CF, collapse_prefix_stack(context), vn)
end

"""
    get_cf_value_nested(::Type{CF}, context, vn)

Same as `get_cf_value(::Type{CF}, ::AbstractContext, ::VarName)` but recursively checks
child contexts.
"""
function get_cf_value_nested(
    ::Type{CF}, context::AbstractContext, vn::VarName
) where {CF<:ConditionOrFix}
    return get_cf_value(CF, context, vn)
end
function get_cf_value_nested(
    ::Type{CF}, context::AbstractParentContext, vn::VarName
) where {CF<:ConditionOrFix}
    return if has_cf_value(CF, context, vn)
        get_cf_value(CF, context, vn)
    else
        get_cf_value_nested(CF, childcontext(context), vn)
    end
end
function get_cf_value_nested(
    ::Type{CF}, context::PrefixContext, vn::VarName
) where {CF<:ConditionOrFix}
    return get_cf_value_nested(CF, collapse_prefix_stack(context), vn)
end

"""
    remove_cf_values(::Type{CF}, context::AbstractContext, syms_or_vns...)

Return `context` but with the specified `Symbol`s or `VarName`s no longer present as
conditioned or fixed values. If no `syms_or_vns` are provided, all conditioned or fixed
values are removed.

The first argument `CF` specifies whether to remove conditioned or fixed values.
Note that this recursively traverses contexts, removing all along the way.
"""
function remove_cf_values(
    ::Type{CF}, context::AbstractContext, args...
) where {CF<:ConditionOrFix}
    return context
end
function remove_cf_values(
    ::Type{CF}, context::AbstractParentContext, args...
) where {CF<:ConditionOrFix}
    return setchildcontext(context, remove_cf_values(CF, childcontext(context), args...))
end
function remove_cf_values(
    ::Type{CF}, context::CondFixContext{CF}, args...
) where {CF<:ConditionOrFix}
    if isempty(args)
        # Remove all conditioned/fixed values
        return remove_cf_values(CF, childcontext(context))
    end
    # TODO(penelopeysm): This would be much better if we had a `delete!!` method on VNT.
    # We don't yet. The reason why it would be good is that it would allow us to decondition
    # on things that weren't exactly keys of the VNT. For example, we could condition on
    # x[1] and x[2], and then decondition on `x[:]`, or something like that.
    ks = keys(context.values)
    vn_args = map(a -> a isa VarName ? a : VarName{a}(), args)
    should_retain(k::VarName) = all(vn_arg -> !subsumes(vn_arg, k), vn_args)
    retained_keys = filter(should_retain, ks)
    new_values = subset(context.values, retained_keys)
    new_childcontext = remove_cf_values(CF, childcontext(context), args...)
    return CondFixContext{CF}(new_values, new_childcontext)
end
function remove_cf_values(
    ::Type{CF}, context::PrefixContext, args...
) where {CF<:ConditionOrFix}
    vn_args = map(a -> a isa VarName ? a : VarName{a}(), args)
    # This is slightly hacky.
    # First, we need to check if any of the args are actually the prefix (or a
    # superset of it). If so, then we just remove everything, since all the 
    # conditioned/fixed variables beneath this PrefixContext will be removed.
    remove_all =
        isempty(vn_args) || any(vn_arg -> subsumes(vn_arg, context.vn_prefix), vn_args)
    if remove_all
        new_childcontext = remove_cf_values(CF, childcontext(context))
        return PrefixContext(context.vn_prefix, new_childcontext)
    else
        # Otherwise, we need to see which of the arguments actually will be carried through.
        sub_args = ()
        for vn_arg in vn_args
            try
                new_arg = AbstractPPL.unprefix(vn_arg, context.vn_prefix)
                sub_args = (sub_args..., new_arg)
            catch e
                # ArgumentError means it couldn't be unprefixed; but that means that the
                # argument is irrelevant as it won't be found in the child context.
                e isa ArgumentError || rethrow(e)
            end
        end
        # if sub_args is empty, we can't pass it through(!!) as that would decondition
        # everything inside. It just means that none of the arguments are relevant.
        return if isempty(sub_args)
            context
        else
            new_childcontext = remove_cf_values(CF, childcontext(context), sub_args...)
            PrefixContext(context.vn_prefix, new_childcontext)
        end
    end
end

"""
    all_cf_values(::Type{CF}, context::AbstractContext)

Return a VarNamedTuple containing all the conditioned or fixed values in `context` and its
descendants. The first argument `CF` specifies whether to get conditioned or fixed values.
"""
function all_cf_values(::Type{CF}, context::AbstractContext) where {CF<:ConditionOrFix}
    return VarNamedTuple()
end
function all_cf_values(
    ::Type{CF}, context::AbstractParentContext
) where {CF<:ConditionOrFix}
    return all_cf_values(CF, childcontext(context))
end
function all_cf_values(::Type{CF}, context::CondFixContext{CF}) where {CF<:ConditionOrFix}
    # Note that `context.values` takes precedence over values from descendants.
    return merge(all_cf_values(CF, childcontext(context)), context.values)
end
function all_cf_values(::Type{CF}, context::PrefixContext) where {CF<:ConditionOrFix}
    return all_cf_values(CF, collapse_prefix_stack(context))
end

#### The rest of the file are just wrappers around the above.

"""
    hasconditioned(context::AbstractContext, vn::VarName)

Return `true` if `vn` has been conditioned in `context`. Note that this only checks
the top-level `context`, and does not recursively checking its descendants.
"""
hasconditioned(context::AbstractContext, vns) = has_cf_value(Condition, context, vns)

"""
    getconditioned(context::AbstractContext, vn::VarName)

Return the value of `vn` in `context`.
"""
getconditioned(context::AbstractContext, vn) = get_cf_value(Condition, context, vn)

"""
    hasconditioned_nested(context, vn)

Return `true` if `vn` is found in `context` or any of its descendants.

This is contrast to [`hasconditioned(::AbstractContext, ::VarName)`](@ref) which only checks
for `vn` in `context`, not recursively checking if `vn` is in any of its descendants.
"""
hasconditioned_nested(context::AbstractContext, vn) =
    has_cf_value_nested(Condition, context, vn)

"""
    getconditioned_nested(context, vn)

Return the value of the parameter corresponding to `vn` from `context` or its descendants.

This is contrast to [`getconditioned`](@ref) which only returns the value `vn` in `context`,
not recursively looking into its descendants.
"""
getconditioned_nested(context::AbstractContext, vn) =
    get_cf_value_nested(Condition, context, vn)

"""
    decondition_context(context::AbstractContext, syms_or_vns...)

Return `context` but with the specified `Symbol`s or `VarName`s no longer conditioned on. If
no `syms_or_vns` are provided, all conditioned values are removed.

Note that this recursively traverses contexts, deconditioning all along the way.
"""
decondition_context(context::AbstractContext, args...) =
    remove_cf_values(Condition, context, args...)

"""
    conditioned(context::AbstractContext)

Return `VarNamedTuple` of values that are conditioned on under `context`.

Note that this will recursively traverse the context stack and return a merged version of
the condition values.
"""
conditioned(ctx::AbstractContext) = all_cf_values(Condition, ctx)

"""
    hasfixed(context::AbstractContext, vn::VarName)

Return `true` if a fixed value for `vn` is found in `context`.
"""
hasfixed(context::AbstractContext, vns) = has_cf_value(Fix, context, vns)

"""
    getfixed(context::AbstractContext, vn::VarName)

Return the fixed value of `vn` in `context`.
"""
getfixed(context::AbstractContext, vn) = get_cf_value(Fix, context, vn)

"""
    hasfixed_nested(context, vn)

Return `true` if a fixed value for `vn` is found in `context` or any of its descendants.

This is contrast to [`hasfixed(::AbstractContext, ::VarName)`](@ref) which only checks
for `vn` in `context`, not recursively checking if `vn` is in any of its descendants.
"""
hasfixed_nested(context::AbstractContext, vn) = has_cf_value_nested(Fix, context, vn)

"""
    getfixed_nested(context, vn)

Return the fixed value of the parameter corresponding to `vn` from `context` or its descendants.

This is contrast to [`getfixed`](@ref) which only returns the value `vn` in `context`,
not recursively looking into its descendants.
"""
getfixed_nested(context::AbstractContext, vn) = get_cf_value_nested(Fix, context, vn)

"""
    unfix_context(context::AbstractContext, syms...)

Return `context` but with `syms` no longer fixed. If no `syms` are provided, all fixed
values are removed.

Note that this recursively traverses contexts, unfixing all along the way.

See also: [`fix`](@ref)
"""
unfix_context(context::AbstractContext, args...) = remove_cf_values(Fix, context, args...)

"""
    fixed(context::AbstractContext)

Return a `VarNamedTuple` containing the values that are fixed under `context`.

Note that this will recursively traverse the context stack and return a merged version of
the fix values.
"""
fixed(ctx::AbstractContext) = all_cf_values(Fix, ctx)

########################################################
### Interaction of PrefixContext with CondFixContext ###
########################################################

"""
    collapse_prefix_stack(context::AbstractContext)

Apply `PrefixContext`s to any conditioned or fixed values inside them, and remove
the `PrefixContext`s from the context stack.

!!! note
    If you are reading this docstring, you might probably be interested in a more
    thorough explanation of how `PrefixContext` and `CondFixContext` interact with one
    another, especially in the context of submodels. The DynamicPPL documentation contains
    [a separate page on this
    topic](https://turinglang.org/DynamicPPL.jl/previews/PR892/internals/submodel_condition/)
    which explains this in much more detail.

```jldoctest
julia> using DynamicPPL: collapse_prefix_stack, PrefixContext, CondFixContext, Condition, @varname

julia> c1 = PrefixContext(@varname(a), CondFixContext{Condition}(VarNamedTuple(x=1,)));

julia> collapse_prefix_stack(c1)
CondFixContext{DynamicPPL.Condition}(VarNamedTuple(a = VarNamedTuple(x = 1,),), DefaultContext())

julia> # Here, `x` gets prefixed only with `a`, whereas `y` is prefixed with both.
       c2 = PrefixContext(@varname(a), CondFixContext{Condition}(VarNamedTuple(x=1, ), PrefixContext(@varname(b), CondFixContext{Condition}(VarNamedTuple(y=2,)))));

julia> collapsed = collapse_prefix_stack(c2)
CondFixContext{DynamicPPL.Condition}(VarNamedTuple(a = VarNamedTuple(b = VarNamedTuple(y = 2,), x = 1),), DefaultContext())

julia> collapsed.values  # In a format that is easier to read.
VarNamedTuple
└─ a => VarNamedTuple
        ├─ b => VarNamedTuple
        │       └─ y => 2
        └─ x => 1
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
collapse_prefix_stack(context::AbstractContext) = context
function collapse_prefix_stack(context::AbstractParentContext)
    new_child_context = collapse_prefix_stack(childcontext(context))
    return setchildcontext(context, new_child_context)
end

"""
    prefix_cond_and_fixed_variables(context::AbstractContext, prefix::VarName)

Prefix all the conditioned and fixed variables in a given context with a single
`prefix`.

```jldoctest
julia> using DynamicPPL: prefix_cond_and_fixed_variables, CondFixContext, Condition, VarNamedTuple, @varname, DefaultContext

julia> c1 = CondFixContext{Condition}(VarNamedTuple(a=1))
CondFixContext{DynamicPPL.Condition}(VarNamedTuple(a = 1,), DefaultContext())

julia> prefix_cond_and_fixed_variables(c1, @varname(y))
CondFixContext{DynamicPPL.Condition}(VarNamedTuple(y = VarNamedTuple(a = 1,),), DefaultContext())
```
"""
function prefix_cond_and_fixed_variables(
    ctx::CondFixContext{CF}, prefix::VarName
) where {CF}
    # Add a prefix to the conditioned or fixed variables
    new_values = VarNamedTuple()
    # TODO(penelopeysm): In principle we should be able to pass down template info
    # from tilde_assume!!. Try to do so.
    new_values = DynamicPPL.setindex!!(new_values, ctx.values, prefix)
    # Prefix the child context as well
    prefixed_child_ctx = prefix_cond_and_fixed_variables(childcontext(ctx), prefix)
    return CondFixContext{CF}(new_values, prefixed_child_ctx)
end
function prefix_cond_and_fixed_variables(context::AbstractContext, ::VarName)
    return context
end
function prefix_cond_and_fixed_variables(context::AbstractParentContext, prefix::VarName)
    return setchildcontext(
        context, prefix_cond_and_fixed_variables(childcontext(context), prefix)
    )
end
