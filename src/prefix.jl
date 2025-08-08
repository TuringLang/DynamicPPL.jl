"""
    maybe_prefix(inner::Union{Nothing,<:VarName}, outer::Union{Nothing,<:VarName})

Prefix `inner` with the prefix `outer`. Both `inner` and `outer` can be either
`VarName`s or `Nothing`.

Note that this differs from `AbstractPPL.prefix` in that it handles `nothing` values.
This can happen e.g. when prefixing a model that is not already prefixed; or when
executing submodels without automatic prefixing.
"""
maybe_prefix(inner::VarName, outer::VarName) = AbstractPPL.prefix(inner, outer)
maybe_prefix(vn::VarName, ::Nothing) = vn
maybe_prefix(::Nothing, vn::VarName) = vn
maybe_prefix(::Nothing, ::Nothing) = nothing

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

"""
    DynamicPPL.prefix(model::Model, x::VarName)
    DynamicPPL.prefix(model::Model, x::Val{sym})
    DynamicPPL.prefix(model::Model, x::Any)

Return `model` but with all random variables prefixed by `x`, where `x` is either:
- a `VarName` (e.g. `@varname(a)`),
- a `Val{sym}` (e.g. `Val(:a)`), or
- for any other type, `x` is converted to a Symbol and then to a `VarName`. Note that
  this will introduce runtime overheads so is not recommended unless absolutely
  necessary.

If `x` is `nothing`, then the model is returned unchanged.

# Examples

```jldoctest
julia> using DynamicPPL: prefix

julia> @model demo() = x ~ Dirac(1)
demo (generic function with 2 methods)

julia> rand(prefix(demo(), @varname(my_prefix)))
(var"my_prefix.x" = 1,)

julia> rand(prefix(demo(), Val(:my_prefix)))
(var"my_prefix.x" = 1,)
```
"""
prefix(model::Model, ::Nothing) = model
function prefix(model::Model, vn::VarName)
    # Add it to the model prefix field
    new_prefix = maybe_prefix(model.prefix, vn)
    # And also make sure to prefix any conditioned and fixed variables stored in the model
    new_context = prefix_cond_and_fixed_variables(model.context, vn)
    return Model(model.f, model.args, model.defaults, new_context, new_prefix)
end
prefix(model::Model, ::Val{sym}) where {sym} = prefix(model, VarName{sym}())
prefix(model::Model, x) = return prefix(model, VarName{Symbol(x)}())
