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
"""
function tilde_assume!!(
    context::AbstractContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    return tilde_assume!!(childcontext(context), right, vn, vi)
end
function tilde_assume!!(
    ::DefaultContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end
function tilde_assume!!(
    context::PrefixContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
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
    return tilde_assume!!(new_context, right, new_vn, vi)
end
"""
    DynamicPPL.tilde_assume!!(
        context::AbstractContext,
        right::DynamicPPL.Submodel,
        vn::VarName,
        vi::AbstractVarInfo
    )

Evaluate the submodel with the given context.
"""
function tilde_assume!!(
    context::AbstractContext, right::DynamicPPL.Submodel, vn::VarName, vi::AbstractVarInfo
)
    return _evaluate!!(right, vi, context, vn)
end

"""
    tilde_observe!!(
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
function tilde_observe!!(
    ::DefaultContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end
function tilde_observe!!(
    ::AbstractContext,
    ::DynamicPPL.Submodel,
    left,
    vn::Union{VarName,Nothing},
    ::AbstractVarInfo,
)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is observed"))
end
