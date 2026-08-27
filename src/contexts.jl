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
    ::AbstractContext, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    return DynamicPPL.map_accumulator!!(
        acc -> store_colon_eq!!(acc, vn, right, template),
        vi,
        Val(DynamicPPL.RAW_VALUE_ACCNAME),
    )
end
