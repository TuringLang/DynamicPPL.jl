"""
    DynamicPPL.tilde_assume!!(
        context::AbstractContext,
        right::Distribution,
        vn::VarName,
        template::Any,
        vi::AbstractVarInfo
    )::Tuple{Any,AbstractVarInfo}

Handle latent variables, excluding conditioned and fixed sites. Accumulate their log
probability and return the sampled value and updated `vi`.

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
