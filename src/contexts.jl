"""
    tilde_assume!!(context::Context, dist::Distribution, vn::VarName, template, vi::AbstractVarInfo)

Obtain a latent value from the context and accumulate its outputs.

Return `(x, vi)`, where `x` is the model-space value and `vi` contains the updated
accumulators. `template` describes the enclosing variable's storage. Extend
[`init`](@ref) for custom value selection, or accumulator methods for custom output handling.
"""
function tilde_assume!! end
