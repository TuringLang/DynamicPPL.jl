"""
    tilde_assume!!(context::Context, dist::Distribution, vn::VarName, template, vi::AbstractVarInfo)

Obtain a latent value from the context and accumulate its outputs.

Return the model-space value and updated `vi`. The template describes the enclosing
variable's storage. Extend `init` for custom value selection, or accumulator methods
for custom output handling.
"""
function tilde_assume!! end
