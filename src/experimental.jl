module Experimental

using DynamicPPL: DynamicPPL

"""
    is_suitable_varinfo(model::Model, context::AbstractContext, varinfo::AbstractVarInfo; kwargs...)

Check if the `model` supports evaluation using the provided `context` and `varinfo`.

!!! warning
    Loading JET.jl is required before calling this function.

# Arguments
- `model`: The model to to verify the support for.
- `context`: The context to use for the model evaluation.
- `varinfo`: The varinfo to verify the support for.

# Keyword Arguments
- `only_ddpl`: If `true`, only consider error reports occuring in the tilde pipeline. Default: `true`.

# Returns
- `issuccess`: `true` if the model supports the varinfo, otherwise `false`.
- `report`: The result of `report_call` from JET.jl.
"""
function is_suitable_varinfo end

# Internal hook for JET.jl to overload.
function _determine_varinfo_jet end

"""
    determine_suitable_varinfo(model[, context]; verbose::Bool=false, only_ddpl::Bool=true)

Return a suitable varinfo for the given `model`.

See also: [`DynamicPPL.is_suitable_varinfo`](@ref).

!!! warning
    For full functionality, this requires JET.jl to be loaded.
    If JET.jl is not loaded, this function will assume the model is compatible with typed varinfo.

# Arguments
- `model`: The model for which to determine the varinfo.
- `context`: The context to use for the model evaluation. Default: `SamplingContext()`.

# Keyword Arguments
- `only_ddpl`: If `true`, only consider error reports within DynamicPPL.jl.
"""
function determine_suitable_varinfo(
    model::DynamicPPL.Model,
    context::DynamicPPL.AbstractContext=DynamicPPL.SamplingContext();
    only_ddpl::Bool=true,
)
    # If JET.jl has been loaded, and thus `determine_varinfo` has been defined, we use that.
    return if Base.get_extension(DynamicPPL, :DynamicPPLJETExt) !== nothing
        _determine_varinfo_jet(model, context; only_ddpl)
    else
        # Warn the user.
        @warn "JET.jl is not loaded. Assumes the model is compatible with typed varinfo."
        # Otherwise, we use the, possibly incorrect, default typed varinfo (to stay backwards compat).
        DynamicPPL.typed_varinfo(model, context)
    end
end

end
