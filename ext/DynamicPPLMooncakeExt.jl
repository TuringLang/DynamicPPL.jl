module DynamicPPLMooncakeExt

__precompile__(false)

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake
import Mooncake: set_to_zero!!
using Mooncake: NoTangent, Tangent, MutableTangent, NoCache, set_to_zero_internal!!

# This is purely an optimisation.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(istrans),Vararg}

"""
Check if a tangent has the expected structure for a given type.
"""
function has_expected_structure(
    x, expected_type::Type{<:Union{Tangent,MutableTangent}}, expected_fields
)
    x isa expected_type || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    if expected_fields isa Tuple
        # Exact match required
        propertynames(fields) == expected_fields || return false
    else
        # All expected fields must be present
        all(f in propertynames(fields) for f in expected_fields) || return false
    end

    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.LogDensityFunction
"""
function is_dppl_ldf_tangent(x)
    has_expected_structure(x, Tangent, (:model, :varinfo, :context, :adtype, :prep)) ||
        return false

    fields = x.fields
    is_dppl_varinfo_tangent(fields.varinfo) || return false
    is_dppl_model_tangent(fields.model) || return false

    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.VarInfo
"""
function is_dppl_varinfo_tangent(x)
    return has_expected_structure(x, Tangent, (:metadata, :logp, :num_produce))
end

"""
Check if a tangent corresponds to a DynamicPPL.Model
"""
function is_dppl_model_tangent(x)
    return has_expected_structure(x, Tangent, (:f, :args, :defaults, :context))
end

"""
Check if a MutableTangent corresponds to DynamicPPL.Metadata
"""
function is_dppl_metadata_tangent(x)
    return has_expected_structure(
        x, MutableTangent, (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags)
    )
end

"""
Check if a model function tangent represents a closure.
"""
function is_closure_model(model_f_tangent)
    model_f_tangent isa MutableTangent && return true

    if model_f_tangent isa Tangent && hasfield(typeof(model_f_tangent), :fields)
        # Check if any field is a MutableTangent with PossiblyUninitTangent{Any}
        for (_, fval) in pairs(model_f_tangent.fields)
            if fval isa MutableTangent &&
                hasfield(typeof(fval), :fields) &&
                hasfield(typeof(fval.fields), :contents) &&
                fval.fields.contents isa Mooncake.PossiblyUninitTangent{Any}
                return true
            end
        end
    end

    return false
end

function Mooncake.set_to_zero!!(x)
    # Check for DynamicPPL types and use NoCache for better performance
    if is_dppl_ldf_tangent(x)
        # Special handling for LogDensityFunction to detect closures
        model_f_tangent = x.fields.model.fields.f
        cache = is_closure_model(model_f_tangent) ? IdDict{Any,Bool}() : NoCache()
        return set_to_zero_internal!!(cache, x)
    elseif is_dppl_varinfo_tangent(x) ||
        is_dppl_model_tangent(x) ||
        is_dppl_metadata_tangent(x)
        # These types can always use NoCache
        return set_to_zero_internal!!(NoCache(), x)
    else
        # Use the original implementation with IdDict for all other types
        return set_to_zero_internal!!(IdDict{Any,Bool}(), x)
    end
end

end # module
