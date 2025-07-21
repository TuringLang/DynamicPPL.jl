module DynamicPPLMooncakeExt

__precompile__(false)

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake
import Mooncake: set_to_zero!!
using Mooncake: NoTangent, Tangent, MutableTangent, NoCache, set_to_zero_internal!!

# This is purely an optimisation.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(istrans),Vararg}

# =======================
# Cache Strategy System
# =======================

"""
    determine_cache_strategy(x)

Determines the appropriate caching strategy for a given tangent.
Returns either `NoCache()` for safe types or `IdDict{Any,Bool}()` for types with circular reference risk.
"""
function determine_cache_strategy(x)
    # Fast path: check for known circular reference patterns
    has_circular_reference_risk(x) && return IdDict{Any,Bool}()

    # Check for DynamicPPL types that can safely use NoCache
    is_safe_dppl_type(x) && return NoCache()

    # Special case: LogDensityFunction without problematic patterns can use NoCache
    if is_dppl_ldf_tangent(x)
        return NoCache()
    end

    # Default to safe caching for unknown types
    return IdDict{Any,Bool}()
end

# =======================
# Type Recognition
# =======================

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

# =======================
# Circular Reference Detection
# =======================

"""
    has_circular_reference_risk(x)

Main entry point for detecting circular reference patterns that require caching.
Optimized for performance with targeted checks instead of recursive traversal.
"""
function has_circular_reference_risk(x)
    # Type-specific targeted checks only
    if is_dppl_ldf_tangent(x)
        # Check model function for closure patterns with circular refs
        model_f = x.fields.model.fields.f
        return is_closure_with_circular_refs(model_f)
    elseif is_dppl_varinfo_tangent(x)
        # Check for Ref fields in VarInfo
        return check_for_ref_fields(x)
    end

    # For unknown types, do a shallow check for PossiblyUninitTangent{Any}
    return x isa Mooncake.PossiblyUninitTangent{Any}
end

"""
Check if a tangent represents a closure with circular reference patterns.
Only returns true for actual problematic patterns, not all MutableTangents.
"""
function is_closure_with_circular_refs(x)
    # Check if MutableTangent contains PossiblyUninitTangent{Any}
    if x isa MutableTangent && hasfield(typeof(x), :fields)
        hasfield(typeof(x.fields), :contents) &&
            x.fields.contents isa Mooncake.PossiblyUninitTangent{Any} &&
            return true
    end

    # For Tangent, only check immediate fields (no deep recursion)
    if x isa Tangent && hasfield(typeof(x), :fields)
        for (_, fval) in pairs(x.fields)
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

"""
Check if a VarInfo tangent has Ref fields that need caching.
"""
function check_for_ref_fields(x)
    # Check if it's a VarInfo tangent
    is_dppl_varinfo_tangent(x) || return false

    # Check if the logp field contains a Ref-like tangent structure
    hasfield(typeof(x.fields), :logp) || return false
    logp_tangent = x.fields.logp

    # Ref types in tangents often appear as MutableTangent with circular references
    return logp_tangent isa MutableTangent
end

"""
Check if a tangent is a safe DynamicPPL type that can use NoCache.
"""
function is_safe_dppl_type(x)
    # Metadata is always safe
    is_dppl_metadata_tangent(x) && return true

    # Model tangents without closures are safe
    if is_dppl_model_tangent(x)
        !is_closure_with_circular_refs(x.fields.f) && return true
    end

    # VarInfo without Ref fields is safe
    if is_dppl_varinfo_tangent(x)
        !check_for_ref_fields(x) && return true
    end

    return false
end

# =======================
# Main Entry Point
# =======================

function Mooncake.set_to_zero!!(x)
    cache = determine_cache_strategy(x)
    return set_to_zero_internal!!(cache, x)
end

end # module
