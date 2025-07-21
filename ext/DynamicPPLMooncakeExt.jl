module DynamicPPLMooncakeExt

__precompile__(false)

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake
import Mooncake: set_to_zero!!
using Mooncake: NoTangent, Tangent, MutableTangent, NoCache, set_to_zero_internal!!

# This is purely an optimisation.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(istrans),Vararg}

# =======================
# `Mooncake.set_to_zero!!` optimization with `NoCache` 
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

function is_dppl_ldf_tangent(x)
    has_expected_structure(x, Tangent, (:model, :varinfo, :context, :adtype, :prep)) ||
        return false

    fields = x.fields
    is_dppl_varinfo_tangent(fields.varinfo) || return false
    is_dppl_model_tangent(fields.model) || return false

    return true
end

function is_dppl_varinfo_tangent(x)
    return has_expected_structure(x, Tangent, (:metadata, :logp, :num_produce))
end

function is_dppl_model_tangent(x)
    return has_expected_structure(x, Tangent, (:f, :args, :defaults, :context))
end

function is_dppl_metadata_tangent(x)
    # Metadata can be either:
    # 1. A MutableTangent with the expected fields (for single metadata)
    # 2. A NamedTuple where each value is a Tangent with the expected fields

    # Check for MutableTangent case
    if has_expected_structure(
        x, MutableTangent, (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags)
    )
        return true
    end

    # Check for NamedTuple case (multiple metadata)
    if x isa NamedTuple
        # Each value should be a Tangent with metadata fields
        for var_metadata in values(x)
            if !has_expected_structure(
                var_metadata,
                Tangent,
                (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags),
            )
                return false
            end
        end
        return true
    end

    return false
end

"""
    has_circular_reference_risk(x)

Main entry point for detecting circular reference patterns that require caching.
"""
function has_circular_reference_risk(x)
    if is_dppl_ldf_tangent(x)
        # Check model function for closure patterns with circular refs
        model_f = x.fields.model.fields.f
        return is_closure_with_circular_refs(model_f)
    elseif is_dppl_varinfo_tangent(x)
        return check_for_ref_fields(x)
    end

    # For unknown types, do a shallow check for PossiblyUninitTangent{Any}
    return x isa Mooncake.PossiblyUninitTangent{Any}
end

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

function check_for_ref_fields(x)
    # Check if it's a VarInfo tangent
    is_dppl_varinfo_tangent(x) || return false

    # Check if the logp field contains a Ref-like tangent structure
    hasfield(typeof(x.fields), :logp) || return false
    logp_tangent = x.fields.logp

    # Ref types in tangents often appear as MutableTangent with circular references
    return logp_tangent isa MutableTangent
end

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

function Mooncake.set_to_zero!!(x)
    cache = determine_cache_strategy(x)
    return set_to_zero_internal!!(cache, x)
end

end # module
