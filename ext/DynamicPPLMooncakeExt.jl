module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(istrans),Vararg}

@static if isdefined(Mooncake, :requires_cache)
    import Mooncake: requires_cache

    function Mooncake.requires_cache(::Type{<:DynamicPPL.Metadata})
        return Val(false)
    end

    function Mooncake.requires_cache(::Type{<:DynamicPPL.TypedVarInfo})
        return Val(false)
    end

    function Mooncake.requires_cache(::Type{<:DynamicPPL.Model})
        # Model has f (function/closure), args, defaults, context
        # Closures can have circular references
        return Val(false)
    end

    function Mooncake.requires_cache(::Type{<:DynamicPPL.LogDensityFunction})
        return Val(false)
    end

    function Mooncake.requires_cache(::Type{<:DynamicPPL.AbstractContext})
        return Val(false)
    end
end

end # module
