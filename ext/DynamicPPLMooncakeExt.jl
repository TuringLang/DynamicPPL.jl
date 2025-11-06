module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL.Experimental.get_range_and_linked),Vararg
}

end # module
