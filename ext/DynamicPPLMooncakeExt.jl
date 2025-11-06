module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed, get_range_and_linked
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(get_range_and_linked),Vararg}

end # module
