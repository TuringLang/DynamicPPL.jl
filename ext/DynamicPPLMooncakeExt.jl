module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}

end # module
