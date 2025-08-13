module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(istrans),Vararg}

end # module
