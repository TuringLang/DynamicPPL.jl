module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake: Mooncake

# These are purely optimisations (although quite significant ones sometimes, especially for
# _get_range_and_linked).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_range_and_linked),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Base.haskey),DynamicPPL.NTVarInfo,DynamicPPL.VarName
}

end # module
