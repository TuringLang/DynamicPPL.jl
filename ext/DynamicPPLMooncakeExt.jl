module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake: Mooncake

# These are purely optimisations (although quite significant ones sometimes, especially for
# _get_range_and_linked).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_range_and_transform),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Base.haskey),DynamicPPL.VarInfo,DynamicPPL.VarName
}

using DynamicPPL: @model, LinkAll, getlogjoint_internal, LogDensityFunction
using ADTypes: AutoMooncake
import DifferentiationInterface
using Distributions: Normal, InverseGamma, Beta
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        for dist in (Normal(), InverseGamma(2, 3), Beta(2, 2))
            @model f() = x ~ dist
            ldf = LogDensityFunction(
                f(), getlogjoint_internal, LinkAll(); adtype=AutoMooncake()
            )
            DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, [0.5])
        end
    end
end

end # module
