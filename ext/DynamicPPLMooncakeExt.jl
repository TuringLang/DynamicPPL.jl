module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using AbstractPPL: AbstractPPL
using Mooncake: Mooncake

# These are purely optimisations (although quite significant ones sometimes, especially for
# _get_range_and_linked).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_range_and_linked),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Base.haskey),DynamicPPL.VarInfo,DynamicPPL.VarName
}

using DynamicPPL: @model, LinkAll, getlogjoint_internal, LogDensityFunction
using ADTypes: AutoMooncake
using Distributions: Normal, InverseGamma, Beta
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        # Julia does not guarantee transitive extensions are loaded while this
        # extension precompiles, so skip the workload unless Mooncake's
        # AbstractPPL methods are already available.
        if !isnothing(Base.get_extension(AbstractPPL, :AbstractPPLMooncakeExt))
            for dist in (Normal(), InverseGamma(2, 3), Beta(2, 2))
                @model f() = x ~ dist
                ldf = LogDensityFunction(
                    f(), getlogjoint_internal, LinkAll(); adtype=AutoMooncake()
                )
                DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, [0.5])
            end
        end
    end
end

end # module
