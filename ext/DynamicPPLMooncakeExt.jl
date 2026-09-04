module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using AbstractPPL: AbstractPPL
using Mooncake: Mooncake

Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    DynamicPPL._StanDifferentiableFunction,<:AbstractArray{<:Real}
}

function Mooncake.rrule!!(
    f::Mooncake.CoDual{<:DynamicPPL._StanDifferentiableFunction},
    x::Mooncake.CoDual{<:AbstractArray{<:Real}},
)
    primal_f = Mooncake.primal(f)
    primal_x, input_fdata = Mooncake.arrayify(x)
    output = Mooncake.zero_fcodual(primal_f(primal_x))

    function pullback_array!!(dy::Mooncake.NoRData)
        _, (dx,) = DynamicPPL._stan_value_and_pullback(primal_f, primal_x, (output.dx,))
        input_fdata .+= dx
        return Mooncake.NoRData(), dy
    end

    function pullback_scalar!!(dy::Number)
        _, (dx,) = DynamicPPL._stan_value_and_pullback(primal_f, primal_x, (dy,))
        input_fdata .+= dx
        return Mooncake.NoRData(), Mooncake.NoRData()
    end

    pullback = if Mooncake.primal(output) isa Number
        pullback_scalar!!
    elseif Mooncake.primal(output) isa AbstractArray
        pullback_array!!
    else
        error("unsupported output type $(typeof(Mooncake.primal(output)))")
    end
    return output, pullback
end

# These are purely optimisations (although quite significant ones sometimes, especially for
# _get_range_and_transform).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_range_and_transform),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Base.haskey),DynamicPPL.VarInfo,DynamicPPL.VarName
}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{
    typeof(DynamicPPL.to_distribution),AbstractString
}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(DynamicPPL.to_distribution),AbstractString
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
