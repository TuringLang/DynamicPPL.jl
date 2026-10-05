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

# Dense scalar overlays need no per-element reverse program for binding metadata.
# Keep nested/block bindings, custom arrays and other numeric types on the generic path.
const ScalarArgumentBinding{T<:Base.IEEEFloat} = Union{
    DynamicPPL.ModelValue{DynamicPPL.ArgumentCondition,T},
    DynamicPPL.ModelValue{DynamicPPL.Condition,T},
    DynamicPPL.ModelValue{DynamicPPL.Fix,T},
}
const DenseArgumentBindings{T} = DynamicPPL.VarNamedTuples.PartialArray{
    B,1,Vector{B},Vector{Bool}
} where {B<:ScalarArgumentBinding{T}}
const ArgumentName = Union{Nothing,DynamicPPL.VarName{S,AbstractPPL.Iden} where {S}}
@static if isdefined(Mooncake, :ReverseMode)
    Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{
        typeof(DynamicPPL._model_argument_value),
        DenseArgumentBindings{T},
        Vector{T},
        ArgumentName,
    } where {T<:Base.IEEEFloat}
else
    Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
        typeof(DynamicPPL._model_argument_value),
        DenseArgumentBindings{T},
        Vector{T},
        ArgumentName,
    } where {T<:Base.IEEEFloat}
end
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(DynamicPPL._model_argument_value)},
    values::Mooncake.CoDual{<:DenseArgumentBindings{T}},
    template::Mooncake.CoDual{Vector{T}},
    vn::Mooncake.CoDual{<:ArgumentName},
) where {T<:Base.IEEEFloat}
    bindings = Mooncake.primal(values)
    original, dtemplate = Mooncake.arrayify(template)
    output = Mooncake.zero_fcodual(
        DynamicPPL._model_argument_value(bindings, original, Mooncake.primal(vn))
    )
    dy = Mooncake.tangent(output)
    dvalues = Mooncake.tangent(values).data.data
    # Each output is either a bound scalar or a surviving template element.
    # Preparation may resize the template; removed elements have zero cotangent.
    function overlay_pullback!!(::Mooncake.NoRData)
        for i in eachindex(dy)
            if bindings.mask[i]
                dvalues[i] = Mooncake.increment!!(
                    dvalues[i], Mooncake.Tangent((value=dy[i],))
                )
            elseif i <= length(dtemplate)
                dtemplate[i] += dy[i]
            end
        end
        return ntuple(_ -> Mooncake.NoRData(), 4)
    end
    return output, overlay_pullback!!
end

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._argument_may_need_adapter),Type
}

# Ownership validation only inspects identity and returns no numerical result.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._check_argument_key_storage),Any,Any
}

# Reconstruction support depends only on types and methods.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._argument_reconstructible),Any,NamedTuple
}

# Storage type selection returns only type metadata; copying payloads stays differentiable.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL.VarNamedTuples._concretised_eltype),
    DynamicPPL.VarNamedTuples.PartialArray,
}

# Logging has no numerical result, even when storage is constructed during evaluation.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL.VarNamedTuples._warn_growable_array_creation),Any
}

# Role queries return only discrete tags, never bound values.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_argument_role),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_model_role),Vararg
}

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
