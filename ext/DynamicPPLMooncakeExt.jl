module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake:
    Mooncake,
    Dual,
    NoTangent,
    prepare_derivative_cache,
    prepare_gradient_cache,
    primal,
    tangent,
    value_and_derivative!!,
    value_and_gradient!!

# These are purely optimisations (although quite significant ones sometimes, especially for
# _get_range_and_linked).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(is_transformed),Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(DynamicPPL._get_range_and_linked),Vararg
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Base.haskey),DynamicPPL.VarInfo,DynamicPPL.VarName
}

using DynamicPPL: @model, LinkAll, LogDensityAt, getlogjoint_internal, LogDensityFunction
using ADTypes: AutoMooncake, AutoMooncakeForward
using Distributions: Normal, InverseGamma, Beta
using PrecompileTools: @setup_workload, @compile_workload

function _cache_config(::Union{AutoMooncake{Nothing},AutoMooncakeForward{Nothing}})
    return Mooncake.Config(; friendly_tangents=false)
end
function _cache_config(adtype::Union{AutoMooncake,AutoMooncakeForward})
    config = adtype.config
    return Mooncake.Config(;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
        friendly_tangents=false,
    )
end

# LogDensityAt is a constant w.r.t. differentiation; NoTangent avoids tangent allocation.
Mooncake.tangent_type(::Type{<:DynamicPPL.LogDensityAt}) = NoTangent

function DynamicPPL._prepare_gradient(
    adtype::AutoMooncake,
    x::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = LogDensityAt(model, getlogdensity, varname_ranges, transform_strategy, accs)
    return prepare_gradient_cache(f, x; config=_cache_config(adtype))
end

function DynamicPPL._prepare_gradient(
    adtype::AutoMooncakeForward,
    x::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = LogDensityAt(model, getlogdensity, varname_ranges, transform_strategy, accs)
    cache = prepare_derivative_cache(f, x; config=_cache_config(adtype))
    return (; cache, dx=similar(x), grad=similar(x))
end

function DynamicPPL._value_and_gradient(
    ::AutoMooncake,
    prep,
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = LogDensityAt(model, getlogdensity, varname_ranges, transform_strategy, accs)
    value, (_, grad) = value_and_gradient!!(prep, f, params; args_to_zero=(false, true))
    return value, copy(grad)
end

function DynamicPPL._value_and_gradient(
    ::AutoMooncakeForward,
    prep,
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = LogDensityAt(model, getlogdensity, varname_ranges, transform_strategy, accs)
    (; cache, dx, grad) = prep
    value = zero(eltype(grad))
    fill!(dx, zero(eltype(dx)))
    @inbounds for i in eachindex(grad, dx)
        dx[i] = one(eltype(dx))
        result = value_and_derivative!!(cache, Dual(f, NoTangent()), Dual(params, dx))
        value = primal(result)
        grad[i] = tangent(result)
        dx[i] = zero(eltype(dx))
    end
    return value, copy(grad)
end

@setup_workload begin
    @compile_workload begin
        for adtype in (AutoMooncake(),)
            for dist in (Normal(), InverseGamma(2, 3), Beta(2, 2))
                @model f() = x ~ dist
                ldf = LogDensityFunction(f(), getlogjoint_internal, LinkAll(); adtype)
                DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, [0.5])
            end
        end
    end
end

end # module
