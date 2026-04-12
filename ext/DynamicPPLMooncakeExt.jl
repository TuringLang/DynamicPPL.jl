module DynamicPPLMooncakeExt

using DynamicPPL: DynamicPPL, is_transformed
using Mooncake:
    Mooncake,
    Dual,
    NoTangent,
    primal,
    prepare_derivative_cache,
    prepare_gradient_cache,
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

_config(::Union{AutoMooncake{Nothing},AutoMooncakeForward{Nothing}}) = Mooncake.Config()
_config(adtype::Union{AutoMooncake,AutoMooncakeForward}) = adtype.config
function _cache_config(adtype::Union{AutoMooncake,AutoMooncakeForward})
    config = _config(adtype)
    # `friendly_tangents=true` rewrites tangent types into named structs at tape-build time,
    # which is incompatible with a reusable cache (the cached tape would be tied to the
    # original tangent struct layout). Force it off so the cache stays valid across calls.
    return Mooncake.Config(;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
        friendly_tangents=false,
    )
end

# LogDensityAt is the function being differentiated through, not a quantity being
# differentiated with respect to. Declaring NoTangent here tells Mooncake to treat it as
# a constant, which is correct and avoids unnecessary tangent allocation.
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
    return (;
        cache=prepare_derivative_cache(f, x; config=_cache_config(adtype)),
        dx=similar(x),
        grad=similar(x),
    )
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
    dx = prep.dx
    grad = prep.grad

    if isempty(grad)
        # Zero-dimensional parameter vector: evaluate primal only. Use a zero tangent so
        # value_and_derivative!! returns the function value without computing any derivative.
        fill!(dx, zero(eltype(dx)))
        value = primal(
            value_and_derivative!!(prep.cache, Dual(f, NoTangent()), Dual(params, dx))
        )
        return value, copy(grad)
    end

    # Standard column-by-column forward-mode sweep: set dx to each unit vector in turn,
    # compute the directional derivative, and accumulate into grad.
    # Each iteration resets dx[i] to zero after use, so dx is all-zeros at loop exit.
    value = zero(eltype(grad))
    @inbounds for i in eachindex(grad, dx)
        dx[i] = oneunit(eltype(dx))
        dual_value = value_and_derivative!!(
            prep.cache, Dual(f, NoTangent()), Dual(params, dx)
        )
        value = primal(dual_value)
        grad[i] = tangent(dual_value)
        dx[i] = zero(eltype(dx))
    end
    return value, copy(grad)
end

@setup_workload begin
    @compile_workload begin
        for adtype in (AutoMooncake(), AutoMooncakeForward())
            for dist in (Normal(), InverseGamma(2, 3), Beta(2, 2))
                @model f() = x ~ dist
                ldf = LogDensityFunction(f(), getlogjoint_internal, LinkAll(); adtype)
                DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, [0.5])
            end
        end
    end
end

end # module
