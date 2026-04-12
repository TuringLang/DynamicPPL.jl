module DynamicPPLForwardDiffExt

using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems
using ForwardDiff

# check if the AD type already has a tag
use_dynamicppl_tag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
use_dynamicppl_tag(::ADTypes.AutoForwardDiff) = false

function DynamicPPL.tweak_adtype(
    ad::ADTypes.AutoForwardDiff{chunk_size}, ::DynamicPPL.Model, params::AbstractVector
) where {chunk_size}
    # Use DynamicPPL tag to improve stack traces
    # https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
    tag = if use_dynamicppl_tag(ad)
        ForwardDiff.Tag(DynamicPPL.DynamicPPLTag(), eltype(params))
    else
        ad.tag
    end

    # Optimise chunk size according to size of model
    chunk = if chunk_size == 0 || chunk_size === nothing
        ForwardDiff.Chunk(params)
    else
        ForwardDiff.Chunk(length(params), chunk_size)
    end

    return ADTypes.AutoForwardDiff(; chunksize=ForwardDiff.chunksize(chunk), tag=tag)
end

function DynamicPPL._prepare_gradient(
    adtype::ADTypes.AutoForwardDiff{chunk_size},
    x::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
) where {chunk_size}
    f = DynamicPPL.LogDensityAt(
        model, getlogdensity, varname_ranges, transform_strategy, accs
    )
    chunk = if chunk_size == 0 || chunk_size === nothing
        ForwardDiff.Chunk(x)
    else
        ForwardDiff.Chunk(length(x), chunk_size)
    end
    cfg = ForwardDiff.GradientConfig(f, x, chunk, adtype.tag)
    grad = similar(x)
    return (; cfg, grad)
end

function DynamicPPL._value_and_gradient(
    ::ADTypes.AutoForwardDiff,
    prep,
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = DynamicPPL.LogDensityAt(
        model, getlogdensity, varname_ranges, transform_strategy, accs
    )
    # Val{false}() skips tag checking, since our DynamicPPLTag is reused across calls
    # with different LogDensityAt instances.
    ForwardDiff.gradient!(prep.grad, f, params, prep.cfg, Val{false}())
    # gradient!(::AbstractArray, ...) doesn't return the value, so evaluate separately.
    value = f(params)
    return value, copy(prep.grad)
end

end # module
