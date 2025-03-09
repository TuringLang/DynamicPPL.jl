module DynamicPPLForwardDiffExt

using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems
using ForwardDiff

# check if the AD type already has a tag
use_dynamicppl_tag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
use_dynamicppl_tag(::ADTypes.AutoForwardDiff) = false

function DynamicPPL.tweak_adtype(
    ad::ADTypes.AutoForwardDiff{chunk_size},
    ::DynamicPPL.Model,
    vi::DynamicPPL.AbstractVarInfo,
    ::DynamicPPL.AbstractContext,
) where {chunk_size}
    params = vi[:]

    # Use DynamicPPL tag to improve stack traces
    # https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
    # NOTE: DifferentiationInterface disables tag checking if the
    # tag inside the AutoForwardDiff type is not nothing. See
    # https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/1df562180bdcc3e91c885aa5f4162a0be2ced850/DifferentiationInterface/ext/DifferentiationInterfaceForwardDiffExt/onearg.jl#L338-L350.
    # So we don't currently need to override ForwardDiff.checktag as well.
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

end # module
