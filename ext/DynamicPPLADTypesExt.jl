module DynamicPPLADTypesExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL, LogDensityProblemsAD
    using ADTypes
else
    using ..DynamicPPL: DynamicPPL, LogDensityProblemsAD
    using ..ADTypes
end

getADType(spl::DynamicPPL.Sampler) = getADType(spl.alg)
getADType(::DynamicPPL.SampleFromPrior) = ADTypes.AutoForwardDiff(; chunksize=0)
getADType(ctx::DynamicPPL.SamplingContext) = getADType(ctx.sampler)
getADType(ctx::DynamicPPL.AbstractContext) = getADType(DynamicPPL.NodeTrait(ctx), ctx)

function getADType(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext)
    return ADTypes.AutoForwardDiff(; chunksize=0)
end
function getADType(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext)
    return getADType(DynamicPPL.childcontext(ctx))
end

function LogDensityProblemsAD.ADgradient(ℓ::DynamicPPL.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(getADType(ℓ.context), ℓ)
end

function LogDensityProblemsAD.ADgradient(ad::AutoReverseDiff, ℓ::DynamicPPL.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(
        Val(:ReverseDiff), ℓ; compile=Val(ad.compile), x=DynamicPPL.getparams(ℓ)
    )
end

end # module
