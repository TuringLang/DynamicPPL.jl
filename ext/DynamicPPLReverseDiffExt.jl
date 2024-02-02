module DynamicPPLReverseDiffExt

if isdefined(Base, :get_extension)
    using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ReverseDiff
else
    using ..DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ..ReverseDiff
end

function LogDensityProblemsAD.ADgradient(
    ad::ADTypes.AutoReverseDiff, ℓ::DynamicPPL.LogDensityFunction
)
    return LogDensityProblemsAD.ADgradient(
        Val(:ReverseDiff), ℓ; compile=Val(ad.compile), x=identity.(DynamicPPL.getparams(ℓ))
    )
end

end # module
