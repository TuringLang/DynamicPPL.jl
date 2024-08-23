module DynamicPPLTapirExt

if isdefined(Base, :get_extension)
    using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using Tapir
else
    using ..DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ..Tapir
end

function DynamicPPL.setmodel(
    f::LogDensityProblemsAD.ADGradientWrapper,
    model::DynamicPPL.Model,
    adtype::ADTypes.AutoTapir,
)
    if !hasfield(typeof(f), :rule)
        @warn "ADGradientWrapper does not have a `rule` field. Please check Tapir version. It is also possible that `adtype` mismatch `ADGradientWrapper` type."
        @warn "Using default rule."
        return LogDensityProblemsAD.ADgradient(
            Val(:Tapir),
            DynamicPPL.setmodel(LogDensityProblemsAD.parent(f), model);
            safety_on=adtype.safe_mode,
            rule=nothing,
        )
    else
        return LogDensityProblemsAD.ADgradient(
            Val(:Tapir),
            DynamicPPL.setmodel(LogDensityProblemsAD.parent(f), model);
            safety_on=adtype.safe_mode,
            rule=f.rule,
        )
    end
end

end # module
