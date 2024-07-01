module DynamicPPLReverseDiffExt

if isdefined(Base, :get_extension)
    using DynamicPPL:
        Accessors, ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ReverseDiff
else
    using ..DynamicPPL: Accessors, ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ..ReverseDiff
end

function LogDensityProblemsAD.ADgradient(
    ad::ADTypes.AutoReverseDiff, ℓ::DynamicPPL.LogDensityFunction
)
    return LogDensityProblemsAD.ADgradient(
        Val(:ReverseDiff),
        ℓ;
        compile=Val(ad.compile),
        # `getparams` can return `Vector{Real}`, in which case, `ReverseDiff` will initialize the gradients to Integer 0
        # because at https://github.com/JuliaDiff/ReverseDiff.jl/blob/c982cde5494fc166965a9d04691f390d9e3073fd/src/tracked.jl#L473
        # `zero(D)` will return 0 when D is Real.
        # here we use `identity` to possibly concretize the type to `Vector{Float64}` in the case of `Vector{Real}`.
        x=map(identity, DynamicPPL.getparams(ℓ)),
    )
end

function DynamicPPL.setmodel(f::LogDensityProblemsAD.ReverseDiffLogDensity{L,Nothing}, model::DynamicPPL.Model) where {L}
    return Accessors.@set f.ℓ = setmodel(f.ℓ, model)
end

function DynamicPPL.setmodel(f::LogDensityProblemsAD.ReverseDiffLogDensity{L,C}, model::DynamicPPL.Model) where {L,C}
    new_f = LogDensityProblemsAD.ADGradient(Val(:ReverseDiff), f.ℓ; compile=Val(true)) # TODO: without a input, can get error
    return Accessors.@set new_f.ℓ = setmodel(f.ℓ, model)
end

end # module
