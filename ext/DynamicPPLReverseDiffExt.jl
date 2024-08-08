module DynamicPPLReverseDiffExt

if isdefined(Base, :get_extension)
    using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ReverseDiff
else
    using ..DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ..ReverseDiff
end

function LogDensityProblemsAD.ADgradient(
    ad::ADTypes.AutoReverseDiff{Tcompile}, ℓ::DynamicPPL.LogDensityFunction
) where {Tcompile}
    return LogDensityProblemsAD.ADgradient(
        Val(:ReverseDiff),
        ℓ;
        compile=Val(Tcompile)
        # `getparams` can return `Vector{Real}`, in which case, `ReverseDiff` will initialize the gradients to Integer 0
        # because at https://github.com/JuliaDiff/ReverseDiff.jl/blob/c982cde5494fc166965a9d04691f390d9e3073fd/src/tracked.jl#L473
        # `zero(D)` will return 0 when D is Real.
        # here we use `identity` to possibly concretize the type to `Vector{Float64}` in the case of `Vector{Real}`.
        x=map(identity, DynamicPPL.getparams(ℓ)),
    )
end

end # module
