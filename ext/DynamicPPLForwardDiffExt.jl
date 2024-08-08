module DynamicPPLForwardDiffExt

if isdefined(Base, :get_extension)
    using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ForwardDiff
else
    using ..DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems, LogDensityProblemsAD
    using ..ForwardDiff
end

getchunksize(::ADTypes.AutoForwardDiff{chunk}) where {chunk} = chunk

standardtag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
standardtag(::ADTypes.AutoForwardDiff) = false

function LogDensityProblemsAD.ADgradient(
    ad::ADTypes.AutoForwardDiff, ℓ::DynamicPPL.LogDensityFunction
)
    θ = DynamicPPL.getparams(ℓ)
    f = Base.Fix1(LogDensityProblems.logdensity, ℓ)

    # Define configuration for ForwardDiff.
    tag = if standardtag(ad)
        ForwardDiff.Tag(DynamicPPL.DynamicPPLTag(), eltype(θ))
    else
        ForwardDiff.Tag(f, eltype(θ))
    end
    chunk_size = getchunksize(ad)
    chunk = if chunk_size == 0 || chunk_size === nothing
        ForwardDiff.Chunk(θ)
    else
        ForwardDiff.Chunk(length(θ), chunk_size)
    end

    return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; chunk, tag, x=θ)
end

# Allow Turing tag in gradient etc. calls of the log density function
function ForwardDiff.checktag(
    ::Type{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag,V}},
    ::DynamicPPL.LogDensityFunction,
    ::AbstractArray{W},
) where {V,W}
    return true
end
function ForwardDiff.checktag(
    ::Type{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag,V}},
    ::Base.Fix1{typeof(LogDensityProblems.logdensity),<:DynamicPPL.LogDensityFunction},
    ::AbstractArray{W},
) where {V,W}
    return true
end

end # module
