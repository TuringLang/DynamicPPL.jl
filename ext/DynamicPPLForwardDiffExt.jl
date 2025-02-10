module DynamicPPLForwardDiffExt

using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems
using ForwardDiff

getchunksize(::ADTypes.AutoForwardDiff{chunk}) where {chunk} = chunk

standardtag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
standardtag(::ADTypes.AutoForwardDiff) = false

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
