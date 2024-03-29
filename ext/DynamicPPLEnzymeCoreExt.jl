module DynamicPPLEnzymeCoreExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using EnzymeCore
else
    using ..DynamicPPL: DynamicPPL
    using ..EnzymeCore
end

@inline EnzymeCore.EnzymeRules.inactive_type(::Type{<:DynamicPPL.SamplingContext}) = true

end
