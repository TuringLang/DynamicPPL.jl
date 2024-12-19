module DynamicPPLEnzymeCoreExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using EnzymeCore
else
    using ..DynamicPPL: DynamicPPL
    using ..EnzymeCore
end

@inline EnzymeCore.EnzymeRules.inactive_type(::Type{<:DynamicPPL.SamplingContext}) = true

# Mark istrans as having 0 derivative. The `nothing` return value is not significant, Enzyme
# only checks whether such a method exists, and never runs it.
@inline EnzymeCore.EnzymeRules.inactive_noinl(::typeof(DynamicPPL.istrans), args...) =
    nothing

end
