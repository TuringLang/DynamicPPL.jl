module DynamicPPLEnzymeCoreExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using EnzymeCore
else
    using ..DynamicPPL: DynamicPPL
    using ..EnzymeCore
end

# Mark is_transformed as having 0 derivative. The `nothing` return value is not significant, Enzyme
# only checks whether such a method exists, and never runs it.
@inline EnzymeCore.EnzymeRules.inactive(::typeof(DynamicPPL.is_transformed), args...) =
    nothing

end
