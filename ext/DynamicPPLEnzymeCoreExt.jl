module DynamicPPLEnzymeCoreExt

using DynamicPPL: DynamicPPL
using EnzymeCore

# Mark the following function signatures as having 0 derivative. The `nothing` return value
# is not significant, Enzyme only checks whether such a method exists, and never runs it.
@inline EnzymeCore.EnzymeRules.inactive(::typeof(DynamicPPL.is_transformed), args...) =
    nothing
@inline EnzymeCore.EnzymeRules.inactive(
    ::typeof(DynamicPPL._get_range_and_linked), args...
) = nothing
@inline EnzymeCore.EnzymeRules.inactive(
    ::typeof(Base.haskey), ::DynamicPPL.NTVarInfo, ::DynamicPPL.VarName
) = nothing

end
