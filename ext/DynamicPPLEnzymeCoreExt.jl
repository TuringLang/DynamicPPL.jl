module DynamicPPLEnzymeCoreExt

using DynamicPPL: DynamicPPL
using EnzymeCore

# Mark is_transformed as having 0 derivative. The `nothing` return value is not significant, Enzyme
# only checks whether such a method exists, and never runs it.
@inline EnzymeCore.EnzymeRules.inactive(::typeof(DynamicPPL.is_transformed), args...) =
    nothing
# Likewise for get_range_and_linked.
@inline EnzymeCore.EnzymeRules.inactive(
    ::typeof(DynamicPPL.get_range_and_linked), args...
) = nothing

end
