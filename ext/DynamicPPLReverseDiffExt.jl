module DynamicPPLReverseDiffExt

using DynamicPPL
using ReverseDiff

@inline DynamicPPL.maybe_view_ad(vect::ReverseDiff.TrackedArray, range) =
    getindex(vect, range)

end # module
