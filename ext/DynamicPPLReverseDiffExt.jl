module DynamicPPLReverseDiffExt

using DynamicPPL
using ReverseDiff

@inline DynamicPPL.maybe_view_ad(vect::ReverseDiff.TrackedArray, range) =
    getindex(vect, range)

# `copy` returns the same non-writable TrackedArray. Keep its scalar tape connections
# while allocating storage that argument reconstruction and latent tildes can update.
function DynamicPPL._copy_model_argument_storage(value::ReverseDiff.TrackedArray)
    return map(identity, value)
end

end # module
